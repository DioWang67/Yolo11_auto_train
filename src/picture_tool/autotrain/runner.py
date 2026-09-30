"""Unattended runs: labelled batches dropped in an inbox, carried to a report.

A person drops a labelled folder into the inbox. ``run`` imports every batch
that has finished arriving, trains a challenger on whatever verified labels
no dataset version has covered yet, evaluates it against the champion and
writes the report. It never deploys, exactly like the cycle it drives.

The one thing it adds over ``import-folder`` followed by ``cycle`` is that it
keeps going when the machine does not. CPU training on the line's PC dies at
random mid-epoch, and a crash takes the whole process with it, so the cycle
runs in a child process. When the child dies with a step still RUNNING --- the
mark a crash leaves, as opposed to a FAILED the cycle recorded itself --- the
same cycle is started again, and it resumes the interrupted training instead
of beginning it over.

Scheduling stays outside: run it by hand, or from Task Scheduler. With nothing
new in the inbox and nothing interrupted, a run imports nothing and trains
nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from picture_tool.autotrain import AutoTrainError
from picture_tool.autotrain.config import AutoTrainConfig
from picture_tool.autotrain.dataset_versions import DatasetVersionStore
from picture_tool.autotrain.labeling import verified_samples
from picture_tool.autotrain.orchestrator import (
    BLOCKED,
    COMPLETED,
    CYCLE_STATE_FILENAME,
    RUNNING,
    STEP_FAILED,
    CycleError,
    CycleState,
    run_training_cycle,
)
from picture_tool.autotrain.paths import AutoTrainPaths
from picture_tool.autotrain.service import AutoTrainService
from picture_tool.dataset_manifest_lock import (
    DatasetManifestLockTimeoutError,
    autotrain_store_lock,
)
from picture_tool.workspace_paths import WORKSPACE_ENV_VAR

LOGGER = logging.getLogger(__name__)

PROCESSED_DIRNAME = "processed"
REJECTED_DIRNAME = "rejected"
IMPORT_RESULT_FILENAME = "import_result.json"
#: A cycle left in one of these states is finished, one way or another.
_SETTLED = frozenset({COMPLETED, BLOCKED, STEP_FAILED})
#: Reported for a cycle whose process kept dying until the attempts ran out.
GAVE_UP = "GAVE_UP"

#: Runs ``command`` in ``cwd`` with extra environment, logging to a file;
#: returns the exit code.
Launcher = Callable[[Sequence[str], Path, Path, Mapping[str, str]], int]


@dataclass(frozen=True)
class BatchOutcome:
    """What happened to one dropped folder."""

    name: str
    accepted: bool
    moved_to: str
    summary: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "accepted": self.accepted,
            "moved_to": self.moved_to,
            "summary": dict(self.summary),
        }


@dataclass(frozen=True)
class CycleOutcome:
    """How one cycle this run started or resumed ended."""

    cycle_id: str
    status: str
    attempts: int
    decision: str = ""
    report: str = ""
    resumed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "status": self.status,
            "attempts": self.attempts,
            "decision": self.decision,
            "report": self.report,
            "resumed": self.resumed,
        }


@dataclass(frozen=True)
class RunOutcome:
    """Everything one ``run`` did, for the console and for the run log."""

    inbox: str
    batches: tuple[BatchOutcome, ...] = ()
    waiting: tuple[str, ...] = ()
    held: tuple[str, ...] = ()
    cycles: tuple[CycleOutcome, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """No batch was refused or held, and no cycle ended short of a result."""
        return (
            not self.held
            and all(batch.accepted for batch in self.batches)
            and all(cycle.status in (COMPLETED, BLOCKED) for cycle in self.cycles)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "inbox": self.inbox,
            "batches": [batch.to_dict() for batch in self.batches],
            "waiting": list(self.waiting),
            "held": list(self.held),
            "cycles": [cycle.to_dict() for cycle in self.cycles],
            "notes": list(self.notes),
            "ok": self.ok,
            "deployed": False,
        }


def run(
    config: AutoTrainConfig,
    paths: AutoTrainPaths,
    *,
    launch: Launcher | None = None,
    now: Callable[[], datetime] | None = None,
) -> RunOutcome:
    """Import the inbox, then finish or start whatever cycle is due.

    Only one run at a time: a scheduled run that fires while an earlier one
    is still training would otherwise take that live cycle for an
    interrupted one and resume it a second time, so it refuses at once
    instead of waiting behind a training that may last hours. The OS
    releases the lock if the process dies, so a crash leaves none behind.
    """
    config.require_enabled()
    try:
        with autotrain_store_lock(paths.root, name="runner", timeout_seconds=0):
            return _run_locked(config, paths, launch=launch, now=now)
    except DatasetManifestLockTimeoutError as exc:
        raise AutoTrainError(
            "Another autotrain run is in progress; this one did nothing."
        ) from exc


def _run_locked(
    config: AutoTrainConfig,
    paths: AutoTrainPaths,
    *,
    launch: Launcher | None,
    now: Callable[[], datetime] | None,
) -> RunOutcome:
    clock = now or (lambda: datetime.now(timezone.utc))
    service = AutoTrainService(config, paths)
    inbox = paths.inbox_dir(config.runner.inbox_dir)
    inbox.mkdir(parents=True, exist_ok=True)

    batches, waiting, held = import_inbox(
        service, inbox, settle_seconds=config.runner.settle_seconds, now=clock
    )
    cycles: list[CycleOutcome] = []
    notes = [f"{name}: left in the inbox, not imported: {reason}" for name, reason in held]

    # An interrupted cycle is finished first: its training is paid for up to
    # the last epoch, and abandoning it would leave a TRAINING candidate
    # behind for ever.
    interrupted = find_interrupted_cycle(paths, config)
    if interrupted:
        cycles.append(
            _supervise(config, paths, interrupted, launch=launch, resumed=True)
        )
    if has_untrained_labels(service):
        cycle_id = clock().strftime("cycle_%Y%m%dT%H%M%SZ")
        cycles.append(_supervise(config, paths, cycle_id, launch=launch))
    elif not cycles and not held:
        notes.append(
            "No cycle run: every verified label is already in a dataset version "
            "and no cycle was left interrupted."
        )

    outcome = RunOutcome(
        inbox=str(inbox),
        batches=batches,
        waiting=waiting,
        held=tuple(name for name, _reason in held),
        cycles=tuple(cycles),
        notes=tuple(notes),
    )
    _write_run_log(paths, outcome, clock())
    return outcome


# ---------------------------------------------------------------------------
# The inbox


def pending_batches(inbox: Path) -> list[Path]:
    """Folders waiting in the inbox, oldest name first."""
    if not inbox.is_dir():
        return []
    return sorted(
        entry
        for entry in inbox.iterdir()
        if entry.is_dir()
        and entry.name not in (PROCESSED_DIRNAME, REJECTED_DIRNAME)
        and not entry.name.startswith(".")
    )


def import_inbox(
    service: AutoTrainService,
    inbox: Path,
    *,
    settle_seconds: int,
    now: Callable[[], datetime],
) -> tuple[tuple[BatchOutcome, ...], tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Import every settled batch and file it under processed/ or rejected/.

    Each batch is all or nothing, as ``import-folder`` is. The folder is moved
    whether or not it was accepted, with the import result written beside it,
    so the inbox only ever holds work that has not been looked at yet. The
    pool keeps its own copy of every image, so moving the source is safe.

    A batch the station itself cannot take --- no class contract because no
    champion is deployed yet --- is *held*, not rejected: nothing is wrong
    with its contents, and once the station is ready the next run imports it
    without anyone moving it back.
    """
    outcomes: list[BatchOutcome] = []
    waiting: list[str] = []
    held: list[tuple[str, str]] = []
    for batch in pending_batches(inbox):
        if _still_arriving(batch, settle_seconds):
            waiting.append(batch.name)
            continue
        try:
            summary = service.import_labelled_folder(batch)
        except AutoTrainError as exc:
            held.append((batch.name, str(exc)))
            continue
        accepted = not summary["errors"]
        stamp = now().strftime("%Y%m%dT%H%M%SZ")
        target_root = inbox / (PROCESSED_DIRNAME if accepted else REJECTED_DIRNAME)
        destination = _unused(target_root / f"{stamp}_{batch.name}")
        moved_to = ""
        try:
            target_root.mkdir(parents=True, exist_ok=True)
            shutil.move(str(batch), str(destination))
            (destination / IMPORT_RESULT_FILENAME).write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            moved_to = str(destination)
        except OSError as exc:
            # Left in place, the batch is simply imported again next time,
            # which is harmless: an unchanged folder re-imports as
            # already_verified.
            LOGGER.warning("Could not file batch %s: %s", batch.name, exc)
        outcomes.append(
            BatchOutcome(
                name=batch.name, accepted=accepted, moved_to=moved_to, summary=summary
            )
        )
    return tuple(outcomes), tuple(waiting), tuple(held)


def has_untrained_labels(service: AutoTrainService) -> bool:
    """Whether some verified label is in no dataset version yet.

    Counting new labels rather than new batches is what makes the trigger
    right in every order of events: labels imported through ``labeling
    import``, a batch imported during a run that was busy resuming an older
    cycle, or a batch whose cycle crashed out all still get trained.
    """
    verified = {sample.sample_id for sample in verified_samples(service._pool())}
    if not verified:
        return False
    store = DatasetVersionStore(
        service.paths.dataset_station_root(service.product, service.area),
        product=service.product,
        area=service.area,
    )
    latest = store.latest()
    return latest is None or bool(verified - set(latest.sample_ids))


# ---------------------------------------------------------------------------
# Cycles, and surviving the process dying under them


def find_interrupted_cycle(paths: AutoTrainPaths, config: AutoTrainConfig) -> str:
    """The newest cycle of this station that died with a step still RUNNING."""
    if not paths.cycles_root.is_dir():
        return ""
    found: list[tuple[str, str]] = []
    for directory in paths.cycles_root.iterdir():
        state = _read_state(directory / CYCLE_STATE_FILENAME)
        if (
            state is not None
            and state.product == config.station.product
            and state.area == config.station.area
            and _died_mid_step(state)
        ):
            found.append((state.created_at, state.cycle_id))
    return max(found)[1] if found else ""


def _supervise(
    config: AutoTrainConfig,
    paths: AutoTrainPaths,
    cycle_id: str,
    *,
    launch: Launcher | None,
    resumed: bool = False,
) -> CycleOutcome:
    """Run one cycle in a child process, starting it again if the child dies."""
    command = [sys.executable, "-m", "picture_tool.autotrain.runner", "--cycle-worker", cycle_id]
    if config.source_path is not None:
        command += ["--config", str(config.source_path)]
    execute = launch or _launch_child
    # The child finds the workspace for itself; pinning it to this one keeps
    # a run launched from anywhere from training against some other tree.
    environment = (
        {WORKSPACE_ENV_VAR: str(paths.workspace.workspace_root)}
        if paths.workspace.manifest_path is not None
        else {}
    )
    directory = paths.cycle_dir(cycle_id)
    state: CycleState | None = None
    attempts = 0
    for attempts in range(1, config.runner.max_attempts + 1):
        paths.ensure_dir(directory)
        code = execute(
            command,
            paths.workspace.training_project,
            directory / f"run_attempt_{attempts}.log",
            environment,
        )
        state = _read_state(directory / CYCLE_STATE_FILENAME)
        if state is not None and state.status in _SETTLED:
            break
        if state is not None and not _died_mid_step(state) and code == 0:
            break
        LOGGER.warning(
            "Cycle %s stopped without finishing (exit code %s); attempt %s of %s",
            cycle_id,
            code,
            attempts,
            config.runner.max_attempts,
        )
    settled = state is not None and state.status in _SETTLED
    report = directory / "report.md"
    return CycleOutcome(
        cycle_id=cycle_id,
        status=state.status if settled and state is not None else GAVE_UP,
        attempts=attempts,
        decision=str(((state.data if state else {}).get("decision") or {}).get("decision", "")),
        report=str(report) if report.is_file() else "",
        resumed=resumed,
    )


def _launch_child(
    command: Sequence[str], cwd: Path, log_path: Path, extra_env: Mapping[str, str]
) -> int:
    """Run the cycle worker, its output going to a log beside the cycle."""
    environment = {**os.environ, **extra_env}
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            list(command),
            cwd=str(cwd),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode


def _died_mid_step(state: CycleState) -> bool:
    return state.status not in _SETTLED and any(
        step.get("status") == RUNNING for step in state.steps.values()
    )


def _read_state(path: Path) -> CycleState | None:
    if not path.is_file():
        return None
    try:
        return CycleState.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Small helpers


def _still_arriving(batch: Path, settle_seconds: int) -> bool:
    if settle_seconds <= 0:
        return False
    newest = max(
        (entry.stat().st_mtime for entry in batch.rglob("*") if entry.is_file()),
        default=batch.stat().st_mtime,
    )
    return time.time() - newest < settle_seconds


def _unused(path: Path) -> Path:
    candidate, counter = path, 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{counter}")
        counter += 1
    return candidate


def _write_run_log(paths: AutoTrainPaths, outcome: RunOutcome, when: datetime) -> None:
    runs = paths.ensure_dir(paths.root / "runs")
    (runs / when.strftime("run_%Y%m%dT%H%M%SZ.json")).write_text(
        json.dumps(outcome.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Entry point that needs nothing beyond the package itself


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m picture_tool.autotrain.runner`` --- ``run`` without typer.

    Also the cycle worker the supervisor launches (``--cycle-worker``).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="Path to the settings file.")
    parser.add_argument("--cycle-worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    config = AutoTrainConfig.load(args.config)

    if args.cycle_worker:
        try:
            run_training_cycle(config, cycle_id=args.cycle_worker)
        except CycleError as exc:
            print(exc, file=sys.stderr)
            return 1
        return 0

    try:
        outcome = run(config, AutoTrainPaths.discover())
    except AutoTrainError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2 if not config.enabled else 1
    print(json.dumps(outcome.to_dict(), ensure_ascii=False, indent=2))
    return 0 if outcome.ok else 1


if __name__ == "__main__":
    sys.exit(main())
