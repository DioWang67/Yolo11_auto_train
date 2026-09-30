"""The unattended run: inbox in, report out, and surviving the process dying.

The child process is replaced by an in-process launcher so each scenario can
decide what the "child" does --- finish, die mid-training the way the line's
PC does, or record a genuine failure --- and the supervisor's reaction to the
state left behind is what gets checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from picture_tool.autotrain import AutoTrainError
from picture_tool.autotrain.config import AutoTrainConfig
from picture_tool.autotrain.orchestrator import (
    COMPLETED,
    STEP_FAILED,
    CycleError,
    run_training_cycle,
)
from picture_tool.autotrain.paths import AutoTrainPaths
from picture_tool.autotrain.promotion import PROMOTION_CANDIDATE
from picture_tool.autotrain.runner import (
    GAVE_UP,
    IMPORT_RESULT_FILENAME,
    PROCESSED_DIRNAME,
    REJECTED_DIRNAME,
    main,
    run,
)
from picture_tool.dataset_manifest_lock import autotrain_store_lock
from picture_tool.workspace_paths import WorkspacePaths

PRODUCT = "Cable1"
AREA = "A"
CLASSES = ["Red", "Orange"]
#: What Windows reports for a process killed by an access violation.
SEGFAULT_EXIT = 3221225477


@pytest.fixture()
def paths(tmp_path) -> AutoTrainPaths:
    root = tmp_path / "workspace"
    training = root / "Yolo11_auto_train"
    inference = root / "yolo11_inference"
    for relative in (
        training / "data",
        inference / "models",
        root / "station_data" / "yolo11_inference",
        root / "Result",
    ):
        relative.mkdir(parents=True, exist_ok=True)
    return AutoTrainPaths.from_workspace(
        WorkspacePaths(
            workspace_root=root.resolve(),
            training_project=training.resolve(),
            inference_project=inference.resolve(),
            training_data=(training / "data").resolve(),
            inference_models=(inference / "models").resolve(),
            station_data=(root / "station_data" / "yolo11_inference").resolve(),
            inference_results=(root / "Result").resolve(),
            inference_artifacts=root.resolve(),
            manifest_path=None,
        )
    )


@pytest.fixture()
def champion(paths) -> Path:
    model_dir = paths.production_model_dir(PRODUCT, AREA)
    (model_dir / "weights").mkdir(parents=True)
    (model_dir / "weights" / "best.training.pt").write_bytes(b"champion-weights")
    (model_dir / "deployment_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "deployed_version": "Cable1_A_v1.2.0",
                "deployed_weight_file": "weights/best.onnx",
                "training_weight_file": "weights/best.training.pt",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "config.yaml").write_text(
        yaml.safe_dump({"class_names": CLASSES}), encoding="utf-8"
    )
    return model_dir


def _config(tmp_path, **runner) -> AutoTrainConfig:
    path = tmp_path / "autonomous_training.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "autonomous_training": {
                    "enabled": True,
                    "station": {"product": PRODUCT, "area": AREA},
                    "collector": {"compute_image_quality": False},
                    "selectors": {"low_confidence": {"enabled": True}},
                    "training": {"epochs": 1, "imgsz": 64, "batch": 1, "device": "cpu"},
                    "promotion": {"require_golden_pass": False},
                    "runner": {"settle_seconds": 0, **runner},
                }
            }
        ),
        encoding="utf-8",
    )
    return AutoTrainConfig.load(path)


def _drop_batch(paths, name="batch1", *, labelled=True) -> Path:
    batch = paths.inbox_dir() / name
    (batch / "images").mkdir(parents=True)
    (batch / "labels").mkdir()
    (batch / "images" / "a.jpg").write_bytes(f"{name}-pixels".encode("utf-8"))
    if labelled:
        (batch / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (batch / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    return batch


# -- what the "child process" does ------------------------------------------------


class _Trains:
    def __call__(self, tasks, config, logger, args):
        project = Path(config["yolo_training"]["project"])
        (project / "candidate" / "weights").mkdir(parents=True, exist_ok=True)
        (project / "candidate" / "weights" / "best.pt").write_bytes(b"challenger")
        split = Path(config["yolo_training"]["dataset_dir"])
        split.mkdir(parents=True, exist_ok=True)
        (split / "data.yaml").write_text(
            yaml.safe_dump({"path": str(split), "val": "val/images", "names": CLASSES}),
            encoding="utf-8",
        )


class _Dies:
    """Outside ``Exception``, so nothing records FAILED --- like a segfault."""

    def __call__(self, tasks, config, logger, args):
        raise KeyboardInterrupt("process killed mid-training")


class _Breaks:
    def __call__(self, tasks, config, logger, args):
        raise RuntimeError("a genuine, recorded failure")


class _Box:
    def __init__(self, score):
        self.mp = self.mr = self.map50 = score
        self.map = score - 0.2
        self.ap_class_index = [0, 1]
        self.p = self.r = self.ap50 = [score, score]
        self.ap = [[score], [score]]


class _Results:
    def __init__(self, score):
        self.box = _Box(score)
        self.names = {0: "Red", 1: "Orange"}
        self.results_dict = {}
        self.confusion_matrix = None


def _launcher(config, paths, champion, *children, seen=None):
    """Each launch plays the next child in ``children``."""
    queue = list(children)
    champion_weights = (champion / "weights" / "best.training.pt").resolve()

    def validator(weights, **kwargs):
        return _Results(0.80 if Path(weights).resolve() == champion_weights else 0.85)

    def launch(command, cwd, log_path, extra_env):
        if seen is not None:
            seen.append((list(command), cwd, dict(extra_env)))
        cycle_id = command[command.index("--cycle-worker") + 1]
        try:
            run_training_cycle(
                config, paths, cycle_id=cycle_id, runner=queue.pop(0), validator=validator
            )
        except KeyboardInterrupt:
            return SEGFAULT_EXIT
        except CycleError:
            return 1
        return 0

    return launch


# ---------------------------------------------------------------------------


def test_nothing_new_means_nothing_runs(paths, champion, tmp_path):
    outcome = run(_config(tmp_path), paths, launch=_launcher(None, paths, champion))

    assert outcome.batches == () and outcome.cycles == ()
    assert "No cycle run" in outcome.notes[0]
    assert outcome.ok
    assert list((paths.root / "runs").glob("run_*.json"))


def test_a_dropped_batch_is_imported_filed_and_carried_to_a_report(
    paths, champion, tmp_path
):
    config = _config(tmp_path)
    _drop_batch(paths)

    outcome = run(config, paths, launch=_launcher(config, paths, champion, _Trains()))

    (batch,) = outcome.batches
    assert batch.accepted and Path(batch.moved_to).parent.name == PROCESSED_DIRNAME
    assert json.loads((Path(batch.moved_to) / IMPORT_RESULT_FILENAME).read_text())[
        "verified"
    ] == 1
    (cycle,) = outcome.cycles
    assert cycle.status == COMPLETED and cycle.attempts == 1
    assert cycle.decision == PROMOTION_CANDIDATE
    assert Path(cycle.report).is_file()
    assert outcome.to_dict()["deployed"] is False
    # The inbox holds only work nobody has looked at yet.
    assert [p.name for p in paths.inbox_dir().iterdir()] == [PROCESSED_DIRNAME]


def test_a_refused_batch_is_filed_with_its_reasons_and_trains_nothing(
    paths, champion, tmp_path
):
    _drop_batch(paths, labelled=False)

    outcome = run(_config(tmp_path), paths, launch=_launcher(None, paths, champion))

    (batch,) = outcome.batches
    assert not batch.accepted and Path(batch.moved_to).parent.name == REJECTED_DIRNAME
    reasons = json.loads((Path(batch.moved_to) / IMPORT_RESULT_FILENAME).read_text())
    assert "no label file" in reasons["errors"][0]
    assert outcome.cycles == ()
    assert not outcome.ok


def test_a_batch_still_being_copied_is_left_for_the_next_run(paths, champion, tmp_path):
    batch = _drop_batch(paths)

    outcome = run(
        _config(tmp_path, settle_seconds=3600), paths, launch=_launcher(None, paths, champion)
    )

    assert outcome.waiting == ("batch1",)
    assert batch.is_dir()


def test_a_child_that_dies_mid_training_is_started_again_and_finishes(
    paths, champion, tmp_path
):
    config = _config(tmp_path)
    _drop_batch(paths)

    outcome = run(
        config, paths, launch=_launcher(config, paths, champion, _Dies(), _Trains())
    )

    (cycle,) = outcome.cycles
    assert cycle.status == COMPLETED
    assert cycle.attempts == 2


def test_a_recorded_failure_is_reported_not_retried(paths, champion, tmp_path):
    config = _config(tmp_path)
    _drop_batch(paths)

    outcome = run(
        config, paths, launch=_launcher(config, paths, champion, _Breaks(), _Trains())
    )

    (cycle,) = outcome.cycles
    assert cycle.status == STEP_FAILED and cycle.attempts == 1
    assert not outcome.ok


def test_it_gives_up_after_the_configured_attempts(paths, champion, tmp_path):
    config = _config(tmp_path, max_attempts=2)
    _drop_batch(paths)

    outcome = run(
        config, paths, launch=_launcher(config, paths, champion, _Dies(), _Dies())
    )

    (cycle,) = outcome.cycles
    assert cycle.status == GAVE_UP and cycle.attempts == 2
    assert not outcome.ok


def test_the_next_run_finishes_a_cycle_that_was_given_up_on(paths, champion, tmp_path):
    config = _config(tmp_path, max_attempts=1)
    _drop_batch(paths)
    first = run(config, paths, launch=_launcher(config, paths, champion, _Dies()))

    second = run(config, paths, launch=_launcher(config, paths, champion, _Trains()))

    (cycle,) = second.cycles
    assert cycle.cycle_id == first.cycles[0].cycle_id
    assert cycle.resumed and cycle.status == COMPLETED
    # Its labels were versioned by that very cycle, so no second one starts.
    assert len(second.cycles) == 1


def test_a_second_run_refuses_while_one_is_in_progress(paths, champion, tmp_path):
    with autotrain_store_lock(paths.root, name="runner"):
        with pytest.raises(AutoTrainError, match="in progress"):
            run(_config(tmp_path), paths, launch=_launcher(None, paths, champion))


def test_the_worker_is_launched_in_the_training_project(paths, champion, tmp_path):
    config = _config(tmp_path)
    _drop_batch(paths)
    seen: list = []

    run(config, paths, launch=_launcher(config, paths, champion, _Trains(), seen=seen))

    ((command, cwd, _env),) = seen
    assert command[1:4] == ["-m", "picture_tool.autotrain.runner", "--cycle-worker"]
    assert command[-2:] == ["--config", str(config.source_path)]
    assert cwd == paths.workspace.training_project


def test_a_station_that_cannot_take_a_batch_yet_holds_it_in_the_inbox(paths, tmp_path):
    """No champion, so no class contract: not the batch's fault, not rejected."""
    config = _config(tmp_path)
    batch = _drop_batch(paths)

    outcome = run(config, paths, launch=_launcher(config, paths, paths.root, _Trains()))

    assert outcome.held == ("batch1",)
    assert "class schema" in outcome.notes[0].lower()
    assert batch.is_dir()
    assert outcome.batches == () and outcome.cycles == ()
    assert not outcome.ok


def test_the_module_entry_point_refuses_while_disabled(tmp_path, capsys):
    settings = tmp_path / "off.yaml"
    settings.write_text(
        yaml.safe_dump({"autonomous_training": {"enabled": False}}), encoding="utf-8"
    )

    assert main(["--config", str(settings)]) == 2
