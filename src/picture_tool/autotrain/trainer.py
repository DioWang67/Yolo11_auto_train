"""Challenger training.

Reuses the existing pipeline rather than reimplementing training: the same
:func:`picture_tool.main_pipeline.run_pipeline`, the same task registry, the
same :func:`picture_tool.train.yolo_trainer.train_yolo`. What differs is the
*task list* and where the output goes.

Two rules define this module:

* **``deploy`` is never in the task list.** Not disabled-by-config, not
  skipped at runtime --- absent. The existing operator workflow ends with a
  deploy task that publishes into the production models directory; this path
  must not be able to reach that code even by misconfiguration, so the
  requested tasks are filtered against an explicit deny list before running.
* **Output lands in the candidate directory.** Weights are written under the
  cycle's own run directory and copied into
  ``models/candidates/<product>/<area>/<version>/``. Nothing is written to
  the inference project.

The base model is the deployed champion's *training* weight, matching what
the operator flow already does: continuing from the deployed model preserves
the classes the line detects today.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from picture_tool.autotrain import AutoTrainError
from picture_tool.autotrain.dataset_versions import DatasetVersion

LOGGER = logging.getLogger(__name__)

#: Tasks this path is allowed to run. Evaluation is deliberately absent:
#: champion/challenger comparison is done by :mod:`.evaluator`, which needs
#: both models measured on one split and must not raise the deployment
#: gate's own exception mid-cycle.
CANDIDATE_TASKS: tuple[str, ...] = (
    "yolo_augmentation",
    "dataset_lint",
    "dataset_splitter",
    "yolo_train",
)

#: Tasks that publish to production. Never runnable from this path.
FORBIDDEN_TASKS = frozenset({"deploy", "artifact_bundle", "anomalib_package"})


class CandidateTrainingError(AutoTrainError):
    """Raised when a challenger could not be produced."""


@dataclass(frozen=True)
class TrainingResult:
    """Where the challenger's artifacts ended up, and how it was trained."""

    model_version: str
    weights_path: Path
    weight_sha256: str
    run_dir: Path
    dataset_version: str
    dataset_content_id: str
    base_model: str
    class_names: tuple[str, ...]
    tasks: tuple[str, ...]
    config_path: Path
    #: False when the pipeline's skip cache reused an earlier attempt's
    #: weights instead of training. The weights are still correct for this
    #: dataset and config --- that is what the cache checks --- but they were
    #: not produced now, and ``metrics`` then carries the earlier run's
    #: ``trained_at``. A cycle report recommending a promotion should be able
    #: to say which of the two happened.
    trained_this_run: bool = True
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "weights_path": str(self.weights_path),
            "weight_sha256": self.weight_sha256,
            "run_dir": str(self.run_dir),
            "dataset_version": self.dataset_version,
            "dataset_content_id": self.dataset_content_id,
            "base_model": self.base_model,
            "class_names": list(self.class_names),
            "tasks": list(self.tasks),
            "config_path": str(self.config_path),
            "trained_this_run": self.trained_this_run,
            "metrics": dict(self.metrics),
        }


def assert_no_forbidden_tasks(tasks: Sequence[str]) -> tuple[str, ...]:
    """Refuse any task that could publish to production.

    Checked as a hard precondition rather than trusted from configuration:
    the whole safety story of this path is that it cannot deploy, and that
    claim should not depend on a YAML file staying correct.
    """
    requested = tuple(str(task).strip().lower() for task in tasks)
    forbidden = sorted(set(requested) & FORBIDDEN_TASKS)
    if forbidden:
        raise CandidateTrainingError(
            "Autonomous training must never run publishing tasks; refusing: "
            + ", ".join(forbidden)
        )
    return requested


def build_candidate_config(
    base_config: Mapping[str, Any],
    *,
    dataset_version: DatasetVersion,
    work_dir: Path,
    run_project: Path,
    run_name: str,
    base_model: str,
    class_names: Sequence[str],
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
) -> dict[str, Any]:
    """Derive a candidate-training config from the station's pipeline config.

    Starts from the real configuration so station-specific settings survive,
    then redirects every path into the cycle's working directory and removes
    anything that could publish.
    """
    config = copy.deepcopy(dict(base_config))

    raw_images = work_dir / "raw" / "images"
    raw_labels = work_dir / "raw" / "labels"
    processed_images = work_dir / "processed" / "images"
    processed_labels = work_dir / "processed" / "labels"
    split_dir = work_dir / "split"

    augmentation = config.setdefault("yolo_augmentation", {})
    augmentation["input"] = {
        "image_dir": str(raw_images),
        "label_dir": str(raw_labels),
    }
    augmentation["output"] = {
        "image_dir": str(processed_images),
        "label_dir": str(processed_labels),
    }
    _force_colour_safe_augmentation(augmentation)

    split = config.setdefault("train_test_split", {})
    split["input"] = {
        "image_dir": str(processed_images),
        "label_dir": str(processed_labels),
    }
    split["output"] = {"output_dir": str(split_dir)}

    training = config.setdefault("yolo_training", {})
    training["dataset_dir"] = str(split_dir)
    training["class_names"] = list(class_names)
    training["model"] = base_model
    training["epochs"] = epochs
    training["imgsz"] = imgsz
    training["batch"] = batch
    training["device"] = device
    training["project"] = str(run_project)
    training["name"] = run_name

    # Position calibration belongs to the operator workflow and can change
    # station behaviour; a challenger has no business touching it.
    position = training.setdefault("position_validation", {})
    position["enabled"] = False
    # Its sample_dir is inherited from the station config and points into a
    # directory this cycle never builds. Disabled means it is never read, but
    # the schema check still reports it on every task, burying the warnings
    # that would matter.
    position.pop("sample_dir", None)

    # Nothing from this path is published, so every publishing block is
    # switched off as well as excluded from the task list.
    for key in ("deploy", "export_runtime", "artifact_bundle"):
        block = training.get(key)
        if isinstance(block, dict):
            block["enabled"] = False
    training.setdefault("export_onnx", {})["enabled"] = False

    # The deployment gate stays out of the way: this path measures champion
    # against challenger itself, and a gate exception here would abort a
    # cycle that produced a perfectly reportable result.
    evaluation = config.setdefault("yolo_evaluation", {})
    evaluation.setdefault("gate", {})["enabled"] = False

    config["autotrain"] = {
        "dataset_version": dataset_version.version,
        "dataset_content_id": dataset_version.content_id,
        "deploy_forbidden": True,
    }
    return config


def prepare_work_dir(dataset_version: DatasetVersion, work_dir: Path) -> Path:
    """Copy an immutable dataset version into a writable training workspace.

    The version itself is read-only, and augmentation and splitting both
    write. Copying is what keeps the version immutable while the pipeline
    works normally.
    """
    raw_images = work_dir / "raw" / "images"
    raw_labels = work_dir / "raw" / "labels"
    raw_images.mkdir(parents=True, exist_ok=True)
    raw_labels.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image in sorted(dataset_version.images_dir.iterdir()):
        if not image.is_file():
            continue
        label = dataset_version.labels_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise CandidateTrainingError(
                f"Dataset version {dataset_version.version} is missing a label "
                f"for {image.name}; refusing to train on an unlabelled image."
            )
        _copy_into_workspace(image, raw_images / image.name)
        _copy_into_workspace(label, raw_labels / label.name)
        copied += 1

    if copied == 0:
        raise CandidateTrainingError(
            f"Dataset version {dataset_version.version} contains no images."
        )
    return work_dir


def train_candidate(
    *,
    dataset_version: DatasetVersion,
    base_config: Mapping[str, Any],
    candidate_dir: Path,
    work_dir: Path,
    model_version: str,
    base_model: str,
    class_names: Sequence[str],
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 4,
    device: str = "cpu",
    tasks: Sequence[str] = CANDIDATE_TASKS,
    runner: Callable[..., Any] | None = None,
    logger: logging.Logger | None = None,
    args_overrides: Mapping[str, Any] | None = None,
) -> TrainingResult:
    """Train one challenger and place its weights in the candidate directory.

    ``runner`` defaults to the existing :func:`run_pipeline`; it is injectable
    so tests can drive the whole flow without invoking ultralytics.
    """
    log = logger or LOGGER
    requested = assert_no_forbidden_tasks(tasks)
    if not base_model:
        raise CandidateTrainingError(
            "No base model resolved. A challenger continues from the deployed "
            "champion so it keeps the classes the line detects today."
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    prepare_work_dir(dataset_version, work_dir)

    run_project = work_dir / "runs"
    run_name = "candidate"
    config = build_candidate_config(
        base_config,
        dataset_version=dataset_version,
        work_dir=work_dir,
        run_project=run_project,
        run_name=run_name,
        base_model=base_model,
        class_names=class_names,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
    )

    config_path = work_dir / "candidate_pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    args = _pipeline_args(config_path, args_overrides)
    execute = runner or _default_runner()
    weights_before = _run_weight_mtimes(run_project)
    log.info(
        "Training challenger %s on %s with tasks %s",
        model_version,
        dataset_version.version,
        ", ".join(requested),
    )
    try:
        execute(list(requested), config, log, args)
    except Exception as exc:  # noqa: BLE001 - the pipeline raises many types
        raise CandidateTrainingError(
            f"Challenger training failed for {model_version}: {exc}"
        ) from exc

    run_dir = _resolve_run_dir(run_project, run_name)
    weights = _resolve_weights(run_dir)
    trained_this_run = weights_before.get(weights) != weights.stat().st_mtime_ns
    if not trained_this_run:
        log.warning(
            "Challenger %s reuses weights from an earlier attempt: the "
            "pipeline's skip cache matched this dataset and config, so no "
            "training ran in this attempt.",
            model_version,
        )
    destination = candidate_dir / "weights" / weights.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, destination)

    return TrainingResult(
        model_version=model_version,
        weights_path=destination,
        weight_sha256=_sha256_file(destination),
        run_dir=run_dir,
        dataset_version=dataset_version.version,
        dataset_content_id=dataset_version.content_id,
        base_model=base_model,
        class_names=tuple(class_names),
        tasks=requested,
        config_path=config_path,
        trained_this_run=trained_this_run,
        metrics=_read_run_metrics(run_dir),
    )


# ---------------------------------------------------------------------------


def _copy_into_workspace(source: Path, destination: Path) -> None:
    """Copy one file in so the pipeline may later write over it.

    ``copy2`` is kept for the mtime it preserves: the existing ``dataset_hash``
    is taken over relative paths, sizes and mtimes, so copying the same
    immutable version twice has to produce the same hash for the trainer's skip
    cache to mean anything.

    What must be undone is the other half of ``copy2``. The version store marks
    its files read-only to enforce immutability, and that bit travels with the
    copy --- leaving this supposedly writable workspace read-only, and making a
    retried cycle die on ``PermissionError`` instead of on whatever actually
    stopped it the first time.
    """
    if destination.exists():
        destination.chmod(destination.stat().st_mode | stat.S_IWUSR)
    shutil.copy2(source, destination)
    destination.chmod(destination.stat().st_mode | stat.S_IWUSR)


def _default_runner() -> Callable[..., Any]:
    """Import the real pipeline lazily.

    ``main_pipeline`` pulls in the whole task registry; importing it at module
    scope would make even ``--help`` pay for it.
    """
    from picture_tool.main_pipeline import run_pipeline

    return run_pipeline


def _pipeline_args(config_path: Path, overrides: Mapping[str, Any] | None):
    """Build the namespace ``run_pipeline`` expects from the CLI."""
    values: dict[str, Any] = {
        "config": str(config_path),
        "tasks": None,
        "exclude_tasks": list(FORBIDDEN_TASKS),
        "task_groups": None,
        "interactive": False,
        "force": False,
        "device": None,
        "epochs": None,
        "imgsz": None,
        "batch": None,
        "model": None,
        "project": None,
        "name": None,
        "weights": None,
        "infer_input": None,
        "infer_output": None,
        "product": None,
        "input_format": None,
        "output_format": None,
    }
    values.update(dict(overrides or {}))
    return type("AutoTrainArgs", (), values)()


def _force_colour_safe_augmentation(augmentation: dict[str, Any]) -> None:
    """Disable augmentations that destroy this domain's class signal.

    Wire colour *is* the label on this station, and orientation carries
    meaning, so hue shifts, flips and perspective would teach the model that
    a red wire is sometimes orange. The operator retraining flow already
    forces exactly this; a challenger trained under looser augmentation would
    not be comparable to the champion.
    """
    operations = augmentation.setdefault("augmentation", {}).setdefault(
        "operations", {}
    )
    operations.setdefault("hue", {})["range"] = [1, 1]
    operations.setdefault("flip", {})["probability"] = 0.0
    operations.setdefault("perspective", {})["scale"] = [0, 0]


def _run_weight_mtimes(project: Path) -> dict[Path, int]:
    """Snapshot the weights already present under a run project.

    Compared against afterwards to tell a real training run from one the
    pipeline's skip cache satisfied with an earlier attempt's output. Done by
    comparing the file against *itself* before and after rather than against
    the wall clock, so it does not depend on the filesystem's timestamp
    granularity agreeing with ``time.time()``.
    """
    if not project.is_dir():
        return {}
    return {
        weight: weight.stat().st_mtime_ns
        for weight in project.glob("*/weights/*.pt")
        if weight.is_file()
    }


def _resolve_run_dir(project: Path, name: str) -> Path:
    """Find the directory ultralytics actually wrote to.

    ``exist_ok=False`` means a rerun becomes ``candidate2``, ``candidate3``
    and so on, so the configured name cannot be trusted.
    """
    if not project.is_dir():
        raise CandidateTrainingError(
            f"Training produced no run directory under {project}."
        )
    candidates = [
        entry
        for entry in project.iterdir()
        if entry.is_dir() and entry.name.startswith(name)
    ]
    if not candidates:
        raise CandidateTrainingError(
            f"No run directory starting with {name!r} under {project}."
        )
    return max(candidates, key=lambda entry: entry.stat().st_mtime)


def _resolve_weights(run_dir: Path) -> Path:
    for relative in ("weights/best.pt", "weights/last.pt"):
        candidate = run_dir / relative
        if candidate.is_file():
            return candidate
    raise CandidateTrainingError(
        f"Training completed but produced no weights under {run_dir / 'weights'}."
    )


def _read_run_metrics(run_dir: Path) -> dict[str, Any]:
    """Best-effort read of the run metadata the trainer writes."""
    path = run_dir / "last_run_metadata.json"
    if not path.is_file():
        return {}
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
