"""Validated handoff contract from yolo11_inference to the training GUI."""

from __future__ import annotations

import copy
import csv
import filecmp
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from picture_tool.pending_annotations import validate_yolo_label_text
from picture_tool.quality.operator_dataset_conflicts import (
    OperatorDatasetConflictError,
    analysis_payload,
    analyze_operator_dataset,
    filter_manifest_rows,
    write_json_atomic as write_conflict_report_atomic,
)


class OperatorHandoffError(ValueError):
    """Raised when an inference-to-training handoff is invalid or unsafe."""


OPERATOR_MIN_INSTANCES_PER_CLASS = 5
OPERATOR_MIN_TRAIN_INSTANCES_PER_CLASS = 3
OPERATOR_MIN_TEST_INSTANCES_PER_CLASS = 5
OPERATOR_MIN_IMAGES_PER_SPLIT = (("train", 3), ("val", 5), ("test", 10))
OPERATOR_MIN_PRECISION = 0.80
OPERATOR_MIN_RECALL = 0.90
OPERATOR_MIN_MAP50 = 0.80
OPERATOR_MIN_MAP50_95 = 0.50
OPERATOR_MAX_METRIC_REGRESSION = 0.02
DEFAULT_PRODUCTION_CONFIDENCE = 0.40
DEFAULT_OPERATOR_AUGMENTATIONS_PER_IMAGE = 20
MAX_OPERATOR_AUGMENTATIONS_PER_IMAGE = 50
YOLO_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
TERMINAL_OPERATOR_JOB_STATES = frozenset({"deployed", "failed", "cancelled"})


@dataclass(frozen=True)
class OperatorTrainingOptions:
    """Validated job-scoped options supplied by the inference UI."""

    epochs: int = 20
    augmentations_per_image: int = 20
    batch: int = 8
    imgsz: int = 640


@dataclass(frozen=True)
class OperatorTarget:
    """One product/area target included in a handoff."""

    product: str
    area: str
    dataset_root: Path
    ready_count: int
    pending_count: int
    total_ready_count: int = 0
    class_names: tuple[str, ...] = ()
    observed_class_map: tuple[tuple[int, str], ...] = ()
    class_schema_hash: str = ""
    class_contract_required: bool = False
    sample_ids: tuple[str, ...] = ()
    pending_sample_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperatorHandoff:
    """Validated operator handoff loaded from JSON."""

    path: Path
    data_root: Path
    inference_models_dir: Path
    ready_count: int
    pending_count: int
    targets: tuple[OperatorTarget, ...]
    job_id: str = ""
    status_path: Path | None = None
    training_options: OperatorTrainingOptions = OperatorTrainingOptions()

    @property
    def selected_target(self) -> OperatorTarget:
        """Return the single product/station handled by the OP workflow."""
        active = [
            target
            for target in self.targets
            if target.ready_count > 0
            or target.total_ready_count > 0
            or target.pending_count > 0
        ]
        if len(active) != 1:
            raise OperatorHandoffError(
                "Operator handoff must contain exactly one trainable or pending product/area."
            )
        return active[0]


def resolve_latest_operator_handoff(training_root: str | Path) -> Path:
    """Resolve the mutable latest pointer to one immutable resumable job.

    Args:
        training_root: Root of this training repository.

    Returns:
        The job-scoped ``handoff.json`` path.

    Raises:
        OperatorHandoffError: If the pointer, job path, or status is unsafe.
    """
    root = Path(training_root).expanduser().resolve()
    operator_root = (root / "data" / ".operator_handoff").resolve()
    latest_path = operator_root / "latest.json"
    if not latest_path.is_file():
        raise OperatorHandoffError(
            "No operator training job is available. Submit reviewed images first."
        )
    try:
        latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperatorHandoffError(
            f"Unable to read the latest operator job: {exc}"
        ) from exc
    if not isinstance(latest_payload, dict):
        raise OperatorHandoffError("The latest operator job pointer is invalid.")

    job_id = str(latest_payload.get("job_id") or "").strip()
    _validate_segment(job_id, "job_id")
    job_root = (operator_root / "jobs" / job_id).resolve()
    jobs_root = (operator_root / "jobs").resolve()
    if not job_root.is_relative_to(jobs_root):
        raise OperatorHandoffError("The latest operator job path is unsafe.")

    handoff_path = job_root / "handoff.json"
    status_path = job_root / "status.json"
    if not handoff_path.is_file() or not status_path.is_file():
        raise OperatorHandoffError(
            f"The latest operator job is incomplete: {job_id}"
        )
    try:
        immutable_payload = json.loads(handoff_path.read_text(encoding="utf-8"))
        status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperatorHandoffError(
            f"Unable to read operator job {job_id}: {exc}"
        ) from exc
    if not isinstance(immutable_payload, dict) or not isinstance(status_payload, dict):
        raise OperatorHandoffError(f"Operator job {job_id} contains invalid JSON.")
    if str(immutable_payload.get("job_id") or "").strip() != job_id:
        raise OperatorHandoffError(
            "The latest operator job does not match its immutable handoff."
        )
    state = str(status_payload.get("state") or "queued").strip().lower()
    if state in TERMINAL_OPERATOR_JOB_STATES:
        raise OperatorHandoffError(
            f"The latest operator job is already terminal ({state}). "
            "Submit or explicitly open another job."
        )
    return handoff_path


def _load_training_options(
    value: Any, *, required: bool
) -> OperatorTrainingOptions:
    """Validate the schema-v4 training options without trusting GUI input."""
    if value is None:
        if required:
            raise OperatorHandoffError("Schema-v4 handoff is missing training_options.")
        return OperatorTrainingOptions()
    if not isinstance(value, dict):
        raise OperatorHandoffError("training_options must be a mapping.")
    allowed = {"epochs", "augmentations_per_image", "batch", "imgsz"}
    unexpected = set(value) - allowed
    if unexpected:
        raise OperatorHandoffError(
            "Unsupported training option(s): " + ", ".join(sorted(unexpected))
        )
    missing = allowed - set(value)
    if required and missing:
        raise OperatorHandoffError(
            "Schema-v4 handoff is missing training option(s): "
            + ", ".join(sorted(missing))
        )
    invalid_types = [key for key in value if type(value[key]) is not int]
    if invalid_types:
        raise OperatorHandoffError(
            "Training option(s) must be integers: "
            + ", ".join(sorted(invalid_types))
        )
    defaults = OperatorTrainingOptions()
    options = OperatorTrainingOptions(
        epochs=value.get("epochs", defaults.epochs),
        augmentations_per_image=value.get(
            "augmentations_per_image", defaults.augmentations_per_image
        ),
        batch=value.get("batch", defaults.batch),
        imgsz=value.get("imgsz", defaults.imgsz),
    )
    ranges = {
        "epochs": (options.epochs, 20, 300),
        "augmentations_per_image": (
            options.augmentations_per_image,
            0,
            MAX_OPERATOR_AUGMENTATIONS_PER_IMAGE,
        ),
        "batch": (options.batch, 1, 64),
        "imgsz": (options.imgsz, 320, 1280),
    }
    for name, (setting, minimum, maximum) in ranges.items():
        if not minimum <= setting <= maximum:
            raise OperatorHandoffError(
                f"training_options.{name} must be between {minimum} and {maximum}."
            )
    if options.imgsz % 32 != 0:
        raise OperatorHandoffError(
            "training_options.imgsz must be a multiple of 32."
        )
    return options


def load_operator_handoff(
    handoff_path: str | Path, *, training_root: str | Path
) -> OperatorHandoff:
    """Load and validate paths/counts from an inference handoff.

    Args:
        handoff_path: JSON file generated by ``yolo11_inference``.
        training_root: Root of this training repository.

    Returns:
        A validated handoff containing exactly one trainable target.

    Raises:
        OperatorHandoffError: If fields, counts, or paths are unsafe or missing.
    """
    path = Path(handoff_path).expanduser().resolve()
    root = Path(training_root).expanduser().resolve()
    if not path.is_file():
        raise OperatorHandoffError(f"Handoff file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperatorHandoffError(f"Unable to read handoff: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2, 3, 4}:
        raise OperatorHandoffError("Unsupported operator handoff schema.")
    schema_version = int(payload.get("schema_version", 0))
    training_options = _load_training_options(
        payload.get("training_options"),
        required=schema_version >= 4,
    )

    data_root = Path(str(payload.get("data_root") or "")).resolve()
    allowed_data_root = (root / "data").resolve()
    if not data_root.is_relative_to(allowed_data_root):
        raise OperatorHandoffError(
            f"Handoff data root is outside training data: {data_root}"
        )
    if not data_root.is_dir():
        raise OperatorHandoffError(f"Handoff data root not found: {data_root}")
    job_id = str(payload.get("job_id") or "").strip()
    status_path: Path | None = None
    if schema_version >= 3:
        _validate_segment(job_id, "job_id")
        expected_job_dir = (data_root / ".operator_handoff" / "jobs" / job_id).resolve()
        if path != expected_job_dir / "handoff.json":
            raise OperatorHandoffError(
                "Job handoff path does not match its immutable job_id."
            )
        status_path = Path(str(payload.get("status_path") or "")).resolve()
        if status_path != expected_job_dir / "status.json":
            raise OperatorHandoffError("Invalid operator job status path.")
    models_value = str(payload.get("inference_models_dir") or "").strip()
    if not models_value:
        raise OperatorHandoffError("Inference models directory is missing.")
    models_dir = Path(models_value).resolve()
    if not models_dir.is_dir():
        raise OperatorHandoffError(
            f"Inference models directory not found: {models_dir}"
        )
    targets_value = payload.get("targets")
    if not isinstance(targets_value, list):
        raise OperatorHandoffError("Handoff targets must be a list.")

    targets: list[OperatorTarget] = []
    for raw in targets_value:
        if not isinstance(raw, dict):
            continue
        product = str(raw.get("product") or "").strip()
        area = str(raw.get("area") or "").strip()
        dataset_root = Path(str(raw.get("dataset_root") or "")).resolve()
        try:
            ready_count = int(raw.get("ready_count", 0))
            pending_count = int(raw.get("pending_count", 0))
            total_ready_count = int(raw.get("total_ready_count", ready_count))
        except (TypeError, ValueError) as exc:
            raise OperatorHandoffError("Invalid target counts in handoff.") from exc
        if not product or not area:
            raise OperatorHandoffError("Handoff product and area are required.")
        _validate_segment(product, "product")
        _validate_segment(area, "area")
        if ready_count < 0 or pending_count < 0 or total_ready_count < 0:
            raise OperatorHandoffError("Handoff counts cannot be negative.")
        if not dataset_root.is_relative_to(data_root):
            raise OperatorHandoffError(
                f"Target dataset is outside handoff data root: {dataset_root}"
            )
        expected_dataset_root = (data_root / product / area).resolve()
        if dataset_root != expected_dataset_root:
            raise OperatorHandoffError(
                "Target dataset does not match product/area: "
                f"expected {expected_dataset_root}, got {dataset_root}"
            )
        if total_ready_count > 0:
            required_paths = (
                dataset_root / "raw" / "images",
                dataset_root / "raw" / "labels",
                dataset_root / "metadata" / "review_dataset_manifest.csv",
            )
            missing = [str(item) for item in required_paths[:2] if not item.is_dir()]
            if not required_paths[2].is_file():
                missing.append(str(required_paths[2]))
            if missing:
                raise OperatorHandoffError(
                    "Trainable dataset is incomplete: " + ", ".join(missing)
                )
        sample_ids = _string_tuple(raw.get("sample_ids"))
        pending_sample_ids = _string_tuple(raw.get("pending_sample_ids"))
        if schema_version >= 3:
            ready_manifest = dataset_root / "metadata" / "review_dataset_manifest.csv"
            ready_ids = _manifest_sample_ids(ready_manifest)
            pending_ids = _manifest_sample_ids(
                dataset_root / "review_pending" / "manifest.csv"
            )
            job_ids = set(sample_ids)
            expected_pending_ids = set(pending_sample_ids)
            overlapping_ids = job_ids & ready_ids & pending_ids
            if overlapping_ids:
                preview = ", ".join(sorted(overlapping_ids)[:5])
                raise OperatorHandoffError(
                    "Handoff samples exist in both ready and pending "
                    f"manifests: {preview}"
                )
            unexpected_pending_ids = (job_ids - expected_pending_ids) & pending_ids
            if unexpected_pending_ids:
                preview = ", ".join(sorted(unexpected_pending_ids)[:5])
                raise OperatorHandoffError(
                    f"Handoff ready samples unexpectedly moved to pending: {preview}"
                )
            missing_ids = job_ids - ready_ids - pending_ids
            if missing_ids:
                preview = ", ".join(sorted(missing_ids)[:5])
                raise OperatorHandoffError(
                    "Handoff samples are missing from both ready and pending "
                    f"manifests: {preview}"
                )
            pending_count = len(expected_pending_ids & pending_ids)
            ready_count = len(job_ids & ready_ids)
            total_ready_count = _csv_row_count(ready_manifest)
        class_names_value = raw.get("class_names")
        class_names = (
            tuple(str(name) for name in class_names_value)
            if isinstance(class_names_value, list)
            else ()
        )
        observed_value = raw.get("observed_class_map")
        observed: list[tuple[int, str]] = []
        if isinstance(observed_value, dict):
            try:
                observed = sorted(
                    (int(class_id), str(class_name))
                    for class_id, class_name in observed_value.items()
                )
            except (TypeError, ValueError) as exc:
                raise OperatorHandoffError(
                    "Invalid observed class mapping in handoff."
                ) from exc
        contract_required = bool(raw.get("class_contract_required", False))
        if contract_required and not (class_names or observed):
            raise OperatorHandoffError("Required class contract is missing.")
        if class_names:
            if any(not name.strip() for name in class_names):
                raise OperatorHandoffError("Class names must not be empty.")
            if len(set(class_names)) != len(class_names):
                raise OperatorHandoffError("Class names must be unique and ordered.")
            expected_schema_hash = _class_schema_hash(class_names)
            actual_schema_hash = str(raw.get("class_schema_hash") or "").strip()
            if schema_version >= 3 and actual_schema_hash != expected_schema_hash:
                raise OperatorHandoffError(
                    "Class contract checksum does not match the ordered class names."
                )
        if schema_version >= 3 and pending_count > 0 and not class_names:
            raise OperatorHandoffError(
                "Pending annotations require the complete ordered class contract."
            )
        targets.append(
            OperatorTarget(
                product=product,
                area=area,
                dataset_root=dataset_root,
                ready_count=ready_count,
                pending_count=pending_count,
                total_ready_count=total_ready_count,
                class_names=class_names,
                observed_class_map=tuple(observed),
                class_schema_hash=str(raw.get("class_schema_hash") or ""),
                class_contract_required=contract_required,
                sample_ids=sample_ids,
                pending_sample_ids=pending_sample_ids,
            )
        )
    try:
        ready_count = int(payload.get("ready_count", 0))
        pending_count = int(payload.get("pending_count", 0))
    except (TypeError, ValueError) as exc:
        raise OperatorHandoffError("Invalid handoff totals.") from exc
    if ready_count < 0 or pending_count < 0:
        raise OperatorHandoffError("Handoff totals cannot be negative.")
    if schema_version >= 3:
        pending_count = sum(target.pending_count for target in targets)
        ready_count = sum(target.ready_count for target in targets)
    handoff = OperatorHandoff(
        path=path,
        data_root=data_root,
        inference_models_dir=models_dir,
        ready_count=ready_count,
        pending_count=pending_count,
        targets=tuple(targets),
        job_id=job_id,
        status_path=status_path,
        training_options=training_options,
    )
    handoff.selected_target
    return handoff


def materialize_job_dataset_snapshot(handoff: OperatorHandoff) -> OperatorHandoff:
    """Create and return a job-scoped immutable copy of the raw dataset.

    Args:
        handoff: Validated schema-v3 operator handoff after annotation completes.

    Returns:
        A copy of the handoff whose selected target points at the job snapshot.

    Raises:
        OperatorHandoffError: If the source dataset is incomplete or copying fails.
    """
    if not handoff.job_id:
        return handoff
    target = handoff.selected_target
    if target.pending_count:
        raise OperatorHandoffError("Pending annotations must be completed first.")
    job_dir = handoff.path.parent.resolve()
    expected_job_dir = (
        handoff.data_root / ".operator_handoff" / "jobs" / handoff.job_id
    ).resolve()
    if job_dir != expected_job_dir:
        raise OperatorHandoffError("Unsafe operator job directory.")

    snapshot_container = job_dir / "dataset"
    snapshot_root = snapshot_container / target.product / target.area
    snapshot_manifest = job_dir / "dataset_snapshot.json"
    if _snapshot_is_complete(snapshot_root, snapshot_manifest):
        return _handoff_with_snapshot(handoff, target, snapshot_root)

    staging = job_dir / f".dataset.tmp-{os.getpid()}-{time.time_ns()}"
    try:
        if snapshot_container.exists():
            shutil.rmtree(snapshot_container)
        staging_root = staging / target.product / target.area
        source_images = target.dataset_root / "raw" / "images"
        source_labels = target.dataset_root / "raw" / "labels"
        source_manifest = (
            target.dataset_root / "metadata" / "review_dataset_manifest.csv"
        )
        if not source_images.is_dir() or not source_labels.is_dir():
            raise OperatorHandoffError("Raw training images or labels are missing.")
        if not source_manifest.is_file():
            raise OperatorHandoffError("Review dataset manifest is missing.")

        image_count = _copy_directory_files(
            source_images, staging_root / "raw" / "images"
        )
        label_count = _copy_directory_files(
            source_labels, staging_root / "raw" / "labels"
        )
        legacy_image_count = _copy_legacy_raw_pairs(
            handoff,
            target,
            staging_root / "raw" / "images",
            staging_root / "raw" / "labels",
        )
        image_count += legacy_image_count
        label_count += legacy_image_count
        if image_count <= 0:
            raise OperatorHandoffError("No training images are available for snapshot.")
        metadata_dir = staging_root / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        staging_manifest = metadata_dir / source_manifest.name
        shutil.copy2(source_manifest, staging_manifest)

        analysis = analyze_operator_dataset(
            staging_root / "raw" / "images",
            staging_root / "raw" / "labels",
            staging_manifest,
            product=target.product,
            area=target.area,
        )
        if analysis.conflicts:
            conflict_report = analysis_payload(
                analysis,
                scope=f"job:{handoff.job_id}",
            )
            report_path = write_conflict_report_atomic(
                job_dir / "dataset_conflict_report.json",
                conflict_report,
            )
            conflict_error = OperatorDatasetConflictError(analysis)
            raise OperatorHandoffError(
                f"{conflict_error} Conflict report: {report_path}"
            )

        deduplication_audit_path: Path | None = None
        if analysis.selections:
            for excluded in analysis.excluded_image_paths:
                excluded.unlink(missing_ok=True)
            for excluded in analysis.excluded_label_paths:
                excluded.unlink(missing_ok=True)
            excluded_ids = {
                item.excluded_sample for item in analysis.selections
            }
            filter_manifest_rows(
                source_manifest,
                staging_manifest,
                excluded_ids,
                excluded_output_image_names={
                    path.name for path in analysis.excluded_image_paths
                },
            )
            deduplication_audit_path = write_conflict_report_atomic(
                job_dir / "dataset_deduplication_audit.json",
                analysis_payload(
                    analysis,
                    scope=f"job:{handoff.job_id}",
                    repair_mode="new_snapshot_canonical_selection",
                ),
            )

        image_count = _directory_file_count(staging_root / "raw" / "images")
        label_count = _directory_file_count(staging_root / "raw" / "labels")
        staging.replace(snapshot_container)
        _write_json_atomic(
            snapshot_manifest,
            {
                "schema_version": 2,
                "job_id": handoff.job_id,
                "product": target.product,
                "area": target.area,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_dataset_root": str(target.dataset_root),
                "snapshot_dataset_root": str(snapshot_root),
                "image_count": image_count,
                "label_count": label_count,
                "legacy_image_count": legacy_image_count,
                "canonical_selection_count": len(analysis.selections),
                "deduplication_audit_path": str(deduplication_audit_path or ""),
            },
        )
    except OperatorHandoffError:
        raise
    except (OSError, shutil.Error, ValueError) as exc:
        raise OperatorHandoffError(f"Unable to create dataset snapshot: {exc}") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return _handoff_with_snapshot(handoff, target, snapshot_root)


def _configure_operator_yolo_augmentation(
    config: dict[str, Any],
    target: OperatorTarget,
    options: OperatorTrainingOptions,
) -> tuple[Path, Path]:
    """Route one immutable job snapshot through color-safe YOLO augmentation."""
    raw_images = target.dataset_root / "raw" / "images"
    raw_labels = target.dataset_root / "raw" / "labels"
    processed_images = target.dataset_root / "processed" / "images"
    processed_labels = target.dataset_root / "processed" / "labels"
    augmentation_cfg = config.setdefault("yolo_augmentation", {})
    augmentation_cfg.setdefault("input", {}).update(
        {"image_dir": str(raw_images), "label_dir": str(raw_labels)}
    )
    augmentation_cfg.setdefault("output", {}).update(
        {"image_dir": str(processed_images), "label_dir": str(processed_labels)}
    )
    processing = augmentation_cfg.setdefault("processing", {})
    processing.setdefault("num_workers", 2)
    # Debug mode intentionally processes only one source image in the legacy
    # augmentor, so it must never leak into an operator retraining job.
    processing["debug_mode"] = False
    policy = augmentation_cfg.setdefault("augmentation", {})
    policy["num_images"] = options.augmentations_per_image
    policy["include_originals"] = True
    policy["target_size"] = options.imgsz
    policy.setdefault("num_operations", [2, 3])
    operations = policy.setdefault("operations", {})
    if not isinstance(operations, dict):
        raise OperatorHandoffError(
            "yolo_augmentation.augmentation.operations must be a mapping."
        )
    safe_defaults: dict[str, dict[str, Any]] = {
        "blur": {"kernel": [0, 1]},
        "contrast": {"range": [0.92, 1.08]},
        "multiply": {"range": [0.88, 1.15]},
        "noise": {"scale": [0, 0.02]},
        "rotate": {"angle": [-2, 2]},
        "scale": {"range": [0.97, 1.03]},
    }
    for operation, values in safe_defaults.items():
        operations.setdefault(operation, values)
    # Wire color is a class signal. Hue, mirroring, and perspective changes
    # are intentionally disabled even if a generic project preset enables them.
    operations.pop("hue", None)
    operations["flip"] = {"probability": 0.0}
    operations["perspective"] = {"scale": [0.0, 0.0]}
    return processed_images, processed_labels


def apply_handoff_to_config(
    config: dict[str, Any], handoff: OperatorHandoff
) -> dict[str, Any]:
    """Return an in-memory config prepared for safe train-and-deploy.

    Args:
        config: Base pipeline configuration selected for the product.
        handoff: Validated operator handoff.

    Returns:
        A deep-copied configuration with dataset readiness and deployment set.
    """
    updated = copy.deepcopy(config)
    target = handoff.selected_target
    updated["operator_handoff"] = {
        "enabled": True,
        "dataset_root": str(target.dataset_root),
        "source_stage": "raw",
        # The immutable review snapshot is the augmentation source, while the
        # splitter must consume the generated variants plus copied originals.
        "split_source_stage": "processed",
    }
    augmented_images, augmented_labels = _configure_operator_yolo_augmentation(
        updated,
        target,
        handoff.training_options,
    )
    split_cfg = updated.setdefault("train_test_split", {})
    split_cfg.setdefault("input", {}).update(
        {
            "image_dir": str(augmented_images),
            "label_dir": str(augmented_labels),
        }
    )
    split_cfg.setdefault("output", {})["output_dir"] = str(
        target.dataset_root / "split"
    )
    configured_group_minimums = split_cfg.get("minimum_source_groups", {})
    if not isinstance(configured_group_minimums, dict):
        configured_group_minimums = {}
    split_cfg["minimum_source_groups"] = {
        "val": max(int(configured_group_minimums.get("val", 0)), 5),
        "test": max(int(configured_group_minimums.get("test", 0)), 10),
    }
    # Every case submitted in this operator job is feedback that must actually
    # reach the optimizer.  The remaining historical source groups still form
    # validation/test cohorts in the splitter.
    split_cfg["force_train_sample_ids"] = list(target.sample_ids)
    training_cfg = updated.setdefault("yolo_training", {})
    configured_names = training_cfg.get("class_names") or []
    if target.class_names:
        if configured_names and list(configured_names) != list(target.class_names):
            raise OperatorHandoffError(
                "Training class order does not match the inference model: "
                f"training={list(configured_names)!r}, "
                f"inference={list(target.class_names)!r}"
            )
        training_cfg["class_names"] = list(target.class_names)
    elif target.observed_class_map:
        if not isinstance(configured_names, list) or not configured_names:
            raise OperatorHandoffError(
                "Training class_names are required to validate legacy detections."
            )
        for class_id, class_name in target.observed_class_map:
            if (
                class_id >= len(configured_names)
                or str(configured_names[class_id]) != class_name
            ):
                raise OperatorHandoffError(
                    "Training class mapping does not match inference: "
                    f"id {class_id} must be {class_name!r}."
                )
    training_cfg["dataset_dir"] = str(target.dataset_root / "split")
    feedback_summary = _summarize_operator_feedback(target)
    updated["operator_feedback"] = feedback_summary
    eligibility_issues: list[str] = []
    if (
        feedback_summary["submitted_count"] > 0
        and feedback_summary["actionable_count"] == 0
        and feedback_summary["unknown_count"] == 0
    ):
        eligibility_issues.append("operator_feedback_not_actionable")
    try:
        deployed_training_weight = find_deployed_training_weight(handoff)
    except OperatorHandoffError as exc:
        deployed_training_weight = None
        eligibility_issues.append(str(exc))
    if eligibility_issues:
        raise OperatorHandoffError(
            "operator_training_preflight_failed: " + " | ".join(eligibility_issues)
        )
    if deployed_training_weight is not None:
        training_cfg["model"] = str(deployed_training_weight)
    configured_epochs = handoff.training_options.epochs
    training_cfg.update(
        {
            "epochs": configured_epochs,
            "batch": handoff.training_options.batch,
            "imgsz": handoff.training_options.imgsz,
            "optimizer": "AdamW",
            "lr0": min(float(training_cfg.get("lr0", 0.0005)), 0.0005),
            "lrf": float(training_cfg.get("lrf", 0.1)),
            "patience": min(max(configured_epochs // 2, 5), 20),
            "freeze": training_cfg.get("freeze", 10),
            "mosaic": 0.0,
            "fliplr": 0.0,
            "close_mosaic": 0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "erasing": 0.0,
            "cos_lr": True,
            "warmup_epochs": min(float(training_cfg.get("warmup_epochs", 1.0)), 1.0),
        }
    )
    deploy_cfg = training_cfg.setdefault("deploy", {})
    deploy_cfg.update(
        {
            "enabled": True,
            "inference_models_dir": str(handoff.inference_models_dir),
            "product": target.product,
            "area": target.area,
            "version": "auto",
            "preserve_station_settings": True,
            "force": False,
        }
    )
    readiness_cfg = updated.setdefault("dataset_readiness", {})
    configured_split_minimums = readiness_cfg.get("min_images_per_split", {})
    if not isinstance(configured_split_minimums, dict):
        configured_split_minimums = {}
    readiness_cfg.update(
        {
            "enabled": True,
            "review_manifest": str(
                target.dataset_root / "metadata" / "review_dataset_manifest.csv"
            ),
            "require_review_manifest": True,
            "min_instances_per_class": max(
                int(readiness_cfg.get("min_instances_per_class", 0)),
                OPERATOR_MIN_INSTANCES_PER_CLASS,
            ),
            "min_train_instances_per_class": max(
                int(readiness_cfg.get("min_train_instances_per_class", 0)),
                OPERATOR_MIN_TRAIN_INSTANCES_PER_CLASS,
            ),
            "min_test_instances_per_class": max(
                int(readiness_cfg.get("min_test_instances_per_class", 0)),
                OPERATOR_MIN_TEST_INSTANCES_PER_CLASS,
            ),
            "min_images_per_split": {
                split: max(int(configured_split_minimums.get(split, 0)), minimum)
                for split, minimum in OPERATOR_MIN_IMAGES_PER_SPLIT
            },
        }
    )
    # Keep QC evidence station-scoped. Path resolution replaces this placeholder
    # with runs/<product>/<area>/quality/qc_summary.json before execution.
    updated.setdefault("qc_summary", {}).setdefault(
        "output_path", "./runs/project/quality/qc_summary.json"
    )
    evaluation_cfg = updated.setdefault("yolo_evaluation", {})
    export_cfg = training_cfg.get("export_detection_config", {}) or {}
    production_confidence = float(
        export_cfg.get("conf_thres", DEFAULT_PRODUCTION_CONFIDENCE)
    )
    if not 0.0 <= production_confidence <= 1.0:
        raise OperatorHandoffError(
            "Production confidence threshold must be between 0 and 1."
        )
    evaluation_cfg.update(
        {
            "split": "test",
            "conf": production_confidence,
            "imgsz": handoff.training_options.imgsz,
        }
    )
    gate_cfg = evaluation_cfg.setdefault("gate", {})
    baseline_weights = find_deployed_runtime_weight(handoff)
    if baseline_weights is None:
        baseline_weights = deployed_training_weight
    if baseline_weights is None:
        raise OperatorHandoffError(
            "The currently deployed model is unavailable for challenger comparison."
        )
    gate_cfg.update(
        {
            "enabled": True,
            "require_metrics": True,
            "require_baseline": True,
            "compare_on_same_dataset": True,
            "baseline_weights": str(baseline_weights),
            "min_precision": min(
                1.0,
                max(
                    float(gate_cfg.get("min_precision", 0.0)),
                    OPERATOR_MIN_PRECISION,
                ),
            ),
            "min_recall": min(
                1.0,
                max(float(gate_cfg.get("min_recall", 0.0)), OPERATOR_MIN_RECALL),
            ),
            "min_map50": min(
                1.0,
                max(float(gate_cfg.get("min_map50", 0.0)), OPERATOR_MIN_MAP50),
            ),
            "min_map50_95": min(
                1.0,
                max(
                    float(gate_cfg.get("min_map50_95", 0.0)),
                    OPERATOR_MIN_MAP50_95,
                ),
            ),
            "max_regression": max(
                0.0,
                min(
                    float(
                        gate_cfg.get("max_regression", OPERATOR_MAX_METRIC_REGRESSION)
                    ),
                    OPERATOR_MAX_METRIC_REGRESSION,
                ),
            ),
            "baseline_manifest": str(
                handoff.inference_models_dir
                / target.product
                / target.area
                / "yolo"
                / "deployment_manifest.yaml"
            ),
        }
    )
    return updated


def find_deployed_training_weight(handoff: OperatorHandoff) -> Path | None:
    """Find the current PyTorch checkpoint used as the fine-tuning base.

    Args:
        handoff: Validated operator handoff identifying the product and area.

    Returns:
        ``best.pt`` or ``last.pt`` from the deployed station, when available.
    """
    target = handoff.selected_target
    station_dir = (
        handoff.inference_models_dir / target.product / target.area / "yolo"
    ).resolve()
    weights_dir = station_dir / "weights"
    runtime_weight = find_deployed_runtime_weight(handoff)
    manifest_path = station_dir / "deployment_manifest.yaml"
    if manifest_path.is_file():
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise OperatorHandoffError(
                f"Unable to read deployment artifact contract: {exc}"
            ) from exc
        if not isinstance(manifest, dict):
            raise OperatorHandoffError("Deployment artifact contract is invalid.")
        training_name = str(manifest.get("training_weight_file") or "").strip()
        training_sha256 = (
            str(manifest.get("training_weight_sha256") or "").strip().lower()
        )
        if training_name and training_sha256:
            candidate = (weights_dir / Path(training_name).name).resolve()
            if (
                candidate.parent != weights_dir.resolve()
                or candidate.suffix.lower() != ".pt"
            ):
                raise OperatorHandoffError(
                    "Deployment training-weight contract contains an unsafe path."
                )
            if not candidate.is_file():
                raise OperatorHandoffError(
                    f"Paired deployed training weight is missing: {candidate}"
                )
            if _sha256_file(candidate) != training_sha256:
                raise OperatorHandoffError(
                    "Paired deployed training weight checksum does not match."
                )
            deployed_file = str(manifest.get("deployed_file") or "").strip()
            runtime_sha256 = str(manifest.get("weight_sha256") or "").strip().lower()
            if runtime_weight is None or runtime_weight.name != deployed_file:
                raise OperatorHandoffError(
                    "Runtime model does not match the deployment artifact contract."
                )
            if runtime_sha256 and _sha256_file(runtime_weight) != runtime_sha256:
                raise OperatorHandoffError(
                    "Runtime model checksum does not match the deployment contract."
                )
            return candidate

    if runtime_weight is not None and runtime_weight.suffix.lower() == ".pt":
        return runtime_weight
    if runtime_weight is not None:
        raise OperatorHandoffError(
            "deployed_training_pair_missing: the current runtime model is "
            f"{runtime_weight.name}, but no exact paired .pt training weight was "
            "recorded. Re-deploy the current model with an artifact contract before "
            "retraining."
        )
    for filename in ("best.pt", "last.pt"):
        candidate = weights_dir / filename
        if candidate.is_file():
            return candidate.resolve()
    versioned = sorted(
        weights_dir.glob("*.pt"),
        key=lambda candidate: candidate.stat().st_mtime_ns,
        reverse=True,
    )
    return versioned[0].resolve() if versioned else None


def _summarize_operator_feedback(target: OperatorTarget) -> dict[str, int]:
    """Count the submitted feedback types used to decide training eligibility."""
    submitted_ids = set(target.sample_ids)
    summary = {
        "submitted_count": len(submitted_ids),
        "corrected_count": 0,
        "verified_empty_count": 0,
        "confirmed_count": 0,
        "unknown_count": 0,
        "actionable_count": 0,
    }
    if not submitted_ids:
        return summary
    manifest_path = target.dataset_root / "metadata" / "review_dataset_manifest.csv"
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if str(row.get("sample_id") or "") in submitted_ids
            ]
    except (OSError, UnicodeDecodeError, csv.Error):
        summary["unknown_count"] = len(submitted_ids)
        return summary

    seen_ids: set[str] = set()
    correction_labels = {"false_positive", "false_negative", "wrong_box", "wrong_class"}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in seen_ids:
            continue
        seen_ids.add(sample_id)
        review_label = str(row.get("review_label") or "").strip().lower()
        annotation_status = str(row.get("annotation_status") or "").strip().lower()
        if review_label in correction_labels:
            summary["corrected_count"] += 1
        elif review_label == "verified_empty" or annotation_status == "verified_empty":
            summary["verified_empty_count"] += 1
        elif review_label == "confirmed_ng":
            summary["confirmed_count"] += 1
        else:
            summary["unknown_count"] += 1
    summary["unknown_count"] += len(submitted_ids - seen_ids)
    summary["actionable_count"] = (
        summary["corrected_count"] + summary["verified_empty_count"]
    )
    return summary


def _sha256_file(path: Path) -> str:
    """Return a stable artifact identity for deployment-contract validation."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_deployed_runtime_weight(handoff: OperatorHandoff) -> Path | None:
    """Resolve the exact runtime artifact referenced by deployed config.yaml."""
    target = handoff.selected_target
    station_dir = (
        handoff.inference_models_dir / target.product / target.area / "yolo"
    ).resolve()
    config_path = station_dir / "config.yaml"
    if not config_path.is_file():
        return None
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    if not isinstance(payload, dict):
        return None
    raw_weights = str(payload.get("weights") or "").strip()
    if not raw_weights:
        return None
    path = Path(raw_weights)
    candidates = (
        [path]
        if path.is_absolute()
        else [handoff.inference_models_dir.parent / path, station_dir / path.name]
    )
    allowed_root = handoff.inference_models_dir.parent.resolve()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_relative_to(allowed_root) and resolved.is_file():
            return resolved
    return None


def _handoff_with_snapshot(
    handoff: OperatorHandoff, source_target: OperatorTarget, snapshot_root: Path
) -> OperatorHandoff:
    snapshot_target = replace(
        source_target,
        dataset_root=snapshot_root.resolve(),
        ready_count=_csv_row_count(
            snapshot_root / "metadata" / "review_dataset_manifest.csv"
        ),
        total_ready_count=_csv_row_count(
            snapshot_root / "metadata" / "review_dataset_manifest.csv"
        ),
        pending_count=0,
    )
    targets = tuple(
        snapshot_target if target is source_target else target
        for target in handoff.targets
    )
    return replace(
        handoff,
        ready_count=snapshot_target.ready_count,
        pending_count=0,
        targets=targets,
    )


def _snapshot_is_complete(snapshot_root: Path, manifest_path: Path) -> bool:
    if not manifest_path.is_file():
        return False
    return all(
        path.exists()
        for path in (
            snapshot_root / "raw" / "images",
            snapshot_root / "raw" / "labels",
            snapshot_root / "metadata" / "review_dataset_manifest.csv",
        )
    )


def _copy_directory_files(source: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for source_path in sorted(source.iterdir()):
        if not source_path.is_file() or source_path.name.startswith("."):
            continue
        shutil.copy2(source_path, destination / source_path.name)
        count += 1
    return count


def _directory_file_count(path: Path) -> int:
    """Count non-temporary files in one job-snapshot directory."""
    return sum(
        candidate.is_file() and not candidate.name.startswith(".")
        for candidate in path.iterdir()
    )


def _copy_legacy_raw_pairs(
    handoff: OperatorHandoff,
    target: OperatorTarget,
    destination_images: Path,
    destination_labels: Path,
) -> int:
    """Overlay validated product/area legacy raw pairs into a job snapshot."""
    legacy_root = handoff.data_root / target.product / "raw"
    legacy_images = legacy_root / "images"
    legacy_labels = legacy_root / "labels"
    if not legacy_images.is_dir() or not legacy_labels.is_dir():
        return 0

    target_token = f"_{target.product}_{target.area}_".casefold()
    copied_count = 0
    for source_image in sorted(legacy_images.iterdir()):
        if (
            not source_image.is_file()
            or source_image.suffix.casefold() not in YOLO_IMAGE_SUFFIXES
            or target_token not in source_image.stem.casefold()
        ):
            continue
        source_label = legacy_labels / f"{source_image.stem}.txt"
        if not source_label.is_file():
            raise OperatorHandoffError(
                f"Legacy training image has no matching label: {source_image.name}"
            )
        try:
            label_text = source_label.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            raise OperatorHandoffError(
                f"Unable to read legacy label {source_label.name}: {exc}"
            ) from exc
        errors = validate_yolo_label_text(label_text, len(target.class_names))
        if errors:
            raise OperatorHandoffError(
                f"Invalid legacy label {source_label.name}: " + "; ".join(errors[:5])
            )

        image_copied = _copy_snapshot_file(
            source_image, destination_images / source_image.name
        )
        _copy_snapshot_file(source_label, destination_labels / source_label.name)
        copied_count += int(image_copied)
    return copied_count


def _copy_snapshot_file(source: Path, destination: Path) -> bool:
    """Copy without overwriting different job-snapshot content."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not filecmp.cmp(source, destination, shallow=False):
            raise OperatorHandoffError(
                f"Conflicting snapshot file name: {destination.name}"
            )
        return False
    shutil.copy2(source, destination)
    return True


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _class_schema_hash(class_names: tuple[str, ...]) -> str:
    """Return the checksum used by the inference-to-training class contract."""
    serialized = json.dumps(
        [str(name) for name in class_names],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _manifest_sample_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return {
                str(row.get("sample_id") or row.get("image_sha256") or "")
                for row in csv.DictReader(handle)
                if str(row.get("sample_id") or row.get("image_sha256") or "")
            }
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise OperatorHandoffError(
            f"Unable to read annotation manifest: {exc}"
        ) from exc


def _csv_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return sum(1 for _row in csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise OperatorHandoffError(f"Unable to read dataset manifest: {exc}") from exc


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_segment(value: str, field_name: str) -> None:
    """Reject path separators and unsafe product/area values."""
    if value in {".", ".."} or any(
        not (character.isalnum() or character in "._-") for character in value
    ):
        raise OperatorHandoffError(f"Invalid {field_name}: {value!r}")
