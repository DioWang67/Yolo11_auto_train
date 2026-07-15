"""Fail-fast checks for YOLO datasets and reviewed production samples."""

from __future__ import annotations

import csv
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from picture_tool.utils.io_utils import DEFAULT_IMAGE_EXTS


class DatasetReadinessError(ValueError):
    """Raised when a dataset is unsafe to use for training."""


@dataclass
class DatasetReadinessReport:
    """Summary of dataset checks performed before training."""

    images: int = 0
    labels: int = 0
    verified_empty: int = 0
    class_counts: dict[int, int] = field(default_factory=dict)
    split_image_counts: dict[str, int] = field(default_factory=dict)
    split_class_counts: dict[str, dict[int, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def validate_training_dataset(
    config: dict[str, Any], logger: logging.Logger | None = None
) -> DatasetReadinessReport:
    """Validate split data and optional review-manifest annotation states.

    Args:
        config: Pipeline configuration containing ``yolo_training`` and
            optional ``dataset_readiness`` sections.
        logger: Optional logger for warnings and summary output.

    Returns:
        A readiness report when no blocking issue is found.

    Raises:
        DatasetReadinessError: If labels are pending, malformed, missing, or
            duplicated across train/validation/test splits.
    """
    logger = logger or logging.getLogger(__name__)
    readiness_cfg = config.get("dataset_readiness", {}) or {}
    if not readiness_cfg.get("enabled", True):
        return DatasetReadinessReport()

    training_cfg = config.get("yolo_training", {}) or {}
    dataset_dir = Path(str(training_cfg.get("dataset_dir") or "")).expanduser()
    class_names = training_cfg.get("class_names") or []
    num_classes = len(class_names) if isinstance(class_names, list) else 0
    report = DatasetReadinessReport()

    if not dataset_dir.exists():
        raise DatasetReadinessError(f"Training dataset does not exist: {dataset_dir}")
    if num_classes <= 0:
        raise DatasetReadinessError("yolo_training.class_names must not be empty")

    manifest_value = readiness_cfg.get("review_manifest")
    manifest_path = (
        Path(str(manifest_value))
        if manifest_value
        else dataset_dir.parent / "metadata" / "review_dataset_manifest.csv"
    )
    verified_empty_stems: set[str] = set()
    if manifest_path.exists():
        verified_empty_stems = _validate_review_manifest(
            manifest_path, num_classes, report
        )
    elif readiness_cfg.get("require_review_manifest", False):
        report.errors.append(f"review_manifest_missing:{manifest_path}")

    split_hashes: dict[str, str] = {}
    source_splits: dict[str, str] = {}
    for split in ("train", "val", "test"):
        _validate_split(
            dataset_dir,
            split,
            num_classes,
            verified_empty_stems,
            split_hashes,
            source_splits,
            report,
        )

    if not (dataset_dir / "train" / "images").exists():
        report.errors.append("train_images_directory_missing")
    if not (dataset_dir / "val" / "images").exists():
        report.errors.append("val_images_directory_missing")

    report.class_counts = _count_class_instances(dataset_dir, num_classes)
    report.split_image_counts = {
        split: _count_split_images(dataset_dir, split)
        for split in ("train", "val", "test")
    }
    report.split_class_counts = {
        split: _count_class_instances_for_split(dataset_dir, split, num_classes)
        for split in ("train", "val", "test")
    }
    minimum_images = readiness_cfg.get("min_images_per_split", {}) or {}
    if isinstance(minimum_images, dict):
        for split in ("train", "val", "test"):
            minimum = int(minimum_images.get(split, 0))
            count = report.split_image_counts.get(split, 0)
            if minimum > 0 and count < minimum:
                report.errors.append(
                    f"split_underrepresented:{split}:{count}:required={minimum}"
                )
    minimum_per_class = int(readiness_cfg.get("min_instances_per_class", 0))
    if minimum_per_class > 0:
        for class_id, class_name in enumerate(class_names):
            count = report.class_counts.get(class_id, 0)
            if count < minimum_per_class:
                report.errors.append(
                    f"class_underrepresented:{class_id}:{class_name}:{count}"
                )
    for split, config_key in (
        ("train", "min_train_instances_per_class"),
        ("test", "min_test_instances_per_class"),
    ):
        minimum = int(readiness_cfg.get(config_key, 0))
        if minimum <= 0:
            continue
        split_counts = report.split_class_counts[split]
        for class_id, class_name in enumerate(class_names):
            count = split_counts.get(class_id, 0)
            if count < minimum:
                report.errors.append(
                    f"{split}_class_underrepresented:"
                    f"{class_id}:{class_name}:{count}:required={minimum}"
                )

    if report.errors:
        preview = "; ".join(report.errors[:10])
        suffix = f" (+{len(report.errors) - 10} more)" if len(report.errors) > 10 else ""
        raise DatasetReadinessError(
            f"Dataset readiness failed with {len(report.errors)} issue(s): "
            f"{preview}{suffix}"
        )

    for warning in report.warnings:
        logger.warning("Dataset readiness: %s", warning)
    logger.info(
        "Dataset ready: images=%d labels=%d verified_empty=%d",
        report.images,
        report.labels,
        report.verified_empty,
    )
    return report


def _validate_review_manifest(
    manifest_path: Path, num_classes: int, report: DatasetReadinessReport
) -> set[str]:
    verified_empty: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            image_path = Path(str(row.get("output_image") or ""))
            label_text = str(row.get("output_label") or "").strip()
            label_path = Path(label_text) if label_text else _expected_label_path(image_path)
            status = str(row.get("annotation_status") or "pending").strip().lower()
            nonempty = label_path.exists() and bool(label_path.read_text(encoding="utf-8").strip())

            if status == "verified_empty":
                if not label_path.exists():
                    report.errors.append(
                        f"manifest:{row_number}:verified_empty_label_missing:{label_path}"
                    )
                elif nonempty:
                    report.errors.append(
                        f"manifest:{row_number}:verified_empty_label_not_empty:{label_path}"
                    )
                else:
                    verified_empty.add(image_path.stem)
                    report.verified_empty += 1
                continue

            if not nonempty:
                report.errors.append(
                    f"manifest:{row_number}:annotation_pending:{image_path}"
                )
                continue
            _validate_label_file(label_path, num_classes, report.errors)
    return verified_empty


def _expected_label_path(image_path: Path) -> Path:
    if image_path.parent.name == "images":
        return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    return image_path.with_suffix(".txt")


def _validate_split(
    dataset_dir: Path,
    split: str,
    num_classes: int,
    verified_empty_stems: set[str],
    split_hashes: dict[str, str],
    source_splits: dict[str, str],
    report: DatasetReadinessReport,
) -> None:
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"
    if not images_dir.exists():
        return

    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in DEFAULT_IMAGE_EXTS:
            continue
        report.images += 1
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            report.errors.append(f"{split}:missing_label:{image_path.name}")
            continue
        report.labels += 1
        if not label_path.read_text(encoding="utf-8").strip():
            if image_path.stem not in verified_empty_stems:
                report.warnings.append(f"{split}:legacy_empty_label:{label_path.name}")
        else:
            _validate_label_file(label_path, num_classes, report.errors)

        digest = _sha256_file(image_path)
        previous_split = split_hashes.get(digest)
        if previous_split is not None and previous_split != split:
            report.errors.append(
                f"split_leakage:{image_path.name}:{previous_split}->{split}"
            )
        else:
            split_hashes[digest] = split

        source_key = re.sub(
            r"(?:_aug_?\d+)$", "", image_path.stem, flags=re.IGNORECASE
        )
        previous_source_split = source_splits.get(source_key)
        if previous_source_split is not None and previous_source_split != split:
            report.errors.append(
                f"source_split_leakage:{source_key}:{previous_source_split}->{split}"
            )
        else:
            source_splits[source_key] = split


def _validate_label_file(
    label_path: Path, num_classes: int, errors: list[str]
) -> None:
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"label_unreadable:{label_path}:{exc}")
        return
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"label_format:{label_path}:{line_number}")
            continue
        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"label_value:{label_path}:{line_number}")
            continue
        if not 0 <= class_id < num_classes:
            errors.append(f"class_out_of_range:{label_path}:{line_number}:{class_id}")
        if any(value < 0.0 or value > 1.0 for value in coords):
            errors.append(f"bbox_out_of_range:{label_path}:{line_number}")
        if coords[2] <= 0.0 or coords[3] <= 0.0:
            errors.append(f"bbox_non_positive:{label_path}:{line_number}")


def _count_class_instances(dataset_dir: Path, num_classes: int) -> dict[int, int]:
    counts = {class_id: 0 for class_id in range(num_classes)}
    for split in ("train", "val", "test"):
        split_counts = _count_class_instances_for_split(
            dataset_dir, split, num_classes
        )
        for class_id, count in split_counts.items():
            counts[class_id] += count
    return counts


def _count_class_instances_for_split(
    dataset_dir: Path, split: str, num_classes: int
) -> dict[int, int]:
    counts = {class_id: 0 for class_id in range(num_classes)}
    labels_dir = dataset_dir / split / "labels"
    if not labels_dir.exists():
        return counts
    for label_path in labels_dir.glob("*.txt"):
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                continue
            if class_id in counts:
                counts[class_id] += 1
    return counts


def _count_split_images(dataset_dir: Path, split: str) -> int:
    images_dir = dataset_dir / split / "images"
    if not images_dir.exists():
        return 0
    return sum(
        path.is_file() and path.suffix.lower() in DEFAULT_IMAGE_EXTS
        for path in images_dir.iterdir()
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
