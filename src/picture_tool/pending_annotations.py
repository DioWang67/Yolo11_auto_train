"""Validate pending YOLO annotations and promote them into the raw dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from picture_tool.operator_job import OperatorJobError, update_job_status


READY_FIELDS = [
    "source_manifest",
    "review_label",
    "review_note",
    "source_image",
    "output_image",
    "output_label",
    "annotation_status",
    "sample_id",
    "image_sha256",
    "product",
    "area",
    "timestamp",
    "status",
    "decision_reasons",
    "model_version",
    "class_names_json",
    "class_map_json",
    "class_schema_hash",
]

PENDING_FIELDS = [
    "sample_id",
    "image_sha256",
    "product",
    "area",
    "timestamp",
    "review_label",
    "reason",
    "annotation_status",
    "review_note",
    "status",
    "decision_reasons",
    "model_version",
    "class_names_json",
    "class_map_json",
    "class_schema_hash",
    "detections_json",
    "source_image",
    "output_image",
    "output_label",
    "label_baseline_sha256",
    "config_snapshot_path",
]

MISSING_LABEL_BASELINE = "missing"
VERIFICATION_RECEIPT_SCHEMA_VERSION = 1
CORRECTION_REASONS = {
    "false_detection_requires_correction",
    "wrong_class_requires_correction",
    "operator_uncertain_requires_review",
}


class PendingAnnotationError(ValueError):
    """Raised when pending annotation data is unsafe to promote."""


@dataclass(frozen=True)
class PendingPromotionReport:
    """Result of validating and promoting completed pending annotations."""

    promoted_count: int
    remaining_count: int
    total_ready_count: int
    handoff_path: Path


def configure_pending_workspace(
    dataset_root: str | Path, class_names: list[str]
) -> tuple[Path, Path, Path]:
    """Prepare LabelImg input/output paths and its ordered class file.

    Args:
        dataset_root: Product/station dataset root.
        class_names: Ordered YOLO class names.

    Returns:
        ``(images_dir, labels_dir, predefined_classes_file)``.

    Raises:
        PendingAnnotationError: If the class contract is empty.
    """
    if not class_names or any(not str(name).strip() for name in class_names):
        raise PendingAnnotationError("The annotation class list is empty or invalid.")
    root = Path(dataset_root).expanduser().resolve()
    pending_root = root / "review_pending"
    images_dir = pending_root / "images"
    labels_dir = pending_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    classes_file = pending_root / "predefined_classes.txt"
    class_text = "\n".join(class_names) + "\n"
    _write_text_atomic(classes_file, class_text)
    # Vendored LabelImg's YoloReader resolves classes.txt relative to each
    # annotation file, independently of the predefined-class CLI argument.
    _write_text_atomic(labels_dir / "classes.txt", class_text)
    return images_dir, labels_dir, classes_file


def promote_completed_pending(
    dataset_root: str | Path,
    class_names: list[str],
    handoff_path: str | Path,
) -> PendingPromotionReport:
    """Promote explicitly saved LabelImg labels into ``raw`` safely.

    A pending image remains pending when its label file does not exist.
    Non-empty labels must pass class and bounding-box validation. An empty
    missed-detection label is rejected; only an explicit verified-background
    review may promote an empty label.
    """
    root = Path(dataset_root).expanduser().resolve()
    handoff = Path(handoff_path).expanduser().resolve()
    images_dir, labels_dir, _classes_file = configure_pending_workspace(
        root, class_names
    )
    pending_manifest = root / "review_pending" / "manifest.csv"
    if not pending_manifest.is_file():
        raise PendingAnnotationError(
            f"Pending annotation manifest not found: {pending_manifest}"
        )

    with _target_lock(root):
        pending_rows = _read_csv(pending_manifest)
        pending_scope = _handoff_pending_scope(handoff, root)
        ready_manifest = root / "metadata" / "review_dataset_manifest.csv"
        ready_by_id = {
            str(row.get("sample_id") or row.get("image_sha256") or ""): row
            for row in _read_csv(ready_manifest)
            if str(row.get("sample_id") or row.get("image_sha256") or "")
        }
        remaining: list[dict[str, str]] = []
        promoted = 0

        for row in pending_rows:
            row_sample_id = str(
                row.get("sample_id") or row.get("image_sha256") or ""
            )
            if pending_scope is not None and row_sample_id not in pending_scope:
                remaining.append(row)
                continue
            image_path = _managed_path(
                row.get("output_image"), images_dir, field_name="output_image"
            )
            if image_path is None or not image_path.is_file():
                remaining.append(row)
                continue
            label_value = str(row.get("output_label") or "")
            label_path = (
                _managed_path(label_value, labels_dir, field_name="output_label")
                if label_value
                else labels_dir / f"{image_path.stem}.txt"
            )
            if label_path is None or not label_path.is_file():
                remaining.append(row)
                continue

            if not _label_changed_since_handoff(row, label_path):
                remaining.append(row)
                continue

            label_text = label_path.read_text(encoding="utf-8")
            errors = validate_yolo_label_text(label_text, len(class_names))
            if errors:
                preview = "; ".join(errors[:5])
                raise PendingAnnotationError(
                    f"Invalid annotation {label_path.name}: {preview}"
                )
            review_label = str(row.get("review_label") or "").strip().lower()
            reason = str(row.get("reason") or "").strip().lower()
            if not label_text.strip() and (
                review_label == "false_negative"
                or reason == "missed_detection_requires_box_annotation"
            ):
                raise PendingAnnotationError(
                    f"漏檢影像 {image_path.name} 尚未框出目標。"
                    "請在標註工具至少畫一個正確框並儲存；"
                    "若影像中其實沒有目標，請回到檢測複核畫面改選「影像中沒有目標」。"
                )

            image_sha256 = str(row.get("image_sha256") or "") or _sha256_file(
                image_path
            )
            sample_id = str(row.get("sample_id") or "") or image_sha256[:24]
            raw_images = root / "raw" / "images"
            raw_labels = root / "raw" / "labels"
            raw_images.mkdir(parents=True, exist_ok=True)
            raw_labels.mkdir(parents=True, exist_ok=True)
            output_image = raw_images / f"review_{sample_id}{image_path.suffix.lower()}"
            output_label = raw_labels / f"review_{sample_id}.txt"
            _copy_atomic(image_path, output_image)
            _write_text_atomic(output_label, label_text)

            normalized_names = json.dumps(
                [str(name) for name in class_names],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            ready_by_id[sample_id] = {
                "source_manifest": str(row.get("config_snapshot_path") or ""),
                "review_label": str(row.get("review_label") or ""),
                "review_note": str(row.get("review_note") or ""),
                "source_image": str(row.get("source_image") or image_path),
                "output_image": str(output_image),
                "output_label": str(output_label),
                "annotation_status": (
                    "verified_annotation" if label_text.strip() else "verified_empty"
                ),
                "sample_id": sample_id,
                "image_sha256": image_sha256,
                "product": str(row.get("product") or ""),
                "area": str(row.get("area") or ""),
                "timestamp": str(row.get("timestamp") or ""),
                "status": str(row.get("status") or ""),
                "decision_reasons": str(row.get("decision_reasons") or ""),
                "model_version": str(row.get("model_version") or ""),
                "class_names_json": normalized_names,
                "class_map_json": str(row.get("class_map_json") or "{}"),
                "class_schema_hash": _class_schema_hash(class_names),
            }
            image_path.unlink()
            label_path.unlink()
            _verification_receipt_path(label_path).unlink(missing_ok=True)
            promoted += 1

        ready_rows = sorted(
            ready_by_id.values(), key=lambda row: str(row.get("sample_id") or "")
        )
        job_remaining_count = (
            len(remaining)
            if pending_scope is None
            else sum(
                str(row.get("sample_id") or row.get("image_sha256") or "")
                in pending_scope
                for row in remaining
            )
        )
        _write_csv_atomic(ready_manifest, READY_FIELDS, ready_rows)
        _write_csv_atomic(pending_manifest, PENDING_FIELDS, remaining)
        _rebuild_master_manifests(root.parents[1])
        _update_handoff_counts(
            handoff,
            root,
            promoted_count=promoted,
            total_ready_count=len(ready_rows),
            remaining_count=job_remaining_count,
        )

    return PendingPromotionReport(
        promoted_count=promoted,
        remaining_count=job_remaining_count,
        total_ready_count=len(ready_rows),
        handoff_path=handoff,
    )


def _handoff_pending_scope(
    handoff_path: Path, dataset_root: Path
) -> set[str] | None:
    """Return schema-v3 sample IDs owned by this immutable operator job."""
    try:
        payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PendingAnnotationError(f"Unable to read handoff: {exc}") from exc
    if not isinstance(payload, dict):
        raise PendingAnnotationError("Invalid handoff payload.")
    try:
        schema_version = int(payload.get("schema_version", 0))
    except (TypeError, ValueError) as exc:
        raise PendingAnnotationError("Invalid handoff schema version.") from exc
    if schema_version < 3:
        return None
    for target in payload.get("targets", []):
        if not isinstance(target, dict):
            continue
        target_root = Path(str(target.get("dataset_root") or "")).resolve()
        if target_root != dataset_root:
            continue
        raw_ids = target.get("pending_sample_ids")
        if not isinstance(raw_ids, list):
            raise PendingAnnotationError("Job pending sample IDs are invalid.")
        return {str(sample_id) for sample_id in raw_ids if str(sample_id)}
    raise PendingAnnotationError("The handoff does not contain this dataset target.")


def validate_yolo_label_text(label_text: str, num_classes: int) -> list[str]:
    """Validate one YOLO detection label, including empty negative labels."""
    errors: list[str] = []
    for line_number, raw_line in enumerate(label_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {line_number}: expected 5 values")
            continue
        try:
            class_id = int(parts[0])
            center_x, center_y, width, height = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"line {line_number}: invalid numeric value")
            continue
        if not 0 <= class_id < num_classes:
            errors.append(f"line {line_number}: class {class_id} is out of range")
        if any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in (center_x, center_y, width, height)
        ):
            errors.append(f"line {line_number}: bounding box is out of range")
        if width <= 0.0 or height <= 0.0:
            errors.append(f"line {line_number}: width and height must be positive")
    return errors


def _label_changed_since_handoff(row: dict[str, str], label_path: Path) -> bool:
    """Return whether an operator produced or changed the pending label.

    Correction cases may contain an automatically generated draft. Merely
    closing LabelImg must not promote that unverified draft into training.
    """
    if _has_matching_verification_receipt(label_path):
        return True
    baseline = str(row.get("label_baseline_sha256") or "").strip().lower()
    if baseline == MISSING_LABEL_BASELINE:
        return True
    if not baseline:
        # Legacy missed-detection rows never had a generated draft. Legacy
        # correction rows are unsafe because save intent cannot be proven.
        return str(row.get("reason") or "") not in CORRECTION_REASONS
    if len(baseline) != 64 or any(
        char not in "0123456789abcdef" for char in baseline
    ):
        raise PendingAnnotationError(
            f"Invalid label baseline for {label_path.name}."
        )
    return _sha256_file(label_path) != baseline


def record_label_verification(
    label_path: str | Path,
    labels_root: str | Path,
) -> Path:
    """Persist an auditable receipt for one label explicitly saved in LabelImg."""
    root = Path(labels_root).expanduser().resolve()
    path = Path(label_path).expanduser().resolve()
    if path.parent != root or path.suffix.lower() != ".txt":
        raise PendingAnnotationError(f"Unsafe verified label path: {path}")
    if path.name.lower() == "classes.txt" or not path.is_file():
        raise PendingAnnotationError(f"Verified label does not exist: {path}")
    receipt_path = _verification_receipt_path(path)
    _write_json_atomic(
        receipt_path,
        {
            "schema_version": VERIFICATION_RECEIPT_SCHEMA_VERSION,
            "label_name": path.name,
            "label_sha256": _sha256_file(path),
            "verified_at_unix_ns": time.time_ns(),
        },
    )
    return receipt_path


def _verification_receipt_path(label_path: Path) -> Path:
    return label_path.parent / ".verified" / f"{label_path.stem}.json"


def _has_matching_verification_receipt(label_path: Path) -> bool:
    receipt_path = _verification_receipt_path(label_path)
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    try:
        schema_version = int(payload.get("schema_version", 0))
    except (TypeError, ValueError):
        return False
    return (
        schema_version == VERIFICATION_RECEIPT_SCHEMA_VERSION
        and str(payload.get("label_name") or "") == label_path.name
        and str(payload.get("label_sha256") or "").lower()
        == _sha256_file(label_path)
    )


def _managed_path(
    raw_value: Any, allowed_root: Path, *, field_name: str
) -> Path | None:
    value = str(raw_value or "").strip()
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(allowed_root.resolve()):
        raise PendingAnnotationError(f"Unsafe {field_name}: {path}")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise PendingAnnotationError(f"Unable to read {path}: {exc}") from exc


def _write_csv_atomic(
    path: Path, fields: list[str], rows: list[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                [{field: str(row.get(field) or "") for field in fields} for row in rows]
            )
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _rebuild_master_manifests(data_root: Path) -> None:
    ready: dict[tuple[str, str, str], dict[str, str]] = {}
    pending: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in sorted(data_root.glob("*/*/metadata/review_dataset_manifest.csv")):
        for row in _read_csv(path):
            key = (
                str(row.get("product") or ""),
                str(row.get("area") or ""),
                str(row.get("sample_id") or row.get("image_sha256") or ""),
            )
            ready[key] = row
    for path in sorted(data_root.glob("*/*/review_pending/manifest.csv")):
        for row in _read_csv(path):
            key = (
                str(row.get("product") or ""),
                str(row.get("area") or ""),
                str(row.get("sample_id") or row.get("config_snapshot_path") or ""),
            )
            pending[key] = row
    _write_csv_atomic(
        data_root / "metadata" / "review_dataset_manifest.csv",
        READY_FIELDS,
        list(ready.values()),
    )
    _write_csv_atomic(
        data_root / ".operator_handoff" / "pending.csv",
        PENDING_FIELDS,
        list(pending.values()),
    )


def _update_handoff_counts(
    handoff_path: Path,
    dataset_root: Path,
    *,
    promoted_count: int,
    total_ready_count: int,
    remaining_count: int,
) -> None:
    if not handoff_path.is_file():
        raise PendingAnnotationError(f"Handoff file not found: {handoff_path}")
    try:
        payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PendingAnnotationError(f"Unable to update handoff: {exc}") from exc
    if not isinstance(payload, dict):
        raise PendingAnnotationError("Invalid handoff payload.")
    if int(payload.get("schema_version", 0)) >= 3:
        status_path = Path(str(payload.get("status_path") or "")).resolve()
        expected_status = handoff_path.parent / "status.json"
        if status_path != expected_status:
            raise PendingAnnotationError("Invalid operator job status path.")
        try:
            update_job_status(
                status_path,
                state="waiting_annotation" if remaining_count else "queued",
                message=(
                    f"尚有 {remaining_count} 張需要補標"
                    if remaining_count
                    else "補標完成，準備建立訓練資料快照"
                ),
                ready_count=total_ready_count,
                pending_count=remaining_count,
                progress=5 if not remaining_count else 0,
            )
        except OperatorJobError as exc:
            raise PendingAnnotationError(str(exc)) from exc
        return
    matched = False
    for target in payload.get("targets", []):
        if not isinstance(target, dict):
            continue
        if Path(str(target.get("dataset_root") or "")).resolve() != dataset_root:
            continue
        target["ready_count"] = promoted_count
        target["total_ready_count"] = total_ready_count
        target["pending_count"] = remaining_count
        matched = True
    if not matched:
        raise PendingAnnotationError("The handoff does not contain this dataset target.")
    payload["ready_count"] = promoted_count
    payload["total_ready_count"] = sum(
        int(target.get("total_ready_count", 0))
        for target in payload.get("targets", [])
        if isinstance(target, dict)
    )
    payload["pending_count"] = sum(
        int(target.get("pending_count", 0))
        for target in payload.get("targets", [])
        if isinstance(target, dict)
    )
    _write_json_atomic(handoff_path, payload)


def _class_schema_hash(class_names: list[str]) -> str:
    serialized = json.dumps(
        [str(name) for name in class_names],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _target_lock(dataset_root: Path, timeout_seconds: float = 10.0) -> Iterator[None]:
    lock_path = dataset_root / ".annotation.lock"
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, str(os.getpid()).encode("ascii"))
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime > 300.0
            except FileNotFoundError:
                continue
            if stale:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise PendingAnnotationError(
                    "This product/station annotation queue is already in use."
                )
            time.sleep(0.1)
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
