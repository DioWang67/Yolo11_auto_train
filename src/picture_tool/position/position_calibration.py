"""Human-verified YOLO annotation reader for position calibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import cv2

from picture_tool.position.yolo_position_validator import _letterbox_transform


IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
)


class PositionCalibrationError(RuntimeError):
    """Raised when human calibration evidence cannot be trusted."""


@dataclass(frozen=True)
class CalibrationSample:
    image_path: Path
    label_path: Path
    image_sha256: str
    label_sha256: str
    class_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "class_counts",
            MappingProxyType(dict(self.class_counts)),
        )

    def as_manifest_record(self) -> dict[str, Any]:
        return {
            "image_path": str(self.image_path.resolve()),
            "label_path": str(self.label_path.resolve()),
            "image_sha256": self.image_sha256,
            "label_sha256": self.label_sha256,
            "class_counts": dict(self.class_counts),
        }


@dataclass(frozen=True)
class CalibrationDataset:
    boxes_by_class: Mapping[str, tuple[tuple[int, int, int, int], ...]]
    per_image_class_counts: Mapping[str, tuple[int, ...]]
    samples: tuple[CalibrationSample, ...]
    excluded_samples: tuple[Mapping[str, str], ...]
    dataset_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "boxes_by_class",
            MappingProxyType(
                {
                    name: tuple(tuple(box) for box in boxes)
                    for name, boxes in self.boxes_by_class.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "per_image_class_counts",
            MappingProxyType(
                {
                    name: tuple(int(count) for count in counts)
                    for name, counts in self.per_image_class_counts.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "excluded_samples",
            tuple(
                MappingProxyType(dict(record))
                for record in self.excluded_samples
            ),
        )

    def manifest_payload(
        self,
        *,
        product: str,
        area: str,
        imgsz: int,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source": "human_verified_yolo_labels",
            "product": product,
            "area": area,
            "imgsz": imgsz,
            "dataset_sha256": self.dataset_sha256,
            "sample_count": len(self.samples),
            "samples": [sample.as_manifest_record() for sample in self.samples],
            "excluded_samples": [
                dict(record) for record in self.excluded_samples
            ],
        }


def collect_yolo_calibration_dataset(
    *,
    image_dir: Path,
    label_dir: Path,
    class_names: Sequence[str],
    imgsz: int,
    require_all_classes: bool = True,
    exclude_augmented: bool = True,
) -> CalibrationDataset:
    """Convert trusted YOLO labels into inference-letterbox coordinates."""

    images = _index_images(image_dir, exclude_augmented=exclude_augmented)
    if not label_dir.is_dir():
        raise PositionCalibrationError(
            f"Position calibration label directory was not found: {label_dir}"
        )
    normalized_names = tuple(str(name).strip() for name in class_names)
    if not normalized_names or any(not name for name in normalized_names):
        raise PositionCalibrationError(
            "Position calibration requires a non-empty ordered class contract."
        )
    if len(set(normalized_names)) != len(normalized_names):
        raise PositionCalibrationError(
            "Position calibration class contract must not contain duplicate names."
        )
    if imgsz <= 0:
        raise PositionCalibrationError("Position calibration imgsz must be positive.")

    boxes_by_class: dict[str, list[tuple[int, int, int, int]]] = {}
    per_image_counts: dict[str, list[int]] = {
        name: [] for name in normalized_names
    }
    samples: list[CalibrationSample] = []
    excluded: list[dict[str, str]] = []
    required_classes = set(normalized_names)
    image_hash_sources: dict[str, Path] = {}

    for stem, image_path in sorted(images.items()):
        label_path = label_dir / f"{stem}.txt"
        if not label_path.is_file():
            excluded.append(
                {"image": str(image_path), "reason": "missing_label"}
            )
            continue
        raw_boxes, class_counts = _read_yolo_boxes(
            image_path=image_path,
            label_path=label_path,
            class_names=normalized_names,
            imgsz=imgsz,
        )
        if require_all_classes and not required_classes.issubset(class_counts):
            excluded.append(
                {
                    "image": str(image_path),
                    "reason": "missing_required_class",
                }
            )
            continue
        for class_name, boxes in raw_boxes.items():
            boxes_by_class.setdefault(class_name, []).extend(boxes)
        for class_name in normalized_names:
            per_image_counts[class_name].append(class_counts.get(class_name, 0))
        image_sha256 = sha256_file(image_path)
        duplicate_source = image_hash_sources.get(image_sha256)
        if duplicate_source is not None:
            raise PositionCalibrationError(
                "Duplicate calibration image content is not allowed: "
                f"{duplicate_source} and {image_path}"
            )
        image_hash_sources[image_sha256] = image_path
        samples.append(
            CalibrationSample(
                image_path=image_path,
                label_path=label_path,
                image_sha256=image_sha256,
                label_sha256=sha256_file(label_path),
                class_counts=class_counts,
            )
        )

    if not samples:
        raise PositionCalibrationError(
            "No complete human-labeled samples were available for position calibration."
        )
    missing_classes = [
        class_name for class_name in normalized_names if class_name not in boxes_by_class
    ]
    if missing_classes:
        raise PositionCalibrationError(
            "Position calibration has no boxes for classes: "
            + ", ".join(missing_classes)
        )

    dataset_sha256 = _dataset_identity(samples)
    return CalibrationDataset(
        boxes_by_class={
            name: tuple(boxes) for name, boxes in boxes_by_class.items()
        },
        per_image_class_counts={
            name: tuple(counts) for name, counts in per_image_counts.items()
        },
        samples=tuple(samples),
        excluded_samples=tuple(excluded),
        dataset_sha256=dataset_sha256,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_calibration_manifest(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _index_images(
    image_dir: Path,
    *,
    exclude_augmented: bool,
) -> dict[str, Path]:
    if not image_dir.is_dir():
        raise PositionCalibrationError(
            f"Position calibration image directory was not found: {image_dir}"
        )
    indexed: dict[str, Path] = {}
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if exclude_augmented and "_aug_" in path.stem.lower():
            continue
        if path.stem in indexed:
            raise PositionCalibrationError(
                f"Multiple calibration images share the same stem: {path.stem}"
            )
        indexed[path.stem] = path
    if not indexed:
        raise PositionCalibrationError(
            f"No calibration images were found under {image_dir}"
        )
    return indexed


def _read_yolo_boxes(
    *,
    image_path: Path,
    label_path: Path,
    class_names: Sequence[str],
    imgsz: int,
) -> tuple[
    dict[str, list[tuple[int, int, int, int]]],
    dict[str, int],
]:
    image = cv2.imread(str(image_path))
    if image is None or image.ndim < 2:
        raise PositionCalibrationError(
            f"Unable to decode position calibration image: {image_path}"
        )
    orig_h, orig_w = image.shape[:2]
    boxes: dict[str, list[tuple[int, int, int, int]]] = {}
    counts: dict[str, int] = {}
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise PositionCalibrationError(
            f"Unable to read position calibration label {label_path}: {exc}"
        ) from exc
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise PositionCalibrationError(
                f"{label_path}:{line_number} must contain exactly 5 YOLO values."
            )
        try:
            class_id = int(parts[0])
            cx, cy, width, height = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise PositionCalibrationError(
                f"{label_path}:{line_number} contains non-numeric YOLO values."
            ) from exc
        if not 0 <= class_id < len(class_names):
            raise PositionCalibrationError(
                f"{label_path}:{line_number} class id {class_id} is outside "
                f"the class contract."
            )
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)):
            raise PositionCalibrationError(
                f"{label_path}:{line_number} has normalized values outside [0, 1]."
            )
        if width <= 0.0 or height <= 0.0:
            raise PositionCalibrationError(
                f"{label_path}:{line_number} width and height must be positive."
            )
        original_box = (
            (cx - width / 2.0) * orig_w,
            (cy - height / 2.0) * orig_h,
            (cx + width / 2.0) * orig_w,
            (cy + height / 2.0) * orig_h,
        )
        transformed = _letterbox_transform(
            original_box,
            orig_w,
            orig_h,
            imgsz,
        )
        class_name = class_names[class_id]
        rounded = (
            int(round(transformed[0])),
            int(round(transformed[1])),
            int(round(transformed[2])),
            int(round(transformed[3])),
        )
        boxes.setdefault(class_name, []).append(rounded)
        counts[class_name] = counts.get(class_name, 0) + 1
    return boxes, counts


def _dataset_identity(samples: Sequence[CalibrationSample]) -> str:
    canonical = [
        {
            "image_sha256": sample.image_sha256,
            "label_sha256": sample.label_sha256,
        }
        for sample in samples
    ]
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
