"""Readiness preview helpers for project/area training inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from picture_tool.path_resolver import parse_project_area_override, resolve_project_paths
from picture_tool.pipeline.artifacts import (
    find_anomalib_run_artifact,
    find_yolo_run_artifact,
)

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ProjectReadiness:
    """Summary of data and artifact availability for a product/area target."""

    product: str
    area: str | None
    raw_images: int
    raw_labels: int
    processed_images: int
    split_train_images: int
    split_val_images: int
    split_test_images: int
    anomalib_normal_images: int
    anomalib_abnormal_images: int
    latest_yolo_run: Path | None
    latest_anomalib_run: Path | None
    package_output_dir: Path
    warnings: tuple[str, ...]

    @property
    def is_ready_for_yolo(self) -> bool:
        """Return whether enough source data exists to start YOLO flow."""

        has_raw_pair = self.raw_images > 0 and self.raw_labels > 0
        has_split = self.split_train_images > 0 and self.split_val_images > 0
        return has_raw_pair or has_split

    @property
    def is_ready_for_anomalib(self) -> bool:
        """Return whether enough normal images exist to start Anomalib flow."""

        return self.anomalib_normal_images > 0


def build_project_readiness(
    config: dict,
    product_override: str,
) -> ProjectReadiness:
    """Build a filesystem readiness summary for a GUI product override.

    Args:
        config: Current pipeline configuration.
        product_override: Product text from GUI, e.g. ``"PCBA1,C"``.

    Returns:
        A readiness summary with counts and warnings.

    Raises:
        ValueError: If the product override is invalid.
    """

    parsed = parse_project_area_override(product_override)
    resolved_config = resolve_project_paths(config or {}, product_override)

    data_root = Path("data") / parsed.project
    runs_root = Path("runs") / parsed.project
    if parsed.area:
        data_root = data_root / parsed.area
        runs_root = runs_root / parsed.area

    raw_root = data_root / "raw"
    processed_root = data_root / "processed"
    split_root = data_root / "split"

    train_cfg = resolved_config.get("anomalib_training", {}) or {}
    package_cfg = resolved_config.get("anomalib_package", {}) or {}
    anomalib_root = Path(str(train_cfg.get("root") or data_root))
    anomalib_normal_dir = anomalib_root / str(train_cfg.get("normal_dir", "train/good"))
    anomalib_abnormal_dir = anomalib_root / str(train_cfg.get("abnormal_dir", "test/bad"))

    yolo_artifact = find_yolo_run_artifact(resolved_config)
    configured_anomalib_project = Path(str(train_cfg.get("project", "runs/anomalib")))
    anomalib_roots = _unique_paths(
        [
            configured_anomalib_project,
            configured_anomalib_project / parsed.project / (parsed.area or ""),
            Path("runs") / "anomalib" / parsed.project / (parsed.area or ""),
            runs_root / "anomalib",
        ]
    )
    anomalib_artifact = find_anomalib_run_artifact(anomalib_roots)

    raw_images = count_images(raw_root / "images")
    raw_labels = count_files(raw_root / "labels", {".txt"})
    processed_images = count_images(processed_root / "images")
    split_train_images = count_images(split_root / "train" / "images")
    split_val_images = count_images(split_root / "val" / "images")
    split_test_images = count_images(split_root / "test" / "images")
    normal_images = count_images(anomalib_normal_dir)
    abnormal_images = count_images(anomalib_abnormal_dir)

    warnings: list[str] = []
    if raw_images and raw_labels and raw_images != raw_labels:
        warnings.append(f"Raw images/labels mismatch: {raw_images} images, {raw_labels} labels.")
    if not raw_images and not split_train_images:
        warnings.append("No raw YOLO images or split training images found.")
    if not normal_images:
        warnings.append(f"No Anomalib normal images found at {anomalib_normal_dir}.")
    if abnormal_images == 0:
        warnings.append("No Anomalib abnormal images found; training may produce baseline-only output.")

    return ProjectReadiness(
        product=parsed.project,
        area=parsed.area,
        raw_images=raw_images,
        raw_labels=raw_labels,
        processed_images=processed_images,
        split_train_images=split_train_images,
        split_val_images=split_val_images,
        split_test_images=split_test_images,
        anomalib_normal_images=normal_images,
        anomalib_abnormal_images=abnormal_images,
        latest_yolo_run=yolo_artifact.run_dir if yolo_artifact else None,
        latest_anomalib_run=anomalib_artifact.run_dir if anomalib_artifact else None,
        package_output_dir=Path(str(package_cfg.get("output_dir", "runs/anomalib_packages"))),
        warnings=tuple(warnings),
    )


def format_readiness_preview(readiness: ProjectReadiness) -> str:
    """Return a compact multi-line status string for the GUI."""

    area = readiness.area or "(config default)"
    yolo_status = "ready" if readiness.is_ready_for_yolo else "missing data"
    anomalib_status = "ready" if readiness.is_ready_for_anomalib else "missing normal images"
    lines = [
        f"Target: {readiness.product} / {area}",
        (
            "YOLO data: "
            f"raw={readiness.raw_images} images/{readiness.raw_labels} labels, "
            f"split train/val/test={readiness.split_train_images}/"
            f"{readiness.split_val_images}/{readiness.split_test_images} "
            f"({yolo_status})"
        ),
        (
            "Anomalib data: "
            f"normal={readiness.anomalib_normal_images}, "
            f"abnormal={readiness.anomalib_abnormal_images} "
            f"({anomalib_status})"
        ),
        f"Latest YOLO run: {readiness.latest_yolo_run or 'none'}",
        f"Latest Anomalib run: {readiness.latest_anomalib_run or 'none'}",
        f"Package output: {readiness.package_output_dir}",
    ]
    if readiness.warnings:
        lines.append("Warnings: " + " | ".join(readiness.warnings))
    return "\n".join(lines)


def count_images(path: Path) -> int:
    """Count supported image files below a directory."""

    return count_files(path, IMAGE_EXTENSIONS)


def count_files(path: Path, extensions: Iterable[str]) -> int:
    """Count files with the given lowercase extensions below a directory."""

    if not path.exists():
        return 0
    allowed = {ext.lower() for ext in extensions}
    return sum(
        1
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in allowed
    )


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        normalized = str(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(path)
    return unique
