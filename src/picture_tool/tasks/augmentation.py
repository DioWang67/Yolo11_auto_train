from pathlib import Path
import shutil
from typing import Any
from picture_tool.augment import ImageAugmentor, YoloDataAugmentor
from picture_tool.quality.dataset_linter import preview_dataset
from picture_tool.pipeline.cache import (
    task_cache_exists,
    task_cache_matches,
    write_task_cache,
)
from picture_tool.pipeline.utils import mtime_latest, exists_and_nonempty
from picture_tool.pipeline.core import Task


class AugmentationOutputIncompleteError(RuntimeError):
    """Requested YOLO augmentation outputs were not completely produced."""


def run_yolo_augmentation(config, args):
    if "yolo_augmentation" not in config:
        # Fallback or friendly error
        raise ValueError(
            "Config is missing 'yolo_augmentation' section. "
            "If using an old config, please add this section or recreate project via Wizard."
        )
    augmentor = YoloDataAugmentor()
    augmentor.config = config.get("yolo_augmentation", {})
    augmentor._setup_output_dirs()
    cfg = config.get("yolo_augmentation", {})
    ic = cfg.get("input", {})
    oc = cfg.get("output", {})
    num_images = int((cfg.get("augmentation", {}) or {}).get("num_images", 0))
    if num_images < 0:
        raise ValueError("yolo_augmentation.augmentation.num_images cannot be negative")
    if num_images > 0:
        augmentor.augmentations = augmentor._create_augmentations()
        augmentor.process_dataset()
    if bool((cfg.get("augmentation", {}) or {}).get("include_originals", False)):
        _copy_original_yolo_pairs(cfg)
    _validate_yolo_augmentation_outputs(cfg)
    write_task_cache(
        Path(oc.get("image_dir", "./data/project/processed/images")).parent,
        "yolo_augmentation",
        cfg,
        [Path(ic["image_dir"]), Path(ic["label_dir"])],
    )


def _copy_original_yolo_pairs(cfg: dict[str, Any]) -> int:
    """Copy original image/label pairs beside augmented variants atomically.

    Including originals preserves verified-empty negatives and ensures offline
    augmentation adds diversity instead of replacing the source dataset.
    """
    input_cfg = cfg.get("input", {}) or {}
    output_cfg = cfg.get("output", {}) or {}
    input_images = Path(str(input_cfg["image_dir"]))
    input_labels = Path(str(input_cfg["label_dir"]))
    output_images = Path(str(output_cfg["image_dir"]))
    output_labels = Path(str(output_cfg["label_dir"]))
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    copied = 0
    for image_path in sorted(input_images.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in {
            ".bmp",
            ".jpeg",
            ".jpg",
            ".png",
            ".tif",
            ".tiff",
            ".webp",
        }:
            continue
        label_path = input_labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        _copy_file_atomic(image_path, output_images / image_path.name)
        _copy_file_atomic(label_path, output_labels / label_path.name)
        copied += 1
    return copied


def _copy_file_atomic(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def skip_yolo_augmentation(config, args):
    cfg = config.get("yolo_augmentation")
    if not cfg:
        # Can't skip if we can't check paths, but run() will raise accurate error
        return None

    ic = cfg.get("input", {})
    oc = cfg.get("output", {})

    if "image_dir" not in ic or "label_dir" not in ic:
        return None  # Let run() handle validation

    in_dirs = [Path(ic["image_dir"]), Path(ic["label_dir"])]
    out_dirs = [
        Path(oc.get("image_dir", "./data/project/processed/images")),
        Path(oc.get("label_dir", "./data/project/processed/labels")),
    ]

    if not all(p.exists() for p in in_dirs):
        # Only raise if we are sure config is intended to be run
        # but if we are here, task is enabled.
        raise FileNotFoundError(f"Augmentation inputs missing: {in_dirs}")

    if all(exists_and_nonempty(p) for p in out_dirs):
        expected_outputs = _expected_yolo_augmentation_outputs(cfg)
        if expected_outputs is not None:
            actual_labels = len(list(out_dirs[1].glob("*.txt")))
            if actual_labels < expected_outputs:
                return None
        cache_dir = out_dirs[0].parent
        if task_cache_matches(cache_dir, "yolo_augmentation", cfg, in_dirs):
            return "Output cache matches inputs and config; skipping."
        if task_cache_exists(cache_dir):
            return None
        if mtime_latest(out_dirs) >= mtime_latest(in_dirs):
            return "Outputs are newer than inputs; skipping."
    return None


def _expected_yolo_augmentation_outputs(cfg: dict[str, Any]) -> int | None:
    """Estimate how many label files a complete YOLO augmentation should produce.

    Args:
        cfg: The ``yolo_augmentation`` configuration block.

    Returns:
        Expected label-file count, or None when paths/config are incomplete.
    """
    try:
        image_dir = Path(str(cfg["input"]["image_dir"]))
        label_dir = Path(str(cfg["input"]["label_dir"]))
        num_images = int(cfg["augmentation"]["num_images"])
    except (KeyError, TypeError, ValueError):
        return None

    if num_images < 0 or not image_dir.exists() or not label_dir.exists():
        return None

    image_stems = {
        path.stem
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    }
    label_stems = {
        path.stem
        for path in label_dir.glob("*.txt")
        if path.name.lower() != "classes.txt"
    }
    matched_stems = image_stems & label_stems
    nonempty_label_count = sum(
        bool((label_dir / f"{stem}.txt").read_text(encoding="utf-8").strip())
        for stem in matched_stems
    )
    original_count = (
        len(matched_stems)
        if bool((cfg.get("augmentation", {}) or {}).get("include_originals", False))
        else 0
    )
    return original_count + nonempty_label_count * num_images


def _validate_yolo_augmentation_outputs(cfg: dict[str, Any]) -> None:
    """Fail before splitting when requested variants are missing."""
    expected = _expected_yolo_augmentation_outputs(cfg)
    if expected is None:
        raise AugmentationOutputIncompleteError(
            "Unable to calculate the requested YOLO augmentation output count."
        )
    output_cfg = cfg.get("output", {}) or {}
    try:
        image_dir = Path(str(output_cfg["image_dir"]))
        label_dir = Path(str(output_cfg["label_dir"]))
    except (KeyError, TypeError) as exc:
        raise AugmentationOutputIncompleteError(
            "YOLO augmentation output paths are missing."
        ) from exc
    image_count = sum(
        path.is_file()
        and path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
        for path in image_dir.iterdir()
    ) if image_dir.is_dir() else 0
    label_count = sum(
        path.is_file() and path.name.lower() != "classes.txt"
        for path in label_dir.glob("*.txt")
    ) if label_dir.is_dir() else 0
    if image_count < expected or label_count < expected:
        requested = int((cfg.get("augmentation", {}) or {}).get("num_images", 0))
        raise AugmentationOutputIncompleteError(
            "YOLO augmentation did not satisfy the job contract: "
            f"requested_variants_per_positive={requested}, expected_pairs={expected}, "
            f"actual_images={image_count}, actual_labels={label_count}."
        )


def run_image_augmentation(config, args):
    augmentor = ImageAugmentor()
    augmentor.config = config["image_augmentation"]
    augmentor._setup_output_dirs()
    augmentor.augmentations = augmentor._create_augmentations()
    augmentor.process_dataset()


def run_aug_preview(config, args):
    preview_dataset(config)


def skip_aug_preview(config, args):
    p = config.get("aug_preview", {})
    img_dir = Path(p.get("image_dir", "./data/project/processed/images"))
    out = Path(p.get("output_dir", "./runs/project/quality/preview")) / "preview.png"
    if out.exists() and out.stat().st_mtime >= mtime_latest([img_dir]):
        return "Preview output is newer; skipping."
    return None


TASKS = [
    Task(
        name="yolo_augmentation",
        run=run_yolo_augmentation,
        skip_fn=skip_yolo_augmentation,
        description="YOLO label-aware augmentation.",
    ),
    Task(
        name="image_augmentation",
        run=run_image_augmentation,
        description="Image-only augmentation.",
    ),
    Task(
        name="aug_preview",
        run=run_aug_preview,
        skip_fn=skip_aug_preview,
        description="Preview augmented samples.",
    ),
]
