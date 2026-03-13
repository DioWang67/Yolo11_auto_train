from pathlib import Path
from picture_tool.augment import ImageAugmentor, YoloDataAugmentor
from picture_tool.quality.dataset_linter import preview_dataset
from picture_tool.pipeline.utils import mtime_latest, exists_and_nonempty
from picture_tool.pipeline.core import Task


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
    augmentor.augmentations = augmentor._create_augmentations()
    augmentor.process_dataset()


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
        if mtime_latest(out_dirs) >= mtime_latest(in_dirs):
            return "Outputs are newer than inputs; skipping."
    return None


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
