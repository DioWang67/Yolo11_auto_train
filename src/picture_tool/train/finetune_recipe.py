"""The recipe for continuing a deployed champion on new station data.

Operator retraining and the autonomous challenger both start from the deployed
weights, and the challenger is then judged against the champion. That
comparison only means something if the two were trained the same way, so both
paths apply these functions instead of each keeping a copy. When they did keep
copies, the challenger copied the colour-safe augmentation operations and
nothing else: it trained on augmented variants alone (no originals, so
verified-empty negatives vanished), at twenty times the champion's learning
rate, with nothing frozen, early stopping that could never fire, and a
"disabled" hue operation that still shifted hue.
"""

from __future__ import annotations

from typing import Any, MutableMapping, Optional

#: Offline augmentation that keeps wire colour intact: photometric changes
#: small enough not to move a class boundary, geometry near identity.
COLOUR_SAFE_OPERATIONS: dict[str, dict[str, Any]] = {
    "blur": {"kernel": [0, 1]},
    "contrast": {"range": [0.92, 1.08]},
    "multiply": {"range": [0.88, 1.15]},
    "noise": {"scale": [0, 0.02]},
    "rotate": {"angle": [-2, 2]},
    "scale": {"range": [0.97, 1.03]},
}


def apply_colour_safe_augmentation(
    augmentation_cfg: MutableMapping[str, Any],
    *,
    imgsz: int,
    augmentations_per_image: Optional[int] = None,
) -> None:
    """Configure offline augmentation in place for a colour-coded station.

    ``augmentations_per_image`` of ``None`` keeps the configured count.
    Originals are always copied beside the variants: without them the model
    never sees an unaltered image, and an empty label file --- a verified
    negative --- produces no variants and would drop out entirely.
    """
    processing = augmentation_cfg.setdefault("processing", {})
    processing.setdefault("num_workers", 2)
    # Debug mode processes only one source image in the legacy augmentor.
    processing["debug_mode"] = False
    policy = augmentation_cfg.setdefault("augmentation", {})
    if augmentations_per_image is not None:
        policy["num_images"] = augmentations_per_image
    policy["include_originals"] = True
    policy["target_size"] = imgsz
    policy.setdefault("num_operations", [2, 3])
    operations = policy.setdefault("operations", {})
    if not isinstance(operations, dict):
        raise ValueError("yolo_augmentation.augmentation.operations must be a mapping.")
    for operation, values in COLOUR_SAFE_OPERATIONS.items():
        operations.setdefault(operation, values)
    # Wire colour is the class signal, so hue changes, mirroring and
    # perspective are off even when a generic preset enables them. Hue is
    # removed rather than narrowed: the augmentor reads its range as a shift,
    # so ``[1, 1]`` shifts every affected image by one step, and enabling the
    # operation also turns on saturation and value jitter.
    operations.pop("hue", None)
    operations["flip"] = {"probability": 0.0}
    operations["perspective"] = {"scale": [0.0, 0.0]}


def apply_finetune_training(
    training_cfg: MutableMapping[str, Any],
    *,
    epochs: int,
    batch: int,
    imgsz: int,
) -> None:
    """Set the Ultralytics arguments for continuing a deployed model, in place.

    A low learning rate with the backbone frozen adapts the head without
    unlearning the station; every Ultralytics augmentation that could move a
    colour or position is zeroed, since the offline pass already varies both
    within safe bounds. ``lr0`` and ``warmup_epochs`` may be lowered by the
    station config but never raised.
    """
    training_cfg.update(
        {
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "optimizer": "AdamW",
            "lr0": min(float(training_cfg.get("lr0", 0.0005)), 0.0005),
            "lrf": float(training_cfg.get("lrf", 0.1)),
            "patience": min(max(epochs // 2, 5), 20),
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
