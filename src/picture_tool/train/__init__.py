from __future__ import annotations

from typing import Any

__all__ = ["train_anomalib", "train_yolo"]


def __getattr__(name: str) -> Any:
    """Load training entrypoints lazily to avoid heavy import side effects."""
    if name == "train_anomalib":
        from .anomalib_trainer import train_anomalib

        return train_anomalib
    if name == "train_yolo":
        from .yolo_trainer import train_yolo

        return train_yolo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
