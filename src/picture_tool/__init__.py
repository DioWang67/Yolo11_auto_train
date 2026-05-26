"""Public package surface for the picture_tool toolkit."""

from __future__ import annotations

from importlib import metadata
from typing import Any

try:
    __version__ = metadata.version("picture-tool")
except metadata.PackageNotFoundError:  # pragma: no cover - local source tree
    __version__ = "0.0.0"

__all__ = [
    "ImageAugmentor",
    "YoloDataAugmentor",
    "process_anomaly_detection",
    "convert_format",
    "split_dataset",
    "run_pipeline",
    "setup_logging",
    "load_config",
    "load_config_if_updated",
    "validate_dependencies",
    "get_tasks_from_groups",
    "interactive_task_selection",
    "__version__",
]


def __getattr__(name: str) -> Any:
    """Load optional public entrypoints lazily to avoid import side effects."""
    if name == "process_anomaly_detection":
        from .anomaly import process_anomaly_detection

        return process_anomaly_detection
    if name == "ImageAugmentor":
        from .augment import ImageAugmentor

        return ImageAugmentor
    if name == "YoloDataAugmentor":
        from .augment import YoloDataAugmentor

        return YoloDataAugmentor
    if name == "convert_format":
        from .format import convert_format

        return convert_format
    if name == "split_dataset":
        from .split import split_dataset

        return split_dataset
    if name in {
        "get_tasks_from_groups",
        "interactive_task_selection",
        "load_config",
        "load_config_if_updated",
        "run_pipeline",
        "setup_logging",
        "validate_dependencies",
    }:
        from . import main_pipeline

        return getattr(main_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
