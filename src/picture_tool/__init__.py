"""Public package surface for the picture_tool toolkit."""

from __future__ import annotations

from importlib import metadata

from .anomaly import process_anomaly_detection
from .augment import ImageAugmentor, YoloDataAugmentor
from .format import convert_format
from .pipeline import (
    get_tasks_from_groups,
    interactive_task_selection,
    load_config,
    load_config_if_updated,
    run_pipeline,
    setup_logging,
    validate_dependencies,
)
from .split import split_dataset

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
