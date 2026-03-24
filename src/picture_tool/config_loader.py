"""Configuration loading utilities for the picture-tool pipeline.

Responsible for reading YAML configuration files, fallback to packaged
defaults, and hot-reload via mtime-based change detection.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_RESOURCE = "default_pipeline.yaml"

logger = logging.getLogger(__name__)


def _load_packaged_default() -> dict[str, Any]:
    """Load the bundled sample config shipped with the package."""
    package_resources = resources.files("picture_tool.resources")
    default_file = package_resources / _DEFAULT_CONFIG_RESOURCE
    with default_file.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load a pipeline configuration file, falling back to the packaged template.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If neither the given path nor the packaged default exists.
    """
    path = Path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    try:
        return _load_packaged_default()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Config file '{config_path}' not found and packaged default is missing."
        ) from exc


@lru_cache(maxsize=4)
def _load_config_snapshot(path: str, mtime: float) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config_if_updated(
    config_path: str | Path,
    config: dict[str, Any],
    log: logging.Logger,
) -> dict[str, Any]:
    """Reload the configuration if the on-disk file changed.

    Args:
        config_path: Path to the YAML configuration file.
        config: Current in-memory configuration dictionary.
        log: Logger instance for status messages.

    Returns:
        Either the existing *config* or a freshly loaded dictionary.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        return config
    current_mtime = config_file.stat().st_mtime
    last_mtime = getattr(load_config_if_updated, "last_mtime", None)

    if last_mtime is None:
        load_config_if_updated.last_mtime = current_mtime  # type: ignore[attr-defined]
        return config

    if current_mtime > last_mtime:
        log.info("Detected configuration change; reloading.")
        load_config_if_updated.last_mtime = current_mtime  # type: ignore[attr-defined]
        return _load_config_snapshot(str(config_file.resolve()), current_mtime)

    return config
