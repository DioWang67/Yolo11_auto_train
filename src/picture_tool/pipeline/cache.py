"""Small metadata cache helpers for pipeline skip decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from picture_tool.utils.hashing import compute_config_hash, compute_dir_hash


CACHE_FILENAME = ".task_metadata.json"


def build_task_cache(
    task_name: str,
    config_section: Mapping[str, Any],
    input_paths: Iterable[Path],
) -> dict[str, str]:
    """Build comparable cache metadata for one task.

    Args:
        task_name: Pipeline task name.
        config_section: Relevant task config section.
        input_paths: Input files/directories that affect task output.

    Returns:
        Metadata dictionary suitable for JSON storage.
    """
    input_hashes = {
        str(path): compute_dir_hash(path) if path.is_dir() else _file_hash(path)
        for path in input_paths
    }
    return {
        "task": task_name,
        "config_hash": compute_config_hash(dict(config_section)),
        "input_hash": compute_config_hash(input_hashes),
    }


def write_task_cache(
    cache_dir: Path,
    task_name: str,
    config_section: Mapping[str, Any],
    input_paths: Iterable[Path],
) -> Path:
    """Write task cache metadata under ``cache_dir``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / CACHE_FILENAME
    cache_path.write_text(
        json.dumps(
            build_task_cache(task_name, config_section, input_paths),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return cache_path


def task_cache_matches(
    cache_dir: Path,
    task_name: str,
    config_section: Mapping[str, Any],
    input_paths: Iterable[Path],
) -> bool:
    """Return whether current inputs/config match stored task metadata."""
    cache_path = cache_dir / CACHE_FILENAME
    if not cache_path.exists():
        return False
    try:
        stored = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return stored == build_task_cache(task_name, config_section, input_paths)


def task_cache_exists(cache_dir: Path) -> bool:
    """Return whether a task metadata cache exists under ``cache_dir``."""
    return (cache_dir / CACHE_FILENAME).exists()


def _file_hash(path: Path) -> str:
    if not path.exists():
        return "empty"
    stat = path.stat()
    return compute_config_hash(
        {
            "path": str(path),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
        }
    )
