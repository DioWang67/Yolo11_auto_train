"""Relocation-safe discovery of the paired YOLO project workspace."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml


WORKSPACE_ENV_VAR = "YOLO11_WORKSPACE_ROOT"
WORKSPACE_MANIFEST_NAME = "workspace.yaml"
_SUPPORTED_SCHEMA_VERSION = 1


class WorkspaceConfigurationError(ValueError):
    """Raised when an explicitly configured workspace is unsafe or malformed."""


@dataclass(frozen=True)
class WorkspacePaths:
    """Resolved paths shared by the training and inference repositories."""

    workspace_root: Path
    training_project: Path
    inference_project: Path
    training_data: Path
    inference_models: Path
    station_data: Path
    inference_results: Path
    inference_artifacts: Path
    manifest_path: Path | None

    @classmethod
    def discover(
        cls,
        start: Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "WorkspacePaths":
        """Discover workspace paths without depending on the process CWD.

        Resolution order is intentional: an explicit environment root wins,
        then the nearest ancestor manifest, followed by the historical sibling
        repository layout.
        """
        environment = os.environ if environ is None else environ
        configured_root = environment.get(WORKSPACE_ENV_VAR, "").strip()
        if configured_root:
            root = Path(configured_root).expanduser().resolve()
            if not root.is_dir():
                raise WorkspaceConfigurationError(
                    f"{WORKSPACE_ENV_VAR} is not a directory: {root}"
                )
            manifest = root / WORKSPACE_MANIFEST_NAME
            if not manifest.is_file():
                raise WorkspaceConfigurationError(
                    f"{WORKSPACE_ENV_VAR} does not contain {WORKSPACE_MANIFEST_NAME}: "
                    f"{root}"
                )
            return cls.from_manifest(manifest)

        search_start = _normalise_search_start(start or _training_project_root())
        for directory in _ancestors_inclusive(search_start):
            manifest = directory / WORKSPACE_MANIFEST_NAME
            if manifest.is_file():
                return cls.from_manifest(manifest)

        legacy = _discover_legacy_siblings(search_start)
        if legacy is not None:
            return legacy
        raise WorkspaceConfigurationError(
            f"No {WORKSPACE_MANIFEST_NAME} or legacy paired repositories were found "
            f"from: {search_start}"
        )

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "WorkspacePaths":
        """Load and validate the versioned workspace contract."""
        manifest = manifest_path.expanduser().resolve()
        if not manifest.is_file():
            raise WorkspaceConfigurationError(
                f"Workspace manifest was not found: {manifest}"
            )
        try:
            payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise WorkspaceConfigurationError(
                f"Unable to read workspace manifest {manifest}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise WorkspaceConfigurationError(
                "Workspace manifest must contain a YAML mapping."
            )
        if payload.get("schema_version") != _SUPPORTED_SCHEMA_VERSION:
            raise WorkspaceConfigurationError(
                "Workspace manifest schema_version must be 1."
            )

        projects = _require_mapping(payload, "projects")
        paths = _require_mapping(payload, "paths")
        root = manifest.parent.resolve()
        inference_project = _resolve_relative_path(root, projects, "inference")
        station_data = _resolve_optional_relative_path(
            root,
            paths,
            "station_data",
            default=inference_project,
        )
        return cls(
            workspace_root=root,
            training_project=_resolve_relative_path(root, projects, "training"),
            inference_project=inference_project,
            training_data=_resolve_relative_path(root, paths, "training_data"),
            inference_models=_resolve_relative_path(root, paths, "inference_models"),
            station_data=station_data,
            inference_results=_resolve_optional_relative_path(
                root,
                paths,
                "inference_results",
                default=station_data / "Result",
            ),
            inference_artifacts=_resolve_optional_relative_path(
                root,
                paths,
                "inference_artifacts",
                default=inference_project,
            ),
            manifest_path=manifest,
        )


def _require_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise WorkspaceConfigurationError(
            f"Workspace manifest field {key!r} must be a mapping."
        )
    return value


def _resolve_relative_path(
    root: Path,
    values: Mapping[str, object],
    key: str,
) -> Path:
    raw_value = values.get(key)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise WorkspaceConfigurationError(
            f"Workspace manifest path {key!r} must be a non-empty string."
        )
    relative = Path(raw_value.strip())
    if relative.is_absolute():
        raise WorkspaceConfigurationError(
            f"Workspace manifest path {key!r} must be relative: {raw_value}"
        )
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise WorkspaceConfigurationError(
            f"Workspace manifest path {key!r} escapes the workspace: {raw_value}"
        )
    return resolved


def _resolve_optional_relative_path(
    root: Path,
    values: Mapping[str, object],
    key: str,
    *,
    default: Path,
) -> Path:
    if key not in values:
        return default.resolve()
    return _resolve_relative_path(root, values, key)


def _normalise_search_start(start: Path) -> Path:
    resolved = start.expanduser().resolve()
    return resolved.parent if resolved.is_file() else resolved


def _ancestors_inclusive(start: Path) -> tuple[Path, ...]:
    return (start, *start.parents)


def _discover_legacy_siblings(start: Path) -> WorkspacePaths | None:
    for candidate in _ancestors_inclusive(start):
        if candidate.name.casefold() == "yolo11_auto_train":
            training_project = candidate
            workspace_root = candidate.parent
        else:
            training_project = candidate / "Yolo11_auto_train"
            workspace_root = candidate
        inference_project = workspace_root / "yolo11_inference"
        if training_project.is_dir() and inference_project.is_dir():
            return WorkspacePaths(
                workspace_root=workspace_root.resolve(),
                training_project=training_project.resolve(),
                inference_project=inference_project.resolve(),
                training_data=(training_project / "data").resolve(),
                inference_models=(inference_project / "models").resolve(),
                station_data=inference_project.resolve(),
                inference_results=(inference_project / "Result").resolve(),
                inference_artifacts=inference_project.resolve(),
                manifest_path=None,
            )
    return None


def _training_project_root() -> Path:
    return Path(__file__).resolve().parents[2]
