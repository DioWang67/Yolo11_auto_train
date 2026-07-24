from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from picture_tool import runtime_pair_deployment
from picture_tool.workspace_paths import (
    WORKSPACE_ENV_VAR,
    WorkspaceConfigurationError,
    WorkspacePaths,
)


def _write_manifest(root: Path, *, inference_models: str = "inference/models") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "workspace.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "projects": {
                    "training": "training",
                    "inference": "inference",
                },
                "paths": {
                    "training_data": "training/data",
                    "inference_models": inference_models,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest


def test_discover_uses_nearest_ancestor_manifest(tmp_path: Path) -> None:
    workspace = tmp_path / "portable-workspace"
    manifest = _write_manifest(workspace)
    nested = workspace / "training" / "src" / "picture_tool"
    nested.mkdir(parents=True)

    paths = WorkspacePaths.discover(nested, environ={})

    assert paths.workspace_root == workspace.resolve()
    assert paths.training_project == (workspace / "training").resolve()
    assert paths.inference_project == (workspace / "inference").resolve()
    assert paths.training_data == (workspace / "training" / "data").resolve()
    assert paths.inference_models == (workspace / "inference" / "models").resolve()
    assert paths.manifest_path == manifest.resolve()


def test_environment_workspace_has_priority_over_start_path(tmp_path: Path) -> None:
    configured = tmp_path / "configured"
    _write_manifest(configured, inference_models="runtime/models")
    nearby = tmp_path / "nearby"
    _write_manifest(nearby)
    nested = nearby / "training"
    nested.mkdir()

    paths = WorkspacePaths.discover(
        nested,
        environ={WORKSPACE_ENV_VAR: str(configured)},
    )

    assert paths.workspace_root == configured.resolve()
    assert paths.inference_models == (configured / "runtime" / "models").resolve()


def test_environment_workspace_requires_manifest(tmp_path: Path) -> None:
    configured = tmp_path / "configured"
    configured.mkdir()

    with pytest.raises(WorkspaceConfigurationError, match="does not contain"):
        WorkspacePaths.discover(
            tmp_path,
            environ={WORKSPACE_ENV_VAR: str(configured)},
        )


def test_manifest_rejects_paths_that_escape_workspace(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, inference_models="../outside/models")

    with pytest.raises(WorkspaceConfigurationError, match="escapes the workspace"):
        WorkspacePaths.from_manifest(manifest)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"schema_version": 2, "projects": {}, "paths": {}},
        {"schema_version": 1, "projects": [], "paths": {}},
        {"schema_version": 1, "projects": {}, "paths": []},
    ],
)
def test_manifest_rejects_invalid_contracts(tmp_path: Path, payload: object) -> None:
    manifest = tmp_path / "workspace.yaml"
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(WorkspaceConfigurationError):
        WorkspacePaths.from_manifest(manifest)


def test_discover_supports_legacy_sibling_repositories(tmp_path: Path) -> None:
    training = tmp_path / "Yolo11_auto_train"
    inference = tmp_path / "yolo11_inference"
    nested = training / "src" / "picture_tool"
    nested.mkdir(parents=True)
    (inference / "models").mkdir(parents=True)

    paths = WorkspacePaths.discover(nested, environ={})

    assert paths.workspace_root == tmp_path.resolve()
    assert paths.training_project == training.resolve()
    assert paths.inference_project == inference.resolve()
    assert paths.inference_models == (inference / "models").resolve()
    assert paths.manifest_path is None


def test_runtime_pair_default_uses_workspace_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    _write_manifest(workspace, inference_models="inference/custom-models")
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(workspace))

    assert runtime_pair_deployment._default_inference_models_dir() == (
        workspace / "inference" / "custom-models"
    ).resolve()


def test_training_launchers_are_relative_to_their_script_directory() -> None:
    repository = Path(__file__).resolve().parents[1]
    for name in ("start.bat", "start_env.bat"):
        content = (repository / "scripts" / name).read_text(
            encoding="utf-8",
            errors="replace",
        )
        direct_relative_cd = 'cd /d "%~dp0.."' in content
        normalized_project_root = (
            'for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"' in content
            and 'cd /d "%PROJECT_ROOT%"' in content
        )
        assert direct_relative_cd or normalized_project_root
        assert "D:\\Git\\robotlearning\\Yolo11_auto_train" not in content
        assert "D:\\Git\\robotlearning\\yolo11_workspace\\Yolo11_auto_train" not in content
