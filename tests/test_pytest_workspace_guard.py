"""Regression tests for pytest's live station-data isolation guard."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import conftest as pytest_config
import pytest
from conftest import (
    _ISOLATED_WORKSPACE_ROOT,
    _LIVE_WORKSPACE_ROOT,
    _TRAINING_PROJECT_ROOT,
    _WORKSPACE_ENV_VAR,
    _assert_safe_workspace_environment,
    _describe_snapshot_changes,
    _snapshot_tree,
    _validated_external_basetemp,
    _WorkspaceGuardError,
)

from picture_tool.workspace_paths import WorkspacePaths

WORKSPACE_MANIFEST = """schema_version: 1
projects:
  training: training
  inference: inference
paths:
  training_data: training/data
  inference_models: inference/models
  station_data: station_data
  inference_results: results
  inference_artifacts: artifacts
"""


def _write_workspace_manifest(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "workspace.yaml").write_text(WORKSPACE_MANIFEST, encoding="utf-8")


def test_default_pytest_workspace_is_isolated_from_live_checkout() -> None:
    configured_root = Path(os.environ[_WORKSPACE_ENV_VAR]).resolve()

    assert configured_root == _ISOLATED_WORKSPACE_ROOT
    assert not configured_root.is_relative_to(_LIVE_WORKSPACE_ROOT)
    assert not _LIVE_WORKSPACE_ROOT.is_relative_to(configured_root)
    assert (configured_root / "workspace.yaml").is_file()


@pytest.mark.parametrize(
    "start",
    (
        None,
        _TRAINING_PROJECT_ROOT,
        _LIVE_WORKSPACE_ROOT,
        _LIVE_WORKSPACE_ROOT / "station_data",
    ),
)
def test_checkout_discovery_is_routed_to_isolated_workspace(
    start: Path | None,
) -> None:
    paths = WorkspacePaths.discover(start)

    assert paths.workspace_root == _ISOLATED_WORKSPACE_ROOT
    assert not paths.station_data.is_relative_to(_LIVE_WORKSPACE_ROOT)


def test_default_external_discovery_uses_nearest_test_manifest(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _write_workspace_manifest(workspace)
    nested = workspace / "training" / "nested"
    nested.mkdir(parents=True)

    paths = WorkspacePaths.discover(nested)

    assert paths.workspace_root == workspace.resolve()
    assert paths.station_data == (workspace / "station_data").resolve()


def test_preimported_workspace_class_alias_uses_installed_isolation() -> None:
    from picture_tool import runtime_pair_deployment
    from picture_tool.gui import operator_handoff

    assert runtime_pair_deployment.WorkspacePaths is WorkspacePaths
    assert operator_handoff.WorkspacePaths is WorkspacePaths
    assert (
        runtime_pair_deployment.WorkspacePaths.discover(
            _TRAINING_PROJECT_ROOT
        ).workspace_root
        == _ISOLATED_WORKSPACE_ROOT
    )
    assert (
        operator_handoff.WorkspacePaths.discover(_TRAINING_PROJECT_ROOT).workspace_root
        == _ISOLATED_WORKSPACE_ROOT
    )


def test_dynamic_live_workspace_environment_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_WORKSPACE_ENV_VAR, str(_LIVE_WORKSPACE_ROOT))

    with pytest.raises(pytest.UsageError, match="Refusing unsafe"):
        WorkspacePaths.discover(_TRAINING_PROJECT_ROOT)


def test_missing_dynamic_environment_keeps_default_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_WORKSPACE_ENV_VAR)

    assert (
        WorkspacePaths.discover(_TRAINING_PROJECT_ROOT).workspace_root
        == _ISOLATED_WORKSPACE_ROOT
    )


@pytest.mark.parametrize(
    "unsafe_root",
    (_LIVE_WORKSPACE_ROOT, _LIVE_WORKSPACE_ROOT / "station_data"),
)
def test_workspace_environment_rejects_live_paths(unsafe_root: Path) -> None:
    with pytest.raises(pytest.UsageError, match="[Uu]nsafe"):
        _assert_safe_workspace_environment(
            {_WORKSPACE_ENV_VAR: str(unsafe_root)},
            live_workspace_root=_LIVE_WORKSPACE_ROOT,
        )


def test_workspace_environment_accepts_valid_isolated_root(tmp_path: Path) -> None:
    isolated_root = tmp_path / "isolated"
    _write_workspace_manifest(isolated_root)

    assert (
        _assert_safe_workspace_environment(
            {_WORKSPACE_ENV_VAR: str(isolated_root)},
            live_workspace_root=_LIVE_WORKSPACE_ROOT,
        )
        == isolated_root.resolve()
    )


def test_workspace_environment_rejects_manifest_symlink_to_live_workspace(
    tmp_path: Path,
) -> None:
    isolated_root = tmp_path / "isolated"
    _write_workspace_manifest(isolated_root)
    station_link = isolated_root / "station_data"
    try:
        station_link.symlink_to(
            _LIVE_WORKSPACE_ROOT / "station_data",
            target_is_directory=True,
        )
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")

    with pytest.raises(pytest.UsageError, match="[Uu]nsafe"):
        _assert_safe_workspace_environment(
            {_WORKSPACE_ENV_VAR: str(isolated_root)},
            live_workspace_root=_LIVE_WORKSPACE_ROOT,
        )


@pytest.mark.parametrize(
    "source",
    ("actual pytest basetemp", "system temporary directory"),
)
def test_temp_root_guard_rejects_workspace_descendant(
    tmp_path: Path,
    source: str,
) -> None:
    workspace = tmp_path / "workspace"
    _write_workspace_manifest(workspace)

    with pytest.raises(pytest.UsageError, match="lies inside the workspace"):
        _validated_external_basetemp(
            workspace / ".tmp" / "pytest",
            source=source,
        )


def test_actual_pytest_basetemp_is_checked_by_pytest_configure(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _write_workspace_manifest(workspace)
    config = SimpleNamespace(
        option=SimpleNamespace(basetemp=workspace / ".tmp" / "pytest")
    )

    with pytest.raises(pytest.UsageError, match="actual pytest basetemp"):
        pytest_config.pytest_configure(config)


def test_temp_root_guard_accepts_external_directory(tmp_path: Path) -> None:
    external = tmp_path / "external" / "pytest"

    assert (
        _validated_external_basetemp(
            external,
            source="actual pytest basetemp",
        )
        == external.resolve()
    )


def test_recursive_snapshot_detects_same_size_existing_file_overwrite(
    tmp_path: Path,
) -> None:
    guarded_root = tmp_path / "station_data"
    guarded_file = guarded_root / "acceptance" / "nested" / "result.json"
    guarded_file.parent.mkdir(parents=True)
    guarded_file.write_bytes(b"original")
    original_stat = guarded_file.stat()
    before = _snapshot_tree(guarded_root)

    guarded_file.write_bytes(b"tampered")
    os.utime(
        guarded_file,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    after = _snapshot_tree(guarded_root)

    assert (
        before["acceptance/nested/result.json"].sha256
        != after["acceptance/nested/result.json"].sha256
    )
    assert "modified: acceptance/nested/result.json" in _describe_snapshot_changes(
        before,
        after,
    )


def test_recursive_snapshot_detects_nested_addition(tmp_path: Path) -> None:
    guarded_root = tmp_path / "station_data"
    guarded_root.mkdir()
    before = _snapshot_tree(guarded_root)

    nested_file = guarded_root / "acceptance" / "new-run" / "report.json"
    nested_file.parent.mkdir(parents=True)
    nested_file.write_text("{}", encoding="utf-8")
    after = _snapshot_tree(guarded_root)

    assert "added: acceptance/new-run/report.json" in _describe_snapshot_changes(
        before,
        after,
    )


def test_recursive_snapshot_detects_metadata_only_change(tmp_path: Path) -> None:
    guarded_root = tmp_path / "station_data"
    guarded_file = guarded_root / "result.json"
    guarded_root.mkdir()
    guarded_file.write_text("{}", encoding="utf-8")
    before = _snapshot_tree(guarded_root)

    original_stat = guarded_file.stat()
    os.utime(
        guarded_file,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000_000),
    )
    after = _snapshot_tree(guarded_root)

    assert before["result.json"].sha256 == after["result.json"].sha256
    assert "modified: result.json" in _describe_snapshot_changes(before, after)


def test_recursive_snapshot_records_permission_denied_directory_as_opaque(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guarded_root = tmp_path / "station_data"
    opaque_directory = guarded_root / "acceptance" / "opaque-run"
    opaque_directory.mkdir(parents=True)
    (opaque_directory / "result.json").write_text("{}", encoding="utf-8")
    original_scandir = os.scandir

    def deny_opaque_directory(path: str | os.PathLike[str]):
        if Path(path).resolve() == opaque_directory.resolve():
            raise PermissionError(13, "simulated access denial", str(path))
        return original_scandir(path)

    monkeypatch.setattr(pytest_config.os, "scandir", deny_opaque_directory)
    before = _snapshot_tree(guarded_root)

    original_stat = opaque_directory.stat()
    os.utime(
        opaque_directory,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000_000),
    )
    after = _snapshot_tree(guarded_root)

    opaque_path = "acceptance/opaque-run"
    assert before[opaque_path].kind == "opaque_directory"
    assert after[opaque_path].kind == "opaque_directory"
    assert f"modified: {opaque_path}" in _describe_snapshot_changes(before, after)


def test_recursive_snapshot_fails_on_non_permission_scan_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guarded_root = tmp_path / "station_data"
    guarded_root.mkdir()

    def fail_scan(path: str | os.PathLike[str]):
        raise OSError(5, "simulated I/O failure", str(path))

    monkeypatch.setattr(pytest_config.os, "scandir", fail_scan)

    with pytest.raises(_WorkspaceGuardError, match="Unable to enumerate"):
        _snapshot_tree(guarded_root)


def test_session_finish_fails_when_guarded_tree_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guarded_root = tmp_path / "station_data"
    guarded_root.mkdir()
    before = _snapshot_tree(guarded_root)
    (guarded_root / "nested").mkdir()

    monkeypatch.setattr(
        pytest_config,
        "_LIVE_STATION_DATA_ROOTS",
        (guarded_root,),
    )
    session = SimpleNamespace(
        config=SimpleNamespace(
            stash={pytest_config._LIVE_SNAPSHOT_KEY: {guarded_root: before}},
            pluginmanager=SimpleNamespace(get_plugin=lambda name: None),
        ),
        exitstatus=int(pytest.ExitCode.OK),
    )

    pytest_config.pytest_sessionfinish(session, int(pytest.ExitCode.OK))

    assert session.exitstatus == int(pytest.ExitCode.TESTS_FAILED)


def test_pytest_startup_rejects_live_workspace_environment() -> None:
    environment = os.environ.copy()
    environment[_WORKSPACE_ENV_VAR] = str(_LIVE_WORKSPACE_ROOT)
    environment.pop("PYTEST_ADDOPTS", None)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            str(Path(__file__).resolve()),
        ],
        cwd=_TRAINING_PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == int(pytest.ExitCode.USAGE_ERROR), output
    assert f"Refusing unsafe {_WORKSPACE_ENV_VAR}" in output
