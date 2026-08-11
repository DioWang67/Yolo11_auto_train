from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from picture_tool.tasks import conversion, data_sync
from picture_tool.tracking import experiment_tracker
from picture_tool.utils import experiment, hashing, io_utils


@pytest.mark.parametrize(
    ("config", "input_format", "output_format", "expected"),
    [
        ({}, None, None, {}),
        ({"format_conversion": {"quality": 90}}, "png", "jpg", {"quality": 90, "input_formats": ["png"], "output_format": "jpg"}),
        ({"format_conversion": {"output_format": "png"}}, None, None, {"output_format": "png"}),
    ],
)
def test_format_conversion_merges_cli_overrides_without_mutating_config(
    config: dict[str, object],
    input_format: str | None,
    output_format: str | None,
    expected: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = MagicMock()
    monkeypatch.setattr(conversion, "convert_format", converter)
    original = json.loads(json.dumps(config))

    conversion.run_format_conversion(
        config,
        SimpleNamespace(input_format=input_format, output_format=output_format),
    )

    converter.assert_called_once_with(expected)
    assert config == original


@pytest.mark.parametrize(
    ("installed", "repo", "pull_result", "expected_log"),
    [
        (False, True, True, "warning"),
        (True, False, True, "info"),
        (True, True, True, "info"),
        (True, True, False, "warning"),
    ],
)
def test_data_sync_handles_unavailable_and_successful_dvc_states(
    installed: bool,
    repo: bool,
    pull_result: bool,
    expected_log: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = SimpleNamespace(is_installed=installed, is_dvc_repo=repo, pull=MagicMock(return_value=pull_result))
    logger = MagicMock()
    monkeypatch.setattr(data_sync, "DVCWrapper", lambda: wrapper)
    monkeypatch.setattr(data_sync, "logger", logger)

    data_sync.run_data_sync({}, SimpleNamespace())

    if installed and repo:
        wrapper.pull.assert_called_once()
    else:
        wrapper.pull.assert_not_called()
    assert getattr(logger, expected_log).called


def test_list_images_normalizes_extensions_filters_and_sorts(tmp_path: Path) -> None:
    assert io_utils.list_images(tmp_path / "missing") == []
    (tmp_path / "B.PNG").write_bytes(b"image")
    (tmp_path / "a.jpg").write_bytes(b"image")
    (tmp_path / "ignore.txt").write_text("text", encoding="utf-8")
    (tmp_path / "nested.png").mkdir()
    assert io_utils.list_images(tmp_path, ["PNG", ".jpg"]) == ["B.PNG", "a.jpg"]


def test_hashing_is_deterministic_and_handles_missing_or_bad_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert hashing.compute_dir_hash(tmp_path / "missing") == "empty"
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("bb", encoding="utf-8")
    first = hashing.compute_dir_hash(tmp_path)
    second = hashing.compute_dir_hash(tmp_path)
    assert first == second
    assert first != "empty"

    original_stat = Path.stat
    stat_calls: dict[Path, int] = {}

    def selective_stat(path: Path, *args: object, **kwargs: object):
        stat_calls[path] = stat_calls.get(path, 0) + 1
        if path.name == "b.txt" and stat_calls[path] > 1:
            raise PermissionError("denied")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", selective_stat)
    assert hashing.compute_dir_hash(tmp_path) != "empty"

    assert hashing.compute_config_hash({"b": 2, "a": 1}) == hashing.compute_config_hash({"a": 1, "b": 2})
    assert hashing.compute_config_hash({"invalid": object()}) == "unknown"


def test_experiment_git_revision_success_empty_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="abc123\n"),
    )
    assert experiment._git_rev() == "abc123"
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=""),
    )
    assert experiment._git_rev() is None
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("git missing")),
    )
    assert experiment._git_rev() is None


def test_experiment_environment_info_for_test_and_real_dependency_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_IS_RUNNING", "1")
    info = experiment._env_info()
    assert info["torch_version"] == "mocked_for_test"
    assert info["cuda_available"] is False

    monkeypatch.delenv("PYTEST_IS_RUNNING", raising=False)
    fake_torch = SimpleNamespace(
        __version__="2.0",
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    fake_ultralytics = SimpleNamespace(__version__="8.0")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    info = experiment._env_info()
    assert info["torch_version"] == "2.0"
    assert info["cuda_available"] is True
    assert info["ultralytics_version"] == "8.0"

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    info = experiment._env_info()
    assert info["torch_version"] is None
    assert info["cuda_available"] is None
    assert info["ultralytics_version"] is None


def test_metrics_csv_and_json_conversion_cover_empty_nested_and_fallback_values(
    tmp_path: Path,
) -> None:
    assert experiment._load_metrics_csv(tmp_path / "missing.csv") == {}
    empty = tmp_path / "empty.csv"
    empty.write_text("metric,value\n", encoding="utf-8")
    assert experiment._load_metrics_csv(empty) == {}
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("metric,value\nloss,0.2\naccuracy,0.9\n", encoding="utf-8")
    assert experiment._load_metrics_csv(metrics) == {"metric": "accuracy", "value": "0.9"}

    class _BrokenString:
        def __str__(self) -> str:
            raise ValueError("no string")

        def __repr__(self) -> str:
            return "broken-repr"

    converted = experiment._jsonable(
        {
            Path("path"): (
                Path("value"),
                "text",
                1,
                2.0,
                True,
                None,
                {"set"},
                _BrokenString(),
            )
        }
    )
    assert converted["path"][-1] == "broken-repr"


def test_write_experiment_persists_yaml_json_metrics_and_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment, "_git_rev", lambda: "commit")
    monkeypatch.setattr(experiment, "_env_info", lambda: {"python": "test"})
    metrics_csv = tmp_path / "results.csv"
    metrics_csv.write_text("metric,value\naccuracy,0.95\n", encoding="utf-8")

    yaml_path = experiment.write_experiment(
        "training",
        {"path": Path("dataset")},
        tmp_path / "run",
        metrics={"loss": 0.1},
        artifacts={"weights": Path("best.pt")},
        extra={"tags": ("candidate",)},
        output_dir=tmp_path / "reports",
        results_csv=metrics_csv,
    )

    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert payload["git_commit"] == "commit"
    assert payload["metrics_csv"]["value"] == "0.95"
    assert payload["extra"] == {"tags": ["candidate"]}
    json_path = yaml_path.with_suffix(".json")
    assert json.loads(json_path.read_text(encoding="utf-8"))["artifacts"]["weights"] == "best.pt"


def test_mlflow_tracker_disabled_and_successful_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment_tracker, "mlflow", None)
    disabled = experiment_tracker.MLflowTracker()
    disabled.start_run("run")
    disabled.log_params({"a": 1})
    disabled.log_metrics({"m": 0.5}, step=2)
    disabled.log_artifact(str(tmp_path / "missing"))
    disabled.end_run()

    artifact = tmp_path / "artifact.txt"
    artifact.write_text("result", encoding="utf-8")
    fake_mlflow = MagicMock()
    monkeypatch.setattr(experiment_tracker, "mlflow", fake_mlflow)
    tracker = experiment_tracker.MLflowTracker("experiment", "http://tracking")
    tracker.start_run("run")
    tracker.log_params({"a": 1})
    tracker.log_metrics({"m": 0.5}, step=2)
    tracker.log_artifact(str(artifact), "models")
    tracker.end_run()
    fake_mlflow.set_tracking_uri.assert_called_once_with("http://tracking")
    fake_mlflow.set_experiment.assert_called_once_with("experiment")
    fake_mlflow.log_artifact.assert_called_once_with(str(artifact), "models")


def test_mlflow_tracker_default_uri_missing_artifact_and_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = MagicMock()
    monkeypatch.setattr(experiment_tracker, "mlflow", fake_mlflow)
    tracker = experiment_tracker.MLflowTracker("experiment")
    assert fake_mlflow.set_tracking_uri.call_args.args[0].startswith("file:///")
    with pytest.raises(experiment_tracker.TrackingDomainError, match="Artifact not found"):
        tracker.log_artifact(str(tmp_path / "missing"))

    assert isinstance(experiment_tracker.get_tracker({}), experiment_tracker.NullTracker)
    assert isinstance(
        experiment_tracker.get_tracker({"tracking": {"enabled": True, "backend": "none"}}),
        experiment_tracker.NullTracker,
    )
    assert isinstance(
        experiment_tracker.get_tracker(
            {
                "tracking": {
                    "enabled": True,
                    "backend": "mlflow",
                    "experiment_name": "custom",
                    "uri": "memory://",
                }
            }
        ),
        experiment_tracker.MLflowTracker,
    )


@pytest.mark.parametrize(
    ("method", "arguments"),
    [
        ("start_run", ("run",)),
        ("log_params", ({"a": 1},)),
        ("log_metrics", ({"m": 1.0},)),
        ("end_run", ()),
    ],
)
@pytest.mark.parametrize("error_type", [experiment_tracker.MlflowException, OSError])
def test_mlflow_tracker_wraps_infrastructure_failures(
    method: str,
    arguments: tuple[object, ...],
    error_type: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = MagicMock()
    monkeypatch.setattr(experiment_tracker, "mlflow", fake_mlflow)
    tracker = experiment_tracker.MLflowTracker("experiment", "memory://")
    getattr(fake_mlflow, method).side_effect = error_type("failure")

    with pytest.raises(experiment_tracker.TrackingInfrastructureError):
        getattr(tracker, method)(*arguments)


@pytest.mark.parametrize("error_type", [experiment_tracker.MlflowException, OSError])
def test_mlflow_artifact_and_setup_wrap_infrastructure_failures(
    tmp_path: Path,
    error_type: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("result", encoding="utf-8")
    fake_mlflow = MagicMock()
    monkeypatch.setattr(experiment_tracker, "mlflow", fake_mlflow)
    tracker = experiment_tracker.MLflowTracker("experiment", "memory://")
    fake_mlflow.log_artifact.side_effect = error_type("failure")
    with pytest.raises(experiment_tracker.TrackingInfrastructureError):
        tracker.log_artifact(str(artifact))

    fake_mlflow.reset_mock()
    fake_mlflow.set_tracking_uri.side_effect = error_type("setup failure")
    with pytest.raises(experiment_tracker.TrackingInfrastructureError):
        experiment_tracker.MLflowTracker("experiment", "memory://")


def test_null_tracker_methods_are_noops() -> None:
    tracker = experiment_tracker.NullTracker()
    assert tracker.start_run("run") is None
    assert tracker.log_params({"a": 1}) is None
    assert tracker.log_metrics({"m": 0.5}, step=1) is None
    assert tracker.log_artifact("missing") is None
    assert tracker.end_run() is None
