from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

typer = pytest.importorskip(
    "typer",
    reason="Typer is required to exercise the production CLI integration",
)

# Imports are intentionally deferred until the optional local test dependency check.
from picture_tool import cli  # noqa: E402
from picture_tool import main_pipeline  # noqa: E402
from picture_tool.exceptions import ConfigurationError, PipelineError  # noqa: E402
from picture_tool.train import anomalib_trainer  # noqa: E402


def _run_command(**overrides: object) -> None:
    arguments: dict[str, object] = {
        "tasks": None,
        "config": "config.yaml",
        "exclude_tasks": None,
        "task_groups": None,
        "interactive": False,
        "force": False,
        "device": None,
        "epochs": None,
        "imgsz": None,
        "batch": None,
        "model": None,
        "project": None,
        "name": None,
        "weights": None,
        "infer_input": None,
        "infer_output": None,
        "product": None,
    }
    arguments.update(overrides)
    cli.run(**arguments)


def test_load_config_or_exit_success_and_supported_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {"pipeline": {"tasks": []}}
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    assert cli._load_config_or_exit("config.yaml") is config

    for error in (FileNotFoundError("missing"), yaml.YAMLError("bad"), OSError("io")):
        monkeypatch.setattr(cli, "load_config", lambda _, error=error: (_ for _ in ()).throw(error))
        with pytest.raises(typer.Exit) as exc_info:
            cli._load_config_or_exit("config.yaml")
        assert exc_info.value.exit_code == 1


@pytest.fixture
def cli_runtime(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, MagicMock, MagicMock]:
    config = {
        "pipeline": {
            "tasks": [
                {"name": "augment", "enabled": True},
                {"name": "disabled", "enabled": False},
                {"name": "train"},
            ]
        }
    }
    logger = MagicMock()
    runner = MagicMock()
    monkeypatch.setattr(cli, "setup_logging", lambda _: logger)
    monkeypatch.setattr(cli, "_load_config_or_exit", lambda _: config)
    monkeypatch.setattr(cli, "run_pipeline", runner)
    monkeypatch.setattr(main_pipeline, "build_task_registry", lambda _: {"augment": object()})
    monkeypatch.setattr(
        main_pipeline,
        "interactive_task_selection",
        lambda config, registry: ["interactive"],
    )
    monkeypatch.setattr(
        main_pipeline,
        "get_tasks_from_groups",
        lambda groups, config: [f"group:{groups[0]}"],
    )
    return config, logger, runner


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, ["augment", "train"]),
        ({"tasks": ["augment", "train"], "exclude_tasks": ["train"]}, ["augment"]),
        ({"interactive": True}, ["interactive"]),
        ({"task_groups": ["fast"]}, ["group:fast"]),
    ],
)
def test_run_selects_tasks_from_each_supported_source(
    cli_runtime: tuple[dict, MagicMock, MagicMock],
    overrides: dict[str, object],
    expected: list[str],
) -> None:
    config, _, runner = cli_runtime

    _run_command(
        **overrides,
        device="cpu",
        epochs=2,
        imgsz=320,
        batch=1,
        model="base.pt",
        project="runs",
        name="test",
        weights="best.pt",
        infer_input="input",
        infer_output="output",
        product="Cable1",
        force=True,
    )

    runner.assert_called_once()
    selected, passed_config, _, args = runner.call_args.args
    assert selected == expected
    assert passed_config is config
    assert args.tasks == expected
    assert args.device == "cpu"
    assert args.force is True


def test_run_returns_without_calling_pipeline_when_selection_is_empty(
    cli_runtime: tuple[dict, MagicMock, MagicMock],
) -> None:
    _, logger, runner = cli_runtime

    _run_command(tasks=["only"], exclude_tasks=["only"])

    runner.assert_not_called()
    logger.warning.assert_called_once()


@pytest.mark.parametrize(
    "error",
    [ConfigurationError("bad config"), PipelineError("failed"), RuntimeError("runtime")],
)
def test_run_translates_pipeline_failures_to_typer_exit(
    cli_runtime: tuple[dict, MagicMock, MagicMock],
    error: Exception,
) -> None:
    _, logger, runner = cli_runtime
    runner.side_effect = error

    with pytest.raises(typer.Exit) as exc_info:
        _run_command(tasks=["augment"])

    assert exc_info.value.exit_code == 1
    logger.exception.assert_called_once()


def test_list_tasks_prints_sorted_registry_and_tolerates_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output: list[tuple[str, bool]] = []
    registry = {
        "zeta": SimpleNamespace(description="last"),
        "alpha": SimpleNamespace(description="first"),
    }
    monkeypatch.setattr(cli.typer, "echo", lambda text, err=False: output.append((text, err)))
    monkeypatch.setattr(main_pipeline, "build_task_registry", lambda cfg: registry)

    monkeypatch.setattr(cli, "load_config", lambda _: {"valid": True})
    cli.list_tasks("config.yaml")
    assert [line[0].strip().split()[0] for line in output[1:]] == ["alpha", "zeta"]

    output.clear()
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: (_ for _ in ()).throw(ConfigurationError("invalid")),
    )
    cli.list_tasks("config.yaml")
    assert output[0][0] == "Available tasks:"


def test_describe_known_and_unknown_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    output: list[tuple[str, bool]] = []
    registry = {
        "train": SimpleNamespace(
            name="train",
            description="Train model",
            dependencies=["split"],
        ),
        "doctor": SimpleNamespace(
            name="doctor",
            description="Check environment",
            dependencies=[],
        ),
    }
    monkeypatch.setattr(cli.typer, "echo", lambda text, err=False: output.append((text, err)))
    monkeypatch.setattr(main_pipeline, "build_task_registry", lambda cfg: registry)
    monkeypatch.setattr(cli, "load_config", lambda _: {})

    cli.describe("train", "config.yaml")
    cli.describe("doctor", "config.yaml")
    assert ("Dependencies: split", False) in output
    assert ("Dependencies: None", False) in output

    monkeypatch.setattr(cli, "load_config", lambda _: (_ for _ in ()).throw(OSError("bad")))
    with pytest.raises(typer.Exit) as exc_info:
        cli.describe("unknown", "config.yaml")
    assert exc_info.value.exit_code == 1
    assert output[-1] == ("Unknown task: unknown", True)


@pytest.mark.parametrize("baseline_only", [False, True])
def test_anomalib_folder_command_forwards_options_and_reports_result(
    tmp_path: Path,
    baseline_only: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = MagicMock()
    trainer = MagicMock(
        return_value=SimpleNamespace(
            run_dir=tmp_path / "run",
            checkpoint_path=None,
            report_path=tmp_path / "report.json",
            normal_image_count=4,
            abnormal_image_count=0,
            baseline_only=baseline_only,
        )
    )
    output: list[str] = []
    monkeypatch.setattr(cli, "setup_logging", lambda _: logger)
    monkeypatch.setattr(anomalib_trainer, "train_anomalib_folder", trainer)
    monkeypatch.setattr(cli.typer, "echo", lambda text, **kwargs: output.append(text))

    cli.anomalib_train_folder(
        input_dir=tmp_path,
        product="PCBA1",
        area="B",
        project=tmp_path / "runs",
        model="padim",
        image_size=64,
        batch_size=2,
        max_epochs=1,
        accelerator="cpu",
        devices="2",
        pre_trained=False,
        require_anomalous_validation=False,
        force=True,
        tmp_dir=tmp_path / "temp",
    )

    assert trainer.call_args.kwargs["devices"] == 2
    assert trainer.call_args.kwargs["force"] is True
    assert any("Checkpoint: not found" in line for line in output)
    expected_status = "baseline_only=true" if baseline_only else "validated layout"
    assert any(expected_status in line for line in output)


@pytest.mark.parametrize("error", [ValueError("bad"), RuntimeError("failed"), ImportError("missing")])
def test_anomalib_folder_command_translates_failures(
    tmp_path: Path,
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = MagicMock()
    monkeypatch.setattr(cli, "setup_logging", lambda _: logger)
    monkeypatch.setattr(
        anomalib_trainer,
        "train_anomalib_folder",
        lambda **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(typer.Exit) as exc_info:
        cli.anomalib_train_folder(
            input_dir=tmp_path,
            product="PCBA1",
            area="B",
            project=None,
            model="padim",
            image_size=64,
            batch_size=2,
            max_epochs=1,
            accelerator="cpu",
            devices="auto",
            pre_trained=False,
            require_anomalous_validation=True,
            force=False,
            tmp_dir=tmp_path,
        )
    assert exc_info.value.exit_code == 1
    logger.exception.assert_called_once()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(" 3 ", 3), ("auto", "auto"), ("0,1", "0,1")],
)
def test_parse_devices(raw: str, expected: str | int) -> None:
    assert cli._parse_devices(raw) == expected
