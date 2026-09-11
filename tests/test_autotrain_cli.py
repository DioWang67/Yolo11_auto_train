"""The command line, exercised through typer's runner.

Mostly one question: with the feature off, does every command refuse without
touching anything?
"""

from __future__ import annotations

import json

import pytest
import yaml

pytest.importorskip(
    "typer",
    reason="Typer is required to exercise the autonomous training CLI",
)

from typer.testing import CliRunner  # noqa: E402

from picture_tool.autotrain.cli import app  # noqa: E402

runner = CliRunner()

#: Exit code the CLI uses for "the feature flag is off".
DISABLED_EXIT = 2


@pytest.fixture()
def disabled_config(tmp_path) -> str:
    path = tmp_path / "autonomous_training.yaml"
    path.write_text(
        yaml.safe_dump({"autonomous_training": {"enabled": False}}), encoding="utf-8"
    )
    return str(path)


@pytest.fixture()
def golden_dir(tmp_path):
    root = tmp_path / "golden"
    root.mkdir()
    (root / "a.jpg").write_bytes(b"image-a")
    return root


# ---------------------------------------------------------------------------
# The feature flag


@pytest.mark.parametrize(
    "command",
    [
        ["cycle"],
        ["collect"],
        ["select", "--cycle-id", "cycle_x"],
        ["dataset"],
        ["labeling", "export", "--classes", "Red"],
    ],
)
def test_every_write_command_refuses_while_disabled(disabled_config, command, tmp_path):
    result = runner.invoke(app, [*command, "--config", disabled_config])

    assert result.exit_code == DISABLED_EXIT, result.output
    assert "disabled" in result.output.lower()


def test_the_refusal_names_the_file_to_edit(disabled_config):
    result = runner.invoke(app, ["cycle", "--config", disabled_config])

    assert "autonomous_training.yaml" in result.output


# ---------------------------------------------------------------------------
# Read commands work regardless


def test_golden_check_reports_not_configured(disabled_config):
    result = runner.invoke(app, ["golden", "check", "--config", disabled_config])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "NOT_CONFIGURED"


def test_golden_register_locks_a_directory(golden_dir):
    result = runner.invoke(
        app,
        [
            "golden",
            "register",
            str(golden_dir),
            "--registered-by",
            "engineer",
            "--description",
            "cable1 evaluation set",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["image_count"] == 1
    assert payload["manifest_sha256"]
    assert (golden_dir / "golden_manifest.json").is_file()


def test_golden_register_requires_an_owner(golden_dir):
    result = runner.invoke(app, ["golden", "register", str(golden_dir)])

    assert result.exit_code != 0


def test_registering_twice_is_refused(golden_dir):
    runner.invoke(
        app, ["golden", "register", str(golden_dir), "--registered-by", "engineer"]
    )

    result = runner.invoke(
        app, ["golden", "register", str(golden_dir), "--registered-by", "engineer"]
    )

    assert result.exit_code == 1
    assert "locked once" in result.output


# ---------------------------------------------------------------------------
# Shape


def test_help_says_it_never_deploys():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Never deploys" in result.output


def test_there_is_no_deploy_command():
    result = runner.invoke(app, ["--help"])

    assert "deploy" not in result.output.lower().replace("never deploys", "")


def test_report_for_an_unknown_cycle_is_a_clean_error():
    result = runner.invoke(app, ["report", "cycle_that_never_ran"])

    assert result.exit_code == 1
    assert "No report" in result.output
