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


@pytest.fixture()
def station_workspace(tmp_path, monkeypatch):
    """A temporary workspace whose station declares a class schema.

    ``golden register`` resolves the class contract from the deployed station
    rather than from the command line, so with no station reachable it refuses
    -- correctly. Pointing ``YOLO11_WORKSPACE_ROOT`` at a workspace built here
    makes that resolution hermetic. Without it these tests read whichever
    station happens to exist on the machine running them, which is why they
    passed on a developer's full workspace and failed in CI, where the child
    repository is checked out alone and ``models/`` is not in version control.

    The station is Cable1/A because that is what ``configs/autonomous_training.yaml``
    declares, and the class list is that station's real contract -- deliberately
    not the order of its ``expected_items``, which is a different thing and is
    never read as a schema.
    """
    root = tmp_path / "workspace"
    (root / "training" / "data").mkdir(parents=True)
    model_dir = root / "inference" / "models" / "Cable1" / "A" / "yolo"
    model_dir.mkdir(parents=True)
    (model_dir / "config.yaml").write_text(
        yaml.safe_dump({"class_names": ["Black", "Green", "Orange", "Red", "Yellow"]}),
        encoding="utf-8",
    )
    (root / "workspace.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "projects": {"training": "training", "inference": "inference"},
                "paths": {
                    "training_data": "training/data",
                    "inference_models": "inference/models",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("YOLO11_WORKSPACE_ROOT", str(root))
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


def test_golden_register_locks_a_directory(golden_dir, station_workspace):
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


def test_golden_register_records_the_split_from_a_candidate_report(
    golden_dir, tmp_path, station_workspace
):
    """The reviewer's own classification, carried through by content hash."""
    import csv
    import hashlib

    digest = hashlib.sha256((golden_dir / "a.jpg").read_bytes()).hexdigest()
    report = tmp_path / "report"
    report.mkdir()
    with open(
        report / "candidates.csv", "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "group", "image_sha256"])
        writer.writerow(["a", "hard_case", digest])
        # An image the reviewer did not keep; it must not reach the manifest.
        writer.writerow(["b", "representative", "b" * 64])

    result = runner.invoke(
        app,
        [
            "golden",
            "register",
            str(golden_dir),
            "--registered-by",
            "engineer",
            "--groups",
            str(report),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["groups"] == {"hard_case": 1}
    assert payload["ungrouped"] == 0
    assert payload["group_assignments_read"] == 2


def test_golden_register_refuses_a_report_that_matches_nothing(
    golden_dir, tmp_path, station_workspace
):
    """Silently registering unsplit would look like a set that has a split."""
    import csv

    report = tmp_path / "report"
    report.mkdir()
    with open(
        report / "candidates.csv", "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "group", "image_sha256"])
        writer.writerow(["other", "hard_case", "c" * 64])

    result = runner.invoke(
        app,
        [
            "golden",
            "register",
            str(golden_dir),
            "--registered-by",
            "engineer",
            "--groups",
            str(report),
        ],
    )

    assert result.exit_code == 1
    assert "match an image in this directory" in result.output
    assert not (golden_dir / "golden_manifest.json").exists()


def test_golden_register_without_groups_registers_unsplit(golden_dir, station_workspace):
    result = runner.invoke(
        app,
        ["golden", "register", str(golden_dir), "--registered-by", "engineer"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["groups"] == {}
    assert payload["group_assignments_read"] == 0


def test_golden_register_requires_an_owner(golden_dir):
    result = runner.invoke(app, ["golden", "register", str(golden_dir)])

    assert result.exit_code != 0


def test_registering_twice_is_refused(golden_dir, station_workspace):
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
    """Asked of the command table, not of the rendered help text.

    The help legitimately uses the word in prose -- "compare it with the
    deployed champion", and ``registry`` describes itself as listing "the
    deployed champion" -- so scanning the text for the substring reports a
    command that does not exist. Subtracting one known phrase did not hold
    either: the help is rendered to the terminal width, so the phrase wraps
    across two lines and stops matching.
    """
    registered = {
        info.name or info.callback.__name__ for info in app.registered_commands
    }
    registered |= {str(info.name) for info in app.registered_groups}
    assert "deploy" not in registered

    result = runner.invoke(app, ["deploy"])
    assert result.exit_code != 0


def test_report_for_an_unknown_cycle_is_a_clean_error():
    result = runner.invoke(app, ["report", "cycle_that_never_ran"])

    assert result.exit_code == 1
    assert "No report" in result.output
