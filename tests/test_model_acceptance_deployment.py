from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest
import yaml

import picture_tool.tasks.deploy as deploy_module
from picture_tool.gui.operator_handoff import (
    OperatorHandoffError,
    _resolve_model_acceptance_gate,
)
from picture_tool.runtime_pair_deployment import (
    NumericalComparison,
    PairVerification,
)
from picture_tool.tasks.deploy import _run_model_acceptance_gate, run_deploy


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _is_color_revision_verification(command: list[str]) -> bool:
    return "app.acceptance.color_revision_contract" in command


def _color_revision_contract(
    command: list[str],
    target: dict[str, str],
) -> dict[str, object]:
    enabled = "--color-model" in command
    payload: dict[str, object] = {
        "schema_version": 1,
        "enabled": enabled,
        "checker_type": "color_qc" if enabled else "",
        "target": target,
        "entries": [],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _acceptance_report_payload(
    command: list[str],
    *,
    snapshot_manifest: Path,
) -> dict[str, object]:
    candidate_weight = Path(_command_value(command, "--candidate-weight"))
    candidate_config = Path(_command_value(command, "--candidate-config"))
    color_model = (
        Path(_command_value(command, "--color-model"))
        if "--color-model" in command
        else None
    )
    target = {
        "product": _command_value(command, "--product"),
        "area": _command_value(command, "--area"),
        "inference_type": _command_value(command, "--inference-type"),
    }
    return {
        "passed": True,
        "failures": [],
        "target": target,
        "candidate": {
            "version": _command_value(command, "--candidate-version"),
            "sha256": _sha256(candidate_weight),
            "runtime_config_sha256": _sha256(candidate_config),
            "color_model_sha256": (
                _sha256(color_model) if color_model is not None else ""
            ),
        },
        "policy": {
            "min_confirmed": int(_command_value(command, "--min-confirmed")),
            "max_false_positives": int(
                _command_value(command, "--max-false-positives")
            ),
            "max_false_negatives": int(
                _command_value(command, "--max-false-negatives")
            ),
            "max_regressions": int(
                _command_value(command, "--max-regressions")
            ),
            "require_all_confirmed": "--allow-pending" not in command,
            "require_no_errors": "--allow-errors" not in command,
            "require_baseline_predictions": True,
        },
        "color_revisions": _color_revision_contract(command, target),
        "metrics": {"fp": 0, "fn": 0},
        "baseline_metrics": {"fp": 0, "fn": 0},
        "dataset": {
            "snapshot_manifest_sha256": _sha256(snapshot_manifest),
        },
    }


def _write_pt_deploy_fixture(
    tmp_path: Path,
    *,
    inference_models_dir: Path,
    inference_project_root: Path,
    enable_color_check: bool = False,
) -> tuple[dict, Path]:
    run_dir = tmp_path / "runs" / "train"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.pt").write_bytes(b"candidate-pt")
    detection_config = {
        "weights": "best.pt",
        "enable_color_check": enable_color_check,
    }
    if enable_color_check:
        detection_config["color_model_path"] = "color_stats.json"
        (run_dir / "color_stats.json").write_text(
            '{"Orange": {"mean": [1, 2, 3]}}',
            encoding="utf-8",
        )
    (run_dir / "detection_config.yaml").write_text(
        yaml.safe_dump(detection_config),
        encoding="utf-8",
    )
    inference_project_root.mkdir(parents=True, exist_ok=True)
    config = {
        "yolo_training": {
            "project": str(tmp_path / "runs"),
            "name": "train",
            "export_onnx": {"enabled": False},
            "deploy": {
                "enabled": True,
                "product": "Cable1",
                "area": "A",
                "inference_models_dir": str(inference_models_dir),
                "inference_project_root": str(inference_project_root),
                "version": "1.0.0",
            },
        }
    }
    return config, run_dir


def _write_onnx_deploy_fixture(
    tmp_path: Path,
    *,
    inference_models_dir: Path,
    inference_project_root: Path,
) -> tuple[dict, Path, Path, Path]:
    config, run_dir = _write_pt_deploy_fixture(
        tmp_path,
        inference_models_dir=inference_models_dir,
        inference_project_root=inference_project_root,
    )
    runtime = run_dir / "weights" / "best.onnx"
    training = run_dir / "weights" / "best.pt"
    runtime.write_bytes(b"verified-onnx")
    (run_dir / "detection_config.yaml").write_text(
        yaml.safe_dump(
            {
                "weights": "best.onnx",
                "enable_color_check": False,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "runtime_export_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runtime_file": "weights/best.onnx",
                "training_weight_file": "weights/best.pt",
                "runtime_sha256": _sha256(runtime),
                "training_weight_sha256": _sha256(training),
            }
        ),
        encoding="utf-8",
    )
    config["yolo_training"]["deploy"]["runtime_pair_verification"] = {
        "enabled": True,
    }
    return config, run_dir, runtime, training


def _passing_pair_verification(
    runtime: Path,
    training: Path,
) -> PairVerification:
    return PairVerification(
        runtime_path=runtime.resolve(),
        training_weight_path=training.resolve(),
        runtime_sha256=_sha256(runtime),
        training_weight_sha256=_sha256(training),
        input_size=640,
        comparison=NumericalComparison(
            runtime_shape=(1, 84, 8400),
            training_shape=(1, 84, 8400),
            max_abs_error=1e-5,
            mean_abs_error=1e-6,
            p99_abs_error=5e-6,
            passed=True,
            class_names=("Cable",),
        ),
    )


def _resolve_inference_runtime_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


@pytest.mark.parametrize("unsafe_value", ["", "..", "../other", "Cable/A"])
def test_acceptance_target_rejects_unsafe_path_segment(unsafe_value: str) -> None:
    with pytest.raises(ValueError, match="not a safe path segment"):
        deploy_module._validate_acceptance_target_segment(
            unsafe_value,
            label="product",
        )


def _write_required_acceptance_snapshot(
    acceptance_root: Path,
    *,
    summary_changes: dict[str, object] | None = None,
) -> Path:
    snapshot_root = acceptance_root / "snapshots" / "current"
    snapshot_root.mkdir(parents=True)
    manifest_path = snapshot_root / "ground_truth.csv"
    manifest_path.write_text(
        "sample_id,review_status,expected_verdict,machine_status\n"
        "sample,confirmed,OK,OK\n",
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "snapshot_id": snapshot_root.name,
        "record_count": 1,
        "confirmed_count": 1,
        "manifest_sha256": _sha256(manifest_path),
        "metrics": {"confirmed": 1, "fp": 0, "fn": 0},
    }
    summary.update(summary_changes or {})
    (snapshot_root / "snapshot.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    return snapshot_root


def test_operator_job_freezes_latest_acceptance_snapshot(tmp_path: Path) -> None:
    acceptance_root = tmp_path / "acceptance" / "Cable1" / "A"
    older = acceptance_root / "snapshots" / "20260729T100000+0800-old"
    latest = acceptance_root / "snapshots" / "20260730T100000+0800-current"
    for root, false_positives in ((older, 40), (latest, 35)):
        root.mkdir(parents=True)
        (root / "ground_truth.csv").write_text(
            "sample_id,review_status\nsample,confirmed\n",
            encoding="utf-8",
        )
        (root / "snapshot.json").write_text(
            json.dumps(
                {
                    "confirmed_count": 250,
                    "metrics": {"fp": false_positives, "fn": 0},
                }
            ),
            encoding="utf-8",
        )

    gate = _resolve_model_acceptance_gate(acceptance_root)

    assert gate["enabled"] is True
    assert gate["snapshot_manifest"] == str(latest / "ground_truth.csv")
    assert gate["min_confirmed"] == 250
    assert gate["max_false_positives"] == 35
    assert gate["max_false_negatives"] == 0
    assert gate["max_regressions"] == 0


def test_operator_job_rejects_acceptance_data_without_snapshot(
    tmp_path: Path,
) -> None:
    acceptance_root = tmp_path / "acceptance" / "Cable1" / "A"
    acceptance_root.mkdir(parents=True)

    with pytest.raises(OperatorHandoffError, match="no snapshots"):
        _resolve_model_acceptance_gate(acceptance_root)


def test_schema6_operator_job_requires_acceptance_data(tmp_path: Path) -> None:
    acceptance_root = tmp_path / "station_data" / "acceptance" / "Cable1" / "A"

    with pytest.raises(OperatorHandoffError, match="acceptance data is missing"):
        _resolve_model_acceptance_gate(acceptance_root, required=True)


def test_schema6_acceptance_snapshot_verifies_producer_contract(
    tmp_path: Path,
) -> None:
    acceptance_root = tmp_path / "station_data" / "acceptance" / "Cable1" / "A"
    snapshot_root = _write_required_acceptance_snapshot(acceptance_root)

    gate = _resolve_model_acceptance_gate(acceptance_root, required=True)

    assert gate["enabled"] is True
    assert gate["snapshot_manifest"] == str(snapshot_root / "ground_truth.csv")
    assert gate["snapshot_manifest_sha256"] == _sha256(
        snapshot_root / "ground_truth.csv"
    )
    assert gate["min_confirmed"] == 1


@pytest.mark.parametrize(
    ("summary_changes", "error_match"),
    (
        ({"schema_version": 2}, "schema is unsupported"),
        ({"snapshot_id": "different"}, "identity does not match"),
        ({"manifest_sha256": "0" * 64}, "checksum does not match"),
        ({"record_count": 2}, "record_count does not match"),
        ({"confirmed_count": 2}, "confirmed_count does not match"),
        (
            {"metrics": {"confirmed": 1, "fp": 1, "fn": 0}},
            "metrics do not match",
        ),
    ),
)
def test_schema6_acceptance_snapshot_rejects_tampered_summary(
    tmp_path: Path,
    summary_changes: dict[str, object],
    error_match: str,
) -> None:
    acceptance_root = tmp_path / "station_data" / "acceptance" / "Cable1" / "A"
    _write_required_acceptance_snapshot(
        acceptance_root,
        summary_changes=summary_changes,
    )

    with pytest.raises(OperatorHandoffError, match=error_match):
        _resolve_model_acceptance_gate(acceptance_root, required=True)


def test_operator_job_rejects_non_numeric_acceptance_counts(tmp_path: Path) -> None:
    acceptance_root = tmp_path / "acceptance" / "Cable1" / "A"
    snapshot = acceptance_root / "snapshots" / "invalid-counts"
    snapshot.mkdir(parents=True)
    (snapshot / "ground_truth.csv").write_text(
        "sample_id,review_status\nsample,confirmed\n",
        encoding="utf-8",
    )
    (snapshot / "snapshot.json").write_text(
        json.dumps(
            {
                "confirmed_count": "not-a-number",
                "metrics": {"fp": 0, "fn": 0},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(OperatorHandoffError, match="counts are invalid"):
        _resolve_model_acceptance_gate(acceptance_root)


def test_deploy_acceptance_runner_publishes_candidate_scoped_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_root = tmp_path / "inference"
    color_revisions_root = tmp_path / "station_data" / ".color_revisions"
    runner = inference_root / "app" / "acceptance" / "headless.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# test runner\n", encoding="utf-8")
    (inference_root / "config.yaml").write_text(
        "weights: placeholder.onnx\n",
        encoding="utf-8",
    )
    dataset_root = inference_root / "acceptance" / "Cable1" / "A"
    snapshot = dataset_root / "snapshots" / "frozen" / "ground_truth.csv"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("sample_id\nsample\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    candidate_weight = run_dir / "best.onnx"
    candidate_weight.write_bytes(b"candidate-onnx")
    written_config: dict[str, object] = {}

    def fake_run(command, **kwargs):
        if _is_color_revision_verification(command):
            return subprocess.CompletedProcess(
                command,
                0,
                "[color-revisions] VERIFIED\n",
                "",
            )
        assert Path(
            command[command.index("--color-revisions-root") + 1]
        ) == color_revisions_root
        report_path = Path(command[command.index("--report") + 1])
        candidate_config = Path(
            command[command.index("--candidate-config") + 1]
        )
        written_config.update(
            yaml.safe_load(candidate_config.read_text(encoding="utf-8"))
        )
        report_path.write_text(
            json.dumps(
                _acceptance_report_payload(
                    command,
                    snapshot_manifest=snapshot,
                )
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "PASSED\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    config = {
        "yolo_training": {
            "deploy": {
                "acceptance_gate": {
                    "enabled": True,
                    "dataset_root": str(dataset_root),
                    "snapshot_manifest": str(snapshot),
                    "snapshot_manifest_sha256": _sha256(snapshot),
                    "min_confirmed": 250,
                    "max_false_positives": 35,
                    "max_false_negatives": 0,
                    "max_regressions": 0,
                }
            }
        }
    }
    candidate_config = {
        "weights": str(candidate_weight),
        "expected_items": {"Cable1": {"A": ["Red"]}},
    }

    report, report_path = _run_model_acceptance_gate(
        config=config,
        run_dir=run_dir,
        inference_project_root=inference_root,
        color_revisions_root=color_revisions_root,
        product="Cable1",
        area="A",
        candidate_config=candidate_config,
        candidate_weight=candidate_weight,
        color_model=None,
        logger=logging.getLogger(__name__),
    )

    assert report["passed"] is True
    assert report_path is not None
    assert report_path.name == "report.json"
    assert report_path.parent.parent == run_dir / ".acceptance_gate_runs"
    runner_log = (report_path.parent / "runner.log").read_text(encoding="utf-8")
    assert "PASSED\n" in runner_log
    assert "[color-revision-verification:post-run]" in runner_log
    assert not (report_path.parent / "candidate_models").exists()
    assert written_config["weights"] == str(candidate_weight)


def test_concurrent_station_gates_isolate_invocation_config_policy_and_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_root = tmp_path / "inference"
    runner = inference_root / "app" / "acceptance" / "headless.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# test runner\n", encoding="utf-8")
    (inference_root / "config.yaml").write_text(
        "weights: placeholder.pt\n",
        encoding="utf-8",
    )
    dataset_root = inference_root / "acceptance" / "Cable1" / "A"
    snapshot = dataset_root / "snapshots" / "frozen" / "ground_truth.csv"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("sample_id\nsample\n", encoding="utf-8")
    run_dir = tmp_path / "shared-training-run"
    run_dir.mkdir()
    first_weight = run_dir / "first.pt"
    second_weight = run_dir / "second.pt"
    first_weight.write_bytes(b"first-candidate")
    second_weight.write_bytes(b"second-candidate")
    entered_runner = Barrier(2)
    reports_written = Barrier(2)
    captured_commands: list[list[str]] = []
    command_lock = Lock()

    def fake_run(command, **kwargs):
        if _is_color_revision_verification(command):
            return subprocess.CompletedProcess(
                command,
                0,
                "[color-revisions] VERIFIED\n",
                "",
            )
        with command_lock:
            captured_commands.append(command)
        entered_runner.wait(timeout=5)
        report_path = Path(_command_value(command, "--report"))
        report_path.write_text(
            json.dumps(
                _acceptance_report_payload(
                    command,
                    snapshot_manifest=snapshot,
                )
            ),
            encoding="utf-8",
        )
        reports_written.wait(timeout=5)
        return subprocess.CompletedProcess(command, 0, "PASSED\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    def run_gate(
        candidate_weight: Path,
        *,
        max_false_positives: int,
        station_marker: str,
        product: str,
        area: str,
    ) -> tuple[dict[str, object], Path | None]:
        config = {
            "yolo_training": {
                "deploy": {
                    "acceptance_gate": {
                        "enabled": True,
                        "dataset_root": str(dataset_root),
                        "snapshot_manifest": str(snapshot),
                        "snapshot_manifest_sha256": _sha256(snapshot),
                        "max_false_positives": max_false_positives,
                    }
                }
            }
        }
        return _run_model_acceptance_gate(
            config=config,
            run_dir=run_dir,
            inference_project_root=inference_root,
            color_revisions_root=tmp_path / f"{station_marker}-color-revisions",
            product=product,
            area=area,
            candidate_config={
                "weights": str(candidate_weight),
                "station_marker": station_marker,
            },
            candidate_weight=candidate_weight,
            color_model=None,
            logger=logging.getLogger(__name__),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(
            run_gate,
            first_weight,
            max_false_positives=1,
            station_marker="station-one",
            product="Cable1",
            area="A",
        )
        second_future = executor.submit(
            run_gate,
            second_weight,
            max_false_positives=2,
            station_marker="station-two",
            product="Cable2",
            area="B",
        )
        first_result = first_future.result(timeout=10)
        second_result = second_future.result(timeout=10)

    report_paths = [first_result[1], second_result[1]]
    assert all(path is not None for path in report_paths)
    resolved_report_paths = {path.resolve() for path in report_paths if path}
    assert len(resolved_report_paths) == 2
    candidate_config_paths = {
        Path(_command_value(command, "--candidate-config")).resolve()
        for command in captured_commands
    }
    assert len(candidate_config_paths) == 2
    assert {
        report["policy"]["max_false_positives"]
        for report, _ in (first_result, second_result)
    } == {1, 2}
    assert {
        report["candidate"]["sha256"]
        for report, _ in (first_result, second_result)
    } == {_sha256(first_weight), _sha256(second_weight)}
    assert {
        (report["target"]["product"], report["target"]["area"])
        for report, _ in (first_result, second_result)
    } == {("Cable1", "A"), ("Cable2", "B")}
    for report_path in resolved_report_paths:
        assert (report_path.parent / "runner.log").is_file()
        assert not (report_path.parent / "candidate_models").exists()


@pytest.mark.parametrize(
    ("mismatch", "expected_message"),
    [
        ("policy", "deployment policy"),
        ("target", "deployment target"),
    ],
)
def test_acceptance_gate_rejects_policy_or_target_cross_talk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
    expected_message: str,
) -> None:
    inference_root = tmp_path / "inference"
    runner = inference_root / "app" / "acceptance" / "headless.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# test runner\n", encoding="utf-8")
    (inference_root / "config.yaml").write_text(
        "weights: placeholder.pt\n",
        encoding="utf-8",
    )
    dataset_root = inference_root / "acceptance" / "Cable1" / "A"
    snapshot = dataset_root / "snapshots" / "frozen" / "ground_truth.csv"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("sample_id\nsample\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    candidate_weight = run_dir / "best.pt"
    candidate_weight.write_bytes(b"candidate-pt")

    def fake_run(command, **kwargs):
        if _is_color_revision_verification(command):
            return subprocess.CompletedProcess(
                command,
                0,
                "[color-revisions] VERIFIED\n",
                "",
            )
        payload = _acceptance_report_payload(
            command,
            snapshot_manifest=snapshot,
        )
        if mismatch == "policy":
            payload["policy"]["max_false_positives"] = 99
        else:
            payload["target"]["area"] = "other-station"
        Path(_command_value(command, "--report")).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "PASSED\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    config = {
        "yolo_training": {
            "deploy": {
                "acceptance_gate": {
                    "enabled": True,
                    "dataset_root": str(dataset_root),
                    "snapshot_manifest": str(snapshot),
                    "snapshot_manifest_sha256": _sha256(snapshot),
                }
            }
        }
    }

    with pytest.raises(ValueError, match=expected_message):
        _run_model_acceptance_gate(
            config=config,
            run_dir=run_dir,
            inference_project_root=inference_root,
            color_revisions_root=tmp_path / "color-revisions",
            product="Cable1",
            area="A",
            candidate_config={"weights": str(candidate_weight)},
            candidate_weight=candidate_weight,
            color_model=None,
            logger=logging.getLogger(__name__),
        )


def test_acceptance_gate_rejects_color_model_mutated_during_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_root = tmp_path / "inference"
    runner = inference_root / "app" / "acceptance" / "headless.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# test runner\n", encoding="utf-8")
    (inference_root / "config.yaml").write_text(
        "weights: placeholder.pt\n",
        encoding="utf-8",
    )
    dataset_root = inference_root / "acceptance" / "Cable1" / "A"
    snapshot = dataset_root / "snapshots" / "frozen" / "ground_truth.csv"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("sample_id\nsample\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    candidate_weight = run_dir / "best.pt"
    candidate_weight.write_bytes(b"candidate-pt")
    color_model = run_dir / "color_stats.json"
    color_model.write_bytes(b"original-color-model")

    def fake_run(command, **kwargs):
        if _is_color_revision_verification(command):
            return subprocess.CompletedProcess(
                command,
                0,
                "[color-revisions] VERIFIED\n",
                "",
            )
        payload = _acceptance_report_payload(
            command,
            snapshot_manifest=snapshot,
        )
        color_model.write_bytes(b"mutated-color-model")
        Path(_command_value(command, "--report")).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "PASSED\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    config = {
        "yolo_training": {
            "deploy": {
                "acceptance_gate": {
                    "enabled": True,
                    "dataset_root": str(dataset_root),
                    "snapshot_manifest": str(snapshot),
                    "snapshot_manifest_sha256": _sha256(snapshot),
                }
            }
        }
    }

    with pytest.raises(ValueError, match="color model changed"):
        _run_model_acceptance_gate(
            config=config,
            run_dir=run_dir,
            inference_project_root=inference_root,
            color_revisions_root=tmp_path / "color-revisions",
            product="Cable1",
            area="A",
            candidate_config={
                "weights": str(candidate_weight),
                "color_model_path": str(color_model),
            },
            candidate_weight=candidate_weight,
            color_model=color_model,
            logger=logging.getLogger(__name__),
        )


def test_deploy_acceptance_runner_rejects_snapshot_changed_during_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_root = tmp_path / "inference"
    runner = inference_root / "app" / "acceptance" / "headless.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# test runner\n", encoding="utf-8")
    (inference_root / "config.yaml").write_text("weights: placeholder.pt\n")
    dataset_root = inference_root / "acceptance" / "Cable1" / "A"
    snapshot = dataset_root / "snapshots" / "frozen" / "ground_truth.csv"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("sample_id\nsample\n", encoding="utf-8")
    expected_snapshot_sha256 = _sha256(snapshot)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    candidate_weight = run_dir / "best.pt"
    candidate_weight.write_bytes(b"candidate-pt")
    invocation_paths: dict[str, Path] = {}

    def fake_run(command, **kwargs):
        if _is_color_revision_verification(command):
            return subprocess.CompletedProcess(
                command,
                0,
                "[color-revisions] VERIFIED\n",
                "",
            )
        snapshot.write_text("sample_id\nchanged\n", encoding="utf-8")
        report_path = Path(command[command.index("--report") + 1])
        candidate_config = Path(
            command[command.index("--candidate-config") + 1]
        )
        invocation_paths["report"] = report_path
        invocation_paths["candidate_config"] = candidate_config
        report_path.write_text(
            json.dumps(
                _acceptance_report_payload(
                    command,
                    snapshot_manifest=snapshot,
                )
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "PASSED\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    config = {
        "yolo_training": {
            "deploy": {
                "acceptance_gate": {
                    "enabled": True,
                    "dataset_root": str(dataset_root),
                    "snapshot_manifest": str(snapshot),
                    "snapshot_manifest_sha256": expected_snapshot_sha256,
                }
            }
        }
    }

    with pytest.raises(ValueError, match="expected snapshot manifest") as exc_info:
        _run_model_acceptance_gate(
            config=config,
            run_dir=run_dir,
            inference_project_root=inference_root,
            color_revisions_root=tmp_path / "color-revisions",
            product="Cable1",
            area="A",
            candidate_config={"weights": str(candidate_weight)},
            candidate_weight=candidate_weight,
            color_model=None,
            logger=logging.getLogger(__name__),
        )

    report_path = invocation_paths["report"]
    assert report_path.is_file()
    assert (report_path.parent / "runner.log").is_file()
    assert invocation_paths["candidate_config"].is_file()
    assert any(
        str(report_path.parent) in note
        for note in getattr(exc_info.value, "__notes__", [])
    )


def test_deploy_acceptance_runner_rejects_snapshot_mismatch_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_root = tmp_path / "inference"
    dataset_root = inference_root / "acceptance" / "Cable1" / "A"
    snapshot = dataset_root / "snapshots" / "frozen" / "ground_truth.csv"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("sample_id\nsample\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    candidate_weight = run_dir / "best.pt"
    candidate_weight.write_bytes(b"candidate-pt")
    subprocess_called = False

    def unexpected_run(command, **kwargs):
        nonlocal subprocess_called
        subprocess_called = True
        raise AssertionError("acceptance subprocess must not run")

    monkeypatch.setattr(subprocess, "run", unexpected_run)
    config = {
        "yolo_training": {
            "deploy": {
                "acceptance_gate": {
                    "enabled": True,
                    "dataset_root": str(dataset_root),
                    "snapshot_manifest": str(snapshot),
                    "snapshot_manifest_sha256": "0" * 64,
                }
            }
        }
    }

    with pytest.raises(ValueError, match="changed before model acceptance"):
        _run_model_acceptance_gate(
            config=config,
            run_dir=run_dir,
            inference_project_root=inference_root,
            color_revisions_root=tmp_path / "color-revisions",
            product="Cable1",
            area="A",
            candidate_config={"weights": str(candidate_weight)},
            candidate_weight=candidate_weight,
            color_model=None,
            logger=logging.getLogger(__name__),
        )

    assert subprocess_called is False


def test_deploy_runtime_paths_follow_custom_models_root(tmp_path: Path) -> None:
    inference_project_root = tmp_path / "inference-source"
    inference_models_dir = tmp_path / "station-runtime" / "custom-models"
    config, _ = _write_pt_deploy_fixture(
        tmp_path,
        inference_models_dir=inference_models_dir,
        inference_project_root=inference_project_root,
        enable_color_check=True,
    )

    run_deploy(config, SimpleNamespace())

    station_dir = inference_models_dir / "Cable1" / "A" / "yolo"
    deployed_config = yaml.safe_load(
        (station_dir / "config.yaml").read_text(encoding="utf-8")
    )
    runtime_path = _resolve_inference_runtime_path(
        inference_project_root,
        deployed_config["weights"],
    )
    color_path = _resolve_inference_runtime_path(
        inference_project_root,
        deployed_config["color_model_path"],
    )

    assert runtime_path.is_file()
    assert runtime_path.is_relative_to(inference_models_dir.resolve())
    assert color_path == (station_dir / "color_stats.json").resolve()
    assert color_path.is_file()
    assert not deployed_config["weights"].startswith("models/")
    assert not deployed_config["color_model_path"].startswith("models/")


def test_deploy_manifest_binds_pair_proof_to_copied_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_project_root = tmp_path / "inference"
    inference_models_dir = inference_project_root / "models"
    config, _, _, _ = _write_onnx_deploy_fixture(
        tmp_path,
        inference_models_dir=inference_models_dir,
        inference_project_root=inference_project_root,
    )

    monkeypatch.setattr(
        deploy_module,
        "verify_runtime_pair",
        lambda runtime_path, training_path, **kwargs: _passing_pair_verification(
            runtime_path,
            training_path,
        ),
    )

    run_deploy(config, SimpleNamespace())

    station_dir = inference_models_dir / "Cable1" / "A" / "yolo"
    manifest = yaml.safe_load(
        (station_dir / "deployment_manifest.yaml").read_text(encoding="utf-8")
    )
    runtime_path = station_dir / "weights" / manifest["deployed_file"]
    training_path = station_dir / "weights" / manifest["training_weight_file"]
    proof = manifest["runtime_pair_verification"]

    assert manifest["runtime_pair_verified"] is True
    assert proof["runtime_sha256"] == _sha256(runtime_path)
    assert proof["training_weight_sha256"] == _sha256(training_path)
    unsigned_proof = dict(proof)
    identity_sha256 = unsigned_proof.pop("identity_sha256")
    encoded = json.dumps(
        unsigned_proof,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert identity_sha256 == hashlib.sha256(encoded).hexdigest()


def test_pair_proof_rejects_mutation_and_recovers_after_exact_restore(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "candidate.onnx"
    training = tmp_path / "candidate.pt"
    runtime_bytes = b"verified-runtime"
    training_bytes = b"verified-training"
    runtime.write_bytes(runtime_bytes)
    training.write_bytes(training_bytes)
    verification = _passing_pair_verification(runtime, training)

    initial_proof = deploy_module._validated_runtime_pair_proof(
        verification,
        deployed_runtime_path=runtime,
        deployed_training_path=training,
    )
    runtime.write_bytes(b"mutated-runtime")

    with pytest.raises(
        ValueError,
        match="artifacts changed after runtime pair verification",
    ):
        deploy_module._validated_runtime_pair_proof(
            verification,
            deployed_runtime_path=runtime,
            deployed_training_path=training,
        )

    runtime.write_bytes(runtime_bytes)
    recovered_proof = deploy_module._validated_runtime_pair_proof(
        verification,
        deployed_runtime_path=runtime,
        deployed_training_path=training,
    )

    assert recovered_proof == initial_proof
    assert recovered_proof["runtime_sha256"] == _sha256(runtime)
    assert recovered_proof["training_weight_sha256"] == _sha256(training)
    assert recovered_proof["class_names"] == ["Cable"]


def test_pair_proof_rejects_non_passing_comparison_without_mutating_artifacts(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "candidate.onnx"
    training = tmp_path / "candidate.pt"
    runtime.write_bytes(b"verified-runtime")
    training.write_bytes(b"verified-training")
    passing = _passing_pair_verification(runtime, training)
    rejected = replace(
        passing,
        comparison=replace(passing.comparison, passed=False),
    )
    before = (runtime.read_bytes(), training.read_bytes())

    with pytest.raises(ValueError, match="does not contain a passing decision"):
        deploy_module._validated_runtime_pair_proof(
            rejected,
            deployed_runtime_path=runtime,
            deployed_training_path=training,
        )

    assert (runtime.read_bytes(), training.read_bytes()) == before


def test_deployment_transaction_rollback_restores_every_destination(
    tmp_path: Path,
) -> None:
    station_dir = tmp_path / "station"
    station_dir.mkdir()
    original_file = station_dir / "config.yaml"
    original_file.write_bytes(b"original-config")
    original_tree = station_dir / "runtime-export"
    original_tree.mkdir()
    (original_tree / "old.bin").write_bytes(b"old-tree")
    source_file = tmp_path / "source.bin"
    source_file.write_bytes(b"new-artifact")
    source_tree = tmp_path / "source-tree"
    source_tree.mkdir()
    (source_tree / "new.bin").write_bytes(b"new-tree")
    new_artifact = station_dir / "weights" / "new.bin"
    new_artifact.parent.mkdir()
    new_manifest = station_dir / "versions" / "manifest.yaml"

    transaction = deploy_module._DeploymentTransaction(station_dir)
    backup_root = transaction._backup_root
    copied_sha256 = transaction.copy_verified(source_file, original_file)
    transaction.copy_verified(source_file, new_artifact)
    transaction.replace_tree(source_tree, original_tree)
    transaction.write_yaml(new_manifest, {"state": "prepared"})

    assert copied_sha256 == hashlib.sha256(b"new-artifact").hexdigest()
    assert original_file.read_bytes() == b"new-artifact"
    assert (original_tree / "new.bin").read_bytes() == b"new-tree"
    assert new_artifact.read_bytes() == b"new-artifact"
    assert yaml.safe_load(new_manifest.read_text(encoding="utf-8")) == {
        "state": "prepared"
    }

    transaction.rollback()
    transaction.rollback()

    assert original_file.read_bytes() == b"original-config"
    assert (original_tree / "old.bin").read_bytes() == b"old-tree"
    assert not (original_tree / "new.bin").exists()
    assert not new_artifact.exists()
    assert not new_manifest.exists()
    assert not backup_root.exists()
    with pytest.raises(RuntimeError, match="already closed"):
        transaction.write_yaml(station_dir / "late.yaml", {"state": "late"})


def test_deployment_transaction_commit_keeps_replacement_and_rejects_escape(
    tmp_path: Path,
) -> None:
    station_dir = tmp_path / "station"
    station_dir.mkdir()
    source_tree = tmp_path / "source-tree"
    source_tree.mkdir()
    (source_tree / "runtime.bin").write_bytes(b"runtime")
    file_target = station_dir / "runtime-export"
    file_target.write_bytes(b"obsolete-file")
    outside = tmp_path / "outside.yaml"
    transaction = deploy_module._DeploymentTransaction(station_dir)

    with pytest.raises(ValueError, match="escapes station directory"):
        transaction.write_yaml(outside, {"unsafe": True})
    transaction.replace_tree(source_tree, file_target)
    transaction.commit()
    transaction.commit()

    assert file_target.is_dir()
    assert (file_target / "runtime.bin").read_bytes() == b"runtime"
    assert not transaction._backup_root.exists()
    assert not outside.exists()


def test_runtime_pair_lineage_accepts_exact_contract_and_direct_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    runtime = weights / "best.onnx"
    training = weights / "best.pt"
    runtime.write_bytes(b"runtime")
    training.write_bytes(b"training")
    (run_dir / "runtime_export_manifest.json").write_text(
        json.dumps(
            {
                "runtime_file": "weights/best.onnx",
                "training_weight_file": "weights/best.pt",
                "runtime_sha256": _sha256(runtime),
                "training_weight_sha256": _sha256(training),
            }
        ),
        encoding="utf-8",
    )

    resolved_pair = deploy_module._resolve_runtime_training_pair(run_dir, runtime)
    direct_pair = deploy_module._resolve_runtime_training_pair(run_dir, training)

    assert resolved_pair == training.resolve()
    assert direct_pair == training.resolve()
    assert runtime.read_bytes() == b"runtime"
    assert training.read_bytes() == b"training"


@pytest.mark.parametrize(
    ("manifest_payload", "expected_error", "expected_message"),
    [
        ("{", ValueError, "lineage contract is invalid"),
        ([], ValueError, "must contain an object"),
        ({}, ValueError, "lineage contract is incomplete"),
        (
            {
                "runtime_file": "../outside.onnx",
                "training_weight_file": "weights/best.pt",
            },
            ValueError,
            "unsafe runtime path",
        ),
        (
            {
                "runtime_file": "weights/best.onnx",
                "training_weight_file": "../outside.pt",
            },
            ValueError,
            "unsafe training path",
        ),
        (
            {
                "runtime_file": "weights/other.onnx",
                "training_weight_file": "weights/best.pt",
            },
            ValueError,
            "Selected runtime artifact does not match",
        ),
        (
            {
                "runtime_file": "weights/best.onnx",
                "training_weight_file": "weights/missing.pt",
            },
            FileNotFoundError,
            "training checkpoint is unavailable",
        ),
    ],
)
def test_runtime_pair_lineage_rejects_malformed_or_unsafe_contracts(
    tmp_path: Path,
    manifest_payload: object,
    expected_error: type[Exception],
    expected_message: str,
) -> None:
    run_dir = tmp_path / "run"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    runtime = weights / "best.onnx"
    training = weights / "best.pt"
    runtime.write_bytes(b"runtime")
    training.write_bytes(b"training")
    contract = run_dir / "runtime_export_manifest.json"
    if isinstance(manifest_payload, str):
        contract.write_text(manifest_payload, encoding="utf-8")
    else:
        contract.write_text(json.dumps(manifest_payload), encoding="utf-8")
    before = (runtime.read_bytes(), training.read_bytes())

    with pytest.raises(expected_error, match=expected_message):
        deploy_module._resolve_runtime_training_pair(run_dir, runtime)

    assert (runtime.read_bytes(), training.read_bytes()) == before


@pytest.mark.parametrize("mismatched_hash", ["runtime", "training"])
def test_runtime_pair_lineage_rejects_checksum_mismatch_without_changes(
    tmp_path: Path,
    mismatched_hash: str,
) -> None:
    run_dir = tmp_path / "run"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    runtime = weights / "best.onnx"
    training = weights / "best.pt"
    runtime.write_bytes(b"runtime")
    training.write_bytes(b"training")
    payload = {
        "runtime_file": "weights/best.onnx",
        "training_weight_file": "weights/best.pt",
        "runtime_sha256": _sha256(runtime),
        "training_weight_sha256": _sha256(training),
    }
    checksum_field = (
        "runtime_sha256"
        if mismatched_hash == "runtime"
        else "training_weight_sha256"
    )
    payload[checksum_field] = "0" * 64
    (run_dir / "runtime_export_manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=f"{mismatched_hash.capitalize()}.*checksum"):
        deploy_module._resolve_runtime_training_pair(run_dir, runtime)

    assert runtime.read_bytes() == b"runtime"
    assert training.read_bytes() == b"training"


def test_runtime_pair_lineage_requires_contract_for_non_pt_runtime(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runtime = run_dir / "best.onnx"
    runtime.write_bytes(b"runtime")

    with pytest.raises(FileNotFoundError, match="lineage contract is missing"):
        deploy_module._resolve_runtime_training_pair(run_dir, runtime)

    assert runtime.read_bytes() == b"runtime"


@pytest.mark.parametrize(
    ("content", "expected_message"),
    [
        (b"\xff", "Unable to read existing station config"),
        (b"- not\n- a\n- mapping\n", "must be a mapping"),
    ],
)
def test_station_config_snapshot_rejects_invalid_bytes_or_shape(
    tmp_path: Path,
    content: bytes,
    expected_message: str,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_bytes(content)

    with pytest.raises(ValueError, match=expected_message):
        deploy_module._load_station_config_snapshot(config_path)

    assert config_path.read_bytes() == content


def test_station_config_snapshot_detects_concurrent_change_and_missing_state(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    missing = deploy_module._load_station_config_snapshot(config_path)
    deploy_module._assert_station_config_unchanged(config_path, missing.identity)
    config_path.write_text("exposure_time: 1000\n", encoding="utf-8")
    snapshot = deploy_module._load_station_config_snapshot(config_path)

    assert missing.values is None
    assert missing.identity == "missing"
    assert snapshot.values == {"exposure_time": 1000}
    assert snapshot.identity.startswith("sha256:")
    deploy_module._assert_station_config_unchanged(config_path, snapshot.identity)

    config_path.write_text("exposure_time: 2000\n", encoding="utf-8")
    with pytest.raises(
        deploy_module.ConcurrentStationConfigChangeError,
        match="changed during deployment",
    ):
        deploy_module._assert_station_config_unchanged(
            config_path,
            snapshot.identity,
        )
    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == {
        "exposure_time": 2000
    }


def test_color_revision_publication_lock_creates_byte_lock_and_releases(
    tmp_path: Path,
) -> None:
    revisions_root = tmp_path / ".color_revisions"

    publication_lock = deploy_module._acquire_color_revision_publication_lock(
        revisions_root,
        timeout_seconds=1.0,
    )
    lock_path = revisions_root / "locks" / "deployment-publication.lock"
    try:
        assert lock_path.is_file()
        assert deploy_module._COLOR_REVISION_PUBLICATION_THREAD_LOCK.locked()
    finally:
        publication_lock.release(logging.getLogger(__name__))
    publication_lock.release(logging.getLogger(__name__))

    assert lock_path.read_bytes() == b"\0"
    assert not deploy_module._COLOR_REVISION_PUBLICATION_THREAD_LOCK.locked()
    reacquired = deploy_module._COLOR_REVISION_PUBLICATION_THREAD_LOCK.acquire(
        timeout=0
    )
    assert reacquired is True
    deploy_module._COLOR_REVISION_PUBLICATION_THREAD_LOCK.release()


def test_color_revision_publication_lock_thread_timeout_releases_owner_state(
    tmp_path: Path,
) -> None:
    thread_lock = deploy_module._COLOR_REVISION_PUBLICATION_THREAD_LOCK
    assert thread_lock.acquire(timeout=0) is True
    try:
        with pytest.raises(
            deploy_module.ColorRevisionPublicationLockTimeoutError,
            match="Another deployment thread",
        ):
            deploy_module._acquire_color_revision_publication_lock(
                tmp_path / ".color_revisions",
                timeout_seconds=0,
            )
        assert thread_lock.locked()
    finally:
        thread_lock.release()
    assert not thread_lock.locked()


def test_color_revision_publication_byte_lock_timeout_closes_handle_and_thread_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[object] = []

    def reject_byte_lock(handle) -> None:
        attempts.append(handle)
        raise OSError("simulated byte-lock contention")

    monkeypatch.setattr(
        deploy_module,
        "_lock_color_revision_publication_byte",
        reject_byte_lock,
    )

    with pytest.raises(
        deploy_module.ColorRevisionPublicationLockTimeoutError,
        match="Timed out waiting for active color revision publication lock",
    ):
        deploy_module._acquire_color_revision_publication_lock(
            tmp_path / ".color_revisions",
            timeout_seconds=0,
        )

    assert len(attempts) == 1
    assert attempts[0].closed is True
    assert not deploy_module._COLOR_REVISION_PUBLICATION_THREAD_LOCK.locked()


def test_color_revision_publication_release_reports_unlock_and_close_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class BrokenHandle:
        def seek(self, offset: int) -> None:
            raise OSError(f"seek failed at {offset}")

        def close(self) -> None:
            raise OSError("close failed")

    thread_lock = deploy_module._COLOR_REVISION_PUBLICATION_THREAD_LOCK
    assert thread_lock.acquire(timeout=0) is True
    publication_lock = deploy_module._ColorRevisionPublicationLock(BrokenHandle())

    with caplog.at_level(logging.WARNING):
        publication_lock.release(logging.getLogger(__name__))
        publication_lock.release(logging.getLogger(__name__))

    assert "unlock was deferred" in caplog.text
    assert "handle cleanup failed" in caplog.text
    assert not thread_lock.locked()


def test_atomic_copy_checksum_failure_removes_temporary_and_keeps_destination_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    destination = tmp_path / "target" / "destination.bin"
    destination.parent.mkdir()
    hashes = iter(("source-sha", "different-copy-sha"))
    monkeypatch.setattr(deploy_module, "_sha256_file", lambda path: next(hashes))

    with pytest.raises(OSError, match="Artifact checksum mismatch"):
        deploy_module._atomic_copy_verified(source, destination)

    assert not destination.exists()
    assert not destination.with_name(".destination.bin.tmp").exists()


def test_runtime_config_path_retains_absolute_path_when_drives_differ(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "models" / "best.onnx"
    artifact.parent.mkdir()
    artifact.write_bytes(b"runtime")

    def reject_relative_path(*args, **kwargs):
        raise ValueError("different drives")

    monkeypatch.setattr(deploy_module.os.path, "relpath", reject_relative_path)

    configured = deploy_module._runtime_config_path(artifact, tmp_path / "project")

    assert Path(configured) == artifact.resolve()


@pytest.mark.parametrize("mutated_artifact", ["runtime", "training"])
def test_deploy_rejects_pair_source_mutated_after_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated_artifact: str,
) -> None:
    inference_project_root = tmp_path / "inference"
    inference_models_dir = inference_project_root / "models"
    config, _, runtime, training = _write_onnx_deploy_fixture(
        tmp_path,
        inference_models_dir=inference_models_dir,
        inference_project_root=inference_project_root,
    )

    def verify_then_mutate(
        runtime_path: Path,
        training_path: Path,
        **kwargs,
    ) -> PairVerification:
        verification = _passing_pair_verification(runtime_path, training_path)
        target = runtime_path if mutated_artifact == "runtime" else training_path
        target.write_bytes(b"mutated-after-pair-verification")
        return verification

    monkeypatch.setattr(
        deploy_module,
        "verify_runtime_pair",
        verify_then_mutate,
    )

    with pytest.raises(
        ValueError,
        match="changed after runtime pair verification",
    ):
        run_deploy(config, SimpleNamespace())

    station_dir = inference_models_dir / "Cable1" / "A" / "yolo"
    expected_mutation = runtime if mutated_artifact == "runtime" else training
    assert expected_mutation.read_bytes() == b"mutated-after-pair-verification"
    assert not (station_dir / "config.yaml").exists()
    assert not (station_dir / "deployment_manifest.yaml").exists()
    assert not list((station_dir / "weights").glob("Cable1_A_v1.0.0_*"))
    assert not list(station_dir.glob(".deploy-rollback-*"))
    assert not (station_dir / ".deploy.lock").exists()


def test_acceptance_gate_holds_lock_and_final_config_uses_same_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_project_root = tmp_path / "inference"
    inference_models_dir = inference_project_root / "custom-models"
    config, _ = _write_pt_deploy_fixture(
        tmp_path,
        inference_models_dir=inference_models_dir,
        inference_project_root=inference_project_root,
    )
    station_dir = inference_models_dir / "Cable1" / "A" / "yolo"
    station_dir.mkdir(parents=True)
    station_config = station_dir / "config.yaml"
    station_config.write_text(
        yaml.safe_dump(
            {
                "weights": "old.pt",
                "exposure_time": "11111",
                "gain": "2.5",
            }
        ),
        encoding="utf-8",
    )
    captured_candidate: dict = {}

    def fake_acceptance_gate(**kwargs):
        assert (station_dir / ".deploy.lock").is_dir()
        captured_candidate.update(kwargs["candidate_config"])
        return {
            "passed": True,
            "candidate": {"sha256": _sha256(kwargs["candidate_weight"])},
        }, None

    monkeypatch.setattr(
        deploy_module,
        "_run_model_acceptance_gate",
        fake_acceptance_gate,
    )

    run_deploy(config, SimpleNamespace())

    deployed_config = yaml.safe_load(station_config.read_text(encoding="utf-8"))
    candidate_contract = {
        key: value
        for key, value in captured_candidate.items()
        if key not in {"weights", "color_model_path"}
    }
    deployed_contract = {
        key: value
        for key, value in deployed_config.items()
        if key not in {"weights", "color_model_path"}
    }
    assert candidate_contract == deployed_contract
    assert deployed_config["exposure_time"] == "11111"
    assert deployed_config["gain"] == "2.5"


def test_publication_lock_spans_final_verification_config_commit_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_project_root = tmp_path / "inference"
    inference_models_dir = inference_project_root / "models"
    config, _ = _write_pt_deploy_fixture(
        tmp_path,
        inference_models_dir=inference_models_dir,
        inference_project_root=inference_project_root,
    )
    station_config = (
        inference_models_dir / "Cable1" / "A" / "yolo" / "config.yaml"
    )
    report_path = tmp_path / "accepted" / "report.json"
    report_path.parent.mkdir(parents=True)
    events: list[str] = []

    class PublicationLock:
        released = False

        def release(self, logger: logging.Logger) -> None:
            assert "transaction_commit" in events
            self.released = True
            events.append("publication_release")

    publication_lock = PublicationLock()

    def fake_acceptance_gate(**kwargs):
        report = {
            "passed": True,
            "candidate": {
                "sha256": _sha256(kwargs["candidate_weight"]),
                "color_model_sha256": "",
            },
            "color_revisions": {"identity_sha256": "a" * 64},
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        deploy_module._bind_validated_acceptance_report(report_path, report)
        return report, report_path

    def fake_acquire_publication_lock(*args, **kwargs):
        events.append("publication_acquire")
        return publication_lock

    def fake_verify_contract(**kwargs) -> None:
        assert kwargs["stage"] == "pre-publish"
        assert not publication_lock.released
        events.append("final_contract_verify")

    original_station_check = deploy_module._assert_station_config_unchanged

    def checked_station_check(path: Path, expected_identity: str) -> None:
        if "publication_acquire" in events:
            assert not publication_lock.released
            events.append("station_cas")
        original_station_check(path, expected_identity)

    original_write_yaml = deploy_module._DeploymentTransaction.write_yaml

    def checked_write_yaml(self, path: Path, payload: dict) -> None:
        if path == station_config:
            assert not publication_lock.released
            events.append("config_publish")
        original_write_yaml(self, path, payload)

    original_commit = deploy_module._DeploymentTransaction.commit

    def checked_commit(self) -> None:
        assert not publication_lock.released
        events.append("transaction_commit")
        original_commit(self)

    monkeypatch.setattr(
        deploy_module,
        "_run_model_acceptance_gate",
        fake_acceptance_gate,
    )
    monkeypatch.setattr(
        deploy_module,
        "_acquire_color_revision_publication_lock",
        fake_acquire_publication_lock,
    )
    monkeypatch.setattr(
        deploy_module,
        "_verify_acceptance_color_revision_contract",
        fake_verify_contract,
    )
    monkeypatch.setattr(
        deploy_module,
        "_assert_station_config_unchanged",
        checked_station_check,
    )
    monkeypatch.setattr(
        deploy_module._DeploymentTransaction,
        "write_yaml",
        checked_write_yaml,
    )
    monkeypatch.setattr(
        deploy_module._DeploymentTransaction,
        "commit",
        checked_commit,
    )

    run_deploy(config, SimpleNamespace())

    assert publication_lock.released is True
    assert events.index("final_contract_verify") < events.index("station_cas")
    assert events.index("station_cas") < events.index("config_publish")
    assert events.index("config_publish") < events.index("transaction_commit")
    assert events.index("transaction_commit") < events.index(
        "publication_release"
    )


def test_deploy_rejects_color_model_mutated_after_acceptance_before_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_project_root = tmp_path / "inference"
    inference_models_dir = inference_project_root / "models"
    config, run_dir = _write_pt_deploy_fixture(
        tmp_path,
        inference_models_dir=inference_models_dir,
        inference_project_root=inference_project_root,
        enable_color_check=True,
    )
    color_model = run_dir / "color_stats.json"

    def fake_acceptance_gate(**kwargs):
        accepted_color_sha256 = _sha256(kwargs["color_model"])
        kwargs["color_model"].write_bytes(b"mutated-after-acceptance")
        return {
            "passed": True,
            "candidate": {
                "sha256": _sha256(kwargs["candidate_weight"]),
                "color_model_sha256": accepted_color_sha256,
            },
        }, None

    monkeypatch.setattr(
        deploy_module,
        "_run_model_acceptance_gate",
        fake_acceptance_gate,
    )

    with pytest.raises(ValueError, match="changed after model acceptance"):
        run_deploy(config, SimpleNamespace())

    station_dir = inference_models_dir / "Cable1" / "A" / "yolo"
    assert color_model.read_bytes() == b"mutated-after-acceptance"
    assert not (station_dir / "config.yaml").exists()
    assert not (station_dir / "deployment_manifest.yaml").exists()
    assert not list((station_dir / "weights").glob("Cable1_A_v1.0.0_*"))
    assert not list(station_dir.glob(".deploy-rollback-*"))
    assert not (station_dir / ".deploy.lock").exists()


def test_acceptance_gate_concurrent_station_edit_aborts_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_project_root = tmp_path / "inference"
    inference_models_dir = inference_project_root / "models"
    config, _ = _write_pt_deploy_fixture(
        tmp_path,
        inference_models_dir=inference_models_dir,
        inference_project_root=inference_project_root,
    )
    station_dir = inference_models_dir / "Cable1" / "A" / "yolo"
    station_dir.mkdir(parents=True)
    station_config = station_dir / "config.yaml"
    station_config.write_text(
        yaml.safe_dump({"weights": "old.pt", "exposure_time": "11111"}),
        encoding="utf-8",
    )
    concurrent_config = {
        "weights": "concurrent.pt",
        "exposure_time": "99999",
        "gain": "9.9",
    }

    def fake_acceptance_gate(**kwargs):
        assert (station_dir / ".deploy.lock").is_dir()
        station_config.write_text(
            yaml.safe_dump(concurrent_config),
            encoding="utf-8",
        )
        return {
            "passed": True,
            "candidate": {"sha256": _sha256(kwargs["candidate_weight"])},
        }, None

    monkeypatch.setattr(
        deploy_module,
        "_run_model_acceptance_gate",
        fake_acceptance_gate,
    )

    with pytest.raises(
        deploy_module.ConcurrentStationConfigChangeError,
        match="without overwriting the concurrent edit",
    ):
        run_deploy(config, SimpleNamespace())

    assert yaml.safe_load(station_config.read_text(encoding="utf-8")) == concurrent_config
    assert not list((station_dir / "weights").glob("Cable1_A_v1.0.0_*"))
    assert not list(station_dir.glob(".deploy-rollback-*"))
    assert not (station_dir / ".deploy.lock").exists()


def test_late_concurrent_station_edit_rolls_back_published_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_project_root = tmp_path / "inference"
    inference_models_dir = inference_project_root / "models"
    config, _ = _write_pt_deploy_fixture(
        tmp_path,
        inference_models_dir=inference_models_dir,
        inference_project_root=inference_project_root,
    )
    station_dir = inference_models_dir / "Cable1" / "A" / "yolo"
    station_dir.mkdir(parents=True)
    station_config = station_dir / "config.yaml"
    deployment_manifest = station_dir / "deployment_manifest.yaml"
    station_config.write_text(
        "weights: old.pt\nexposure_time: '11111'\n",
        encoding="utf-8",
    )
    previous_manifest = b"schema_version: 2\ndeployed_version: 0.9.0\n"
    deployment_manifest.write_bytes(previous_manifest)
    concurrent_bytes = b"weights: concurrent.pt\nexposure_time: '99999'\n"
    real_write_yaml_atomic = deploy_module._write_yaml_atomic

    def edit_after_manifest(path: Path, payload: dict) -> None:
        real_write_yaml_atomic(path, payload)
        if path.resolve() == deployment_manifest.resolve():
            station_config.write_bytes(concurrent_bytes)

    monkeypatch.setattr(
        deploy_module,
        "_write_yaml_atomic",
        edit_after_manifest,
    )

    with pytest.raises(deploy_module.ConcurrentStationConfigChangeError):
        run_deploy(config, SimpleNamespace())

    assert station_config.read_bytes() == concurrent_bytes
    assert deployment_manifest.read_bytes() == previous_manifest
    assert not list((station_dir / "weights").glob("Cable1_A_v1.0.0_*"))
    assert not list(station_dir.glob(".deploy-rollback-*"))
    assert not (station_dir / ".deploy.lock").exists()


def test_late_config_write_failure_rolls_back_manifest_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_project_root = tmp_path / "inference"
    inference_models_dir = inference_project_root / "models"
    config, _ = _write_pt_deploy_fixture(
        tmp_path,
        inference_models_dir=inference_models_dir,
        inference_project_root=inference_project_root,
    )
    station_dir = inference_models_dir / "Cable1" / "A" / "yolo"
    station_dir.mkdir(parents=True)
    station_config = station_dir / "config.yaml"
    deployment_manifest = station_dir / "deployment_manifest.yaml"
    previous_config = b"weights: old.pt\nexposure_time: '54321'\n"
    previous_manifest = b"schema_version: 2\ndeployed_version: 0.9.0\n"
    station_config.write_bytes(previous_config)
    deployment_manifest.write_bytes(previous_manifest)
    real_write_yaml_atomic = deploy_module._write_yaml_atomic
    write_order: list[Path] = []

    def fail_active_config(path: Path, payload: dict) -> None:
        write_order.append(path.resolve())
        if path.resolve() == station_config.resolve():
            assert deployment_manifest.resolve() in write_order
            raise PermissionError("simulated late config failure")
        real_write_yaml_atomic(path, payload)

    monkeypatch.setattr(
        deploy_module,
        "_write_yaml_atomic",
        fail_active_config,
    )

    with pytest.raises(PermissionError, match="simulated late config failure"):
        run_deploy(config, SimpleNamespace())

    assert station_config.read_bytes() == previous_config
    assert deployment_manifest.read_bytes() == previous_manifest
    assert not any(
        path.is_file()
        for path in (station_dir / "weights").glob("Cable1_A_v1.0.0_*")
    )
    assert not any((station_dir / "versions").glob("*.config.yaml"))
    assert not list(station_dir.glob(".deploy-rollback-*"))
    assert not (station_dir / ".deploy.lock").exists()


def test_deploy_preserves_primary_error_and_evidence_when_rollback_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_project_root = tmp_path / "inference"
    inference_models_dir = inference_project_root / "models"
    config, _ = _write_pt_deploy_fixture(
        tmp_path,
        inference_models_dir=inference_models_dir,
        inference_project_root=inference_project_root,
    )
    station_dir = inference_models_dir / "Cable1" / "A" / "yolo"
    station_dir.mkdir(parents=True)
    station_config = station_dir / "config.yaml"
    deployment_manifest = station_dir / "deployment_manifest.yaml"
    previous_config = b"weights: old.pt\nexposure_time: '54321'\n"
    previous_manifest = b"schema_version: 2\ndeployed_version: 0.9.0\n"
    station_config.write_bytes(previous_config)
    deployment_manifest.write_bytes(previous_manifest)
    real_write_yaml_atomic = deploy_module._write_yaml_atomic
    real_atomic_copy_verified = deploy_module._atomic_copy_verified

    def fail_active_config(path: Path, payload: dict) -> None:
        if path.resolve() == station_config.resolve():
            raise PermissionError("simulated primary config failure")
        real_write_yaml_atomic(path, payload)

    def fail_backup_restore(source: Path, destination: Path) -> str:
        if source.parent.name.startswith(".deploy-rollback-"):
            raise PermissionError("simulated rollback restore failure")
        return real_atomic_copy_verified(source, destination)

    monkeypatch.setattr(deploy_module, "_write_yaml_atomic", fail_active_config)
    monkeypatch.setattr(
        deploy_module,
        "_atomic_copy_verified",
        fail_backup_restore,
    )

    with pytest.raises(
        PermissionError,
        match="simulated primary config failure",
    ) as exc_info:
        run_deploy(config, SimpleNamespace())

    notes = getattr(exc_info.value, "__notes__", [])
    assert any(
        "recovery evidence was retained at" in note
        and "simulated rollback restore failure" in note
        for note in notes
    )
    backup_roots = list(station_dir.glob(".deploy-rollback-*"))
    assert len(backup_roots) == 1
    retained_payloads = {
        path.read_bytes() for path in backup_roots[0].iterdir() if path.is_file()
    }
    assert previous_config in retained_payloads
    assert previous_manifest in retained_payloads
    assert not (station_dir / ".deploy.lock").exists()


def test_deploy_commit_cleanup_failure_does_not_report_published_release_as_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    inference_project_root = tmp_path / "inference"
    inference_models_dir = inference_project_root / "models"
    config, _ = _write_pt_deploy_fixture(
        tmp_path,
        inference_models_dir=inference_models_dir,
        inference_project_root=inference_project_root,
    )
    station_dir = inference_models_dir / "Cable1" / "A" / "yolo"
    real_rmtree = deploy_module.shutil.rmtree

    def deny_rollback_evidence_cleanup(path, *args, **kwargs):
        candidate = Path(path)
        if candidate.name.startswith(".deploy-rollback-"):
            raise PermissionError("simulated cleanup denial")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(
        deploy_module.shutil,
        "rmtree",
        deny_rollback_evidence_cleanup,
    )
    caplog.set_level(logging.WARNING, logger=deploy_module.__name__)

    run_deploy(config, SimpleNamespace())

    active_config = yaml.safe_load(
        (station_dir / "config.yaml").read_text(encoding="utf-8")
    )
    assert active_config["weights"].endswith(".pt")
    assert (station_dir / "deployment_manifest.yaml").is_file()
    assert len(list(station_dir.glob(".deploy-rollback-*"))) == 1
    assert not (station_dir / ".deploy.lock").exists()
    assert "rollback evidence cleanup was deferred" in caplog.text
    assert "simulated cleanup denial" in caplog.text


def test_deploy_lock_cleanup_failure_does_not_report_published_release_as_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    inference_project_root = tmp_path / "inference"
    inference_models_dir = inference_project_root / "models"
    config, _ = _write_pt_deploy_fixture(
        tmp_path,
        inference_models_dir=inference_models_dir,
        inference_project_root=inference_project_root,
    )
    station_dir = inference_models_dir / "Cable1" / "A" / "yolo"
    real_rmdir = Path.rmdir

    def deny_lock_cleanup(path):
        if path.name == ".deploy.lock":
            raise PermissionError("simulated lock cleanup denial")
        return real_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", deny_lock_cleanup)
    caplog.set_level(logging.WARNING, logger=deploy_module.__name__)

    run_deploy(config, SimpleNamespace())

    assert (station_dir / "config.yaml").is_file()
    assert (station_dir / "deployment_manifest.yaml").is_file()
    assert (station_dir / ".deploy.lock").is_dir()
    assert "Deployment lock cleanup was deferred" in caplog.text
    assert "simulated lock cleanup denial" in caplog.text


def test_deploy_lock_cleanup_failure_does_not_mask_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    inference_project_root = tmp_path / "inference"
    inference_models_dir = inference_project_root / "models"
    config, _ = _write_pt_deploy_fixture(
        tmp_path,
        inference_models_dir=inference_models_dir,
        inference_project_root=inference_project_root,
    )
    station_dir = inference_models_dir / "Cable1" / "A" / "yolo"
    active_config = station_dir / "config.yaml"
    real_write_yaml_atomic = deploy_module._write_yaml_atomic
    real_rmdir = Path.rmdir

    def fail_active_config(path: Path, payload: dict) -> None:
        if path.resolve() == active_config.resolve():
            raise PermissionError("simulated primary publication failure")
        real_write_yaml_atomic(path, payload)

    def deny_lock_cleanup(path):
        if path.name == ".deploy.lock":
            raise PermissionError("simulated lock cleanup denial")
        return real_rmdir(path)

    monkeypatch.setattr(deploy_module, "_write_yaml_atomic", fail_active_config)
    monkeypatch.setattr(Path, "rmdir", deny_lock_cleanup)
    caplog.set_level(logging.WARNING, logger=deploy_module.__name__)

    with pytest.raises(
        PermissionError,
        match="simulated primary publication failure",
    ):
        run_deploy(config, SimpleNamespace())

    assert not active_config.exists()
    assert (station_dir / ".deploy.lock").is_dir()
    assert "Deployment lock cleanup was deferred" in caplog.text
    assert "simulated lock cleanup denial" in caplog.text


def test_rollback_evidence_cleanup_failure_does_not_mask_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    inference_project_root = tmp_path / "inference"
    inference_models_dir = inference_project_root / "models"
    config, _ = _write_pt_deploy_fixture(
        tmp_path,
        inference_models_dir=inference_models_dir,
        inference_project_root=inference_project_root,
    )
    station_dir = inference_models_dir / "Cable1" / "A" / "yolo"
    active_config = station_dir / "config.yaml"
    real_write_yaml_atomic = deploy_module._write_yaml_atomic
    real_rmtree = deploy_module.shutil.rmtree

    def fail_active_config(path: Path, payload: dict) -> None:
        if path.resolve() == active_config.resolve():
            raise PermissionError("simulated primary publication failure")
        real_write_yaml_atomic(path, payload)

    def deny_rollback_evidence_cleanup(path, *args, **kwargs):
        candidate = Path(path)
        if candidate.name.startswith(".deploy-rollback-"):
            raise PermissionError("simulated rollback cleanup denial")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(deploy_module, "_write_yaml_atomic", fail_active_config)
    monkeypatch.setattr(
        deploy_module.shutil,
        "rmtree",
        deny_rollback_evidence_cleanup,
    )
    caplog.set_level(logging.WARNING, logger=deploy_module.__name__)

    with pytest.raises(
        PermissionError,
        match="simulated primary publication failure",
    ):
        run_deploy(config, SimpleNamespace())

    assert not active_config.exists()
    assert len(list(station_dir.glob(".deploy-rollback-*"))) == 1
    assert "rollback completed; evidence cleanup was deferred" in caplog.text
    assert "simulated rollback cleanup denial" in caplog.text
