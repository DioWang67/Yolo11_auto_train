from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path

import pytest
import yaml

from picture_tool.gui.operator_handoff import (
    OperatorHandoffError,
    _resolve_model_acceptance_gate,
)
from picture_tool.tasks.deploy import _run_model_acceptance_gate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_deploy_acceptance_runner_publishes_candidate_scoped_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_root = tmp_path / "inference"
    inference_models = inference_root / "models"
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

    def fake_run(command, **kwargs):
        report_path = Path(command[command.index("--report") + 1])
        candidate_config = Path(
            command[command.index("--candidate-config") + 1]
        )
        report_path.write_text(
            json.dumps(
                {
                    "passed": True,
                    "failures": [],
                    "candidate": {
                        "sha256": _sha256(candidate_weight),
                        "runtime_config_sha256": _sha256(candidate_config),
                    },
                    "metrics": {"fp": 35, "fn": 0},
                    "baseline_metrics": {"fp": 35, "fn": 0},
                    "dataset": {
                        "snapshot_manifest_sha256": _sha256(snapshot)
                    },
                }
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
        inference_models_dir=inference_models,
        product="Cable1",
        area="A",
        candidate_config=candidate_config,
        candidate_weight=candidate_weight,
        color_model=None,
        logger=logging.getLogger(__name__),
    )

    written_config = yaml.safe_load(
        (
            run_dir
            / "acceptance_candidate"
            / "models"
            / "Cable1"
            / "A"
            / "yolo"
            / "config.yaml"
        ).read_text(encoding="utf-8")
    )
    assert report["passed"] is True
    assert report_path == run_dir / "model_acceptance_gate.json"
    assert written_config["weights"] == str(candidate_weight)
