from __future__ import annotations

import hashlib

import pytest
import yaml

from picture_tool.operator_acceptance import (
    OperatorAcceptanceError,
    load_operator_acceptance_summary,
)


def _write_manifest(tmp_path, **overrides):
    station = tmp_path / "Cable1" / "A" / "yolo"
    station.mkdir(parents=True)
    versions = station / "versions"
    versions.mkdir()
    gate_evidence = versions / "model.position_gate.json"
    validation_evidence = versions / "model.position_validation.json"
    gate_evidence.write_text('{"passed":true}', encoding="utf-8")
    validation_evidence.write_text('{"summary":{}}', encoding="utf-8")
    payload = {
        "deployed_version": "1.0.3",
        "evaluation_gate_passed": True,
        "evaluation_metrics": {
            "precision": 0.91,
            "recall": 0.92,
            "map50": 0.93,
            "map50_95": 0.81,
        },
        "color_model_source": "existing_station",
        "position_runtime_enabled": False,
        "position_gate_required": True,
        "position_gate_passed": True,
        "position_gate_report": "versions/model.position_gate.json",
        "position_gate_sha256": hashlib.sha256(
            gate_evidence.read_bytes()
        ).hexdigest(),
        "position_validation_report": "versions/model.position_validation.json",
        "position_validation_sha256": hashlib.sha256(
            validation_evidence.read_bytes()
        ).hexdigest(),
        "position_metrics": {
            "ok_false_reject_rate": 0.0,
            "ng_recall": 0.95,
        },
    }
    payload.update(overrides)
    (station / "deployment_manifest.yaml").write_text(
        yaml.safe_dump(payload), encoding="utf-8"
    )


def test_acceptance_summary_exposes_gate_metrics_and_color_source(tmp_path):
    _write_manifest(tmp_path)

    summary = load_operator_acceptance_summary(
        tmp_path, product="Cable1", area="A"
    )

    assert summary.gate_passed is True
    assert summary.deployed_version == "1.0.3"
    assert summary.map50_95 == pytest.approx(0.81)
    assert summary.position_gate_passed is True
    assert summary.position_runtime_enabled is False
    text = summary.to_operator_text()
    assert "Evaluation Gate：通過" in text
    assert "Position Gate：通過" in text
    assert "維持停用" in text
    assert "這次沒有重新校正" in text
    assert "不等於產線驗收完成" in text


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"evaluation_gate_passed": False}, "no passing evaluation gate"),
        ({"evaluation_metrics": {"precision": 2.0}}, "precision"),
        ({"deployed_version": ""}, "version is missing"),
        ({"position_gate_passed": False}, "position gate evidence"),
        ({"position_metrics": {}}, "false-reject evidence"),
    ],
)
def test_acceptance_summary_rejects_incomplete_or_unsafe_evidence(
    tmp_path, overrides, message
):
    _write_manifest(tmp_path, **overrides)

    with pytest.raises(OperatorAcceptanceError, match=message):
        load_operator_acceptance_summary(tmp_path, product="Cable1", area="A")


def test_acceptance_summary_rejects_tampered_position_evidence(tmp_path):
    _write_manifest(tmp_path)
    evidence = (
        tmp_path
        / "Cable1"
        / "A"
        / "yolo"
        / "versions"
        / "model.position_validation.json"
    )
    evidence.write_text('{"tampered":true}', encoding="utf-8")

    with pytest.raises(OperatorAcceptanceError, match="checksum does not match"):
        load_operator_acceptance_summary(tmp_path, product="Cable1", area="A")


def test_acceptance_summary_exposes_fixed_set_metrics(tmp_path):
    _write_manifest(tmp_path)
    station = tmp_path / "Cable1" / "A" / "yolo"
    evidence = station / "versions" / "model.acceptance.json"
    evidence.write_text('{"passed":true}', encoding="utf-8")
    manifest_path = station / "deployment_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "model_acceptance_gate_passed": True,
            "model_acceptance_report": "versions/model.acceptance.json",
            "model_acceptance_report_sha256": hashlib.sha256(
                evidence.read_bytes()
            ).hexdigest(),
            "model_acceptance_metrics": {
                "confirmed": 250,
                "fp": 35,
                "fn": 0,
                "accuracy": 0.86,
                "overkill_rate": 35 / 173,
            },
        }
    )
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    summary = load_operator_acceptance_summary(
        tmp_path, product="Cable1", area="A"
    )

    assert summary.model_acceptance_gate_passed is True
    assert summary.model_acceptance_confirmed == 250
    assert summary.model_acceptance_accuracy == pytest.approx(0.86)
    assert summary.model_acceptance_false_positives == 35
    assert summary.model_acceptance_false_negatives == 0
    assert "固定驗收集：通過" in summary.to_operator_text()
