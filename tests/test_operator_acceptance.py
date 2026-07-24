from __future__ import annotations

import pytest
import yaml

from picture_tool.operator_acceptance import (
    OperatorAcceptanceError,
    load_operator_acceptance_summary,
)


def _write_manifest(tmp_path, **overrides):
    station = tmp_path / "Cable1" / "A" / "yolo"
    station.mkdir(parents=True)
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
    text = summary.to_operator_text()
    assert "Evaluation Gate：通過" in text
    assert "這次沒有重新校正" in text
    assert "不等於產線驗收完成" in text


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"evaluation_gate_passed": False}, "no passing evaluation gate"),
        ({"evaluation_metrics": {"precision": 2.0}}, "precision"),
        ({"deployed_version": ""}, "version is missing"),
    ],
)
def test_acceptance_summary_rejects_incomplete_or_unsafe_evidence(
    tmp_path, overrides, message
):
    _write_manifest(tmp_path, **overrides)

    with pytest.raises(OperatorAcceptanceError, match=message):
        load_operator_acceptance_summary(tmp_path, product="Cable1", area="A")
