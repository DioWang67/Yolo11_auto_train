from __future__ import annotations

import json

import pytest

from picture_tool.position.position_gate import (
    PositionGateError,
    PositionGatePolicy,
    evaluate_position_gate,
    load_json_mapping,
    position_metrics_from_report,
    write_position_gate_report,
)


def _report(
    statuses: list[tuple[str, str]],
    *,
    hashes: list[str] | None = None,
) -> dict:
    records = []
    for index, (expected, actual) in enumerate(statuses):
        records.append(
            {
                "expected_status": expected,
                "validation": {"status": actual},
                "image_sha256": (
                    hashes[index] if hashes and index < len(hashes) else ""
                ),
            }
        )
    return {"summary": {}, "records": records}


def test_position_gate_passes_absolute_ok_and_ng_thresholds() -> None:
    report = _report(
        [
            ("PASS", "PASS"),
            ("PASS", "PASS"),
            ("FAIL", "FAIL"),
            ("FAIL", "FAIL"),
        ]
    )

    decision = evaluate_position_gate(
        report,
        policy=PositionGatePolicy(
            min_ok_samples=2,
            max_ok_false_reject_rate=0.0,
            min_ng_samples=2,
            min_ng_recall=1.0,
            require_disjoint_calibration=False,
        ),
    )

    assert decision.passed is True
    assert decision.metrics.ok_false_reject_rate == 0.0
    assert decision.metrics.ng_recall == 1.0


def test_position_gate_blocks_false_reject_and_missing_ng_evidence() -> None:
    report = _report([("PASS", "FAIL"), ("PASS", "PASS")])

    decision = evaluate_position_gate(
        report,
        policy=PositionGatePolicy(
            min_ok_samples=2,
            max_ok_false_reject_rate=0.0,
            min_ng_samples=1,
            min_ng_recall=0.9,
            require_disjoint_calibration=False,
        ),
    )

    assert decision.passed is False
    assert any("ok_false_reject_rate_exceeded" in item for item in decision.failures)
    assert any("insufficient_ng_samples" in item for item in decision.failures)
    assert "ng_recall_missing" in decision.failures


def test_position_gate_compares_candidate_with_same_dataset_baseline() -> None:
    baseline = _report(
        [
            ("PASS", "PASS"),
            ("PASS", "PASS"),
            ("FAIL", "FAIL"),
        ]
    )
    candidate = _report(
        [
            ("PASS", "PASS"),
            ("PASS", "FAIL"),
            ("FAIL", "PASS"),
        ]
    )

    decision = evaluate_position_gate(
        candidate,
        baseline_report=baseline,
        policy=PositionGatePolicy(
            max_ok_false_reject_rate=1.0,
            max_ok_false_reject_regression=0.1,
            max_ng_recall_regression=0.1,
            require_disjoint_calibration=False,
        ),
    )

    assert decision.passed is False
    assert any("ok_false_reject_regression" in item for item in decision.failures)
    assert any("ng_recall_regression" in item for item in decision.failures)


def test_position_gate_blocks_calibration_golden_hash_overlap() -> None:
    shared_hash = "a" * 64
    report = _report([("PASS", "PASS")], hashes=[shared_hash])
    calibration = {"samples": [{"image_sha256": shared_hash}]}

    decision = evaluate_position_gate(
        report,
        calibration_manifest=calibration,
        policy=PositionGatePolicy(require_disjoint_calibration=True),
    )

    assert decision.passed is False
    assert decision.calibration_overlap_count == 1
    assert any("calibration_golden_overlap" in item for item in decision.failures)


def test_position_gate_requires_calibration_manifest_for_disjoint_policy() -> None:
    decision = evaluate_position_gate(
        _report([("PASS", "PASS")]),
        policy=PositionGatePolicy(require_disjoint_calibration=True),
    )

    assert decision.passed is False
    assert "calibration_manifest_missing" in decision.failures


def test_position_gate_rejects_malformed_calibration_manifest() -> None:
    with pytest.raises(PositionGateError, match="no sample records"):
        evaluate_position_gate(
            _report([("PASS", "PASS")]),
            calibration_manifest={"samples": []},
            policy=PositionGatePolicy(require_disjoint_calibration=True),
        )


def test_position_gate_blocks_baseline_from_different_golden_set() -> None:
    decision = evaluate_position_gate(
        _report([("PASS", "PASS")], hashes=["a" * 64]),
        baseline_report=_report([("PASS", "PASS")], hashes=["b" * 64]),
        policy=PositionGatePolicy(
            require_baseline=True,
            require_disjoint_calibration=False,
        ),
    )

    assert decision.passed is False
    assert any("baseline_golden_set_mismatch" in item for item in decision.failures)


def test_position_gate_rejects_duplicate_schema_v2_golden_images() -> None:
    report = _report(
        [("PASS", "PASS"), ("PASS", "PASS")],
        hashes=["a" * 64, "a" * 64],
    )
    report["schema_version"] = 2

    with pytest.raises(PositionGateError, match="duplicate image content"):
        evaluate_position_gate(
            report,
            policy=PositionGatePolicy(require_disjoint_calibration=False),
        )


def test_position_metrics_accepts_materialized_summary() -> None:
    metrics = position_metrics_from_report(
        {
            "summary": {
                "metrics": {
                    "ok_samples": 10,
                    "ok_false_rejects": 1,
                    "ok_false_reject_rate": 0.1,
                    "ng_samples": 5,
                    "ng_detected": 4,
                    "ng_recall": 0.8,
                }
            }
        }
    )

    assert metrics.ok_samples == 10
    assert metrics.ng_recall == pytest.approx(0.8)


@pytest.mark.parametrize(
    "report",
    [
        {},
        {"summary": {}, "records": "bad"},
        {
            "summary": {},
            "records": [{"expected_status": "UNKNOWN", "validation": {"status": "PASS"}}],
        },
        {
            "summary": {},
            "records": [{"expected_status": "PASS", "validation": {"status": "UNKNOWN"}}],
        },
    ],
)
def test_position_metrics_rejects_malformed_reports(report: dict) -> None:
    with pytest.raises(PositionGateError):
        position_metrics_from_report(report)


def test_position_metrics_rejects_inconsistent_materialized_summary() -> None:
    report = {
        "summary": {
            "metrics": {
                "ok_samples": 10,
                "ok_false_rejects": 1,
                "ok_false_reject_rate": 0.2,
                "ng_samples": 2,
                "ng_detected": 2,
                "ng_recall": 1.0,
            }
        }
    }

    with pytest.raises(PositionGateError, match="does not match"):
        position_metrics_from_report(report)


def test_position_gate_report_is_atomic_and_reloadable(tmp_path) -> None:
    candidate_path = tmp_path / "position_validation.json"
    candidate_path.write_text("{}", encoding="utf-8")
    decision = evaluate_position_gate(
        _report([("PASS", "PASS")]),
        policy=PositionGatePolicy(require_disjoint_calibration=False),
    )
    output = tmp_path / "position_gate.json"

    write_position_gate_report(
        output,
        decision,
        product="Cable1",
        area="A",
        candidate_report_path=candidate_path,
        baseline_report_path=None,
        calibration_manifest_path=None,
    )

    loaded = load_json_mapping(output, "position gate")
    assert loaded["passed"] is True
    assert loaded["product"] == "Cable1"
    assert len(loaded["candidate_report_sha256"]) == 64
    assert loaded["baseline_report_sha256"] is None
    assert loaded["calibration_manifest_sha256"] is None
    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1


def test_position_gate_policy_rejects_invalid_rate() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        PositionGatePolicy(min_ng_recall=1.1)
