import csv
import json

import pytest
import yaml

from picture_tool.color.threshold_calibration import (
    CalibrationPolicy,
    apply_recommendations,
    build_recommendation,
    load_feedback,
    main,
)


FIELDS = [
    "product",
    "area",
    "model_type",
    "checker_type",
    "threshold_key",
    "failure_kind",
    "sample_id",
    "item_index",
    "diff",
    "threshold",
    "actual_is_ok",
]


def _write_feedback(path, *, ok_count=6, ng_count=6):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(ok_count):
        rows.append(
            {
                "product": "Cable1",
                "area": "A",
                "model_type": "yolo",
                "checker_type": "stats",
                "threshold_key": "red",
                "sample_id": f"ok-{index}",
                "item_index": "0",
                "diff": str(0.61 + index * 0.01),
                "threshold": "0.6",
                "actual_is_ok": "1",
            }
        )
    for index in range(ng_count):
        rows.append(
            {
                "product": "Cable1",
                "area": "A",
                "model_type": "yolo",
                "checker_type": "stats",
                "threshold_key": "red",
                "sample_id": f"ng-{index}",
                "item_index": "0",
                "diff": str(0.80 + index * 0.01),
                "threshold": "0.6",
                "actual_is_ok": "0",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_recommendation_stays_in_shadow_mode_and_improves_false_rejects(tmp_path):
    feedback = tmp_path / "feedback.csv"
    report_path = tmp_path / "report.json"
    _write_feedback(feedback)
    policy = CalibrationPolicy(minimum_total=10, minimum_ok=5, minimum_ng=5)

    report = build_recommendation(
        [feedback], policy=policy, output_path=report_path
    )

    recommendation = report["recommendations"][0]
    assert report["mode"] == "shadow"
    assert recommendation["status"] == "ready"
    assert recommendation["current_metrics"]["false_reject"] == 6
    assert recommendation["suggested_metrics"]["false_reject"] == 0
    assert recommendation["suggested_metrics"]["false_accept"] == 0
    assert recommendation["suggested_public_threshold"] == pytest.approx(0.66)
    assert recommendation["suggested_config_value"] == pytest.approx(0.34)
    assert json.loads(report_path.read_text(encoding="utf-8"))["report_id"]


def test_recommendation_requires_balanced_minimum_evidence(tmp_path):
    feedback = tmp_path / "feedback.csv"
    _write_feedback(feedback, ok_count=2, ng_count=6)

    report = build_recommendation(
        [feedback],
        policy=CalibrationPolicy(minimum_total=8, minimum_ok=5, minimum_ng=5),
    )

    recommendation = report["recommendations"][0]
    assert recommendation["status"] == "insufficient_data"
    assert "actual-OK" in " ".join(recommendation["reasons"])


def test_named_approval_atomically_updates_override_and_writes_audit(tmp_path):
    feedback = tmp_path / "feedback.csv"
    report_path = tmp_path / "report.json"
    models_root = tmp_path / "models"
    config_path = models_root / "Cable1" / "A" / "yolo" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "color_checker_type": "stats",
                "color_score_threshold": 0.4,
                "enable_color_check": True,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_feedback(feedback)
    build_recommendation(
        [feedback],
        policy=CalibrationPolicy(minimum_total=10, minimum_ok=5, minimum_ng=5),
        output_path=report_path,
    )

    receipt = apply_recommendations(
        report_path, models_root=models_root, approver="OP-王小明"
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["color_threshold_overrides"]["red"] == pytest.approx(0.34)
    assert config["color_score_threshold"] == 0.4
    assert receipt["approved_by"] == "OP-王小明"
    assert list((config_path.parent / "color_threshold_backups").glob("*_config.yaml"))
    history = json.loads(
        (config_path.parent / "color_threshold_history.json").read_text(
            encoding="utf-8"
        )
    )
    assert history[-1]["report_id"] == receipt["report_id"]


def test_approval_fails_closed_when_config_changed_after_report(tmp_path):
    feedback = tmp_path / "feedback.csv"
    report_path = tmp_path / "report.json"
    models_root = tmp_path / "models"
    config_path = models_root / "Cable1" / "A" / "yolo" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("color_score_threshold: 0.5\n", encoding="utf-8")
    _write_feedback(feedback)
    build_recommendation(
        [feedback],
        policy=CalibrationPolicy(minimum_total=10, minimum_ok=5, minimum_ng=5),
        output_path=report_path,
    )
    original = config_path.read_bytes()

    with pytest.raises(ValueError, match="changed after the report"):
        apply_recommendations(
            report_path, models_root=models_root, approver="OP-1"
        )

    assert config_path.read_bytes() == original


@pytest.mark.parametrize(
    "values",
    [
        {"minimum_total": 0},
        {"false_accept_cost": 0},
        {"maximum_false_accept_rate": 1.1},
    ],
)
def test_policy_rejects_unsafe_values(values):
    with pytest.raises(ValueError):
        CalibrationPolicy(**values)


def test_feedback_directory_loading_deduplicates_last_review(tmp_path):
    first = tmp_path / "one" / "feedback.csv"
    second = tmp_path / "two" / "feedback.csv"
    _write_feedback(first, ok_count=1, ng_count=0)
    _write_feedback(second, ok_count=1, ng_count=0)
    with second.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["actual_is_ok"] = "0"
    with second.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    samples = load_feedback([tmp_path])

    assert len(samples) == 1
    assert samples[0].actual_is_ok is False


def test_overlapping_distributions_are_blocked_by_false_accept_policy(tmp_path):
    feedback = tmp_path / "feedback.csv"
    _write_feedback(feedback, ok_count=5, ng_count=5)
    with feedback.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["diff"] = "0.5" if row["actual_is_ok"] == "0" else "0.8"
    with feedback.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = build_recommendation(
        [feedback],
        policy=CalibrationPolicy(minimum_total=10, minimum_ok=5, minimum_ng=5),
    )

    assert report["recommendations"][0]["status"] == "blocked_by_safety_policy"


def test_optimal_current_threshold_reports_no_change(tmp_path):
    feedback = tmp_path / "feedback.csv"
    _write_feedback(feedback, ok_count=5, ng_count=5)
    with feedback.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["diff"] = "0.5" if row["actual_is_ok"] == "1" else "0.8"
    with feedback.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = build_recommendation(
        [feedback],
        policy=CalibrationPolicy(minimum_total=10, minimum_ok=5, minimum_ng=5),
    )

    assert report["recommendations"][0]["status"] == "no_change"


def test_apply_requires_name_and_untampered_report(tmp_path):
    feedback = tmp_path / "feedback.csv"
    report_path = tmp_path / "report.json"
    _write_feedback(feedback)
    build_recommendation(
        [feedback],
        policy=CalibrationPolicy(minimum_total=10, minimum_ok=5, minimum_ng=5),
        output_path=report_path,
    )

    with pytest.raises(ValueError, match="approver"):
        apply_recommendations(report_path, models_root=tmp_path, approver=" ")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["generated_at"] = "tampered"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        apply_recommendations(report_path, models_root=tmp_path, approver="OP-1")


def test_cli_builds_shadow_report(tmp_path, capsys):
    feedback = tmp_path / "feedback.csv"
    report_path = tmp_path / "report.json"
    _write_feedback(feedback)

    exit_code = main(
        [
            "recommend",
            str(feedback),
            "--output",
            str(report_path),
            "--minimum-total",
            "10",
            "--minimum-ok",
            "5",
            "--minimum-ng",
            "5",
        ]
    )

    assert exit_code == 0
    assert report_path.is_file()
    assert '"mode": "shadow"' in capsys.readouterr().out


def test_rule_failures_are_reported_but_never_used_to_tune_threshold(tmp_path):
    feedback = tmp_path / "feedback.csv"
    _write_feedback(feedback, ok_count=5, ng_count=5)
    with feedback.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["failure_kind"] = "rule"
    with feedback.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = build_recommendation([feedback])

    assert report["feedback_item_count"] == 10
    assert report["rule_feedback_item_count"] == 10
    assert report["threshold_feedback_item_count"] == 0
    assert report["recommendations"] == []


def test_invalid_feedback_kind_is_rejected_with_row_context(tmp_path):
    feedback = tmp_path / "feedback.csv"
    _write_feedback(feedback, ok_count=1, ng_count=0)
    with feedback.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["failure_kind"] = "unknown"
    with feedback.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="feedback.csv:2.*failure_kind"):
        load_feedback([feedback])


def test_apply_rejects_report_without_ready_recommendation(tmp_path):
    feedback = tmp_path / "feedback.csv"
    report_path = tmp_path / "report.json"
    _write_feedback(feedback, ok_count=1, ng_count=1)
    build_recommendation([feedback], output_path=report_path)

    with pytest.raises(ValueError, match="no ready recommendation"):
        apply_recommendations(report_path, models_root=tmp_path, approver="OP-1")


def test_audit_write_failure_rolls_config_back(tmp_path):
    feedback = tmp_path / "feedback.csv"
    report_path = tmp_path / "report.json"
    models_root = tmp_path / "models"
    config_path = models_root / "Cable1" / "A" / "yolo" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "color_checker_type: stats\ncolor_score_threshold: 0.4\n",
        encoding="utf-8",
    )
    history_path = config_path.parent / "color_threshold_history.json"
    history_path.write_text("{}", encoding="utf-8")
    _write_feedback(feedback)
    build_recommendation(
        [feedback],
        policy=CalibrationPolicy(minimum_total=10, minimum_ok=5, minimum_ng=5),
        output_path=report_path,
    )
    original = config_path.read_bytes()

    with pytest.raises(ValueError, match="history must be a JSON array"):
        apply_recommendations(
            report_path, models_root=models_root, approver="OP-1"
        )

    assert config_path.read_bytes() == original
