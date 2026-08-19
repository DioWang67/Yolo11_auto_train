import json

import yaml

from picture_tool.eval.yolo_evaluator import _evaluate_gate
from picture_tool.tasks.deploy import TASKS, _load_evaluation_gate


def test_evaluation_gate_blocks_metric_below_threshold(tmp_path):
    config = {
        "yolo_evaluation": {
            "gate": {
                "enabled": True,
                "require_metrics": True,
                "min_precision": 0.8,
                "min_recall": 0.8,
                "min_map50": 0.8,
                "min_map50_95": 0.4,
            }
        }
    }

    report = _evaluate_gate(
        config,
        {"precision": 0.9, "recall": 0.7, "map50": 0.9, "map50_95": 0.5},
        tmp_path,
    )

    assert report.passed is False
    assert any("recall" in failure for failure in report.failures)
    assert json.loads(report.report_path.read_text(encoding="utf-8"))["passed"] is False


def test_evaluation_gate_blocks_regression_against_deployed_model(tmp_path):
    baseline = tmp_path / "deployment_manifest.yaml"
    baseline.write_text(
        yaml.safe_dump(
            {
                "evaluation_metrics": {
                    "precision": 0.9,
                    "recall": 0.9,
                    "map50": 0.9,
                    "map50_95": 0.6,
                }
            }
        ),
        encoding="utf-8",
    )
    config = {
        "yolo_evaluation": {
            "gate": {
                "enabled": True,
                "require_metrics": True,
                "max_regression": 0.05,
                "baseline_manifest": str(baseline),
            }
        }
    }

    report = _evaluate_gate(
        config,
        {"precision": 0.8, "recall": 0.9, "map50": 0.9, "map50_95": 0.6},
        tmp_path,
    )

    assert report.passed is False
    assert any("regressed" in failure for failure in report.failures)


def test_evaluation_gate_requires_complete_incumbent_metrics(tmp_path):
    config = {
        "yolo_evaluation": {
            "gate": {
                "enabled": True,
                "require_metrics": True,
                "require_baseline": True,
            }
        }
    }
    challenger = {
        "precision": 0.9,
        "recall": 0.9,
        "map50": 0.9,
        "map50_95": 0.6,
    }

    report = _evaluate_gate(config, challenger, tmp_path, baseline_metrics={})

    assert report.passed is False
    assert any("missing incumbent baseline metrics" in item for item in report.failures)


def test_evaluation_gate_uses_same_dataset_baseline_over_manifest(tmp_path):
    config = {
        "yolo_evaluation": {
            "gate": {
                "enabled": True,
                "require_baseline": True,
                "max_regression": 0.02,
                "baseline_manifest": str(tmp_path / "missing.yaml"),
            }
        }
    }
    challenger = {
        "precision": 0.9,
        "recall": 0.88,
        "map50": 0.9,
        "map50_95": 0.6,
    }
    incumbent = {
        "precision": 0.9,
        "recall": 0.92,
        "map50": 0.9,
        "map50_95": 0.6,
    }

    report = _evaluate_gate(
        config, challenger, tmp_path, baseline_metrics=incumbent
    )

    assert report.passed is False
    assert report.baseline_metrics == incumbent
    payload = json.loads(report.report_path.read_text(encoding="utf-8"))
    assert payload["comparison_mode"] == "same_dataset"


def test_deploy_task_requires_yolo_evaluation():
    deploy_task = next(task for task in TASKS if task.name == "deploy")

    assert deploy_task.dependencies == ["yolo_evaluation"]


def test_deploy_gate_loader_accepts_passed_report(tmp_path):
    (tmp_path / "evaluation_gate.json").write_text(
        json.dumps({"passed": True, "metrics": {"recall": 0.9}}),
        encoding="utf-8",
    )

    report = _load_evaluation_gate(tmp_path, required=True)

    assert report["metrics"]["recall"] == 0.9
