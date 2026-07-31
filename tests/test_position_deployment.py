from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import yaml

from picture_tool.tasks.deploy import run_deploy


def _prepare_position_deployment(tmp_path, *, position_activation: str = "preserve"):
    run_dir = tmp_path / "runs" / "train"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"candidate-pt")
    position_config = {
        "Cable1": {
            "A": {
                "enabled": True,
                "mode": "center",
                "imgsz": 640,
                "tolerance": 8,
                "tolerance_unit": "pixel",
                "expected_boxes": {
                    "Red": {"x1": 10, "y1": 20, "x2": 30, "y2": 40}
                },
            }
        }
    }
    (run_dir / "detection_config.yaml").write_text(
        yaml.safe_dump(
            {
                "weights": "best.pt",
                "enable_color_check": False,
                "position_config": position_config,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report = run_dir / "position_validation.json"
    report.write_text(
        json.dumps(
            {
                "summary": {
                    "metrics": {
                        "ok_samples": 10,
                        "ok_false_rejects": 0,
                        "ok_false_reject_rate": 0.0,
                        "ng_samples": 0,
                        "ng_detected": 0,
                        "ng_recall": None,
                    }
                },
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    gate = run_dir / "position_gate.json"
    report_sha256 = hashlib.sha256(report.read_bytes()).hexdigest()
    gate.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "product": "Cable1",
                "area": "A",
                "passed": True,
                "failures": [],
                "metrics": {
                    "ok_samples": 10,
                    "ok_false_rejects": 0,
                    "ok_false_reject_rate": 0.0,
                    "ng_samples": 0,
                    "ng_detected": 0,
                    "ng_recall": None,
                },
                "baseline_metrics": None,
                "candidate_report": str(report.resolve()),
                "candidate_report_sha256": report_sha256,
            }
        ),
        encoding="utf-8",
    )

    models = tmp_path / "models"
    station = models / "Cable1" / "A" / "yolo"
    station.mkdir(parents=True)
    incumbent_position = json.loads(json.dumps(position_config))
    incumbent_position["Cable1"]["A"]["enabled"] = False
    (station / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "weights": "old.pt",
                "position_config": incumbent_position,
                "exposure_time": "1000",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = {
        "yolo_training": {
            "project": str(tmp_path / "runs"),
            "name": "train",
            "position_validation": {
                "enabled": True,
                "gate": {"enabled": True},
            },
            "deploy": {
                "enabled": True,
                "product": "Cable1",
                "area": "A",
                "inference_models_dir": str(models),
                "version": "1.2.3",
                "position_activation": position_activation,
            },
        }
    }
    return config, run_dir, station


def test_deploy_preserves_position_activation_and_versions_gate_evidence(
    tmp_path,
) -> None:
    config, _run_dir, station = _prepare_position_deployment(tmp_path)

    run_deploy(config, SimpleNamespace())

    deployed = yaml.safe_load(
        (station / "config.yaml").read_text(encoding="utf-8")
    )
    manifest = yaml.safe_load(
        (station / "deployment_manifest.yaml").read_text(encoding="utf-8")
    )
    assert deployed["position_config"]["Cable1"]["A"]["enabled"] is False
    assert deployed["exposure_time"] == "1000"
    assert manifest["position_runtime_enabled"] is False
    assert manifest["position_gate_required"] is True
    assert manifest["position_gate_passed"] is True
    assert len(manifest["position_config_sha256"]) == 64
    gate_copy = station / manifest["position_gate_report"]
    validation_copy = station / manifest["position_validation_report"]
    assert gate_copy.is_file()
    assert validation_copy.is_file()
    assert hashlib.sha256(gate_copy.read_bytes()).hexdigest() == manifest[
        "position_gate_sha256"
    ]


def test_deploy_blocks_active_position_config_without_gate(tmp_path) -> None:
    config, run_dir, _station = _prepare_position_deployment(tmp_path)
    (run_dir / "position_gate.json").unlink()

    with pytest.raises(ValueError, match="Position gate report not found"):
        run_deploy(config, SimpleNamespace())


def test_deploy_blocks_disabled_candidate_position_config_without_gate(
    tmp_path,
) -> None:
    config, run_dir, _station = _prepare_position_deployment(tmp_path)
    detection_path = run_dir / "detection_config.yaml"
    detection = yaml.safe_load(detection_path.read_text(encoding="utf-8"))
    detection["position_config"]["Cable1"]["A"]["enabled"] = False
    detection_path.write_text(
        yaml.safe_dump(detection, sort_keys=False),
        encoding="utf-8",
    )
    config["yolo_training"]["position_validation"]["gate"]["enabled"] = False
    (run_dir / "position_gate.json").unlink()

    with pytest.raises(ValueError, match="Position gate report not found"):
        run_deploy(config, SimpleNamespace())


def test_deploy_can_preserve_disabled_station_contract_without_position_goldens(
    tmp_path,
) -> None:
    config, run_dir, station = _prepare_position_deployment(tmp_path)
    station_before = yaml.safe_load(
        (station / "config.yaml").read_text(encoding="utf-8")
    )["position_config"]
    config["yolo_training"]["position_validation"]["enabled"] = False
    config["yolo_training"]["position_validation"]["gate"]["enabled"] = False
    config["yolo_training"]["deploy"][
        "position_contract_policy"
    ] = "preserve_disabled_station"
    (run_dir / "position_gate.json").unlink()

    run_deploy(config, SimpleNamespace())

    deployed = yaml.safe_load(
        (station / "config.yaml").read_text(encoding="utf-8")
    )
    manifest = yaml.safe_load(
        (station / "deployment_manifest.yaml").read_text(encoding="utf-8")
    )
    assert deployed["position_config"] == station_before
    assert manifest["position_runtime_enabled"] is False
    assert manifest["position_gate_required"] is False
    assert manifest["position_gate_passed"] is None


def test_deploy_can_explicitly_enable_position_after_gate(tmp_path) -> None:
    config, _run_dir, station = _prepare_position_deployment(
        tmp_path,
        position_activation="enable",
    )

    run_deploy(config, SimpleNamespace())

    deployed = yaml.safe_load(
        (station / "config.yaml").read_text(encoding="utf-8")
    )
    assert deployed["position_config"]["Cable1"]["A"]["enabled"] is True


def test_deploy_blocks_position_report_changed_after_gate(tmp_path) -> None:
    config, run_dir, _station = _prepare_position_deployment(tmp_path)
    (run_dir / "position_validation.json").write_text(
        '{"summary":{"metrics":{"ok_samples":0}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checksum changed after gate"):
        run_deploy(config, SimpleNamespace())


def test_deploy_rejects_preserving_deployment_owned_position_config(
    tmp_path,
) -> None:
    config, _run_dir, _station = _prepare_position_deployment(tmp_path)
    config["yolo_training"]["deploy"]["station_fields"] = [
        "exposure_time",
        "position_config",
    ]

    with pytest.raises(ValueError, match="deployment-owned fields"):
        run_deploy(config, SimpleNamespace())
