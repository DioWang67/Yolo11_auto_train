from types import SimpleNamespace

import pytest
import yaml

from picture_tool.gui.constants import DEFAULT_PRESETS, TASK_OPTIONS_MAP
from picture_tool.gui.workflows import ordered_task_keys, workflow_tasks
from picture_tool.pipeline.artifacts import (
    find_anomalib_run_artifact,
    find_yolo_run_artifact,
)


def test_gui_workflow_presets_use_real_task_keys():
    """User-facing workflows should map to executable pipeline task names."""
    known_tasks = set(TASK_OPTIONS_MAP)

    for preset_name, task_keys in DEFAULT_PRESETS.items():
        assert task_keys, f"{preset_name} should not be empty"
        assert set(task_keys) <= known_tasks

    assert workflow_tasks("YOLO: train and package") == [
        "dataset_splitter",
        "yolo_train",
        "artifact_bundle",
    ]
    assert workflow_tasks("YOLO: train and deploy") == [
        "dataset_splitter",
        "yolo_train",
        "yolo_evaluation",
        "deploy",
    ]
    assert workflow_tasks("Anomalib: train and package") == [
        "anomalib_train",
        "anomalib_package",
    ]


def test_ordered_task_keys_keeps_stable_gui_order():
    tasks = ["artifact_bundle", "dataset_splitter", "yolo_train"]

    assert ordered_task_keys(tasks) == [
        "dataset_splitter",
        "yolo_train",
        "artifact_bundle",
    ]


def test_find_yolo_run_artifact_prefers_latest_matching_run(tmp_path):
    older = tmp_path / "runs" / "detect" / "train"
    newer = tmp_path / "runs" / "detect" / "train2"
    for run_dir in (older, newer):
        (run_dir / "weights").mkdir(parents=True)
    (older / "weights" / "best.pt").write_bytes(b"old")
    latest_weight = newer / "weights" / "best.pt"
    latest_weight.write_bytes(b"new")
    latest_mtime = latest_weight.stat().st_mtime + 10
    import os

    os.utime(latest_weight, (latest_mtime, latest_mtime))

    artifact = find_yolo_run_artifact(
        {"yolo_training": {"project": str(tmp_path / "runs" / "detect"), "name": "train"}}
    )

    assert artifact is not None
    assert artifact.kind == "yolo"
    assert artifact.run_dir == newer.resolve()
    assert artifact.primary_artifact == latest_weight.resolve()


def test_find_anomalib_run_artifact_returns_run_dir_not_checkpoint_parent(tmp_path):
    run_dir = tmp_path / "runs" / "anomalib" / "PCBA1" / "B" / "EfficientAd" / "latest"
    checkpoint = run_dir / "weights" / "lightning" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"ckpt")

    artifact = find_anomalib_run_artifact([tmp_path / "runs" / "anomalib"])

    assert artifact is not None
    assert artifact.kind == "anomalib"
    assert artifact.run_dir == run_dir.resolve()
    assert artifact.primary_artifact == checkpoint.resolve()


def test_artifact_bundle_fails_when_detection_config_missing(tmp_path):
    from picture_tool.tasks.bundle import run_artifact_bundle

    run_dir = tmp_path / "runs" / "detect" / "train"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"pt")

    config = {
        "yolo_training": {
            "project": str(tmp_path / "runs" / "detect"),
            "name": "train",
            "artifact_bundle": {
                "enabled": True,
                "product": "PCBA1",
                "area": "A",
                "base_dir": str(tmp_path),
            },
        }
    }

    with pytest.raises(FileNotFoundError, match="detection_config.yaml"):
        run_artifact_bundle(config, SimpleNamespace(product=None))


def test_deploy_requires_inference_models_dir(tmp_path):
    from picture_tool.tasks.deploy import run_deploy

    run_dir = tmp_path / "runs" / "detect" / "train"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"pt")
    (run_dir / "detection_config.yaml").write_text(
        yaml.safe_dump({"weights": "best.pt", "enable_color_check": False}),
        encoding="utf-8",
    )

    config = {
        "yolo_training": {
            "project": str(tmp_path / "runs" / "detect"),
            "name": "train",
            "deploy": {"enabled": True, "product": "PCBA1", "area": "A"},
        }
    }

    with pytest.raises(ValueError, match="inference_models_dir"):
        run_deploy(config, SimpleNamespace(product=None))
