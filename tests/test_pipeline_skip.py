from pathlib import Path
from types import SimpleNamespace

import pytest

from picture_tool.tasks import augmentation, quality, training


@pytest.fixture()
def temp_dirs(tmp_path):
    # set up minimal directory tree used by skip checks
    raw_images = tmp_path / "raw" / "images"
    raw_labels = tmp_path / "raw" / "labels"
    raw_images.mkdir(parents=True)
    raw_labels.mkdir(parents=True)
    (raw_images / "a.png").write_bytes(b"img")
    (raw_labels / "a.txt").write_text("0 0.5 0.5 1 1", encoding="utf-8")

    augmented_images = tmp_path / "augmented" / "images"
    augmented_labels = tmp_path / "augmented" / "labels"
    augmented_images.mkdir(parents=True)
    augmented_labels.mkdir(parents=True)

    split_root = tmp_path / "split"
    for subset in ("train", "val", "test"):
        d = split_root / subset / "images"
        d.mkdir(parents=True)
        (d / "dummy.jpg").write_bytes(b"img")

    runs_root = tmp_path / "runs" / "detect" / "train"
    (runs_root / "weights").mkdir(parents=True)
    (runs_root / "weights" / "best.pt").write_bytes(b"weights")
    (runs_root / "auto_position_config.yaml").write_text("config", encoding="utf-8")

    lint_out = tmp_path / "reports" / "lint"
    lint_out.mkdir(parents=True)
    (lint_out / "lint.csv").write_text("file,status\n", encoding="utf-8")

    preview_out = tmp_path / "reports" / "preview"
    preview_out.mkdir(parents=True)
    (preview_out / "preview.png").write_bytes(b"png")

    infer_in = tmp_path / "inference" / "images"
    infer_in.mkdir(parents=True)
    (infer_in / "b.png").write_bytes(b"img")
    infer_out = tmp_path / "reports" / "infer"
    infer_out.mkdir(parents=True)
    (infer_out / "predictions.csv").write_text("file,conf\n", encoding="utf-8")

    return {
        "raw_images": raw_images,
        "raw_labels": raw_labels,
        "aug_images": augmented_images,
        "aug_labels": augmented_labels,
        "split": split_root,
        "runs": runs_root,
        "lint_out": lint_out,
        "preview_out": preview_out,
        "infer_in": infer_in,
        "infer_out": infer_out,
    }


@pytest.fixture()
def base_config(temp_dirs):
    cfg = {
        "pipeline": {"log_file": "logs/pipeline.log"},
        "yolo_augmentation": {
            "input": {
                "image_dir": str(temp_dirs["raw_images"]),
                "label_dir": str(temp_dirs["raw_labels"]),
            },
            "output": {
                "image_dir": str(temp_dirs["aug_images"]),
                "label_dir": str(temp_dirs["aug_labels"]),
            },
        },
        "train_test_split": {
            "input": {
                "image_dir": str(temp_dirs["aug_images"]),
                "label_dir": str(temp_dirs["aug_labels"]),
            },
            "output": {"output_dir": str(temp_dirs["split"])},
        },
        "yolo_training": {
            "dataset_dir": str(temp_dirs["split"]),
            "class_names": ["class"],
            "model": "model.pt",
            "epochs": 1,
            "imgsz": 32,
            "batch": 1,
            "device": "cpu",
            "project": str(temp_dirs["runs"].parent),
            "name": "train",
        },
        "dataset_lint": {
            "image_dir": str(temp_dirs["aug_images"]),
            "output_dir": str(temp_dirs["lint_out"]),
        },
        "aug_preview": {
            "image_dir": str(temp_dirs["aug_images"]),
            "output_dir": str(temp_dirs["preview_out"]),
        },
        "batch_inference": {
            "input_dir": str(temp_dirs["infer_in"]),
            "output_dir": str(temp_dirs["infer_out"]),
        },
    }
    return cfg


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("dummy", encoding="utf-8")


def test_should_skip_yolo_augmentation_when_outputs_newer(base_config, temp_dirs):
    # ensure output timestamps newer than input
    for out_dir in (temp_dirs["aug_images"], temp_dirs["aug_labels"]):
        _touch(out_dir / "marker.txt")

    reason = augmentation.skip_yolo_augmentation(
        base_config, SimpleNamespace(force=False)
    )
    assert reason is not None


def test_should_skip_dataset_splitter_when_split_ready(base_config):
    reason = quality.skip_dataset_splitter(base_config, SimpleNamespace(force=False))
    assert reason is not None


def test_should_skip_yolo_train_when_weights_fresh(base_config, temp_dirs):
    # Ensure weights are newer than dataset files
    import time

    time.sleep(1.1)
    (temp_dirs["runs"] / "weights" / "best.pt").touch()
    (temp_dirs["runs"] / "last_run_metadata.json").write_text("{}", encoding="utf-8")

    reason = training.skip_yolo_train(base_config, SimpleNamespace(force=False))
    assert reason is not None


def test_should_skip_dataset_lint_when_csv_fresh(base_config):
    reason = quality.skip_dataset_lint(base_config, SimpleNamespace(force=False))
    assert reason is not None


def test_should_skip_aug_preview_when_preview_exists(base_config):
    reason = augmentation.skip_aug_preview(base_config, SimpleNamespace(force=False))
    assert reason is not None


def test_should_skip_batch_inference_when_predictions_exist(base_config):
    reason = quality.skip_batch_infer(base_config, SimpleNamespace(force=False))
    assert reason is not None
