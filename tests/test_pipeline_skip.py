from pathlib import Path
from types import SimpleNamespace

import pytest

from picture_tool.tasks import augmentation, quality, training
from picture_tool.pipeline.cache import write_task_cache
from picture_tool.pipeline.preflight import PreflightChecker


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
    for index, subset in enumerate(("train", "val", "test")):
        image_dir = split_root / subset / "images"
        label_dir = split_root / subset / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (image_dir / f"dummy_{subset}.jpg").write_bytes(f"img-{index}".encode())
        (label_dir / f"dummy_{subset}.txt").write_text(
            "0 0.5 0.5 1 1", encoding="utf-8"
        )

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


def test_yolo_augmentation_does_not_skip_when_expected_outputs_missing(
    base_config, temp_dirs
):
    base_config["yolo_augmentation"]["augmentation"] = {"num_images": 3}
    _touch(temp_dirs["aug_images"] / "a_aug_1.png")
    _touch(temp_dirs["aug_labels"] / "a_aug_1.txt")

    reason = augmentation.skip_yolo_augmentation(
        base_config, SimpleNamespace(force=False)
    )

    assert reason is None


def test_yolo_augmentation_skips_when_expected_outputs_complete(
    base_config, temp_dirs
):
    base_config["yolo_augmentation"]["augmentation"] = {"num_images": 2}
    for index in (1, 2):
        _touch(temp_dirs["aug_images"] / f"a_aug_{index}.png")
        _touch(temp_dirs["aug_labels"] / f"a_aug_{index}.txt")

    reason = augmentation.skip_yolo_augmentation(
        base_config, SimpleNamespace(force=False)
    )

    assert reason is not None


def test_yolo_augmentation_cache_mismatch_forces_rerun(base_config, temp_dirs):
    base_config["yolo_augmentation"]["augmentation"] = {"num_images": 2}
    for index in (1, 2):
        _touch(temp_dirs["aug_images"] / f"a_aug_{index}.png")
        _touch(temp_dirs["aug_labels"] / f"a_aug_{index}.txt")
    old_cfg = dict(base_config["yolo_augmentation"])
    old_cfg["augmentation"] = {"num_images": 1}
    write_task_cache(
        temp_dirs["aug_images"].parent,
        "yolo_augmentation",
        old_cfg,
        [temp_dirs["raw_images"], temp_dirs["raw_labels"]],
    )

    reason = augmentation.skip_yolo_augmentation(
        base_config, SimpleNamespace(force=False)
    )

    assert reason is None


def test_yolo_augmentation_original_count_includes_empty_negative(tmp_path):
    image_dir = tmp_path / "raw" / "images"
    label_dir = tmp_path / "raw" / "labels"
    output_images = tmp_path / "processed" / "images"
    output_labels = tmp_path / "processed" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "positive.jpg").write_bytes(b"positive")
    (label_dir / "positive.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (image_dir / "negative.jpg").write_bytes(b"negative")
    (label_dir / "negative.txt").write_text("", encoding="utf-8")
    config = {
        "input": {"image_dir": str(image_dir), "label_dir": str(label_dir)},
        "output": {
            "image_dir": str(output_images),
            "label_dir": str(output_labels),
        },
        "augmentation": {"num_images": 3, "include_originals": True},
    }

    copied = augmentation._copy_original_yolo_pairs(config)

    assert copied == 2
    assert (output_images / "positive.jpg").read_bytes() == b"positive"
    assert (output_labels / "negative.txt").read_text(encoding="utf-8") == ""
    assert augmentation._expected_yolo_augmentation_outputs(config) == 5


def test_yolo_augmentation_zero_variants_counts_original_pairs(tmp_path):
    image_dir = tmp_path / "raw" / "images"
    label_dir = tmp_path / "raw" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "one.jpg").write_bytes(b"one")
    (label_dir / "one.txt").write_text("", encoding="utf-8")
    config = {
        "input": {"image_dir": str(image_dir), "label_dir": str(label_dir)},
        "output": {
            "image_dir": str(tmp_path / "processed" / "images"),
            "label_dir": str(tmp_path / "processed" / "labels"),
        },
        "augmentation": {"num_images": 0, "include_originals": True},
    }

    assert augmentation._expected_yolo_augmentation_outputs(config) == 1


def test_yolo_augmentation_output_contract_rejects_missing_variants(tmp_path):
    image_dir = tmp_path / "raw" / "images"
    label_dir = tmp_path / "raw" / "labels"
    output_images = tmp_path / "processed" / "images"
    output_labels = tmp_path / "processed" / "labels"
    for directory in (image_dir, label_dir, output_images, output_labels):
        directory.mkdir(parents=True)
    (image_dir / "one.jpg").write_bytes(b"one")
    (label_dir / "one.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    config = {
        "input": {"image_dir": str(image_dir), "label_dir": str(label_dir)},
        "output": {
            "image_dir": str(output_images),
            "label_dir": str(output_labels),
        },
        "augmentation": {"num_images": 2, "include_originals": True},
    }
    augmentation._copy_original_yolo_pairs(config)

    with pytest.raises(
        augmentation.AugmentationOutputIncompleteError,
        match="expected_pairs=3",
    ):
        augmentation._validate_yolo_augmentation_outputs(config)


def test_yolo_augmentation_output_contract_accepts_complete_pairs(tmp_path):
    image_dir = tmp_path / "raw" / "images"
    label_dir = tmp_path / "raw" / "labels"
    output_images = tmp_path / "processed" / "images"
    output_labels = tmp_path / "processed" / "labels"
    for directory in (image_dir, label_dir, output_images, output_labels):
        directory.mkdir(parents=True)
    (image_dir / "one.jpg").write_bytes(b"one")
    (label_dir / "one.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    config = {
        "input": {"image_dir": str(image_dir), "label_dir": str(label_dir)},
        "output": {
            "image_dir": str(output_images),
            "label_dir": str(output_labels),
        },
        "augmentation": {"num_images": 2, "include_originals": True},
    }
    for stem in ("one", "one_aug_1", "one_aug_2"):
        (output_images / f"{stem}.jpg").write_bytes(b"image")
        (output_labels / f"{stem}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )

    augmentation._validate_yolo_augmentation_outputs(config)


def test_should_skip_dataset_splitter_when_split_ready(base_config):
    reason = quality.skip_dataset_splitter(base_config, SimpleNamespace(force=False))
    assert reason is not None


def test_dataset_splitter_cache_mismatch_forces_rerun(base_config, temp_dirs):
    old_cfg = dict(base_config["train_test_split"])
    old_cfg["split_ratios"] = {"train": 0.8, "val": 0.1, "test": 0.1}
    write_task_cache(
        temp_dirs["split"],
        "dataset_splitter",
        old_cfg,
        [temp_dirs["aug_images"], temp_dirs["aug_labels"]],
    )
    base_config["train_test_split"]["split_ratios"] = {
        "train": 0.7,
        "val": 0.2,
        "test": 0.1,
    }

    reason = quality.skip_dataset_splitter(base_config, SimpleNamespace(force=False))

    assert reason is None


def test_resolve_split_input_dirs_falls_back_to_raw(tmp_path):
    raw_images = tmp_path / "data" / "PCBA1" / "raw" / "images"
    raw_labels = tmp_path / "data" / "PCBA1" / "raw" / "labels"
    raw_images.mkdir(parents=True)
    raw_labels.mkdir(parents=True)
    config = {
        "train_test_split": {
            "input": {
                "image_dir": str(tmp_path / "data" / "PCBA1" / "processed" / "images"),
                "label_dir": str(tmp_path / "data" / "PCBA1" / "processed" / "labels"),
            }
        }
    }

    image_dir, label_dir, used_fallback = quality.resolve_split_input_dirs(config)

    assert image_dir == raw_images
    assert label_dir == raw_labels
    assert used_fallback is True


def test_operator_handoff_forbids_raw_fallback_when_augmentation_is_missing(tmp_path):
    raw_images = tmp_path / "data" / "PCBA1" / "raw" / "images"
    raw_labels = tmp_path / "data" / "PCBA1" / "raw" / "labels"
    raw_images.mkdir(parents=True)
    raw_labels.mkdir(parents=True)
    config = {
        "operator_handoff": {
            "enabled": True,
            "split_source_stage": "processed",
        },
        "train_test_split": {
            "input": {
                "image_dir": str(tmp_path / "data" / "PCBA1" / "processed" / "images"),
                "label_dir": str(tmp_path / "data" / "PCBA1" / "processed" / "labels"),
            }
        },
    }

    with pytest.raises(
        quality.OperatorAugmentationMissingError,
        match="raw-data fallback is forbidden",
    ):
        quality.resolve_split_input_dirs(config)


def test_operator_preflight_blocks_missing_augmentation_outputs(tmp_path):
    from picture_tool.pipeline.preflight import PreflightChecker, Severity

    raw_images = tmp_path / "data" / "PCBA1" / "raw" / "images"
    raw_labels = tmp_path / "data" / "PCBA1" / "raw" / "labels"
    raw_images.mkdir(parents=True)
    raw_labels.mkdir(parents=True)
    config = {
        "operator_handoff": {
            "enabled": True,
            "split_source_stage": "processed",
        },
        "train_test_split": {
            "input": {
                "image_dir": str(tmp_path / "data" / "PCBA1" / "processed" / "images"),
                "label_dir": str(tmp_path / "data" / "PCBA1" / "processed" / "labels"),
            }
        },
        "yolo_training": {
            "dataset_dir": str(tmp_path / "data" / "PCBA1" / "split"),
            "class_names": ["part"],
            "model": "yolo11n.pt",
        },
    }

    issues = PreflightChecker().run(["dataset_splitter"], config)

    blocking = [issue for issue in issues if issue.task == "dataset_splitter"]
    assert len(blocking) == 1
    assert blocking[0].severity == Severity.ERROR
    assert "raw-data fallback is forbidden" in blocking[0].message


def test_operator_preflight_allows_processed_outputs_to_be_created_by_augmentation(
    tmp_path,
):
    raw_images = tmp_path / "data" / "PCBA1" / "raw" / "images"
    raw_labels = tmp_path / "data" / "PCBA1" / "raw" / "labels"
    raw_images.mkdir(parents=True)
    raw_labels.mkdir(parents=True)
    config = {
        "operator_handoff": {
            "enabled": True,
            "split_source_stage": "processed",
        },
        "yolo_augmentation": {
            "input": {
                "image_dir": str(raw_images),
                "label_dir": str(raw_labels),
            }
        },
        "train_test_split": {
            "input": {
                "image_dir": str(tmp_path / "data" / "PCBA1" / "processed" / "images"),
                "label_dir": str(tmp_path / "data" / "PCBA1" / "processed" / "labels"),
            }
        },
        "yolo_training": {
            "dataset_dir": str(tmp_path / "data" / "PCBA1" / "split"),
            "class_names": ["part"],
            "model": "yolo11n.pt",
        },
    }

    issues = PreflightChecker().run(
        ["yolo_augmentation", "dataset_splitter"], config
    )

    assert not [issue for issue in issues if issue.task == "dataset_splitter"]


def test_operator_preflight_blocks_when_planned_augmentation_raw_inputs_are_missing(
    tmp_path,
):
    # The operator guard is reached only when the split input has a usable raw
    # fallback.  The planned augmentation itself deliberately points elsewhere
    # so preflight must validate (and reject) its actual configured inputs.
    (tmp_path / "raw" / "images").mkdir(parents=True)
    (tmp_path / "raw" / "labels").mkdir(parents=True)
    config = {
        "operator_handoff": {
            "enabled": True,
            "split_source_stage": "processed",
        },
        "yolo_augmentation": {
            "input": {
                "image_dir": str(tmp_path / "missing" / "images"),
                "label_dir": str(tmp_path / "missing" / "labels"),
            }
        },
        "train_test_split": {
            "input": {
                "image_dir": str(tmp_path / "processed" / "images"),
                "label_dir": str(tmp_path / "processed" / "labels"),
            }
        },
        "yolo_training": {
            "dataset_dir": str(tmp_path / "split"),
            "class_names": ["part"],
            "model": "yolo11n.pt",
        },
    }

    issues = PreflightChecker().run(
        ["yolo_augmentation", "dataset_splitter"], config
    )

    blocking = [issue for issue in issues if issue.task == "dataset_splitter"]
    assert len(blocking) == 2
    assert all("找不到" in issue.message for issue in blocking)
    assert any(str(tmp_path / "missing" / "images") in issue.message for issue in blocking)
    assert any(str(tmp_path / "missing" / "labels") in issue.message for issue in blocking)


def test_preflight_accepts_raw_fallback_when_processed_missing(tmp_path):
    from picture_tool.pipeline.preflight import PreflightChecker

    raw_images = tmp_path / "data" / "PCBA1" / "raw" / "images"
    raw_labels = tmp_path / "data" / "PCBA1" / "raw" / "labels"
    raw_images.mkdir(parents=True)
    raw_labels.mkdir(parents=True)
    config = {
        "train_test_split": {
            "input": {
                "image_dir": str(tmp_path / "data" / "PCBA1" / "processed" / "images"),
                "label_dir": str(tmp_path / "data" / "PCBA1" / "processed" / "labels"),
            }
        },
        "yolo_training": {
            "dataset_dir": str(tmp_path / "data" / "PCBA1" / "split"),
            "class_names": ["part"],
            "model": "yolo11n.pt",
        },
    }

    issues = PreflightChecker().run(["dataset_splitter"], config)

    assert not [issue for issue in issues if issue.task == "dataset_splitter"]


def test_should_skip_yolo_train_when_weights_fresh(base_config, temp_dirs):
    # Ensure weights are newer than dataset files
    import time

    time.sleep(1.1)
    (temp_dirs["runs"] / "weights" / "best.pt").touch()
    (temp_dirs["runs"] / "last_run_metadata.json").write_text("{}", encoding="utf-8")

    reason = training.skip_yolo_train(base_config, SimpleNamespace(force=False))
    assert reason is not None


def test_completed_operator_checkpoint_skips_training_despite_config_change(
    base_config, tmp_path
):
    checkpoint = tmp_path / "train10" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"completed")
    base_config["yolo_training"]["completed_job_checkpoint"] = str(checkpoint)
    base_config["yolo_training"]["position_validation"] = {"enabled": False}

    reason = training.skip_yolo_train(
        base_config,
        SimpleNamespace(force=False),
    )

    assert reason is not None
    assert "already completed successfully" in reason


def test_should_skip_dataset_lint_when_csv_fresh(base_config):
    reason = quality.skip_dataset_lint(base_config, SimpleNamespace(force=False))
    assert reason is not None


def test_should_skip_aug_preview_when_preview_exists(base_config):
    reason = augmentation.skip_aug_preview(base_config, SimpleNamespace(force=False))
    assert reason is not None


def test_should_skip_batch_inference_when_predictions_exist(base_config):
    reason = quality.skip_batch_infer(base_config, SimpleNamespace(force=False))
    assert reason is not None


def test_batch_inference_cache_mismatch_forces_rerun(base_config, temp_dirs):
    old_cfg = dict(base_config["batch_inference"])
    old_cfg["conf"] = 0.25
    write_task_cache(
        temp_dirs["infer_out"],
        "batch_inference",
        old_cfg,
        [temp_dirs["infer_in"]],
    )
    base_config["batch_inference"]["conf"] = 0.5

    reason = quality.skip_batch_infer(base_config, SimpleNamespace(force=False))

    assert reason is None


# --- _find_latest_run_dir & versioning tests ---

def test_find_latest_run_dir_returns_base_when_only_one(tmp_path):
    from picture_tool.tasks.training import _find_latest_run_dir

    (tmp_path / "train" / "weights").mkdir(parents=True)
    result = _find_latest_run_dir(tmp_path, "train")
    assert result is not None
    assert result.name == "train"


def test_find_latest_run_dir_returns_most_recent_versioned(tmp_path):
    import time
    from picture_tool.tasks.training import _find_latest_run_dir

    for name in ("train", "train2"):
        (tmp_path / name / "weights").mkdir(parents=True)

    time.sleep(0.05)
    (tmp_path / "train3" / "weights").mkdir(parents=True)
    (tmp_path / "train3" / "weights" / "best.pt").write_bytes(b"w")

    result = _find_latest_run_dir(tmp_path, "train")
    assert result is not None
    assert result.name == "train3"


def test_find_latest_run_dir_ignores_unrelated_dirs(tmp_path):
    from picture_tool.tasks.training import _find_latest_run_dir

    (tmp_path / "train" / "weights").mkdir(parents=True)
    (tmp_path / "train_backup").mkdir()
    (tmp_path / "other").mkdir()

    result = _find_latest_run_dir(tmp_path, "train")
    assert result is not None
    assert result.name == "train"


def test_find_latest_run_dir_returns_none_when_project_missing(tmp_path):
    from picture_tool.tasks.training import _find_latest_run_dir

    result = _find_latest_run_dir(tmp_path / "nonexistent", "train")
    assert result is None


def test_skip_yolo_train_uses_latest_versioned_dir(base_config, temp_dirs):
    """skip_yolo_train should check the most recently modified versioned dir."""
    import time

    project = temp_dirs["runs"].parent  # runs/detect

    # Create a newer versioned run directory: train2
    train2 = project / "train2"
    (train2 / "weights").mkdir(parents=True)
    (train2 / "auto_position_config.yaml").write_text("cfg", encoding="utf-8")

    time.sleep(1.1)
    (train2 / "weights" / "best.pt").write_bytes(b"w")
    (train2 / "last_run_metadata.json").write_text("{}", encoding="utf-8")

    reason = training.skip_yolo_train(base_config, SimpleNamespace(force=False))
    assert reason is not None
    assert "train2" in reason


def test_skip_yolo_train_force_never_skips(base_config, temp_dirs):
    import time

    time.sleep(1.1)
    (temp_dirs["runs"] / "weights" / "best.pt").touch()
    (temp_dirs["runs"] / "last_run_metadata.json").write_text("{}", encoding="utf-8")

    reason = training.skip_yolo_train(base_config, SimpleNamespace(force=True))
    assert reason is None
