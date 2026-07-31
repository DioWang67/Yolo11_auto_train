import csv
from pathlib import Path

import pytest

from picture_tool.quality.dataset_readiness import (
    DatasetReadinessError,
    validate_training_dataset,
)
from picture_tool.split.dataset_splitter import (
    _deterministic_group_split,
    split_dataset,
)


def test_operator_split_reserves_absolute_validation_and_test_groups() -> None:
    train, val, test = _deterministic_group_split(
        list(range(19)),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        has_forced_train=True,
        minimum_val_groups=5,
        minimum_test_groups=10,
    )

    assert len(train) == 4
    assert len(val) == 5
    assert len(test) == 10
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)


def test_operator_split_rejects_insufficient_historical_groups() -> None:
    with pytest.raises(ValueError, match="independent historical source groups"):
        _deterministic_group_split(
            list(range(14)),
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            has_forced_train=True,
            minimum_val_groups=5,
            minimum_test_groups=10,
        )


def _dataset_config(root: Path) -> dict:
    return {
        "yolo_training": {
            "dataset_dir": str(root / "split"),
            "class_names": ["part"],
        }
    }


def _write_pair(root: Path, split: str, stem: str, image: bytes, label: str) -> None:
    images = root / "split" / split / "images"
    labels = root / "split" / split / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    (images / f"{stem}.png").write_bytes(image)
    (labels / f"{stem}.txt").write_text(label, encoding="utf-8")


def test_valid_dataset_passes_readiness(tmp_path):
    _write_pair(tmp_path, "train", "train_part", b"train", "0 0.5 0.5 0.2 0.2\n")
    _write_pair(tmp_path, "val", "val_part", b"val", "0 0.5 0.5 0.2 0.2\n")

    report = validate_training_dataset(_dataset_config(tmp_path))

    assert report.images == 2
    assert report.errors == []


def test_pending_review_label_blocks_training(tmp_path):
    _write_pair(tmp_path, "train", "review", b"train", "")
    _write_pair(tmp_path, "val", "val", b"val", "0 0.5 0.5 0.2 0.2\n")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    image_path = tmp_path / "raw" / "images" / "review.png"
    with (metadata / "review_dataset_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["output_image", "output_label", "annotation_status"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "output_image": str(image_path),
                "output_label": str(tmp_path / "raw" / "labels" / "review.txt"),
                "annotation_status": "pending",
            }
        )

    with pytest.raises(DatasetReadinessError, match="annotation_pending"):
        validate_training_dataset(_dataset_config(tmp_path))


def test_verified_empty_review_label_is_explicitly_allowed(tmp_path):
    _write_pair(tmp_path, "train", "negative", b"train", "")
    _write_pair(tmp_path, "val", "val", b"val", "0 0.5 0.5 0.2 0.2\n")
    raw_images = tmp_path / "raw" / "images"
    raw_labels = tmp_path / "raw" / "labels"
    metadata = tmp_path / "metadata"
    raw_images.mkdir(parents=True)
    raw_labels.mkdir(parents=True)
    metadata.mkdir()
    (raw_images / "negative.png").write_bytes(b"negative")
    (raw_labels / "negative.txt").write_text("", encoding="utf-8")
    (metadata / "review_dataset_manifest.csv").write_text(
        "output_image,output_label,annotation_status\n"
        f"{raw_images / 'negative.png'},{raw_labels / 'negative.txt'},verified_empty\n",
        encoding="utf-8",
    )

    report = validate_training_dataset(_dataset_config(tmp_path))

    assert report.verified_empty == 1


def test_identical_image_across_splits_is_rejected(tmp_path):
    _write_pair(tmp_path, "train", "a", b"duplicate", "0 0.5 0.5 0.2 0.2\n")
    _write_pair(tmp_path, "val", "b", b"duplicate", "0 0.5 0.5 0.2 0.2\n")

    with pytest.raises(DatasetReadinessError, match="split_leakage"):
        validate_training_dataset(_dataset_config(tmp_path))


def test_augmented_family_across_splits_is_rejected(tmp_path):
    _write_pair(tmp_path, "train", "board_aug_1", b"one", "0 0.5 0.5 0.2 0.2\n")
    _write_pair(tmp_path, "val", "board_aug_2", b"two", "0 0.5 0.5 0.2 0.2\n")

    with pytest.raises(DatasetReadinessError, match="source_split_leakage"):
        validate_training_dataset(_dataset_config(tmp_path))


def test_operator_minimum_class_coverage_blocks_missing_class(tmp_path):
    _write_pair(tmp_path, "train", "train_part", b"train", "0 0.5 0.5 0.2 0.2\n")
    _write_pair(tmp_path, "val", "val_part", b"val", "0 0.5 0.5 0.2 0.2\n")
    config = _dataset_config(tmp_path)
    config["yolo_training"]["class_names"] = ["part", "missing"]
    config["dataset_readiness"] = {"min_instances_per_class": 1}

    with pytest.raises(DatasetReadinessError, match="class_underrepresented:1"):
        validate_training_dataset(config)


def test_operator_split_minimums_block_too_small_test_cohort(tmp_path):
    _write_pair(tmp_path, "train", "train_part", b"train", "0 0.5 0.5 0.2 0.2\n")
    _write_pair(tmp_path, "val", "val_part", b"val", "0 0.5 0.5 0.2 0.2\n")
    _write_pair(tmp_path, "test", "test_part", b"test", "0 0.5 0.5 0.2 0.2\n")
    config = _dataset_config(tmp_path)
    config["dataset_readiness"] = {
        "min_images_per_split": {"train": 1, "val": 1, "test": 2},
        "min_test_instances_per_class": 2,
    }

    with pytest.raises(DatasetReadinessError, match="split_underrepresented:test"):
        validate_training_dataset(config)


def test_splitter_keeps_augmented_family_in_one_split(tmp_path):
    processed_images = tmp_path / "processed" / "images"
    processed_labels = tmp_path / "processed" / "labels"
    processed_images.mkdir(parents=True)
    processed_labels.mkdir(parents=True)
    for source_index in range(10):
        for augmentation_index in range(3):
            stem = f"board_{source_index}_aug_{augmentation_index}"
            (processed_images / f"{stem}.png").write_bytes(
                f"{source_index}-{augmentation_index}".encode()
            )
            (processed_labels / f"{stem}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
    output_dir = tmp_path / "split"
    config = {
        "train_test_split": {
            "input": {
                "image_dir": str(processed_images),
                "label_dir": str(processed_labels),
            },
            "output": {"output_dir": str(output_dir)},
            "split_ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
            "stratified": False,
        },
        "yolo_training": {"class_names": ["part"]},
    }

    split_dataset(config)

    source_locations: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        for image_path in (output_dir / split / "images").glob("*.png"):
            source = image_path.stem.rsplit("_aug_", 1)[0]
            source_locations.setdefault(source, set()).add(split)
    assert all(len(locations) == 1 for locations in source_locations.values())


def test_splitter_forces_submitted_feedback_into_train(tmp_path):
    processed_images = tmp_path / "processed" / "images"
    processed_labels = tmp_path / "processed" / "labels"
    processed_images.mkdir(parents=True)
    processed_labels.mkdir(parents=True)
    sample_id = "feedback123"
    stems = [f"review_{sample_id}", "history_a", "history_b", "history_c"]
    for index, stem in enumerate(stems):
        (processed_images / f"{stem}.png").write_bytes(f"image-{index}".encode())
        (processed_labels / f"{stem}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    output_dir = tmp_path / "split"
    config = {
        "train_test_split": {
            "input": {
                "image_dir": str(processed_images),
                "label_dir": str(processed_labels),
            },
            "output": {"output_dir": str(output_dir)},
            "split_ratios": {"train": 0.5, "val": 0.25, "test": 0.25},
            "stratified": False,
            "force_train_sample_ids": [sample_id],
        },
        "yolo_training": {"class_names": ["part"]},
    }

    split_dataset(config)

    assert (output_dir / "train" / "images" / f"review_{sample_id}.png").is_file()
    assert not (output_dir / "val" / "images" / f"review_{sample_id}.png").exists()
    assert not (output_dir / "test" / "images" / f"review_{sample_id}.png").exists()


def test_splitter_rejects_missing_submitted_feedback(tmp_path):
    processed_images = tmp_path / "processed" / "images"
    processed_labels = tmp_path / "processed" / "labels"
    processed_images.mkdir(parents=True)
    processed_labels.mkdir(parents=True)
    for index in range(3):
        (processed_images / f"history_{index}.png").write_bytes(
            f"image-{index}".encode()
        )
        (processed_labels / f"history_{index}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    config = {
        "train_test_split": {
            "input": {
                "image_dir": str(processed_images),
                "label_dir": str(processed_labels),
            },
            "output": {"output_dir": str(tmp_path / "split")},
            "split_ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
            "stratified": False,
            "force_train_sample_ids": ["missing-feedback"],
        },
        "yolo_training": {"class_names": ["part"]},
    }

    with pytest.raises(ValueError, match="missing from the split input"):
        split_dataset(config)


def test_splitter_reserves_position_golden_family_for_test(tmp_path):
    processed_images = tmp_path / "processed" / "images"
    processed_labels = tmp_path / "processed" / "labels"
    processed_images.mkdir(parents=True)
    processed_labels.mkdir(parents=True)
    sample_id = "position-golden"
    stems = [
        f"review_{sample_id}",
        f"review_{sample_id}_aug_1",
        "history_a",
        "history_b",
    ]
    for index, stem in enumerate(stems):
        (processed_images / f"{stem}.png").write_bytes(
            f"image-{index}".encode()
        )
        (processed_labels / f"{stem}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    output_dir = tmp_path / "split"
    config = {
        "train_test_split": {
            "input": {
                "image_dir": str(processed_images),
                "label_dir": str(processed_labels),
            },
            "output": {"output_dir": str(output_dir)},
            "split_ratios": {"train": 0.5, "val": 0.25, "test": 0.25},
            "stratified": False,
            "force_test_sample_ids": [sample_id],
        },
        "yolo_training": {"class_names": ["part"]},
    }

    split_dataset(config)

    assert (
        output_dir / "test" / "images" / f"review_{sample_id}.png"
    ).is_file()
    assert (
        output_dir / "test" / "images" / f"review_{sample_id}_aug_1.png"
    ).is_file()
    assert not (
        output_dir / "train" / "images" / f"review_{sample_id}.png"
    ).exists()
