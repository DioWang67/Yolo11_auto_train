from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from picture_tool.position.position_calibration import (
    PositionCalibrationError,
    collect_yolo_calibration_dataset,
    write_calibration_manifest,
)


def _write_image(path, *, width: int = 200, height: int = 100) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_collect_calibration_uses_human_labels_and_letterbox_coordinates(
    tmp_path,
) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    _write_image(images / "board.jpg")
    (labels / "board.txt").write_text(
        "0 0.25 0.5 0.2 0.4\n"
        "1 0.75 0.5 0.2 0.4\n",
        encoding="utf-8",
    )

    dataset = collect_yolo_calibration_dataset(
        image_dir=images,
        label_dir=labels,
        class_names=["Red", "Green"],
        imgsz=640,
    )

    assert len(dataset.samples) == 1
    red = dataset.boxes_by_class["Red"][0]
    green = dataset.boxes_by_class["Green"][0]
    assert red == (96, 256, 224, 384)
    assert green == (416, 256, 544, 384)
    assert dataset.per_image_class_counts["Red"] == (1,)
    assert len(dataset.dataset_sha256) == 64


def test_collect_calibration_excludes_incomplete_and_augmented_samples(
    tmp_path,
) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for name in ("complete", "missing_class", "complete_aug_1"):
        _write_image(images / f"{name}.jpg")
    (labels / "complete.txt").write_text(
        "0 0.25 0.5 0.2 0.4\n1 0.75 0.5 0.2 0.4\n",
        encoding="utf-8",
    )
    (labels / "missing_class.txt").write_text(
        "0 0.25 0.5 0.2 0.4\n",
        encoding="utf-8",
    )
    (labels / "complete_aug_1.txt").write_text(
        "0 0.25 0.5 0.2 0.4\n1 0.75 0.5 0.2 0.4\n",
        encoding="utf-8",
    )

    dataset = collect_yolo_calibration_dataset(
        image_dir=images,
        label_dir=labels,
        class_names=["Red", "Green"],
        imgsz=640,
    )

    assert [sample.image_path.stem for sample in dataset.samples] == ["complete"]
    assert dataset.excluded_samples[0]["reason"] == "missing_required_class"


@pytest.mark.parametrize(
    ("label", "message"),
    [
        ("2 0.5 0.5 0.2 0.2\n", "outside the class contract"),
        ("0 1.2 0.5 0.2 0.2\n", r"outside \[0, 1\]"),
        ("0 0.5 0.5 0.0 0.2\n", "must be positive"),
        ("0 0.5 0.5 0.2\n", "exactly 5"),
    ],
)
def test_collect_calibration_fails_closed_on_invalid_labels(
    tmp_path,
    label,
    message,
) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    _write_image(images / "board.jpg")
    (labels / "board.txt").write_text(label, encoding="utf-8")

    with pytest.raises(PositionCalibrationError, match=message):
        collect_yolo_calibration_dataset(
            image_dir=images,
            label_dir=labels,
            class_names=["Red"],
            imgsz=640,
        )


def test_collect_calibration_rejects_duplicate_class_contract(tmp_path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    _write_image(images / "board.jpg")
    (labels / "board.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(PositionCalibrationError, match="duplicate"):
        collect_yolo_calibration_dataset(
            image_dir=images,
            label_dir=labels,
            class_names=["Black", "Black"],
            imgsz=640,
        )


def test_collect_calibration_rejects_duplicate_image_content(tmp_path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for name in ("board_a", "board_b"):
        _write_image(images / f"{name}.jpg")
        (labels / f"{name}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )

    with pytest.raises(PositionCalibrationError, match="Duplicate calibration"):
        collect_yolo_calibration_dataset(
            image_dir=images,
            label_dir=labels,
            class_names=["Black"],
            imgsz=640,
        )


def test_write_calibration_manifest_is_atomic(tmp_path) -> None:
    path = tmp_path / "position_calibration_manifest.json"

    write_calibration_manifest(path, {"schema_version": 1, "samples": []})

    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert not list(tmp_path.glob("*.tmp"))
