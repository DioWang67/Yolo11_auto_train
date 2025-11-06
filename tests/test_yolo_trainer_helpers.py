from pathlib import Path

import yaml

from picture_tool.train import yolo_trainer


def test_ensure_data_yaml_creates_expected_structure(tmp_path):
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "train" / "images").mkdir(parents=True)
    (dataset_dir / "val" / "images").mkdir(parents=True)
    (dataset_dir / "test" / "images").mkdir(parents=True)

    output = yolo_trainer._ensure_data_yaml(dataset_dir, ["cat", "dog"])

    data = yaml.safe_load(Path(output).read_text(encoding="utf-8"))
    assert data["path"] == str(dataset_dir.resolve())
    assert data["names"] == ["cat", "dog"]
    assert data["train"] == "train/images"


def test_normalize_imgsz_handles_sequences_and_scalars():
    assert yolo_trainer._normalize_imgsz(["", "512", "256"]) == [512, 256]
    assert yolo_trainer._normalize_imgsz("640") == [640, 640]
    assert yolo_trainer._normalize_imgsz(None) is None


def test_normalize_name_sequence_accepts_mappings_and_lists():
    mapping = {"1": "first", "0": "zero"}
    assert yolo_trainer._normalize_name_sequence(mapping) == ["zero", "first"]
    assert yolo_trainer._normalize_name_sequence(["one", None, "two"]) == [
        "one",
        "two",
    ]
