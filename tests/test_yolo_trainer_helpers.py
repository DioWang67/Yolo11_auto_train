from pathlib import Path
import json
import logging
import threading
from types import SimpleNamespace

import yaml

from picture_tool.train import yolo_trainer
from picture_tool.utils.normalization import normalize_imgsz, normalize_name_sequence


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
    assert normalize_imgsz(["", "512", "256"]) == [512, 256]
    assert normalize_imgsz("640") == [640, 640]
    assert normalize_imgsz(None) is None


def test_normalize_name_sequence_accepts_mappings_and_lists():
    mapping = {"1": "first", "0": "zero"}
    assert normalize_name_sequence(mapping) == ["zero", "first"]
    assert normalize_name_sequence(["one", None, "two"]) == [
        "one",
        "two",
    ]


def test_stop_callback_preserves_native_resume_checkpoint(tmp_path):
    callbacks = {}

    class FakeModel:
        def add_callback(self, name, callback):
            callbacks[name] = callback

    run_dir = tmp_path / "runs" / "train"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    last_checkpoint = weights_dir / "last.pt"
    last_checkpoint.write_bytes(b"optimizer-state")
    data_yaml = tmp_path / "dataset" / "data.yaml"
    data_yaml.parent.mkdir()
    data_yaml.write_text("names: [item]\n", encoding="utf-8")
    stop_event = threading.Event()
    stop_event.set()

    yolo_trainer._attach_yolo_callbacks(
        FakeModel(),
        logging.getLogger("resume-test"),
        stop_event,
        data_yaml=data_yaml,
        requested_total_epochs=20,
        completed_before_run=2,
    )
    trainer = SimpleNamespace(save_dir=run_dir, last=last_checkpoint)

    callbacks["on_train_start"](trainer)
    callbacks["on_model_save"](trainer)

    assert (weights_dir / "last.resume.pt").read_bytes() == b"optimizer-state"
    lineage = json.loads(
        (run_dir / "operator_resume_lineage.json").read_text(encoding="utf-8")
    )
    assert lineage["requested_total_epochs"] == 20
    assert lineage["completed_before_run"] == 2
