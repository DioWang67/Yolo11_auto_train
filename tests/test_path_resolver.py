"""Tests for picture_tool.path_resolver module.

Validates that resolve_project_paths is a pure function (returns new dict,
does not mutate input) and correctly maps all path sections.
"""

import copy
from pathlib import Path

import pytest

from picture_tool.path_resolver import (
    parse_project_area_override,
    resolve_project_paths,
)


@pytest.fixture
def base_config():
    """Minimal config containing all sections that get path-resolved."""
    return {
        "pipeline": {"log_file": "old.log", "task_groups": {}},
        "format_conversion": {"input_dir": "", "output_dir": ""},
        "anomaly_detection": {
            "output_folder": "",
            "reference_folder": "",
            "test_folder": "",
        },
        "yolo_augmentation": {
            "input": {"image_dir": "", "label_dir": ""},
            "output": {"image_dir": "", "label_dir": ""},
        },
        "image_augmentation": {},
        "train_test_split": {},
        "yolo_training": {
            "project": "",
            "name": "",
            "dataset_dir": "",
            "position_validation": {},
            "export_detection_config": {},
        },
        "batch_inference": {"input_dir": "./data/project/test", "output_dir": ""},
        "color_inspection": {"input_dir": "", "output_json": ""},
        "color_verification": {
            "input_dir": "",
            "color_stats": "",
            "output_json": "",
            "output_csv": "",
        },
        "dataset_lint": {"image_dir": "", "label_dir": "", "output_dir": ""},
        "aug_preview": {"image_dir": "", "label_dir": "", "output_dir": ""},
        "report": {"output_dir": ""},
    }


class TestResolveProjectPaths:
    def test_parses_product_area_override(self):
        parsed = parse_project_area_override("PCBA1,B")

        assert parsed.project == "PCBA1"
        assert parsed.area == "B"

    def test_parses_product_without_area(self):
        parsed = parse_project_area_override("PCBA1")

        assert parsed.project == "PCBA1"
        assert parsed.area is None

    def test_rejects_empty_product_before_area(self):
        with pytest.raises(ValueError):
            parse_project_area_override(",B")

    def test_rejects_path_separator_in_product(self):
        with pytest.raises(ValueError):
            parse_project_area_override("../PCBA1,B")

    def test_rejects_path_separator_in_area(self):
        with pytest.raises(ValueError):
            parse_project_area_override("PCBA1,B/side")

    def test_returns_new_dict(self, base_config):
        """Must not mutate the input dict."""
        original = copy.deepcopy(base_config)
        result = resolve_project_paths(base_config, "TestProduct")
        assert result is not base_config
        assert base_config == original

    def test_format_conversion_paths(self, base_config):
        result = resolve_project_paths(base_config, "Cable1")
        fc = result["format_conversion"]
        assert "Cable1" in fc["input_dir"]
        assert "raw" in fc["input_dir"]
        assert "Cable1" in fc["output_dir"]
        assert "processed" in fc["output_dir"]

    def test_yolo_training_paths(self, base_config):
        result = resolve_project_paths(base_config, "LED")
        yt = result["yolo_training"]
        assert "LED" in yt["project"]
        assert yt["name"] == "train"
        assert "LED" in yt["dataset_dir"]
        assert yt["position_validation"]["product"] == "LED"

    def test_product_area_override_updates_area_settings(self, base_config):
        base_config["yolo_training"]["deploy"] = {
            "enabled": True,
            "inference_models_dir": "../yolo11_inference/models",
        }
        base_config["yolo_training"]["artifact_bundle"] = {"enabled": True}

        result = resolve_project_paths(base_config, "PCBA1,B")
        yt = result["yolo_training"]

        assert "PCBA1" in yt["project"]
        assert "PCBA1,B" not in yt["project"]
        assert str(Path("data") / "PCBA1" / "B" / "split") == yt["dataset_dir"]
        assert str(Path("runs") / "PCBA1" / "B") == yt["project"]
        assert (
            result["yolo_augmentation"]["input"]["image_dir"]
            == str(Path("data") / "PCBA1" / "B" / "raw" / "images")
        )
        assert (
            result["train_test_split"]["input"]["image_dir"]
            == str(Path("data") / "PCBA1" / "B" / "processed" / "images")
        )
        assert yt["position_validation"]["product"] == "PCBA1"
        assert yt["position_validation"]["area"] == "B"
        assert yt["export_detection_config"]["current_product"] == "PCBA1"
        assert yt["export_detection_config"]["area"] == "B"
        assert yt["deploy"]["product"] == "PCBA1"
        assert yt["deploy"]["area"] == "B"
        assert yt["artifact_bundle"]["area"] == "B"

    def test_operator_handoff_can_augment_raw_then_split_processed(
        self, base_config, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        dataset_root = tmp_path / "data" / "Cable1" / "A"
        base_config["operator_handoff"] = {
            "enabled": True,
            "dataset_root": str(dataset_root),
            "source_stage": "raw",
            "split_source_stage": "processed",
        }

        result = resolve_project_paths(base_config, "Cable1,A")

        assert Path(result["yolo_augmentation"]["input"]["image_dir"]) == (
            dataset_root / "raw" / "images"
        )
        assert Path(result["train_test_split"]["input"]["image_dir"]) == (
            dataset_root / "processed" / "images"
        )

    def test_pipeline_log_path(self, base_config):
        result = resolve_project_paths(base_config, "MyProduct")
        assert "MyProduct" in result["pipeline"]["log_file"]
        assert "logs" in result["pipeline"]["log_file"]

    def test_batch_inference_replaces_project_placeholder(self, base_config):
        result = resolve_project_paths(base_config, "Cable1")
        bi = result["batch_inference"]
        assert "/project/" not in bi["input_dir"]
        assert "Cable1" in bi["input_dir"]

    def test_color_inspection_paths(self, base_config):
        result = resolve_project_paths(base_config, "PCBA")
        ci = result["color_inspection"]
        assert "PCBA" in ci["input_dir"]
        assert "qc" in ci["input_dir"]
        assert "PCBA" in ci["output_json"]

    def test_color_verification_paths(self, base_config):
        result = resolve_project_paths(base_config, "PCBA")
        cv = result["color_verification"]
        assert "PCBA" in cv["output_json"]
        assert "PCBA" in cv["output_csv"]

    def test_dataset_lint_paths(self, base_config):
        result = resolve_project_paths(base_config, "X")
        dl = result["dataset_lint"]
        assert "X" in dl["image_dir"]

    def test_missing_sections_are_skipped(self):
        """Config without optional sections should not raise."""
        minimal = {"pipeline": {"log_file": "x.log"}}
        result = resolve_project_paths(minimal, "P")
        assert "P" in result["pipeline"]["log_file"]

    def test_aug_preview_paths(self, base_config):
        result = resolve_project_paths(base_config, "LED")
        ap = result["aug_preview"]
        assert "LED" in ap["image_dir"]
        assert "LED" in ap["output_dir"]

    def test_report_paths(self, base_config):
        result = resolve_project_paths(base_config, "LED")
        assert "LED" in result["report"]["output_dir"]

    def test_loads_project_classes_from_labelimg_classes_file(
        self, base_config, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        labels_dir = Path("data") / "PCBA1" / "raw" / "labels"
        labels_dir.mkdir(parents=True)
        (labels_dir / "classes.txt").write_text(
            "J5-1\nJ5-2\nC22B\nJ6\nJ7\n",
            encoding="utf-8",
        )
        base_config["yolo_training"]["export_detection_config"] = {
            "expected_items": {"project": {"A": ["Black", "Green"]}},
            "steps": {
                "sequence_check": {
                    "expected": ["Black", "Green"],
                }
            },
        }

        result = resolve_project_paths(base_config, "PCBA1")
        yolo_cfg = result["yolo_training"]
        export_cfg = yolo_cfg["export_detection_config"]

        assert yolo_cfg["class_names"] == ["J5-1", "J5-2", "C22B", "J6", "J7"]
        assert export_cfg["expected_items"] == {
            "PCBA1": {"A": ["J5-1", "J5-2", "C22B", "J6", "J7"]}
        }
        assert export_cfg["steps"]["sequence_check"]["expected"] == [
            "J5-1",
            "J5-2",
            "C22B",
            "J6",
            "J7",
        ]

    def test_loads_area_scoped_project_classes(self, base_config, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        labels_dir = Path("data") / "PCBA1" / "B" / "raw" / "labels"
        labels_dir.mkdir(parents=True)
        (labels_dir / "classes.txt").write_text("J5-1\nJ5-2\n", encoding="utf-8")

        result = resolve_project_paths(base_config, "PCBA1,B")

        assert result["yolo_training"]["class_names"] == ["J5-1", "J5-2"]
