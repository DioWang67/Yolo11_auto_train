"""Tests for picture_tool.path_resolver module.

Validates that resolve_project_paths is a pure function (returns new dict,
does not mutate input) and correctly maps all path sections.
"""

import copy

import pytest

from picture_tool.path_resolver import resolve_project_paths


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
