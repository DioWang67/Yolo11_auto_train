"""
Comprehensive tests for utils/detection_config.py module.
Coverage target: 4% → 70%+
"""

import logging
import yaml

from picture_tool.utils.detection_config import (
    _load_class_names_from_run,
    _load_mapping_from_source,
    _apply_area_overrides,
    _prepare_position_config,
    DetectionConfigExporter,
)


class TestLoadClassNamesFromRun:
    """Test _load_class_names_from_run function."""

    def test_loads_from_args_yaml_names_field(self, tmp_path):
        """Should load class names from args.yaml names field."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        args_file = run_dir / "args.yaml"
        args_file.write_text(
            yaml.safe_dump({"names": ["class1", "class2", "class3"]}), encoding="utf-8"
        )

        logger = logging.getLogger("test")
        result = _load_class_names_from_run(run_dir, logger, fallback=[])

        assert result == ["class1", "class2", "class3"]

    def test_loads_from_data_yaml_via_args(self, tmp_path):
        """Should follow data path in args.yaml to load names."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Create data.yaml
        data_file = run_dir / "data.yaml"
        data_file.write_text(
            yaml.safe_dump({"names": ["dog", "cat", "bird"]}), encoding="utf-8"
        )

        # Create args.yaml pointing to data.yaml
        args_file = run_dir / "args.yaml"
        args_file.write_text(yaml.safe_dump({"data": "data.yaml"}), encoding="utf-8")

        logger = logging.getLogger("test")
        result = _load_class_names_from_run(run_dir, logger, fallback=[])

        assert result == ["dog", "cat", "bird"]

    def test_handles_absolute_data_path(self, tmp_path):
        """Should handle absolute path to data.yaml."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        data_file = tmp_path / "external_data.yaml"
        data_file.write_text(
            yaml.safe_dump({"names": ["item1", "item2"]}), encoding="utf-8"
        )

        args_file = run_dir / "args.yaml"
        args_file.write_text(yaml.safe_dump({"data": str(data_file)}), encoding="utf-8")

        logger = logging.getLogger("test")
        result = _load_class_names_from_run(run_dir, logger, fallback=[])

        assert result == ["item1", "item2"]

    def test_falls_back_when_args_missing(self, tmp_path):
        """Should use fallback when args.yaml doesn't exist."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        logger = logging.getLogger("test")
        fallback = ["fallback1", "fallback2"]
        result = _load_class_names_from_run(run_dir, logger, fallback=fallback)

        assert result == fallback

    def test_falls_back_when_data_yaml_missing(self, tmp_path):
        """Should use fallback when data.yaml is referenced but missing."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        args_file = run_dir / "args.yaml"
        args_file.write_text(yaml.safe_dump({"data": "missing.yaml"}), encoding="utf-8")

        logger = logging.getLogger("test")
        fallback = ["fallback_class"]
        result = _load_class_names_from_run(run_dir, logger, fallback=fallback)

        assert result == fallback

    def test_handles_corrupt_args_yaml(self, tmp_path):
        """Should handle corrupted args.yaml gracefully."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        args_file = run_dir / "args.yaml"
        args_file.write_text("invalid: yaml: content: [[[", encoding="utf-8")

        logger = logging.getLogger("test")
        fallback = ["safe"]
        result = _load_class_names_from_run(run_dir, logger, fallback=fallback)

        assert result == fallback

    def test_handles_dict_names_format(self, tmp_path):
        """Should normalize dict-based names format."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        args_file = run_dir / "args.yaml"
        args_file.write_text(
            yaml.safe_dump({"names": {0: "zero", 1: "one", 2: "two"}}), encoding="utf-8"
        )

        logger = logging.getLogger("test")
        result = _load_class_names_from_run(run_dir, logger, fallback=[])

        assert result == ["zero", "one", "two"]


class TestLoadMappingFromSource:
    """Test _load_mapping_from_source function."""

    def test_returns_empty_for_none(self):
        """Should return empty dict for None source."""
        logger = logging.getLogger("test")
        result = _load_mapping_from_source(None, logger)
        assert result == {}

    def test_returns_empty_for_empty_string(self):
        """Should return empty dict for empty string."""
        logger = logging.getLogger("test")
        result = _load_mapping_from_source("", logger)
        assert result == {}

    def test_returns_empty_for_empty_dict(self):
        """Should return empty dict for empty dict."""
        logger = logging.getLogger("test")
        result = _load_mapping_from_source({}, logger)
        assert result == {}

    def test_loads_from_dict_mapping(self):
        """Should convert Mapping to str-keyed dict."""
        logger = logging.getLogger("test")
        source = {"key1": "value1", "key2": {"nested": "value"}}
        result = _load_mapping_from_source(source, logger)

        assert result == {"key1": "value1", "key2": {"nested": "value"}}

    def test_loads_from_yaml_file(self, tmp_path):
        """Should load mapping from YAML file."""
        src_file = tmp_path / "config.yaml"
        src_file.write_text(
            yaml.safe_dump({"product": "Cable1", "area": "A"}), encoding="utf-8"
        )

        logger = logging.getLogger("test")
        result = _load_mapping_from_source(src_file, logger)

        assert result == {"product": "Cable1", "area": "A"}

    def test_handles_missing_file(self, tmp_path):
        """Should return empty dict for missing file."""
        missing_file = tmp_path / "nonexistent.yaml"
        logger = logging.getLogger("test")
        result = _load_mapping_from_source(missing_file, logger)

        assert result == {}

    def test_handles_corrupt_yaml_file(self, tmp_path):
        """Should return empty dict for corrupted YAML."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("invalid: [[[yaml", encoding="utf-8")

        logger = logging.getLogger("test")
        result = _load_mapping_from_source(bad_file, logger)

        assert result == {}

    def test_handles_non_mapping_yaml(self, tmp_path):
        """Should return empty dict if YAML doesn't contain mapping."""
        list_file = tmp_path / "list.yaml"
        list_file.write_text(yaml.safe_dump([1, 2, 3]), encoding="utf-8")

        logger = logging.getLogger("test")
        result = _load_mapping_from_source(list_file, logger)

        assert result == {}


class TestApplyAreaOverrides:
    """Test _apply_area_overrides function."""

    def test_applies_tolerance_override(self):
        """Should override tolerance value."""
        area_cfg = {"tolerance": 5, "imgsz": 640}
        export_cfg = {"tolerance": 10}
        logger = logging.getLogger("test")
        warned = {}

        result = _apply_area_overrides(area_cfg, export_cfg, None, logger, warned)

        assert result["tolerance"] == 10.0

    def test_applies_tolerance_unit_override(self):
        """Should override tolerance_unit."""
        area_cfg = {"tolerance": 5}
        export_cfg = {"tolerance_unit": "percent"}
        logger = logging.getLogger("test")
        warned = {}

        result = _apply_area_overrides(area_cfg, export_cfg, None, logger, warned)

        assert result["tolerance_unit"] == "percent"

    def test_applies_imgsz_override(self):
        """Should override imgsz from parameter."""
        area_cfg = {"tolerance": 5}
        export_cfg = {}
        logger = logging.getLogger("test")
        warned = {}
        imgsz = [1280, 1280]

        result = _apply_area_overrides(area_cfg, export_cfg, imgsz, logger, warned)

        # Should be scalar when both dimensions equal
        assert result["imgsz"] == 1280

    def test_preserves_list_imgsz_when_different(self):
        """Should keep list format when dimensions differ."""
        area_cfg = {}
        export_cfg = {}
        logger = logging.getLogger("test")
        warned = {}
        imgsz = [640, 480]

        result = _apply_area_overrides(area_cfg, export_cfg, imgsz, logger, warned)

        assert result["imgsz"] == [640, 480]

    def test_handles_invalid_tolerance(self):
        """Should skip invalid tolerance and warn once."""
        area_cfg = {"tolerance": 5}
        export_cfg = {"tolerance": "invalid"}
        logger = logging.getLogger("test")
        warned = {}

        result = _apply_area_overrides(area_cfg, export_cfg, None, logger, warned)

        assert result["tolerance"] == 5  # Original unchanged
        assert warned.get("tolerance") is True

    def test_does_not_warn_twice_for_same_issue(self):
        """Should only warn once per issue."""
        area_cfg = {"tolerance": 5}
        export_cfg = {"tolerance": "invalid"}
        logger = logging.getLogger("test")
        warned = {"tolerance": True}  # Already warned

        # Should not raise or log again
        result = _apply_area_overrides(area_cfg, export_cfg, None, logger, warned)
        assert result["tolerance"] == 5


class TestPreparePositionConfig:
    """Test _prepare_position_config function."""

    def test_returns_single_product_area_when_specified(self):
        """Should return only specified product/area."""
        raw_config = {
            "Cable1": {
                "A": {"tolerance": 10, "expected_boxes": {}},
                "B": {"tolerance": 15, "expected_boxes": {}},
            },
            "Cable2": {"A": {"tolerance": 20, "expected_boxes": {}}},
        }
        export_cfg = {}
        logger = logging.getLogger("test")

        result = _prepare_position_config(
            raw_config,
            export_cfg,
            product="Cable1",
            area="A",
            imgsz=None,
            logger=logger,
        )

        assert "Cable1" in result
        assert "A" in result["Cable1"]
        assert "B" not in result["Cable1"]
        assert "Cable2" not in result

    def test_returns_all_products_when_include_all_enabled(self):
        """Should return all products when include_all_products=True."""
        raw_config = {
            "Product1": {"AreaA": {"tolerance": 5}},
            "Product2": {"AreaB": {"tolerance": 10}},
        }
        export_cfg = {"include_all_products": True}
        logger = logging.getLogger("test")

        result = _prepare_position_config(
            raw_config,
            export_cfg,
            product="Product1",
            area="AreaA",
            imgsz=None,
            logger=logger,
        )

        assert "Product1" in result
        assert "Product2" in result

    def test_returns_all_when_product_or_area_missing(self):
        """Should include all if product/area not specified."""
        raw_config = {"ProductX": {"Area1": {"tolerance": 8}}}
        export_cfg = {}
        logger = logging.getLogger("test")

        result = _prepare_position_config(
            raw_config,
            export_cfg,
            product=None,
            area="Area1",
            imgsz=None,
            logger=logger,
        )

        assert "ProductX" in result

    def test_returns_empty_when_product_not_found(self):
        """Should return empty dict if product doesn't exist."""
        raw_config = {"OtherProduct": {"A": {}}}
        export_cfg = {}
        logger = logging.getLogger("test")

        result = _prepare_position_config(
            raw_config,
            export_cfg,
            product="MissingProduct",
            area="A",
            imgsz=None,
            logger=logger,
        )

        assert result == {}

    def test_returns_empty_when_area_not_found(self):
        """Should return empty dict if area doesn't exist in product."""
        raw_config = {"Cable1": {"AreaX": {}}}
        export_cfg = {}
        logger = logging.getLogger("test")

        result = _prepare_position_config(
            raw_config,
            export_cfg,
            product="Cable1",
            area="AreaY",
            imgsz=None,
            logger=logger,
        )

        assert result == {}

    def test_applies_overrides_to_selected_area(self):
        """Should apply overrides to the specific area."""
        raw_config = {"P1": {"A1": {"tolerance": 5}}}
        export_cfg = {"tolerance": 12}
        logger = logging.getLogger("test")

        result = _prepare_position_config(
            raw_config,
            export_cfg,
            product="P1",
            area="A1",
            imgsz=[800, 800],
            logger=logger,
        )

        assert result["P1"]["A1"]["tolerance"] == 12.0
        assert result["P1"]["A1"]["imgsz"] == 800


class TestDetectionConfigExporter:
    """Test DetectionConfigExporter.export method."""

    def test_exports_basic_config_successfully(self, tmp_path):
        """Should export basic detection config."""
        run_dir = tmp_path / "runs" / "detect" / "train"
        run_dir.mkdir(parents=True)

        # Create weights
        weights_dir = run_dir / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_text("fake_weights")

        # Create args.yaml
        (run_dir / "args.yaml").write_text(
            yaml.safe_dump({"names": ["class1", "class2"]}), encoding="utf-8"
        )

        config = {
            "yolo_training": {
                "class_names": ["class1", "class2"],
                "device": "cpu",
                "imgsz": 640,
                "export_detection_config": {
                    "enabled": True,
                    "weights_name": "best.pt",
                    "conf_thres": 0.3,
                    "iou_thres": 0.5,
                },
            }
        }

        logger = logging.getLogger("test")
        result_path = DetectionConfigExporter.export(
            config, run_dir, logger, include_position=False
        )

        assert result_path is not None
        assert result_path.exists()

        # Verify content
        with open(result_path, encoding="utf-8") as f:
            exported = yaml.safe_load(f)

        assert exported["conf_thres"] == 0.3
        assert exported["iou_thres"] == 0.5
        assert exported["device"] == "cpu"
        assert "weights" in exported

    def test_skips_when_disabled(self, tmp_path):
        """Should return None when export is disabled."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        config = {"yolo_training": {"export_detection_config": {"enabled": False}}}

        logger = logging.getLogger("test")
        result = DetectionConfigExporter.export(
            config, run_dir, logger, include_position=False
        )

        assert result is None

    def test_skips_when_weights_missing(self, tmp_path):
        """Should skip export when weights file doesn't exist."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "weights").mkdir()

        config = {
            "yolo_training": {
                "export_detection_config": {"enabled": True, "weights_name": "best.pt"}
            }
        }

        logger = logging.getLogger("test")
        result = DetectionConfigExporter.export(
            config, run_dir, logger, include_position=False
        )

        assert result is None

    def test_includes_position_config_when_enabled(self, tmp_path):
        """Should include position config when requested."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        weights_dir = run_dir / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_text("weights")

        # Create position config file
        pos_file = tmp_path / "position.yaml"
        pos_file.write_text(
            yaml.safe_dump(
                {
                    "Cable1": {
                        "A": {
                            "tolerance": 10,
                            "expected_boxes": {
                                "Red": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
                            },
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        config = {
            "yolo_training": {
                "class_names": ["Red"],
                "export_detection_config": {
                    "enabled": True,
                    "current_product": "Cable1",
                    "area": "A",
                    "position_config_path": str(pos_file),
                },
                "position_validation": {
                    "enabled": True,
                    "product": "Cable1",
                    "area": "A",
                },
            }
        }

        logger = logging.getLogger("test")
        result_path = DetectionConfigExporter.export(
            config, run_dir, logger, include_position=True
        )

        assert result_path is not None
        with open(result_path, encoding="utf-8") as f:
            exported = yaml.safe_load(f)

        assert "position_config" in exported
        assert "Cable1" in exported["position_config"]

    def test_uses_custom_output_path(self, tmp_path):
        """Should use custom output path if specified."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        weights_dir = run_dir / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_text("weights")

        custom_output = tmp_path / "custom" / "config.yaml"

        config = {
            "yolo_training": {
                "class_names": ["test"],
                "export_detection_config": {
                    "enabled": True,
                    "output_path": str(custom_output),
                },
            }
        }

        logger = logging.getLogger("test")
        result_path = DetectionConfigExporter.export(
            config, run_dir, logger, include_position=False
        )

        assert result_path == custom_output.resolve()
        assert result_path.exists()

    def test_normalizes_imgsz_from_config(self, tmp_path):
        """Should normalize imgsz correctly."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        weights_dir = run_dir / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_text("weights")

        config = {
            "yolo_training": {
                "class_names": ["x"],
                "imgsz": 1024,  # Scalar
                "export_detection_config": {"enabled": True},
            }
        }

        logger = logging.getLogger("test")
        result_path = DetectionConfigExporter.export(
            config, run_dir, logger, include_position=False
        )

        with open(result_path, encoding="utf-8") as f:
            exported = yaml.safe_load(f)

        assert exported["imgsz"] == [1024, 1024]

    def test_extracts_expected_items_from_position_config(self, tmp_path):
        """Should extract expected_items from position config."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        weights_dir = run_dir / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_text("weights")

        config = {
            "yolo_training": {
                "class_names": ["A", "B"],
                "export_detection_config": {
                    "enabled": True,
                    "current_product": "P1",
                    "area": "Area1",
                    "position_config": {
                        "P1": {"Area1": {"expected_boxes": {"ItemA": {}, "ItemB": {}}}}
                    },
                },
            }
        }

        logger = logging.getLogger("test")
        result_path = DetectionConfigExporter.export(
            config, run_dir, logger, include_position=True
        )

        with open(result_path, encoding="utf-8") as f:
            exported = yaml.safe_load(f)

        assert "expected_items" in exported
        assert exported["expected_items"]["P1"]["Area1"] == ["ItemA", "ItemB"]

    def test_generates_expected_items_from_class_names(self, tmp_path):
        """Should generate expected_items from class_names if no position config."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        weights_dir = run_dir / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_text("weights")
        (run_dir / "args.yaml").write_text(
            yaml.safe_dump({"names": ["Red", "Green", "Blue"]}), encoding="utf-8"
        )

        config = {
            "yolo_training": {
                "class_names": ["Red", "Green", "Blue"],
                "export_detection_config": {
                    "enabled": True,
                    "current_product": "TestProduct",
                    "area": "TestArea",
                },
            }
        }

        logger = logging.getLogger("test")
        result_path = DetectionConfigExporter.export(
            config, run_dir, logger, include_position=False
        )

        with open(result_path, encoding="utf-8") as f:
            exported = yaml.safe_load(f)

        assert "expected_items" in exported
        assert exported["expected_items"]["TestProduct"]["TestArea"] == [
            "Red",
            "Green",
            "Blue",
        ]
