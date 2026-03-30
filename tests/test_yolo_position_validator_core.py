import pytest

from picture_tool.position.yolo_position_validator import (
    ExpectedBox,
    PositionAreaConfig,
    PositionValidationResult,
    _base_class_name,
    _imgsz_value,
    validate_detections_against_area,
)


def test_imgsz_value_prefers_first_numeric_entry():
    value = _imgsz_value(["", "512", "256"])
    assert value == 512


def test_imgsz_value_rejects_missing_entries():
    with pytest.raises(ValueError):
        _imgsz_value(["bad", None])


def _make_config(tolerance: float = 5.0) -> PositionAreaConfig:
    return PositionAreaConfig(
        enabled=True,
        tolerance=tolerance,
        expected_boxes={"widget": ExpectedBox(100.0, 100.0, 200.0, 200.0)},
    )


def _sample_detection(cx: float, cy: float, name: str = "widget"):
    return {
        "class": name,
        "class_id": 0,
        "confidence": 0.9,
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "cx": cx,
        "cy": cy,
    }


def test_validate_detections_skips_when_disabled():
    cfg = PositionAreaConfig(
        enabled=False,
        tolerance=5.0,
        expected_boxes={"widget": ExpectedBox(100.0, 100.0, 200.0, 200.0)},
    )
    result = validate_detections_against_area([], cfg, 640, "prod", "area")

    assert result.status == "SKIPPED"
    assert isinstance(result, PositionValidationResult)


def test_validate_detections_marks_missing_and_unexpected():
    cfg = _make_config()
    detections = [_sample_detection(400.0, 400.0, name="alien")]

    result = validate_detections_against_area(detections, cfg, 640, "prod", "area")

    assert result.status == "FAIL"
    assert result.missing == ["widget"]
    assert "alien" in result.unexpected


def test_validate_detections_flags_wrong_location_with_tolerance_override():
    cfg = _make_config()
    detections = [_sample_detection(350.0, 350.0)]

    result = validate_detections_against_area(
        detections, cfg, 640, "prod", "area", tolerance_override=1.0
    )

    assert result.status == "FAIL"
    assert result.wrong == ["widget"]
    entry = next(item for item in result.results if item["class"] == "widget")
    assert entry["status"] == "WRONG"
    assert "allowed_box" in entry


def test_validate_detections_passes_within_allowed_region():
    cfg = _make_config(tolerance=10.0)
    detections = [_sample_detection(150.0, 150.0)]

    result = validate_detections_against_area(detections, cfg, 640, "prod", "area")

    assert result.status == "PASS"
    assert result.wrong == []
    assert result.missing == []


# -----------------------------------------------------------------------
# New tests: statistical fields, multi-instance, per-class tolerance
# -----------------------------------------------------------------------


class TestExpectedBoxCenter:
    """Test ExpectedBox.center() with and without precomputed cx/cy."""

    def test_center_from_bbox(self):
        box = ExpectedBox(100.0, 200.0, 300.0, 400.0)
        assert box.center() == (200.0, 300.0)

    def test_center_precomputed(self):
        box = ExpectedBox(100.0, 200.0, 300.0, 400.0, cx=195.5, cy=298.3)
        assert box.center() == (195.5, 298.3)

    def test_statistical_fields_defaults(self):
        box = ExpectedBox(0.0, 0.0, 10.0, 10.0)
        assert box.sigma_cx == 0.0
        assert box.sigma_cy == 0.0
        assert box.count == 0
        assert box.tolerance is None


class TestBaseClassName:
    """Test _base_class_name stripping #N suffix."""

    def test_no_suffix(self):
        assert _base_class_name("Black") == "Black"

    def test_indexed(self):
        assert _base_class_name("Black#0") == "Black"
        assert _base_class_name("Black#1") == "Black"
        assert _base_class_name("LED#12") == "LED"

    def test_hash_in_name_non_numeric(self):
        assert _base_class_name("C#Sharp") == "C#Sharp"

    def test_hash_at_start(self):
        assert _base_class_name("#0") == "#0"  # # at position 0, idx <= 0


class TestMultiInstanceMatching:
    """Test multi-instance validation with #N indexed keys."""

    def test_two_instances_both_correct(self):
        cfg = PositionAreaConfig(
            enabled=True,
            tolerance=10.0,
            expected_boxes={
                "Black#0": ExpectedBox(100.0, 100.0, 200.0, 200.0),
                "Black#1": ExpectedBox(400.0, 400.0, 500.0, 500.0),
            },
        )
        detections = [
            _sample_detection(150.0, 150.0, name="Black"),
            _sample_detection(450.0, 450.0, name="Black"),
        ]
        result = validate_detections_against_area(detections, cfg, 640, "P", "A")

        assert result.status == "PASS"
        assert result.missing == []
        assert result.wrong == []
        correct_count = sum(1 for r in result.results if r["status"] == "CORRECT")
        assert correct_count == 2

    def test_two_instances_one_missing(self):
        cfg = PositionAreaConfig(
            enabled=True,
            tolerance=10.0,
            expected_boxes={
                "Black#0": ExpectedBox(100.0, 100.0, 200.0, 200.0),
                "Black#1": ExpectedBox(400.0, 400.0, 500.0, 500.0),
            },
        )
        detections = [_sample_detection(150.0, 150.0, name="Black")]
        result = validate_detections_against_area(detections, cfg, 640, "P", "A")

        assert result.status == "FAIL"
        assert len(result.missing) == 1

    def test_two_instances_one_wrong(self):
        cfg = PositionAreaConfig(
            enabled=True,
            tolerance=2.0,  # tight tolerance
            expected_boxes={
                "LED#0": ExpectedBox(100.0, 100.0, 120.0, 120.0),
                "LED#1": ExpectedBox(300.0, 300.0, 320.0, 320.0),
            },
        )
        detections = [
            _sample_detection(110.0, 110.0, name="LED"),  # correct
            _sample_detection(500.0, 500.0, name="LED"),  # way off
        ]
        result = validate_detections_against_area(detections, cfg, 640, "P", "A")

        assert result.status == "FAIL"
        assert len(result.wrong) == 1

    def test_greedy_matches_nearest(self):
        """Greedy matcher should pair each detection to the nearest expected box."""
        cfg = PositionAreaConfig(
            enabled=True,
            tolerance=5.0,
            expected_boxes={
                "X#0": ExpectedBox(90.0, 90.0, 110.0, 110.0),   # center (100, 100)
                "X#1": ExpectedBox(290.0, 290.0, 310.0, 310.0),  # center (300, 300)
            },
        )
        # Detection near X#1 first, then near X#0 — order shouldn't matter
        detections = [
            _sample_detection(300.0, 300.0, name="X"),
            _sample_detection(100.0, 100.0, name="X"),
        ]
        result = validate_detections_against_area(detections, cfg, 640, "P", "A")
        assert result.status == "PASS"
        assert result.wrong == []


class TestPerClassTolerance:
    """Test per-class tolerance override via ExpectedBox.tolerance."""

    def test_per_class_tighter_tolerance_causes_fail(self):
        cfg = PositionAreaConfig(
            enabled=True,
            tolerance=20.0,  # global: generous (128px expansion)
            expected_boxes={
                "widget": ExpectedBox(
                    100.0, 100.0, 200.0, 200.0,
                    tolerance=1.0,  # per-class: tight (6.4px expansion)
                ),
            },
        )
        # Detection outside box + 6.4px but inside box + 128px
        detections = [_sample_detection(220.0, 220.0)]
        result = validate_detections_against_area(detections, cfg, 640, "P", "A")

        assert result.status == "FAIL"
        assert result.wrong == ["widget"]

    def test_per_class_generous_tolerance_allows_pass(self):
        cfg = PositionAreaConfig(
            enabled=True,
            tolerance=1.0,  # global: tight
            expected_boxes={
                "widget": ExpectedBox(
                    100.0, 100.0, 200.0, 200.0,
                    tolerance=30.0,  # per-class: generous
                ),
            },
        )
        detections = [_sample_detection(250.0, 250.0)]
        result = validate_detections_against_area(detections, cfg, 640, "P", "A")

        assert result.status == "PASS"


class TestLoadPositionConfig:
    """Tests for load_position_config() parsing."""

    def test_none_returns_empty(self):
        from picture_tool.position.yolo_position_validator import load_position_config
        assert load_position_config(None) == {}

    def test_dict_parses_basic_config(self):
        from picture_tool.position.yolo_position_validator import load_position_config
        raw = {
            "P1": {
                "A1": {
                    "enabled": True,
                    "tolerance": 5.0,
                    "expected_boxes": {
                        "LED": {"x1": 10, "y1": 20, "x2": 30, "y2": 40},
                    },
                }
            }
        }
        result = load_position_config(raw)
        assert "P1" in result
        assert "A1" in result["P1"]
        cfg = result["P1"]["A1"]
        assert cfg.enabled is True
        assert cfg.tolerance == 5.0
        assert "LED" in cfg.expected_boxes
        box = cfg.expected_boxes["LED"]
        assert (box.x1, box.y1, box.x2, box.y2) == (10.0, 20.0, 30.0, 40.0)

    def test_parses_statistical_fields(self):
        from picture_tool.position.yolo_position_validator import load_position_config
        raw = {
            "P": {
                "A": {
                    "expected_boxes": {
                        "W": {
                            "x1": 100, "y1": 100, "x2": 200, "y2": 200,
                            "cx": 148.5, "cy": 151.2,
                            "sigma_cx": 3.1, "sigma_cy": 2.8,
                            "count": 20,
                            "tolerance": 8.0,
                        }
                    }
                }
            }
        }
        result = load_position_config(raw)
        box = result["P"]["A"].expected_boxes["W"]
        assert box.cx == 148.5
        assert box.cy == 151.2
        assert box.sigma_cx == 3.1
        assert box.sigma_cy == 2.8
        assert box.count == 20
        assert box.tolerance == 8.0

    def test_parses_indexed_keys(self):
        from picture_tool.position.yolo_position_validator import load_position_config
        raw = {
            "P": {
                "A": {
                    "expected_boxes": {
                        "Black#0": {"x1": 10, "y1": 10, "x2": 50, "y2": 50},
                        "Black#1": {"x1": 100, "y1": 100, "x2": 150, "y2": 150},
                    }
                }
            }
        }
        result = load_position_config(raw)
        boxes = result["P"]["A"].expected_boxes
        assert "Black#0" in boxes
        assert "Black#1" in boxes

    def test_default_mode_is_center(self):
        from picture_tool.position.yolo_position_validator import load_position_config
        raw = {"P": {"A": {"expected_boxes": {}}}}
        result = load_position_config(raw)
        assert result["P"]["A"].mode == "center"

    def test_non_mapping_raises_type_error(self):
        from picture_tool.position.yolo_position_validator import load_position_config
        with pytest.raises(TypeError):
            load_position_config([1, 2, 3])

    def test_malformed_box_skipped(self):
        from picture_tool.position.yolo_position_validator import load_position_config
        raw = {
            "P": {
                "A": {
                    "expected_boxes": {
                        "Good": {"x1": 10, "y1": 10, "x2": 50, "y2": 50},
                        "Bad": {"x1": "not_a_number"},  # will fail float()
                    }
                }
            }
        }
        result = load_position_config(raw)
        boxes = result["P"]["A"].expected_boxes
        assert "Good" in boxes
        assert "Bad" not in boxes

    def test_skips_non_mapping_areas(self):
        from picture_tool.position.yolo_position_validator import load_position_config
        raw = {"P": {"A": "not_a_mapping", "B": {"expected_boxes": {}}}}
        result = load_position_config(raw)
        assert "A" not in result.get("P", {})
        assert "B" in result["P"]

    def test_yaml_file_source(self, tmp_path):
        from picture_tool.position.yolo_position_validator import load_position_config
        import yaml as _yaml
        cfg_file = tmp_path / "pos.yaml"
        cfg_file.write_text(_yaml.safe_dump({
            "P": {"A": {"tolerance": 3.0, "expected_boxes": {"X": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}}}
        }), encoding="utf-8")
        result = load_position_config(str(cfg_file))
        assert result["P"]["A"].tolerance == 3.0


class TestGreedyMatchEdgeCases:
    """Edge cases for _greedy_match()."""

    def test_both_empty(self):
        from picture_tool.position.yolo_position_validator import _greedy_match
        matched, unmatched_keys, unmatched_dets = _greedy_match([], [], 640, 10.0)
        assert matched == []
        assert unmatched_keys == []
        assert unmatched_dets == []

    def test_no_detections(self):
        from picture_tool.position.yolo_position_validator import _greedy_match
        entries = [("W#0", ExpectedBox(90, 90, 110, 110)), ("W#1", ExpectedBox(200, 200, 220, 220))]
        matched, unmatched_keys, unmatched_dets = _greedy_match(entries, [], 640, 10.0)
        assert matched == []
        assert unmatched_keys == ["W#0", "W#1"]
        assert unmatched_dets == []

    def test_no_expected(self):
        from picture_tool.position.yolo_position_validator import _greedy_match
        dets = [_sample_detection(100, 100)]
        matched, unmatched_keys, unmatched_dets = _greedy_match([], dets, 640, 10.0)
        assert matched == []
        assert unmatched_keys == []
        assert len(unmatched_dets) == 1

    def test_more_detections_than_expected(self):
        from picture_tool.position.yolo_position_validator import _greedy_match
        entries = [("W#0", ExpectedBox(90, 90, 110, 110))]
        dets = [_sample_detection(100, 100), _sample_detection(300, 300)]
        matched, _, unmatched_dets = _greedy_match(entries, dets, 640, 10.0)
        assert len(matched) == 1
        assert len(unmatched_dets) == 1

    def test_more_expected_than_detections(self):
        from picture_tool.position.yolo_position_validator import _greedy_match
        entries = [
            ("W#0", ExpectedBox(90, 90, 110, 110)),
            ("W#1", ExpectedBox(200, 200, 220, 220)),
            ("W#2", ExpectedBox(300, 300, 320, 320)),
        ]
        dets = [_sample_detection(100, 100)]
        matched, unmatched_keys, _ = _greedy_match(entries, dets, 640, 10.0)
        assert len(matched) == 1
        assert len(unmatched_keys) == 2

    def test_detection_missing_cx_cy_skipped(self):
        from picture_tool.position.yolo_position_validator import _greedy_match
        entries = [("W#0", ExpectedBox(90, 90, 110, 110))]
        bad_det = {"class": "W", "class_id": 0, "confidence": 0.9, "bbox": [0, 0, 10, 10]}
        matched, unmatched_keys, unmatched_dets = _greedy_match(entries, [bad_det], 640, 10.0)
        assert len(matched) == 0
        assert unmatched_keys == ["W#0"]
        assert len(unmatched_dets) == 1


class TestValidateEdgeCases:
    """Edge cases for validate_detections_against_area()."""

    def test_invalid_mode_raises(self):
        cfg = PositionAreaConfig(
            enabled=True, mode="invalid_mode", tolerance=5.0, expected_boxes={}
        )
        with pytest.raises(ValueError, match="Unsupported"):
            validate_detections_against_area([], cfg, 640, "P", "A")

    def test_empty_detections_all_missing(self):
        cfg = _make_config()
        result = validate_detections_against_area([], cfg, 640, "P", "A")
        assert result.status == "FAIL"
        assert result.missing == ["widget"]

    def test_detection_without_cx_cy_is_unknown(self):
        cfg = _make_config()
        bad_det = {"class": "widget", "class_id": 0, "confidence": 0.9, "bbox": [0, 0, 10, 10]}
        # no cx/cy
        result = validate_detections_against_area([bad_det], cfg, 640, "P", "A")
        assert result.unknown == ["widget"]

    def test_multiple_unexpected_classes(self):
        cfg = _make_config()
        dets = [
            _sample_detection(100, 100, name="alien1"),
            _sample_detection(200, 200, name="alien2"),
        ]
        result = validate_detections_against_area(dets, cfg, 640, "P", "A")
        assert "alien1" in result.unexpected
        assert "alien2" in result.unexpected
        assert "widget" in result.missing

    def test_extra_multi_instance_detections_unexpected(self):
        """3 detections for 2 expected slots → 1 unexpected."""
        cfg = PositionAreaConfig(
            enabled=True, tolerance=10.0,
            expected_boxes={
                "B#0": ExpectedBox(90, 90, 110, 110),
                "B#1": ExpectedBox(200, 200, 220, 220),
            },
        )
        dets = [
            _sample_detection(100, 100, name="B"),
            _sample_detection(210, 210, name="B"),
            _sample_detection(500, 500, name="B"),  # extra
        ]
        result = validate_detections_against_area(dets, cfg, 640, "P", "A")
        assert len(result.unexpected) == 1
        assert "B" in result.unexpected


class TestImgszValueExtended:
    """Extended edge cases for _imgsz_value()."""

    def test_direct_integer(self):
        assert _imgsz_value(640) == 640

    def test_direct_string_number(self):
        assert _imgsz_value("512") == 512

    def test_single_element_list(self):
        assert _imgsz_value([1024]) == 1024

    def test_none_raises(self):
        with pytest.raises(ValueError):
            _imgsz_value(None)

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            _imgsz_value(0)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _imgsz_value("")


class TestExpectedBoxPartialCenter:
    """Partial precomputed cx or cy only."""

    def test_only_cx_precomputed(self):
        box = ExpectedBox(100.0, 200.0, 300.0, 400.0, cx=195.5)
        cx, cy = box.center()
        assert cx == 195.5
        assert cy == 300.0  # (200+400)/2

    def test_only_cy_precomputed(self):
        box = ExpectedBox(100.0, 200.0, 300.0, 400.0, cy=295.0)
        cx, cy = box.center()
        assert cx == 200.0  # (100+300)/2
        assert cy == 295.0


class TestCenterModeAccepted:
    """Test that mode='center' is accepted (not just 'bbox')."""

    def test_center_mode_passes(self):
        cfg = PositionAreaConfig(
            enabled=True,
            mode="center",
            tolerance=10.0,
            expected_boxes={"widget": ExpectedBox(100.0, 100.0, 200.0, 200.0)},
        )
        detections = [_sample_detection(150.0, 150.0)]
        result = validate_detections_against_area(detections, cfg, 640, "P", "A")
        assert result.status == "PASS"
