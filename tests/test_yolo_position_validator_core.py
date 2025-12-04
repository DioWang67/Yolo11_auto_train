import pytest

from picture_tool.position.yolo_position_validator import (
    ExpectedBox,
    PositionAreaConfig,
    PositionValidationResult,
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
