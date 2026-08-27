import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from picture_tool.color import color_verifier


def _make_color_stats(path: Path, mapping: dict[str, tuple[int, int, int]]) -> Path:
    summary = {}
    for name, bgr in mapping.items():
        patch = np.zeros((1, 1, 3), dtype=np.uint8)
        patch[0, 0] = np.array(bgr, dtype=np.uint8)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[0, 0].astype(float).tolist()
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[0, 0].astype(float).tolist()
        summary[name] = {
            "count": 1,
            "hsv_mean": hsv,
            "hsv_min": hsv,
            "hsv_max": hsv,
            "lab_mean": lab,
            "lab_min": lab,
            "lab_max": lab,
        }
    payload = {"summary": summary, "items": []}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _roi(
    path: Path, color: tuple[int, int, int], bg: tuple[int, int, int] = (120, 120, 120)
) -> None:
    canvas = np.full((80, 80, 3), bg, dtype=np.uint8)
    cv2.rectangle(canvas, (20, 20), (60, 60), color, thickness=-1)
    cv2.imwrite(str(path), canvas)


def test_verify_directory_matches_expected_colors(tmp_path):
    stats_path = tmp_path / "color_stats.json"
    _make_color_stats(stats_path, {"Orange": (0, 140, 255), "Green": (0, 255, 0)})

    orange_dir = tmp_path / "orange"
    orange_dir.mkdir()
    _roi(orange_dir / "Cable_orange.png", (0, 140, 255))
    _, orange_results = color_verifier.verify_directory(
        input_dir=orange_dir,
        color_stats=stats_path,
        output_json=None,
        output_csv=None,
        infer_expected_from_name=False,
    )
    assert orange_results[0].predicted_color == "Orange"

    green_dir = tmp_path / "green"
    green_dir.mkdir()
    _roi(green_dir / "Cable_green.png", (0, 255, 0))
    _, green_results = color_verifier.verify_directory(
        input_dir=green_dir,
        color_stats=stats_path,
        output_json=None,
        output_csv=None,
        orientation="vertical",
        edge_margin=0.0,
        sat_threshold=10,
        val_threshold=250,
    )
    assert green_results[0].predicted_color == "Green"


def test_strip_segments_handles_thin_wire(tmp_path):
    stats_path = tmp_path / "color_stats.json"
    _make_color_stats(stats_path, {"Orange": (0, 120, 255), "Black": (0, 0, 0)})

    input_dir = tmp_path / "thin"
    input_dir.mkdir()
    canvas = np.full((60, 120, 3), (120, 120, 120), dtype=np.uint8)
    cv2.rectangle(canvas, (55, 5), (65, 55), (0, 120, 255), -1)
    cv2.imwrite(str(input_dir / "wire.png"), canvas)

    summary, results = color_verifier.verify_directory(
        input_dir=input_dir,
        color_stats=stats_path,
        output_json=None,
        output_csv=None,
        orientation="vertical",
        segments=8,
        min_strip_ratio=0.05,
        edge_margin=0.05,
        sat_threshold=5,
        val_threshold=255,
    )

    assert summary["predicted_only"] == 1
    assert results[0].predicted_color == "Orange"


def test_ratio_threshold_flags_low_confidence(tmp_path):
    stats_path = tmp_path / "color_stats.json"
    _make_color_stats(stats_path, {"Orange": (0, 140, 255)})
    input_dir = tmp_path / "noisy"
    input_dir.mkdir()
    canvas = np.full((80, 80, 3), (120, 120, 120), dtype=np.uint8)
    cv2.rectangle(canvas, (25, 25), (55, 55), (0, 140, 255), -1)
    cv2.imwrite(str(input_dir / "roi.png"), canvas)

    summary, results = color_verifier.verify_directory(
        input_dir=input_dir,
        color_stats=stats_path,
        output_json=None,
        output_csv=None,
        ratio_threshold=0.9,
        sat_threshold=80,
        infer_expected_from_name=False,
    )

    assert summary["low_confidence"] == 1
    assert results[0].status in {"low_confidence", "predicted_only_low_conf"}


def test_edge_margin_filters_border(tmp_path):
    stats_path = tmp_path / "color_stats.json"
    _make_color_stats(stats_path, {"Orange": (0, 140, 255)})
    input_dir = tmp_path / "edge"
    input_dir.mkdir()
    canvas = np.full((80, 80, 3), (0, 255, 0), dtype=np.uint8)
    cv2.rectangle(canvas, (5, 20), (15, 60), (0, 140, 255), -1)
    cv2.imwrite(str(input_dir / "edge.png"), canvas)

    summary, results = color_verifier.verify_directory(
        input_dir=input_dir,
        color_stats=stats_path,
        output_json=None,
        output_csv=None,
        edge_margin=0.2,
    )

    assert summary["predicted_only"] == 1
    assert results[0].confidence == 0.0


# ---------------------------------------------------------------------------
# Regressions from the color-detection review
# ---------------------------------------------------------------------------


def _flat_ranges():
    from picture_tool.color.strategies.base import ColorRange

    def _one(name):
        return ColorRange(
            name,
            np.array([0.0, 0.0, 0.0]),
            np.array([180.0, 255.0, 255.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([255.0, 255.0, 255.0]),
            hsv_mean=np.array([30.0, 100.0, 100.0]),
            lab_mean=np.array([100.0, 120.0, 110.0]),
        )

    return {name: _one(name) for name in ("Black", "Green", "Orange", "Red", "Yellow")}


def test_too_few_valid_pixels_does_not_invent_a_black_verdict():
    """A washed-out region carries no color evidence.

    It used to answer with a hard-coded ``Black: 0.7`` -- above black's own
    0.45 threshold -- so an unlit or overexposed image passed the color gate.
    """
    import numpy as np

    from picture_tool.color import color_verifier

    grey = np.full((40, 40, 3), 128, dtype=np.uint8)
    hsv = cv2.cvtColor(grey, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(grey, cv2.COLOR_BGR2LAB).astype(np.float32)

    ratios, _masks, debug = color_verifier._evaluate_image_improved(
        grey, hsv, lab, _flat_ranges()
    )

    assert debug.get("insufficient_pixels") is True
    assert set(ratios) == {"Black", "Green", "Orange", "Red", "Yellow"}
    assert all(value == 0.0 for value in ratios.values())


def test_post_correction_is_skipped_when_there_is_no_evidence():
    """With every ratio at 0 the Orange/Red tie test is trivially satisfied."""
    import numpy as np

    from picture_tool.color import color_verifier

    context = color_verifier.DecisionContext(
        ratios=dict.fromkeys(("Black", "Orange", "Red"), 0.0),
        debug_info={"insufficient_pixels": True},
        hsv_img=np.full((40, 40, 3), 128, dtype=np.float32),
        lab_img=np.full((40, 40, 3), 128, dtype=np.float32),
    )

    color, confidence = color_verifier._apply_color_rules("Orange", 0.0, context)

    assert (color, confidence) == ("Orange", 0.0)
    assert "post_corrected" not in context.debug_info


def test_baseline_hue_is_aggregated_on_the_circle(tmp_path):
    """Both aggregation levels used a linear mean over a periodic channel."""
    import numpy as np

    from picture_tool.color.color_inspection import (
        ColorStatsRecorder,
        compute_hsv_lab_stats,
    )

    hsv = np.zeros((20, 20, 3), np.uint8)
    hsv[:10, :, 0] = 3
    hsv[10:, :, 0] = 178
    hsv[:, :, 1] = 200
    hsv[:, :, 2] = 150
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    per_image = compute_hsv_lab_stats(image, np.ones((20, 20), np.uint8))
    assert per_image["hsv_mean"][0] == pytest.approx(0.5, abs=0.6)

    recorder = ColorStatsRecorder()
    for hue in (3.0, 178.0):
        recorder.record(
            tmp_path / "x.png", "Red",
            [hue, 200, 150], [0, 0, 0], [180, 255, 255],
            [50, 150, 150], [0, 0, 0], [255, 255, 255],
            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], 1.0,
        )
    aggregate = recorder.to_json()["summary"]["Red"]["hsv_mean"][0]
    assert aggregate == pytest.approx(0.5, abs=1e-6)
