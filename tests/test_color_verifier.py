import json
from pathlib import Path

import cv2
import numpy as np

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


def _roi(path: Path, color: tuple[int, int, int], bg: tuple[int, int, int] = (120, 120, 120)) -> None:
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
