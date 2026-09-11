"""Read-only collection of production inspection records.

The load-bearing tests here are the robustness ones: a corrupt, truncated or
half-written record must cost the pass one sample and nothing else, and the
collector must never write into the production tree.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is a pinned runtime dependency
    cv2 = None

from picture_tool.autotrain import AutoTrainDisabledError
from picture_tool.autotrain.collector import (
    collect_from_results_root,
    collect_production_records,
)
from picture_tool.autotrain.config import AutoTrainConfig
from picture_tool.autotrain.paths import AutoTrainPaths
from picture_tool.workspace_paths import WorkspacePaths

PRODUCT = "Cable1"
AREA = "A"
REFERENCE = datetime(2026, 9, 10, 12, 0, 0)


# ---------------------------------------------------------------------------
# Fixture tree helpers


def _snapshot_payload(inspection_id: str, *, confidences, status="PASS", image=""):
    return {
        "schema_version": 2,
        "inspection_id": inspection_id,
        "timestamp": REFERENCE.isoformat(),
        "status": status,
        "detector": "yolo",
        "product": PRODUCT,
        "area": AREA,
        "equipment": {"machine_id": "line-1", "station": AREA, "camera_id": "cam0"},
        "fail_reasons": ["COLOR_MISMATCH"] if status != "PASS" else [],
        "model_info": {
            "weights": "models/Cable1/A/yolo/weights/best.onnx",
            "model_version": "1.2.0",
            "conf_thres": 0.4,
            "class_names": ["Black", "Green", "Orange", "Red", "Yellow"],
        },
        "inference_time": 0.031,
        "detections": [
            {"class": "Red", "confidence": score, "bbox": [1.0, 2.0, 3.0, 4.0]}
            for score in confidences
        ],
        "artifacts": {"original_path": image, "preprocessed_path": "", "annotated_path": ""},
        "config_hash": "abc123def456",
        "config": {"exposure_time": "9687.0", "gain": "23.0", "light_brightness": 120},
    }


def _write_snapshot(results_root: Path, payload, *, day="20260910", status="PASS") -> Path:
    directory = results_root / day / PRODUCT / AREA / status / "metadata" / "yolo"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{payload['inspection_id']}_config_snapshot.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((16, 16, 3), dtype=np.uint8)
    array[:, :8] = (10, 20, 200)
    assert cv2 is not None
    cv2.imwrite(str(path), array)
    return path


def _collect(results_root: Path, **overrides):
    kwargs = {
        "product": PRODUCT,
        "area": AREA,
        "lookback_days": 7,
        "max_records": 100,
        "compute_image_quality": False,
        "now": REFERENCE,
    }
    kwargs.update(overrides)
    return collect_from_results_root(results_root, **kwargs)


# ---------------------------------------------------------------------------
# Happy path


def test_reads_every_field_the_request_asked_to_preserve(tmp_path):
    results = tmp_path / "Result"
    _write_snapshot(results, _snapshot_payload("insp-1", confidences=[0.91]))

    result = _collect(results)

    assert len(result.records) == 1
    record = result.records[0]
    assert record.inspection_id == "insp-1"
    assert record.timestamp == REFERENCE
    assert record.model_version == "1.2.0"
    assert record.model_weights.endswith("best.onnx")
    assert record.conf_threshold == pytest.approx(0.4)
    assert record.class_names == ("Black", "Green", "Orange", "Red", "Yellow")
    assert record.detections[0].class_name == "Red"
    assert record.detections[0].confidence == pytest.approx(0.91)
    assert record.detections[0].bbox == (1.0, 2.0, 3.0, 4.0)
    assert record.equipment["camera_id"] == "cam0"
    assert record.config_hash == "abc123def456"


def test_camera_parameters_come_from_the_embedded_config(tmp_path):
    results = tmp_path / "Result"
    _write_snapshot(results, _snapshot_payload("insp-1", confidences=[0.5]))

    record = _collect(results).records[0]

    assert record.camera_parameters == {
        "exposure_time": "9687.0",
        "gain": "23.0",
        "light_brightness": 120,
    }


def test_min_confidence_ignores_records_without_predictions(tmp_path):
    results = tmp_path / "Result"
    _write_snapshot(results, _snapshot_payload("with", confidences=[0.9, 0.3, 0.6]))
    _write_snapshot(results, _snapshot_payload("without", confidences=[]))

    by_id = {r.inspection_id: r for r in _collect(results).records}

    assert by_id["with"].min_confidence == pytest.approx(0.3)
    assert by_id["without"].min_confidence is None


def test_only_the_configured_station_is_read(tmp_path):
    results = tmp_path / "Result"
    _write_snapshot(results, _snapshot_payload("mine", confidences=[0.5]))
    other = results / "20260910" / "PCBA1" / "B" / "PASS" / "metadata" / "yolo"
    other.mkdir(parents=True)
    (other / "theirs_config_snapshot.json").write_text(
        json.dumps(_snapshot_payload("theirs", confidences=[0.5])), encoding="utf-8"
    )

    result = _collect(results)

    assert [r.inspection_id for r in result.records] == ["mine"]


def test_lookback_window_excludes_older_days(tmp_path):
    results = tmp_path / "Result"
    _write_snapshot(results, _snapshot_payload("recent", confidences=[0.5]), day="20260910")
    _write_snapshot(results, _snapshot_payload("old", confidences=[0.5]), day="20260801")

    result = _collect(results, lookback_days=3)

    assert [r.inspection_id for r in result.records] == ["recent"]


def test_max_records_bounds_the_pass(tmp_path):
    results = tmp_path / "Result"
    for index in range(5):
        _write_snapshot(results, _snapshot_payload(f"insp-{index}", confidences=[0.5]))

    result = _collect(results, max_records=2)

    assert len(result.records) == 2


def test_both_pass_and_fail_directories_are_read(tmp_path):
    results = tmp_path / "Result"
    _write_snapshot(results, _snapshot_payload("ok", confidences=[0.9]), status="PASS")
    _write_snapshot(
        results,
        _snapshot_payload("ng", confidences=[0.4], status="DETECTION_FAIL"),
        status="DETECTION_FAIL",
    )

    statuses = {r.inspection_id: r.status for r in _collect(results).records}

    assert statuses == {"ok": "PASS", "ng": "DETECTION_FAIL"}


# ---------------------------------------------------------------------------
# Robustness: a bad record costs one sample, not the pass


@pytest.mark.parametrize(
    "content",
    [
        "{ not json at all",
        '{"inspection_id": "x", "detections": "not-a-list"}',
        "[]",
        "",
    ],
    ids=["truncated", "wrong-types", "not-a-mapping", "empty"],
)
def test_a_broken_record_is_logged_and_skipped(tmp_path, content):
    results = tmp_path / "Result"
    _write_snapshot(results, _snapshot_payload("good", confidences=[0.5]))
    directory = results / "20260910" / PRODUCT / AREA / "PASS" / "metadata" / "yolo"
    (directory / "broken_config_snapshot.json").write_text(content, encoding="utf-8")

    result = _collect(results)

    assert [r.inspection_id for r in result.records] == ["good"]
    assert result.scanned_files == 2
    assert len(result.unreadable) == 1


def test_missing_results_root_yields_nothing_rather_than_raising(tmp_path):
    result = _collect(tmp_path / "does-not-exist")

    assert result.records == ()
    assert result.scanned_files == 0


def test_records_whose_image_is_gone_are_reported_not_dropped(tmp_path):
    """Retention may delete the image before a cycle runs; say so explicitly."""
    results = tmp_path / "Result"
    _write_snapshot(
        results,
        _snapshot_payload("gone", confidences=[0.5], image=str(tmp_path / "nope.jpg")),
    )

    result = _collect(results)

    assert len(result.records) == 1
    assert result.records[0].has_image is False
    assert result.missing_images == ("gone",)


def test_collection_never_writes_into_the_production_tree(tmp_path):
    results = tmp_path / "Result"
    image = _write_image(tmp_path / "images" / "insp-1.jpg")
    _write_snapshot(results, _snapshot_payload("insp-1", confidences=[0.5], image=str(image)))
    before = sorted(p.relative_to(results).as_posix() for p in results.rglob("*"))

    _collect(results, compute_image_quality=True)

    after = sorted(p.relative_to(results).as_posix() for p in results.rglob("*"))
    assert before == after


# ---------------------------------------------------------------------------
# Image quality


@pytest.mark.skipif(cv2 is None, reason="cv2 is required for image quality")
def test_image_quality_is_measured_offline_when_the_image_exists(tmp_path):
    results = tmp_path / "Result"
    image = _write_image(tmp_path / "images" / "insp-1.jpg")
    _write_snapshot(results, _snapshot_payload("insp-1", confidences=[0.5], image=str(image)))

    record = _collect(results, compute_image_quality=True).records[0]

    assert record.image_quality is not None
    assert record.image_quality.brightness > 0
    assert record.image_quality.saturation > 0
    assert record.image_quality.blur_score >= 0


def test_image_quality_is_skipped_when_switched_off(tmp_path):
    results = tmp_path / "Result"
    image = _write_image(tmp_path / "images" / "insp-1.jpg")
    _write_snapshot(results, _snapshot_payload("insp-1", confidences=[0.5], image=str(image)))

    record = _collect(results, compute_image_quality=False).records[0]

    assert record.image_quality is None


# ---------------------------------------------------------------------------
# Review outcomes from the production database


def _write_inspection_db(path: Path, rows) -> Path:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE inspections (inspection_id TEXT PRIMARY KEY, "
            "review_outcome TEXT, review_label TEXT, failure_category TEXT, "
            "action_route TEXT)"
        )
        connection.executemany(
            "INSERT INTO inspections VALUES (?, ?, ?, ?, ?)", rows
        )
        connection.commit()
    finally:
        connection.close()
    return path


def test_review_outcomes_are_attached_from_the_production_database(tmp_path):
    results = tmp_path / "Result"
    _write_snapshot(results, _snapshot_payload("insp-1", confidences=[0.5]))
    database = _write_inspection_db(
        tmp_path / "inspection_records.sqlite3",
        [("insp-1", "false_reject", "Red", "COLOR", "color")],
    )

    record = _collect(results, database_path=database).records[0]

    assert record.review_outcome == "false_reject"
    assert record.failure_category == "COLOR"
    assert record.action_route == "color"


def test_a_missing_database_degrades_to_no_review_information(tmp_path):
    results = tmp_path / "Result"
    _write_snapshot(results, _snapshot_payload("insp-1", confidences=[0.5]))

    result = _collect(results, database_path=tmp_path / "absent.sqlite3")

    assert result.records[0].review_outcome == ""
    assert result.review_lookup_error == ""


def test_a_broken_database_is_reported_without_failing_the_pass(tmp_path):
    results = tmp_path / "Result"
    _write_snapshot(results, _snapshot_payload("insp-1", confidences=[0.5]))
    corrupt = tmp_path / "inspection_records.sqlite3"
    corrupt.write_text("this is not a database", encoding="utf-8")

    result = _collect(results, database_path=corrupt)

    assert len(result.records) == 1
    assert result.review_lookup_error != ""


def test_the_production_database_is_opened_read_only(tmp_path):
    """A read-only URI is what stops this from locking or migrating the DB."""
    results = tmp_path / "Result"
    _write_snapshot(results, _snapshot_payload("insp-1", confidences=[0.5]))
    database = _write_inspection_db(
        tmp_path / "inspection_records.sqlite3", [("insp-1", "ok", "", "", "")]
    )
    before = database.read_bytes()

    _collect(results, database_path=database)

    assert database.read_bytes() == before


# ---------------------------------------------------------------------------
# Feature flag


def test_collection_refuses_while_the_feature_is_disabled(tmp_path):
    config_path = tmp_path / "autonomous_training.yaml"
    config_path.write_text(
        yaml.safe_dump({"autonomous_training": {"enabled": False}}), encoding="utf-8"
    )
    config = AutoTrainConfig.load(config_path)
    paths = AutoTrainPaths.from_workspace(WorkspacePaths.discover())

    with pytest.raises(AutoTrainDisabledError):
        collect_production_records(paths, config)
