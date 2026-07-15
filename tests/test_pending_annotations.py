import csv
import hashlib
import json
from pathlib import Path

import pytest

from picture_tool.pending_annotations import (
    PENDING_FIELDS,
    PendingAnnotationError,
    configure_pending_workspace,
    promote_completed_pending,
    record_label_verification,
)


def _pending_case(
    tmp_path: Path,
    label_text: str | None,
    *,
    review_label: str = "false_negative",
    reason: str = "missed_detection_requires_box_annotation",
    label_baseline_sha256: str = "",
) -> tuple[Path, Path]:
    dataset = tmp_path / "data" / "Cable1" / "A"
    images, labels, _classes = configure_pending_workspace(dataset, ["Black", "Red"])
    image = images / "review_sample1.jpg"
    image.write_bytes(b"image")
    label = labels / "review_sample1.txt"
    if label_text is not None:
        label.write_text(label_text, encoding="utf-8")
    manifest = dataset / "review_pending" / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PENDING_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample1",
                "image_sha256": "",
                "product": "Cable1",
                "area": "A",
                "review_label": review_label,
                "reason": reason,
                "annotation_status": "pending",
                "source_image": str(image),
                "output_image": str(image),
                "output_label": str(label),
                "label_baseline_sha256": label_baseline_sha256,
                "config_snapshot_path": "case.json",
            }
        )
    handoff = tmp_path / "data" / ".operator_handoff" / "latest.json"
    handoff.parent.mkdir(parents=True)
    handoff.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "ready_count": 0,
                "pending_count": 1,
                "targets": [
                    {
                        "product": "Cable1",
                        "area": "A",
                        "dataset_root": str(dataset.resolve()),
                        "ready_count": 0,
                        "total_ready_count": 0,
                        "pending_count": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return dataset, handoff


def test_configure_pending_workspace_writes_labelimg_yolo_classes(tmp_path):
    dataset = tmp_path / "data" / "Cable1" / "A"

    _images, labels, predefined = configure_pending_workspace(
        dataset, ["Black", "Green"]
    )

    expected = "Black\nGreen\n"
    assert predefined.read_text(encoding="utf-8") == expected
    assert (labels / "classes.txt").read_text(encoding="utf-8") == expected


def test_promote_completed_pending_moves_valid_annotation_to_raw(tmp_path):
    dataset, handoff = _pending_case(tmp_path, "0 0.5 0.5 0.2 0.2\n")

    report = promote_completed_pending(dataset, ["Black", "Red"], handoff)

    assert report.promoted_count == 1
    assert report.remaining_count == 0
    assert (dataset / "raw" / "images" / "review_sample1.jpg").is_file()
    assert (
        dataset / "raw" / "labels" / "review_sample1.txt"
    ).read_text(encoding="utf-8").startswith("0 ")
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    assert payload["targets"][0]["pending_count"] == 0
    assert payload["targets"][0]["total_ready_count"] == 1


def test_promote_pending_keeps_case_without_saved_label(tmp_path):
    dataset, handoff = _pending_case(tmp_path, None)

    report = promote_completed_pending(dataset, ["Black", "Red"], handoff)

    assert report.promoted_count == 0
    assert report.remaining_count == 1
    assert not (dataset / "raw" / "images").exists()


def test_promote_pending_rejects_empty_false_negative_label(tmp_path):
    dataset, handoff = _pending_case(tmp_path, "")

    with pytest.raises(PendingAnnotationError, match="至少框選一個"):
        promote_completed_pending(dataset, ["Black", "Red"], handoff)

    assert not (dataset / "raw" / "images").exists()


def test_promote_pending_accepts_explicit_verified_empty_background(tmp_path):
    dataset, handoff = _pending_case(
        tmp_path,
        "",
        review_label="verified_empty",
        reason="operator_verified_empty_background",
    )

    report = promote_completed_pending(dataset, ["Black", "Red"], handoff)

    assert report.promoted_count == 1
    with (
        dataset / "metadata" / "review_dataset_manifest.csv"
    ).open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["annotation_status"] == "verified_empty"


def test_promote_pending_rejects_out_of_range_class(tmp_path):
    dataset, handoff = _pending_case(tmp_path, "9 0.5 0.5 0.2 0.2\n")

    with pytest.raises(PendingAnnotationError, match="out of range"):
        promote_completed_pending(dataset, ["Black", "Red"], handoff)

    assert (dataset / "review_pending" / "images" / "review_sample1.jpg").is_file()


def test_promote_pending_rejects_non_finite_coordinates(tmp_path):
    dataset, handoff = _pending_case(tmp_path, "0 nan 0.5 0.2 0.2\n")

    with pytest.raises(PendingAnnotationError, match="out of range"):
        promote_completed_pending(dataset, ["Black", "Red"], handoff)

    assert (dataset / "review_pending" / "images" / "review_sample1.jpg").is_file()


def test_promote_pending_keeps_unchanged_generated_draft(tmp_path):
    draft = "0 0.5 0.5 0.2 0.2\n"
    baseline = hashlib.sha256(draft.encode("utf-8")).hexdigest()
    dataset, handoff = _pending_case(
        tmp_path,
        draft,
        reason="false_detection_requires_correction",
        label_baseline_sha256=baseline,
    )

    report = promote_completed_pending(dataset, ["Black", "Red"], handoff)

    assert report.promoted_count == 0
    assert report.remaining_count == 1
    assert not (dataset / "raw" / "images").exists()


def test_promote_pending_accepts_operator_change_to_generated_draft(tmp_path):
    draft = "0 0.5 0.5 0.2 0.2\n"
    baseline = hashlib.sha256(draft.encode("utf-8")).hexdigest()
    dataset, handoff = _pending_case(
        tmp_path,
        "1 0.5 0.5 0.2 0.2\n",
        reason="wrong_class_requires_correction",
        label_baseline_sha256=baseline,
    )

    report = promote_completed_pending(dataset, ["Black", "Red"], handoff)

    assert report.promoted_count == 1
    assert report.remaining_count == 0


def test_promote_pending_accepts_explicit_save_when_content_is_unchanged(tmp_path):
    draft = "0 0.5 0.5 0.2 0.2\n"
    baseline = hashlib.sha256(draft.encode("utf-8")).hexdigest()
    dataset, handoff = _pending_case(
        tmp_path,
        draft,
        reason="false_detection_requires_correction",
        label_baseline_sha256=baseline,
    )
    label = dataset / "review_pending" / "labels" / "review_sample1.txt"
    receipt = record_label_verification(label, label.parent)

    report = promote_completed_pending(dataset, ["Black", "Red"], handoff)

    assert report.promoted_count == 1
    assert report.remaining_count == 0
    assert not receipt.exists()


def test_promote_pending_rejects_receipt_for_different_label_content(tmp_path):
    draft = "0 0.5 0.5 0.2 0.2\n"
    baseline = hashlib.sha256(draft.encode("utf-8")).hexdigest()
    dataset, handoff = _pending_case(
        tmp_path,
        draft,
        reason="false_detection_requires_correction",
        label_baseline_sha256=baseline,
    )
    label = dataset / "review_pending" / "labels" / "review_sample1.txt"
    record_label_verification(label, label.parent)
    receipt = label.parent / ".verified" / "review_sample1.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["label_sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    report = promote_completed_pending(dataset, ["Black", "Red"], handoff)

    assert report.promoted_count == 0
    assert report.remaining_count == 1


def test_schema3_promotion_only_processes_samples_owned_by_job(tmp_path):
    dataset, handoff = _pending_case(
        tmp_path, "0 0.5 0.5 0.2 0.2\n"
    )
    other_image = dataset / "review_pending" / "images" / "review_sample2.jpg"
    other_label = dataset / "review_pending" / "labels" / "review_sample2.txt"
    other_image.write_bytes(b"other-image")
    other_label.write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    pending_manifest = dataset / "review_pending" / "manifest.csv"
    with pending_manifest.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PENDING_FIELDS)
        writer.writerow(
            {
                "sample_id": "sample2",
                "product": "Cable1",
                "area": "A",
                "reason": "missed_detection_requires_box_annotation",
                "annotation_status": "pending",
                "output_image": str(other_image),
                "output_label": str(other_label),
                "label_baseline_sha256": "missing",
            }
        )
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    status_path = handoff.parent / "status.json"
    payload["schema_version"] = 3
    payload["status_path"] = str(status_path.resolve())
    payload["targets"][0]["pending_sample_ids"] = ["sample1"]
    handoff.write_text(json.dumps(payload), encoding="utf-8")
    status_path.write_text("{}", encoding="utf-8")

    report = promote_completed_pending(dataset, ["Black", "Red"], handoff)

    assert report.promoted_count == 1
    assert report.remaining_count == 0
    with pending_manifest.open("r", encoding="utf-8", newline="") as handle:
        pending_rows = list(csv.DictReader(handle))
    assert [row["sample_id"] for row in pending_rows] == ["sample2"]
    assert other_image.is_file()
