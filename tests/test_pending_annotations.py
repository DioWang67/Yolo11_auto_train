import csv
import hashlib
import json
from pathlib import Path

import pytest

from picture_tool.pending_annotations import (
    PENDING_FIELDS,
    PendingAnnotationError,
    configure_pending_job_workspace,
    configure_pending_workspace,
    inspect_pending_annotation_progress,
    promote_completed_pending,
    reconcile_pending_label_sidecars,
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
        label.write_text(label_text, encoding="utf-8", newline="")
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


def _scoped_handoff(tmp_path: Path, dataset: Path) -> Path:
    job_dir = tmp_path / "data" / ".operator_handoff" / "jobs" / "job-sidecar"
    job_dir.mkdir(parents=True)
    handoff = job_dir / "handoff.json"
    status = job_dir / "status.json"
    handoff.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "status_path": str(status.resolve()),
                "targets": [
                    {
                        "product": "Cable1",
                        "area": "A",
                        "dataset_root": str(dataset.resolve()),
                        "pending_sample_ids": ["sample1"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    status.write_text("{}", encoding="utf-8")
    return handoff


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


def test_reconcile_imports_job_sidecar_into_managed_labels(tmp_path):
    draft = "0 0.5 0.5 0.2 0.2\n"
    baseline = hashlib.sha256(draft.encode("utf-8")).hexdigest()
    dataset, _legacy_handoff = _pending_case(
        tmp_path,
        draft,
        reason="false_detection_requires_correction",
        label_baseline_sha256=baseline,
    )
    handoff = _scoped_handoff(tmp_path, dataset)
    workspace, labels, _classes = configure_pending_job_workspace(
        dataset, ["Black", "Red"], handoff
    )
    sidecar = workspace / "review_sample1.txt"
    corrected = "1 0.4 0.5 0.3 0.2\n"
    sidecar.write_text(corrected, encoding="utf-8")

    reconciled_count = reconcile_pending_label_sidecars(
        dataset, ["Black", "Red"], handoff
    )

    managed_label = labels / "review_sample1.txt"
    assert reconciled_count == 1
    assert managed_label.read_text(encoding="utf-8") == corrected
    assert sidecar.read_text(encoding="utf-8") == corrected
    assert (labels / ".verified" / "review_sample1.json").is_file()
    progress = inspect_pending_annotation_progress(
        dataset, ["Black", "Red"], handoff
    )
    assert progress.completed_count == 1
    assert progress.pending_items == ()


def test_reconcile_rejects_invalid_job_sidecar_without_overwriting_draft(tmp_path):
    draft = "0 0.5 0.5 0.2 0.2\n"
    baseline = hashlib.sha256(draft.encode("utf-8")).hexdigest()
    dataset, _legacy_handoff = _pending_case(
        tmp_path,
        draft,
        reason="false_detection_requires_correction",
        label_baseline_sha256=baseline,
    )
    handoff = _scoped_handoff(tmp_path, dataset)
    workspace, labels, _classes = configure_pending_job_workspace(
        dataset, ["Black", "Red"], handoff
    )
    (workspace / "review_sample1.txt").write_text(
        "9 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )

    with pytest.raises(PendingAnnotationError, match="out of range"):
        reconcile_pending_label_sidecars(dataset, ["Black", "Red"], handoff)

    assert (labels / "review_sample1.txt").read_text(encoding="utf-8") == draft
    assert not (labels / ".verified" / "review_sample1.json").exists()


def test_promote_automatically_recovers_labelimg_job_sidecar(tmp_path):
    draft = "0 0.5 0.5 0.2 0.2\n"
    baseline = hashlib.sha256(draft.encode("utf-8")).hexdigest()
    dataset, _legacy_handoff = _pending_case(
        tmp_path,
        draft,
        reason="false_detection_requires_correction",
        label_baseline_sha256=baseline,
    )
    handoff = _scoped_handoff(tmp_path, dataset)
    workspace, _labels, _classes = configure_pending_job_workspace(
        dataset, ["Black", "Red"], handoff
    )
    corrected = "1 0.4 0.5 0.3 0.2\n"
    sidecar = workspace / "review_sample1.txt"
    sidecar.write_text(corrected, encoding="utf-8")

    report = promote_completed_pending(dataset, ["Black", "Red"], handoff)

    assert report.promoted_count == 1
    assert report.remaining_count == 0
    assert (
        dataset / "raw" / "labels" / "review_sample1.txt"
    ).read_text(encoding="utf-8") == corrected
    assert sidecar.is_file()


def test_promote_pending_rejects_empty_false_negative_label(tmp_path):
    dataset, handoff = _pending_case(tmp_path, "")

    with pytest.raises(PendingAnnotationError, match="review_sample1.jpg"):
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


@pytest.mark.parametrize(
    "reason",
    ["false_detection_requires_correction", "box_geometry_requires_correction"],
)
def test_promote_pending_keeps_unchanged_generated_draft(tmp_path, reason):
    draft = "0 0.5 0.5 0.2 0.2\n"
    baseline = hashlib.sha256(draft.encode("utf-8")).hexdigest()
    dataset, handoff = _pending_case(
        tmp_path,
        draft,
        reason=reason,
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


def test_pending_progress_does_not_count_unchanged_prefill_as_complete(tmp_path):
    draft = "0 0.5 0.5 0.2 0.2\n"
    baseline = hashlib.sha256(draft.encode("utf-8")).hexdigest()
    dataset, handoff = _pending_case(
        tmp_path,
        draft,
        reason="false_detection_requires_correction",
        label_baseline_sha256=baseline,
    )

    progress = inspect_pending_annotation_progress(
        dataset, ["Black", "Red"], handoff
    )

    assert progress.total_count == 1
    assert progress.completed_count == 0
    assert len(progress.pending_items) == 1
    assert progress.pending_items[0].status == "unchanged_draft"
    assert "Ctrl+S" in progress.pending_items[0].detail


def test_pending_progress_counts_explicit_same_content_save(tmp_path):
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

    progress = inspect_pending_annotation_progress(
        dataset, ["Black", "Red"], handoff
    )

    assert progress.total_count == 1
    assert progress.completed_count == 1
    assert progress.pending_items == ()


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


def test_schema3_annotation_workspace_contains_only_job_owned_images(tmp_path):
    dataset, handoff = _pending_case(tmp_path, None)
    shared_images = dataset / "review_pending" / "images"
    other_image = shared_images / "review_sample2.jpg"
    other_image.write_bytes(b"other-image")
    manifest = dataset / "review_pending" / "manifest.csv"
    with manifest.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PENDING_FIELDS)
        writer.writerow(
            {
                "sample_id": "sample2",
                "product": "Cable1",
                "area": "A",
                "reason": "missed_detection_requires_box_annotation",
                "annotation_status": "pending",
                "output_image": str(other_image),
                "output_label": str(
                    dataset / "review_pending" / "labels" / "review_sample2.txt"
                ),
                "label_baseline_sha256": "missing",
            }
        )
    job_dir = tmp_path / "data" / ".operator_handoff" / "jobs" / "job-1"
    job_dir.mkdir(parents=True)
    handoff = job_dir / "handoff.json"
    status = job_dir / "status.json"
    handoff.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "status_path": str(status.resolve()),
                "targets": [
                    {
                        "dataset_root": str(dataset.resolve()),
                        "pending_sample_ids": ["sample1"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    status.write_text("{}", encoding="utf-8")

    workspace, labels, _classes = configure_pending_job_workspace(
        dataset, ["Black", "Red"], handoff
    )

    assert [path.name for path in workspace.glob("*.jpg")] == ["review_sample1.jpg"]
    assert labels == dataset / "review_pending" / "labels"
    progress = inspect_pending_annotation_progress(
        dataset, ["Black", "Red"], handoff
    )
    assert progress.total_count == 1

    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload["targets"][0]["pending_sample_ids"] = ["sample2"]
    handoff.write_text(json.dumps(payload), encoding="utf-8")
    workspace, _labels, _classes = configure_pending_job_workspace(
        dataset, ["Black", "Red"], handoff
    )
    assert [path.name for path in workspace.glob("*.jpg")] == ["review_sample2.jpg"]
