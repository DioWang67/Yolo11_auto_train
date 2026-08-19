import csv
import hashlib
from pathlib import Path

from picture_tool.quality.operator_dataset_conflicts import (
    analyze_operator_dataset,
    filter_manifest_rows,
)
from scripts.audit_operator_dataset_conflicts import audit_operator_data


def _write_duplicate_dataset(
    root: Path,
    *,
    first_status: str,
    second_status: str,
    first_label: str,
    second_label: str,
) -> tuple[Path, Path, Path]:
    images = root / "raw" / "images"
    labels = root / "raw" / "labels"
    metadata = root / "metadata"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    metadata.mkdir(parents=True)
    (images / "review_first.jpg").write_bytes(b"identical-image")
    (images / "review_second.jpg").write_bytes(b"identical-image")
    (labels / "review_first.txt").write_text(first_label, encoding="utf-8")
    (labels / "review_second.txt").write_text(second_label, encoding="utf-8")
    manifest = metadata / "review_dataset_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "output_image",
                "output_label",
                "annotation_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "first",
                "output_image": str(images / "review_first.jpg"),
                "output_label": str(labels / "review_first.txt"),
                "annotation_status": first_status,
            }
        )
        writer.writerow(
            {
                "sample_id": "second",
                "output_image": str(images / "review_second.jpg"),
                "output_label": str(labels / "review_second.txt"),
                "annotation_status": second_status,
            }
        )
    return images, labels, manifest


def test_human_annotation_is_canonical_over_ai_snapshot(tmp_path):
    images, labels, manifest = _write_duplicate_dataset(
        tmp_path / "Cable1" / "A",
        first_status="verified_annotation",
        second_status="verified_snapshot",
        first_label="0 0.5 0.5 0.2 0.2\n",
        second_label="0 0.4 0.5 0.2 0.2\n",
    )

    analysis = analyze_operator_dataset(
        images, labels, manifest, product="Cable1", area="A"
    )

    assert analysis.is_safe
    assert len(analysis.selections) == 1
    selection = analysis.selections[0]
    assert selection.kept_sample == "first"
    assert selection.excluded_sample == "second"
    assert selection.reason == "human_annotation_over_ai_snapshot"
    assert selection.kept_source_type == "human_annotation"
    assert selection.excluded_source_type == "ai_snapshot"
    assert len(selection.kept_label_sha256) == 64
    assert len(selection.excluded_label_sha256) == 64


def test_two_conflicting_human_annotations_have_no_canonical_selection(tmp_path):
    images, labels, manifest = _write_duplicate_dataset(
        tmp_path / "Cable1" / "A",
        first_status="verified_annotation",
        second_status="verified_annotation",
        first_label="0 0.5 0.5 0.2 0.2\n",
        second_label="1 0.5 0.5 0.2 0.2\n",
    )

    analysis = analyze_operator_dataset(
        images, labels, manifest, product="Cable1", area="A"
    )

    assert not analysis.is_safe
    assert analysis.selections == ()
    assert analysis.conflicts[0].reason == "conflicting_human_labels"
    assert analysis.conflicts[0].image_sha256 == hashlib.sha256(
        b"identical-image"
    ).hexdigest()


def test_unknown_legacy_label_cannot_be_overridden_by_human_rule(tmp_path):
    images, labels, manifest = _write_duplicate_dataset(
        tmp_path / "Cable1" / "A",
        first_status="verified_annotation",
        second_status="",
        first_label="0 0.5 0.5 0.2 0.2\n",
        second_label="1 0.5 0.5 0.2 0.2\n",
    )

    analysis = analyze_operator_dataset(
        images, labels, manifest, product="Cable1", area="A"
    )

    assert not analysis.is_safe
    assert analysis.selections == ()
    assert analysis.conflicts[0].reason == (
        "conflicting_labels_without_canonical_authority"
    )


def test_manifest_filter_uses_output_name_when_duplicate_sample_ids_match(tmp_path):
    images, labels, manifest = _write_duplicate_dataset(
        tmp_path / "Cable1" / "A",
        first_status="verified_snapshot",
        second_status="verified_snapshot",
        first_label="0 0.5 0.5 0.2 0.2\n",
        second_label="0 0.5 0.5 0.2 0.2\n",
    )
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["sample_id"] = "shared-sample-id"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    analysis = analyze_operator_dataset(
        images, labels, manifest, product="Cable1", area="A"
    )
    filtered = tmp_path / "filtered.csv"

    filter_manifest_rows(
        manifest,
        filtered,
        {item.excluded_sample for item in analysis.selections},
        excluded_output_image_names={
            path.name for path in analysis.excluded_image_paths
        },
    )

    with filtered.open("r", encoding="utf-8", newline="") as handle:
        retained = list(csv.DictReader(handle))
    assert [Path(row["output_image"]).name for row in retained] == [
        "review_first.jpg"
    ]


def test_identical_human_and_snapshot_labels_keep_human(tmp_path):
    label = "0 0.5 0.5 0.2 0.2\n"
    images, labels, manifest = _write_duplicate_dataset(
        tmp_path / "Cable1" / "A",
        first_status="verified_snapshot",
        second_status="verified_annotation",
        first_label=label,
        second_label=label,
    )

    analysis = analyze_operator_dataset(
        images, labels, manifest, product="Cable1", area="A"
    )

    assert analysis.is_safe
    assert analysis.selections[0].kept_sample == "second"
    assert analysis.selections[0].reason == "identical_label_prefer_human_over_ai"


def test_dry_run_audits_ready_manifests_and_failed_job_snapshots(tmp_path):
    data_root = tmp_path / "data"
    _write_duplicate_dataset(
        data_root / "Cable1" / "A",
        first_status="verified_annotation",
        second_status="verified_annotation",
        first_label="0 0.5 0.5 0.2 0.2\n",
        second_label="1 0.5 0.5 0.2 0.2\n",
    )
    job_root = (
        data_root
        / ".operator_handoff"
        / "jobs"
        / "failed-job"
    )
    (job_root / "status.json").parent.mkdir(parents=True)
    (job_root / "status.json").write_text(
        '{"state":"failed"}', encoding="utf-8"
    )
    _write_duplicate_dataset(
        job_root / "dataset" / "Cable1" / "A",
        first_status="verified_annotation",
        second_status="verified_snapshot",
        first_label="0 0.5 0.5 0.2 0.2\n",
        second_label="0 0.4 0.5 0.2 0.2\n",
    )

    report = audit_operator_data(data_root)

    assert report["mutation_performed"] is False
    assert report["summary"] == {
        "scopes_checked": 2,
        "canonical_selection_count": 1,
        "blocking_conflict_count": 1,
        "error_count": 0,
    }
    assert {item["scope"] for item in report["reports"]} == {
        "ready:Cable1/A",
        "job:failed-job:failed",
    }
