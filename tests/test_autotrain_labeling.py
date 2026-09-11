"""The human labelling queue between selection and training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from picture_tool.autotrain.candidate_pool import (
    NEEDS_LABEL,
    VERIFIED,
    CandidatePool,
    CandidateSample,
)
from picture_tool.autotrain.labeling import (
    LabelingError,
    export_request,
    import_request,
    load_request,
    request_statistics,
    verified_samples,
)

CLASSES = ["Red", "Orange"]


def _pool_with(tmp_path, count=2, selector="low_confidence") -> CandidatePool:
    pool = CandidatePool(tmp_path / "pool")
    samples = []
    for index in range(count):
        source = tmp_path / "src" / f"img{index}.jpg"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"pixels-{index}".encode("utf-8"))
        samples.append(
            CandidateSample(
                sample_id=f"sample{index}",
                inspection_id=f"insp{index}",
                product="Cable1",
                area="A",
                selector=selector,
                selected_reason="uncertain",
                score=0.3,
                source_model="best.onnx",
                source_model_version="1.2.0",
                selected_at=f"2026-09-10T00:0{index}:00+00:00",
                production_timestamp="",
                image_path="",
                source_image_path=str(source),
            )
        )
    pool.add(samples)
    return pool


def _export(tmp_path, pool, **kwargs):
    options = {
        "product": "Cable1",
        "area": "A",
        "class_names": CLASSES,
        "request_id": "req_test",
    }
    options.update(kwargs)
    return export_request(pool, tmp_path / "labeling", **options)


def _write_label(request, name: str, text: str) -> Path:
    path = request.labels_dir / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Export


def test_pending_candidates_are_exported_with_their_classes(tmp_path):
    pool = _pool_with(tmp_path)

    request = _export(tmp_path, pool)

    assert len(request.sample_ids) == 2
    assert sorted(p.name for p in request.images_dir.iterdir()) == [
        "low_confidence-sample0.jpg",
        "low_confidence-sample1.jpg",
    ]
    assert (request.root / "classes.txt").read_text(encoding="utf-8").split() == CLASSES


def test_filenames_carry_the_selector_so_work_can_be_batched(tmp_path):
    pool = _pool_with(tmp_path, count=1, selector="review_correction")

    request = _export(tmp_path, pool)

    assert next(request.images_dir.iterdir()).name.startswith("review_correction-")


def test_export_requires_a_class_list(tmp_path):
    pool = _pool_with(tmp_path)

    with pytest.raises(LabelingError, match="class_names is required"):
        _export(tmp_path, pool, class_names=[])


def test_exporting_an_empty_queue_is_refused(tmp_path):
    pool = CandidatePool(tmp_path / "pool")

    with pytest.raises(LabelingError, match="no candidates waiting"):
        _export(tmp_path, pool)


def test_export_can_be_limited(tmp_path):
    pool = _pool_with(tmp_path, count=5)

    request = _export(tmp_path, pool, limit=2)

    assert len(request.sample_ids) == 2


def test_a_request_round_trips_through_disk(tmp_path):
    pool = _pool_with(tmp_path)
    request = _export(tmp_path, pool)

    loaded = load_request(request.root)

    assert loaded.request_id == request.request_id
    assert loaded.class_names == tuple(CLASSES)
    assert loaded.sample_ids == request.sample_ids


def test_loading_a_directory_that_is_not_a_request_fails(tmp_path):
    (tmp_path / "not-a-request").mkdir()

    with pytest.raises(LabelingError, match="Not a labelling request"):
        load_request(tmp_path / "not-a-request")


# ---------------------------------------------------------------------------
# Import


def test_valid_labels_mark_their_candidates_verified(tmp_path):
    pool = _pool_with(tmp_path)
    request = _export(tmp_path, pool)
    _write_label(request, "sample0.txt", "0 0.5 0.5 0.2 0.2\n")
    _write_label(request, "sample1.txt", "1 0.4 0.4 0.1 0.1\n")

    result = import_request(pool, request.root)

    assert set(result.verified) == {"sample0", "sample1"}
    assert result.errors == ()
    assert all(sample.label_state == VERIFIED for sample in pool.load())


def test_the_exported_filename_is_also_accepted(tmp_path):
    pool = _pool_with(tmp_path, count=1)
    request = _export(tmp_path, pool)
    _write_label(request, "low_confidence-sample0.txt", "0 0.5 0.5 0.2 0.2\n")

    result = import_request(pool, request.root)

    assert result.verified == ("sample0",)


def test_an_empty_label_is_a_valid_verified_negative(tmp_path):
    """Only an explicit empty file means 'no objects'; a missing file does not."""
    pool = _pool_with(tmp_path, count=1)
    request = _export(tmp_path, pool)
    _write_label(request, "sample0.txt", "")

    result = import_request(pool, request.root)

    assert result.verified == ("sample0",)


def test_an_unlabelled_sample_stays_pending(tmp_path):
    pool = _pool_with(tmp_path)
    request = _export(tmp_path, pool)
    _write_label(request, "sample0.txt", "0 0.5 0.5 0.2 0.2\n")

    result = import_request(pool, request.root)

    assert result.verified == ("sample0",)
    assert result.skipped_unlabelled == ("sample1",)
    states = {s.sample_id: s.label_state for s in pool.load()}
    assert states == {"sample0": VERIFIED, "sample1": NEEDS_LABEL}


@pytest.mark.parametrize(
    "label,reason",
    [
        ("0 0.5 0.5", "expected 5 values"),
        ("9 0.5 0.5 0.2 0.2", "out of range"),
        ("0 0.5 0.5 0.0 0.2", "must be positive"),
        ("0 1.5 0.5 0.2 0.2", "out of range"),
        ("x 0.5 0.5 0.2 0.2", "invalid numeric"),
    ],
)
def test_invalid_labels_are_rejected_with_the_shared_validator(tmp_path, label, reason):
    pool = _pool_with(tmp_path, count=1)
    request = _export(tmp_path, pool)
    _write_label(request, "sample0.txt", label)

    result = import_request(pool, request.root)

    assert result.verified == ()
    assert any(reason in error for error in result.errors)


def test_one_bad_label_leaves_the_whole_pool_untouched(tmp_path):
    """All-or-nothing, so a retry starts from a pool the person recognises."""
    pool = _pool_with(tmp_path)
    request = _export(tmp_path, pool)
    _write_label(request, "sample0.txt", "0 0.5 0.5 0.2 0.2\n")
    _write_label(request, "sample1.txt", "99 0.5 0.5 0.2 0.2\n")

    result = import_request(pool, request.root)

    assert result.verified == ()
    assert all(sample.label_state == NEEDS_LABEL for sample in pool.load())


def test_a_sample_that_left_the_pool_is_reported(tmp_path):
    pool = _pool_with(tmp_path, count=1)
    request = _export(tmp_path, pool)
    _write_label(request, "sample0.txt", "0 0.5 0.5 0.2 0.2\n")
    manifest = json.loads((request.root / "request.json").read_text(encoding="utf-8"))
    manifest["sample_ids"].append("vanished")
    (request.root / "request.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = import_request(pool, request.root)

    assert any("no longer in the candidate pool" in error for error in result.errors)


# ---------------------------------------------------------------------------
# Verified samples feeding training


def test_verified_samples_are_the_only_training_input(tmp_path):
    pool = _pool_with(tmp_path)
    request = _export(tmp_path, pool)
    _write_label(request, "sample0.txt", "0 0.5 0.5 0.2 0.2\n")
    import_request(pool, request.root)

    usable = verified_samples(pool)

    assert [sample.sample_id for sample in usable] == ["sample0"]


def test_a_verified_sample_whose_label_vanished_is_excluded(tmp_path):
    pool = _pool_with(tmp_path, count=1)
    request = _export(tmp_path, pool)
    _write_label(request, "sample0.txt", "0 0.5 0.5 0.2 0.2\n")
    import_request(pool, request.root)
    Path(pool.load()[0].label_path).unlink()

    assert verified_samples(pool) == ()


def test_request_statistics_track_annotation_progress(tmp_path):
    pool = _pool_with(tmp_path, count=3)
    request = _export(tmp_path, pool)
    _write_label(request, "sample0.txt", "0 0.5 0.5 0.2 0.2\n")

    stats = request_statistics(request)

    assert stats == {"requested": 3, "labelled": 1, "outstanding": 2}
