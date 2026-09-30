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
    sample_id_for_image,
)
from picture_tool.autotrain.class_schema import normalize_class_names
from picture_tool.autotrain.labeling import (
    IMPORTED_SELECTOR,
    LabelingError,
    export_request,
    import_labelled_folder,
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


# ---------------------------------------------------------------------------
# Importing a folder that is already labelled

STATION = normalize_class_names(CLASSES, source="station")


def _labelled_folder(root: Path, items: dict[str, str | None], classes=CLASSES) -> Path:
    """``items`` maps an image's relative path to its label text, or None."""
    for relative, label in items.items():
        image = root / "images" / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"pixels-of-{relative}".encode("utf-8"))
        if label is not None:
            target = (root / "labels" / relative).with_suffix(".txt")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(label, encoding="utf-8")
    (root / "labels").mkdir(parents=True, exist_ok=True)
    if classes is not None:
        (root / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    return root


def _import_folder(tmp_path, folder, pool=None, **kwargs):
    pool = pool or CandidatePool(tmp_path / "pool")
    options = {"class_schema": STATION, "product": "Cable1", "area": "A"}
    options.update(kwargs)
    return pool, import_labelled_folder(pool, folder, **options)


def test_a_labelled_folder_becomes_verified_training_input(tmp_path):
    folder = _labelled_folder(
        tmp_path / "batch",
        {"a.jpg": "0 0.5 0.5 0.2 0.2\n", "b.png": "1 0.4 0.4 0.1 0.1\n", "neg.jpg": ""},
    )

    pool, result = _import_folder(tmp_path, folder)

    assert result.errors == ()
    assert len(result.verified) == 3
    usable = verified_samples(pool)
    assert {sample.selector for sample in usable} == {IMPORTED_SELECTOR}
    # The empty label is kept as a verified negative, not dropped.
    stored = {Path(sample.label_path).read_text(encoding="utf-8") for sample in usable}
    assert "" in stored
    # The pool keeps its own copy, so the source folder can be deleted.
    assert all(Path(sample.image_path).is_relative_to(pool.images_dir) for sample in usable)


def test_nested_split_folders_are_accepted(tmp_path):
    folder = _labelled_folder(
        tmp_path / "batch",
        {"train/a.jpg": "0 0.5 0.5 0.2 0.2\n", "val/b.jpg": "1 0.5 0.5 0.2 0.2\n"},
    )

    _, result = _import_folder(tmp_path, folder)

    assert result.errors == ()
    assert len(result.verified) == 2


def test_the_class_list_may_come_from_a_data_yaml(tmp_path):
    folder = _labelled_folder(tmp_path / "batch", {"a.jpg": "0 0.5 0.5 0.2 0.2\n"}, classes=None)
    (folder / "data.yaml").write_text("names:\n  0: Red\n  1: Orange\n", encoding="utf-8")

    _, result = _import_folder(tmp_path, folder)

    assert result.errors == ()


@pytest.mark.parametrize(
    ("classes", "reason"),
    [(None, "declares no classes"), (["Orange", "Red"], "order matters")],
)
def test_a_class_list_that_is_absent_or_reordered_is_refused(tmp_path, classes, reason):
    folder = _labelled_folder(tmp_path / "batch", {"a.jpg": "0 0.5 0.5 0.2 0.2\n"}, classes=classes)

    pool, result = _import_folder(tmp_path, folder)

    assert reason in result.errors[0]
    assert pool.load() == ()


@pytest.mark.parametrize(
    ("items", "reason"),
    [
        ({"a.jpg": "0 0.5 0.5 0.2 0.2\n", "b.jpg": None}, "no label file"),
        ({"a.jpg": "0 0.5 0.5 0.2 0.2\n", "b.jpg": "7 0.5 0.5 0.2 0.2\n"}, "b.jpg"),
    ],
)
def test_one_bad_image_leaves_the_pool_untouched(tmp_path, items, reason):
    folder = _labelled_folder(tmp_path / "batch", items)

    pool, result = _import_folder(tmp_path, folder)

    assert any(reason in error for error in result.errors)
    assert result.verified == ()
    assert pool.load() == ()


def test_the_same_image_twice_is_a_duplicate_or_a_conflict(tmp_path):
    folder = _labelled_folder(tmp_path / "batch", {"a.jpg": "0 0.5 0.5 0.2 0.2\n"})
    # Byte-identical copy under another name, as public datasets contain.
    copy = folder / "images" / "copy.jpg"
    copy.write_bytes((folder / "images" / "a.jpg").read_bytes())
    (folder / "labels" / "copy.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    _, same = _import_folder(tmp_path / "first", folder)

    assert same.errors == ()
    assert same.duplicates == ("copy.jpg",)
    assert len(same.verified) == 1

    (folder / "labels" / "copy.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    pool, conflict = _import_folder(tmp_path / "second", folder)

    assert "different label" in conflict.errors[0]
    assert pool.load() == ()


def test_golden_images_are_refused_at_the_door(tmp_path):
    folder = _labelled_folder(tmp_path / "batch", {"a.jpg": "0 0.5 0.5 0.2 0.2\n"})
    golden_id = sample_id_for_image(folder / "images" / "a.jpg")

    pool, result = _import_folder(tmp_path, folder, golden_sample_ids=[golden_id])

    assert "golden set" in result.errors[0]
    assert pool.load() == ()


def test_reimporting_the_same_folder_changes_nothing(tmp_path):
    folder = _labelled_folder(tmp_path / "batch", {"a.jpg": "0 0.5 0.5 0.2 0.2\n"})
    pool, _ = _import_folder(tmp_path, folder)

    _, again = _import_folder(tmp_path, folder, pool=pool)

    assert again.errors == ()
    assert again.verified == ()
    assert len(again.already_verified) == 1


def test_a_different_label_for_a_verified_image_is_a_conflict(tmp_path):
    folder = _labelled_folder(tmp_path / "batch", {"a.jpg": "0 0.5 0.5 0.2 0.2\n"})
    pool, _ = _import_folder(tmp_path, folder)
    (folder / "labels" / "a.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    _, result = _import_folder(tmp_path, folder, pool=pool)

    assert "already verified" in result.errors[0]
    stored = Path(pool.load()[0].label_path).read_text(encoding="utf-8")
    assert stored == "0 0.5 0.5 0.2 0.2\n"


def test_a_pending_production_candidate_is_verified_by_the_import(tmp_path):
    folder = _labelled_folder(tmp_path / "batch", {"a.jpg": "0 0.5 0.5 0.2 0.2\n"})
    image = folder / "images" / "a.jpg"
    content_id = sample_id_for_image(image)
    pool = CandidatePool(tmp_path / "pool")
    pool.add(
        [
            CandidateSample(
                sample_id=content_id,
                inspection_id="insp0",
                product="Cable1",
                area="A",
                selector="low_confidence",
                selected_reason="uncertain",
                score=0.3,
                source_model="best.onnx",
                source_model_version="1.2.0",
                selected_at="2026-09-10T00:00:00+00:00",
                production_timestamp="",
                image_path="",
                source_image_path=str(image),
            )
        ]
    )

    _, result = _import_folder(tmp_path, folder, pool=pool)

    assert result.verified == (content_id,)
    merged = {sample.sample_id: sample for sample in pool.load()}[content_id]
    assert merged.label_state == VERIFIED
    assert IMPORTED_SELECTOR in merged.also_selected_by
