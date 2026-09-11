"""Immutable dataset versions and their lineage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from picture_tool.autotrain.dataset_versions import (
    DatasetVersionError,
    DatasetVersionStore,
    LabelledSample,
    content_id_for,
)
from picture_tool.autotrain.class_schema import normalize_class_names

#: The real Cable1/A contract, in the order the station's champion and
#: every handoff manifest record it.
SCHEMA = normalize_class_names(
    ["Black", "Green", "Orange", "Red", "Yellow"], source="test"
)


def _sample(tmp_path: Path, sample_id: str, *, label: str | None = "0 0.5 0.5 0.2 0.2"):
    image = tmp_path / "src" / f"{sample_id}.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(sample_id.encode("utf-8"))
    label_path = None
    if label is not None:
        label_path = tmp_path / "src" / f"{sample_id}.txt"
        label_path.write_text(label, encoding="utf-8")
    return LabelledSample(
        sample_id=sample_id, image_path=image, label_path=label_path, origin="pool"
    )


def _store(tmp_path: Path) -> DatasetVersionStore:
    return DatasetVersionStore(tmp_path / "datasets", product="Cable1", area="A")


def _create(store, tmp_path, ids, **kwargs):
    options = {
        "source": "pool",
        "label_source": "operator",
        "class_schema": SCHEMA,
    }
    options.update(kwargs)
    return store.create([_sample(tmp_path, i) for i in ids], **options)


# ---------------------------------------------------------------------------
# Versioning


def test_versions_increment_from_v001(tmp_path):
    store = _store(tmp_path)

    first = _create(store, tmp_path, ["a"])
    second = _create(store, tmp_path, ["a", "b"])

    assert first.version == "dataset_v001"
    assert second.version == "dataset_v002"
    assert store.list_versions() == ("dataset_v001", "dataset_v002")


def test_version_ordering_survives_double_digits(tmp_path):
    store = _store(tmp_path)
    for index in range(11):
        _create(store, tmp_path, [f"s{index}"])

    assert store.list_versions()[-1] == "dataset_v011"
    assert store.latest().version == "dataset_v011"


def test_the_parent_link_and_the_diff_are_recorded(tmp_path):
    store = _store(tmp_path)
    _create(store, tmp_path, ["a", "b"])

    second = _create(store, tmp_path, ["b", "c"])

    assert second.parent_version == "dataset_v001"
    assert second.added_samples == ("c",)
    assert second.removed_samples == ("a",)
    assert second.sample_ids == ("b", "c")


def test_the_first_version_has_no_parent(tmp_path):
    first = _create(_store(tmp_path), tmp_path, ["a"])

    assert first.parent_version == ""
    assert first.removed_samples == ()
    assert first.added_samples == ("a",)


def test_lineage_records_everything_the_request_asked_for(tmp_path):
    store = _store(tmp_path)

    version = _create(
        store,
        tmp_path,
        ["a"],
        description="first cycle",
        split_policy={"minimum_source_groups": {"val": 5, "test": 10}},
    )
    payload = json.loads((version.root / "lineage.json").read_text(encoding="utf-8"))

    for key in (
        "version",
        "parent_version",
        "created_at",
        "added_samples",
        "removed_samples",
        "sample_ids",
        "source",
        "label_source",
        "split_policy",
        "description",
        "content_id",
    ):
        assert key in payload, key
    assert payload["product"] == "Cable1"
    assert payload["area"] == "A"
    assert payload["sample_count"] == 1


# ---------------------------------------------------------------------------
# Immutability


def test_creating_a_version_never_edits_an_earlier_one(tmp_path):
    store = _store(tmp_path)
    first = _create(store, tmp_path, ["a"])
    before = json.loads((first.root / "lineage.json").read_text(encoding="utf-8"))
    first_images = sorted(p.name for p in first.images_dir.iterdir())

    _create(store, tmp_path, ["a", "b"])

    after = json.loads((first.root / "lineage.json").read_text(encoding="utf-8"))
    assert after == before
    assert sorted(p.name for p in first.images_dir.iterdir()) == first_images


def test_a_damaged_version_stops_creation_instead_of_being_stepped_over(tmp_path):
    """Numbering past a lineage-less directory would hide real corruption."""
    store = _store(tmp_path)
    _create(store, tmp_path, ["a"])
    (store.root / "dataset_v002").mkdir(parents=True)

    assert store.incomplete_versions() == ("dataset_v002",)
    with pytest.raises(DatasetVersionError, match="no lineage record"):
        _create(store, tmp_path, ["b"])


def test_a_damaged_version_does_not_break_latest(tmp_path):
    store = _store(tmp_path)
    _create(store, tmp_path, ["a"])
    (store.root / "dataset_v002").mkdir(parents=True)

    assert store.latest().version == "dataset_v001"


def test_verify_accepts_an_untouched_version(tmp_path):
    version = _create(_store(tmp_path), tmp_path, ["a", "b"])

    assert _store(tmp_path).verify(version.version) == ()


def test_verify_notices_a_deleted_image(tmp_path):
    store = _store(tmp_path)
    version = _create(store, tmp_path, ["a", "b"])
    target = next(p for p in version.images_dir.iterdir() if p.stem == "a")
    target.chmod(0o666)
    target.unlink()

    problems = store.verify(version.version)

    assert any("missing on disk" in problem for problem in problems)


def test_verify_notices_an_extra_image(tmp_path):
    store = _store(tmp_path)
    version = _create(store, tmp_path, ["a"])
    (version.images_dir / "smuggled.jpg").write_bytes(b"x")

    problems = store.verify(version.version)

    assert any("undeclared file" in problem for problem in problems)


def test_files_are_marked_read_only(tmp_path):
    import os
    import stat

    version = _create(_store(tmp_path), tmp_path, ["a"])
    image = next(version.images_dir.iterdir())

    mode = stat.S_IMODE(os.stat(image).st_mode)
    assert not mode & stat.S_IWOTH
    assert not mode & stat.S_IWGRP


# ---------------------------------------------------------------------------
# Content


def test_content_id_is_independent_of_order_and_paths(tmp_path):
    assert content_id_for(["b", "a"]) == content_id_for(["a", "b"])
    assert content_id_for(["a"]) != content_id_for(["a", "b"])


def test_the_same_samples_produce_the_same_content_id(tmp_path):
    one = _create(DatasetVersionStore(tmp_path / "one"), tmp_path, ["a", "b"])
    two = _create(DatasetVersionStore(tmp_path / "two"), tmp_path, ["b", "a"])

    assert one.content_id == two.content_id


def test_images_and_labels_are_copied_in(tmp_path):
    version = _create(_store(tmp_path), tmp_path, ["a", "b"])

    assert sorted(p.stem for p in version.images_dir.iterdir()) == ["a", "b"]
    assert sorted(p.stem for p in version.labels_dir.iterdir()) == ["a", "b"]
    assert (version.labels_dir / "a.txt").read_text(encoding="utf-8").startswith("0 ")


def test_a_verified_negative_gets_an_explicit_empty_label(tmp_path):
    """An absent label file must never stand in for 'no objects here'."""
    store = _store(tmp_path)

    version = store.create(
        [_sample(tmp_path, "empty", label=None)],
        source="pool",
        label_source="operator",
        class_schema=SCHEMA,
    )

    assert (version.labels_dir / "empty.txt").is_file()
    assert (version.labels_dir / "empty.txt").read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# Refusals


def test_an_empty_dataset_version_is_refused(tmp_path):
    with pytest.raises(DatasetVersionError, match="nothing to"):
        _store(tmp_path).create(
            [], source="pool", label_source="operator", class_schema=SCHEMA
        )


def test_duplicate_sample_ids_are_refused(tmp_path):
    store = _store(tmp_path)
    duplicate = _sample(tmp_path, "a")

    with pytest.raises(DatasetVersionError, match="Duplicate sample id"):
        store.create(
            [duplicate, duplicate],
            source="pool",
            label_source="operator",
            class_schema=SCHEMA,
        )


def test_a_missing_image_aborts_without_leaving_a_partial_version(tmp_path):
    store = _store(tmp_path)
    good = _sample(tmp_path, "good")
    broken = _sample(tmp_path, "broken")
    broken.image_path.unlink()

    with pytest.raises(DatasetVersionError, match="no image"):
        store.create(
            [good, broken],
            source="pool",
            label_source="operator",
            class_schema=SCHEMA,
        )

    assert store.list_versions() == ()
    assert not (store.root / "dataset_v001").exists()


def test_loading_an_unknown_version_is_an_error(tmp_path):
    with pytest.raises(DatasetVersionError, match="No such dataset version"):
        _store(tmp_path).load("dataset_v999")


def test_an_empty_store_reports_no_versions(tmp_path):
    store = _store(tmp_path)

    assert store.list_versions() == ()
    assert store.latest() is None
    assert store.next_version_name() == "dataset_v001"
    assert store.history() == ()
