"""The locked evaluation dataset: registration, locking, contamination.

The behaviour these tests pin down is fail-closed. Every way a golden set can
be absent, changed or polluted must produce a non-passing status, because a
promotion recommendation is only worth anything if the yardstick held still.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from picture_tool.autotrain import golden
from picture_tool.autotrain.golden import (
    CONTAMINATED,
    INVALID,
    MISMATCH,
    MISSING,
    NOT_CONFIGURED,
    OK,
    GoldenDatasetError,
)


def _golden_dir(tmp_path: Path, contents: dict[str, bytes]) -> Path:
    root = tmp_path / "golden"
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return root


def _registered(tmp_path: Path, contents=None):
    root = _golden_dir(
        tmp_path, contents or {"a.jpg": b"image-a", "b.jpg": b"image-b"}
    )
    dataset = golden.register(root, registered_by="engineer", description="v1 set")
    return root, dataset


# ---------------------------------------------------------------------------
# Not configured is a first-class state


def test_an_unconfigured_golden_set_is_not_an_error_but_never_passes():
    status = golden.resolve("")

    assert status.status == NOT_CONFIGURED
    assert status.is_passing is False
    assert status.is_configured is False
    assert "cannot be recommended for promotion" in status.detail


def test_blank_whitespace_counts_as_unconfigured():
    assert golden.resolve("   ").status == NOT_CONFIGURED


# ---------------------------------------------------------------------------
# Registration does not choose data


def test_registration_records_what_is_there_without_moving_it(tmp_path):
    root = _golden_dir(tmp_path, {"a.jpg": b"image-a", "sub/b.png": b"image-b"})
    before = sorted(p.name for p in root.rglob("*") if p.is_file())

    dataset = golden.register(root, registered_by="engineer")

    after = sorted(p.name for p in root.rglob("*") if p.is_file())
    assert after == sorted([*before, "golden_manifest.json"])
    assert dataset.image_count == 2
    assert dataset.registered_by == "engineer"


def test_registration_requires_an_owner(tmp_path):
    root = _golden_dir(tmp_path, {"a.jpg": b"image-a"})

    with pytest.raises(GoldenDatasetError, match="registered_by is required"):
        golden.register(root, registered_by="  ")


def test_registration_refuses_an_empty_directory(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()

    with pytest.raises(GoldenDatasetError, match="nothing to register"):
        golden.register(root, registered_by="engineer")


def test_registration_refuses_a_missing_directory(tmp_path):
    with pytest.raises(GoldenDatasetError, match="not found"):
        golden.register(tmp_path / "nope", registered_by="engineer")


def test_re_registration_is_refused_unless_explicitly_intended(tmp_path):
    root, _ = _registered(tmp_path)

    with pytest.raises(GoldenDatasetError, match="locked once"):
        golden.register(root, registered_by="engineer")

    replacement = golden.register(root, registered_by="engineer", overwrite=True)
    assert replacement.image_count == 2


# ---------------------------------------------------------------------------
# Resolution


def test_a_registered_untouched_set_resolves_ok(tmp_path):
    root, dataset = _registered(tmp_path)

    status = golden.resolve(str(root), dataset.manifest_sha256)

    assert status.status == OK
    assert status.is_passing is True
    assert status.dataset.image_count == 2


def test_resolution_works_without_a_pinned_hash(tmp_path):
    root, _ = _registered(tmp_path)

    assert golden.resolve(str(root)).status == OK


def test_a_missing_directory_is_reported_not_raised(tmp_path):
    status = golden.resolve(str(tmp_path / "absent"))

    assert status.status == MISSING
    assert status.is_passing is False


def test_a_directory_without_a_manifest_is_missing(tmp_path):
    root = _golden_dir(tmp_path, {"a.jpg": b"image-a"})

    status = golden.resolve(str(root))

    assert status.status == MISSING
    assert "Register the directory" in status.detail


def test_an_unreadable_manifest_is_invalid(tmp_path):
    root, _ = _registered(tmp_path)
    (root / "golden_manifest.json").write_text("{ broken", encoding="utf-8")

    assert golden.resolve(str(root)).status == INVALID


def test_a_manifest_of_the_wrong_shape_is_invalid(tmp_path):
    root, _ = _registered(tmp_path)
    (root / "golden_manifest.json").write_text(
        json.dumps({"schema_version": 1}), encoding="utf-8"
    )

    assert golden.resolve(str(root)).status == INVALID


# ---------------------------------------------------------------------------
# Locking


def test_a_changed_manifest_fails_against_the_pinned_hash(tmp_path):
    root, dataset = _registered(tmp_path)
    manifest = root / "golden_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["description"] = "tampered"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    status = golden.resolve(str(root), dataset.manifest_sha256)

    assert status.status == MISMATCH
    assert "no longer comparable" in status.detail


def test_a_changed_image_fails_even_with_an_intact_manifest(tmp_path):
    root, dataset = _registered(tmp_path)
    (root / "a.jpg").write_bytes(b"different-pixels")

    status = golden.resolve(str(root), dataset.manifest_sha256)

    assert status.status == MISMATCH
    assert any("changed on disk" in problem for problem in status.problems)


def test_a_deleted_image_fails(tmp_path):
    root, dataset = _registered(tmp_path)
    (root / "a.jpg").unlink()

    status = golden.resolve(str(root), dataset.manifest_sha256)

    assert status.status == MISMATCH
    assert any("is missing" in problem for problem in status.problems)


def test_an_added_image_does_not_silently_join_the_golden_set(tmp_path):
    """Growth must go through registration, so the pinned hash still means something."""
    root, dataset = _registered(tmp_path)
    (root / "c.jpg").write_bytes(b"image-c")

    status = golden.resolve(str(root), dataset.manifest_sha256)

    assert status.status == OK
    assert status.dataset.image_count == 2
    assert "c.jpg" not in status.dataset.images.values()


# ---------------------------------------------------------------------------
# Contamination


def test_overlap_with_training_data_is_contamination(tmp_path):
    root, dataset = _registered(tmp_path)
    shared = dataset.sample_ids[0]

    status = golden.resolve(
        str(root), dataset.manifest_sha256, training_sample_ids=[shared, "unrelated"]
    )

    assert status.status == CONTAMINATED
    assert status.is_passing is False
    assert status.contaminated_samples == (shared,)


def test_disjoint_training_data_is_clean(tmp_path):
    root, dataset = _registered(tmp_path)

    status = golden.resolve(
        str(root), dataset.manifest_sha256, training_sample_ids=["something-else"]
    )

    assert status.status == OK


def test_contamination_is_detected_across_renamed_files(tmp_path):
    """Identity is the image content hash, so a rename cannot hide a reuse."""
    root, dataset = _registered(tmp_path)
    training_copy = tmp_path / "train" / "renamed.jpg"
    training_copy.parent.mkdir(parents=True)
    training_copy.write_bytes((root / "a.jpg").read_bytes())

    from picture_tool.autotrain.candidate_pool import sample_id_for_image

    status = golden.resolve(
        str(root),
        dataset.manifest_sha256,
        training_sample_ids=[sample_id_for_image(training_copy)],
    )

    assert status.status == CONTAMINATED


def test_status_serialises_for_reports(tmp_path):
    root, dataset = _registered(tmp_path)

    payload = golden.resolve(str(root), dataset.manifest_sha256).to_dict()

    assert payload["status"] == OK
    assert payload["image_count"] == 2
    assert payload["root"] == str(root.resolve())
