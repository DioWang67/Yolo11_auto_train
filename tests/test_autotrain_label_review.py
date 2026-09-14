"""Human labels, and the gap between a file existing and someone agreeing.

The load-bearing property is that a label file is not an approval. Most of
what follows is an attempt to break that in the ways it would actually break
in practice: approve then edit, drop in an image nobody vetted, stop drawing
boxes halfway.
"""

from __future__ import annotations

import json

import pytest
import yaml

from picture_tool.autotrain.class_schema import normalize_class_names
from picture_tool.autotrain.label_review import (
    APPROVED,
    LABELED,
    NEEDS_LABEL,
    NEEDS_REVIEW,
    PROBLEM_APPROVAL_STALE,
    PROBLEM_EMPTY_LABEL,
    PROBLEM_INVALID_SYNTAX,
    PROBLEM_LABEL_INCOMPLETE,
    PROBLEM_NOT_IN_PACK,
    PROBLEM_TRAINING_CONTAMINATION,
    REJECTED,
    LabelReviewError,
    PackSample,
    coverage,
    load_decisions,
    read_pack_samples,
    record_decision,
    stage_approved,
    validate_labels,
)

SCHEMA = normalize_class_names(
    ["Black", "Green", "Orange", "Red", "Yellow"], source="test"
)
EXPECTED = {"Black": 2, "Green": 1, "Orange": 1, "Red": 1, "Yellow": 1}

#: Six boxes: the station's real expectation, two of them Black.
COMPLETE_LABEL = (
    "0 0.2 0.2 0.1 0.1\n"
    "0 0.3 0.3 0.1 0.1\n"
    "1 0.4 0.4 0.1 0.1\n"
    "2 0.5 0.5 0.1 0.1\n"
    "3 0.6 0.6 0.1 0.1\n"
    "4 0.7 0.7 0.1 0.1\n"
)


def _labelling_dir(tmp_path, samples: dict[str, str | None]):
    """Build images/ and labels/; a None label means none was drawn."""
    root = tmp_path / "golden_labeling" / "v1"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for name, label in samples.items():
        (root / "images" / f"{name}.jpg").write_bytes(f"image-{name}".encode())
        if label is not None:
            (root / "labels" / f"{name}.txt").write_text(label, encoding="utf-8")
    return root


def _pack(tmp_path, samples, groups=None):
    """A pack index keyed by image content hash, as the real one is."""
    import hashlib

    groups = groups or {}
    entries = {}
    for name in samples:
        digest = hashlib.sha256(f"image-{name}".encode()).hexdigest()
        entries[digest] = PackSample(
            source_image_id=name,
            group=groups.get(name, "hard_case"),
            image_sha256=digest,
        )
    return entries


def _validate(root, pack=None, **kwargs):
    return validate_labels(
        root,
        schema=SCHEMA,
        expected_counts=EXPECTED,
        pack_samples=pack,
        **kwargs,
    )


def _state(report, source_id):
    for sample in report.samples:
        if sample.source_image_id == source_id:
            return sample
    raise AssertionError(f"{source_id} not in report")


# ---------------------------------------------------------------------------
# Derived states


def test_an_image_with_no_label_needs_one(tmp_path):
    root = _labelling_dir(tmp_path, {"a": None})

    report = _validate(root, _pack(tmp_path, ["a"]))

    assert _state(report, "a").state == NEEDS_LABEL


def test_a_complete_label_is_labeled_not_approved(tmp_path):
    """The load-bearing test. Drawing boxes is not agreeing they are right."""
    root = _labelling_dir(tmp_path, {"a": COMPLETE_LABEL})

    report = _validate(root, _pack(tmp_path, ["a"]))

    sample = _state(report, "a")
    assert sample.state == LABELED
    assert sample.is_approved is False
    assert sample.is_eligible is False


def test_a_short_label_goes_to_review_and_is_not_corrected(tmp_path):
    """Five boxes where the station has six: a person decides, not the tool."""
    root = _labelling_dir(
        tmp_path, {"a": "\n".join(COMPLETE_LABEL.splitlines()[:5]) + "\n"}
    )
    before = (root / "labels" / "a.txt").read_text(encoding="utf-8")

    report = _validate(root, _pack(tmp_path, ["a"]))

    sample = _state(report, "a")
    assert sample.state == NEEDS_REVIEW
    assert PROBLEM_LABEL_INCOMPLETE in sample.problems
    # Names the class that is short, not just the total: "5 of 6" leaves a
    # reviewer hunting for which box they did not draw.
    assert "Yellow 0/1" in sample.detail
    assert (root / "labels" / "a.txt").read_text(encoding="utf-8") == before


def test_an_extra_box_also_goes_to_review(tmp_path):
    root = _labelling_dir(tmp_path, {"a": COMPLETE_LABEL + "3 0.8 0.8 0.1 0.1\n"})

    assert PROBLEM_LABEL_INCOMPLETE in _state(
        _validate(root, _pack(tmp_path, ["a"])), "a"
    ).problems


def test_an_empty_label_is_not_silently_a_negative_sample(tmp_path):
    root = _labelling_dir(tmp_path, {"a": "\n"})

    sample = _state(_validate(root, _pack(tmp_path, ["a"])), "a")

    assert sample.state == NEEDS_REVIEW
    assert PROBLEM_EMPTY_LABEL in sample.problems


@pytest.mark.parametrize(
    "label",
    [
        "0 0.2 0.2\n",
        "not-a-number 0.2 0.2 0.1 0.1\n",
        "9 0.2 0.2 0.1 0.1\n",
        "0 0.2 0.2 nan 0.1\n",
        "0 0.2 0.2 inf 0.1\n",
        "0 0.2 0.2 0.0 0.1\n",
        "0 0.2 0.2 -0.1 0.1\n",
        "0 1.4 0.2 0.1 0.1\n",
    ],
    ids=[
        "too-few-values",
        "non-numeric",
        "class-out-of-range",
        "nan",
        "infinity",
        "zero-width",
        "negative-width",
        "outside-the-image",
    ],
)
def test_malformed_labels_go_to_review(tmp_path, label):
    root = _labelling_dir(tmp_path, {"a": label})

    sample = _state(_validate(root, _pack(tmp_path, ["a"])), "a")

    assert sample.state == NEEDS_REVIEW
    assert PROBLEM_INVALID_SYNTAX in sample.problems


def test_an_image_that_was_never_in_the_pack_is_flagged(tmp_path):
    """The pack is where the contamination checks were made."""
    root = _labelling_dir(tmp_path, {"a": COMPLETE_LABEL, "smuggled": COMPLETE_LABEL})

    report = _validate(root, _pack(tmp_path, ["a"]))

    assert PROBLEM_NOT_IN_PACK in _state(report, "smuggled").problems
    assert _state(report, "a").state == LABELED


def test_a_trained_image_is_never_eligible_however_good_the_label(tmp_path):
    import hashlib

    root = _labelling_dir(tmp_path, {"a": COMPLETE_LABEL})
    digest = hashlib.sha256(b"image-a").hexdigest()

    report = _validate(root, _pack(tmp_path, ["a"]), trained_sha256={digest})

    sample = _state(report, "a")
    assert PROBLEM_TRAINING_CONTAMINATION in sample.problems
    assert sample.state == NEEDS_REVIEW


def test_a_label_with_no_image_is_reported_as_an_orphan(tmp_path):
    root = _labelling_dir(tmp_path, {"a": COMPLETE_LABEL})
    (root / "labels" / "ghost.txt").write_text(COMPLETE_LABEL, encoding="utf-8")

    assert _validate(root, _pack(tmp_path, ["a"])).orphan_labels == ("ghost.txt",)


def test_a_missing_images_directory_is_refused(tmp_path):
    with pytest.raises(LabelReviewError, match="not found"):
        _validate(tmp_path / "nothing")


# ---------------------------------------------------------------------------
# Decisions


def test_approval_needs_a_name(tmp_path):
    root = _labelling_dir(tmp_path, {"a": COMPLETE_LABEL})
    report = _validate(root, _pack(tmp_path, ["a"]))

    with pytest.raises(LabelReviewError, match="reviewed_by is required"):
        record_decision(
            root, report, source_image_ids=["a"], state=APPROVED, reviewed_by="  "
        )


def test_a_derived_state_cannot_be_set_by_hand(tmp_path):
    root = _labelling_dir(tmp_path, {"a": COMPLETE_LABEL})
    report = _validate(root, _pack(tmp_path, ["a"]))

    with pytest.raises(LabelReviewError, match="derived from"):
        record_decision(
            root, report, source_image_ids=["a"], state=LABELED, reviewed_by="Dio"
        )


def test_approving_a_clean_sample_makes_it_eligible(tmp_path):
    root = _labelling_dir(tmp_path, {"a": COMPLETE_LABEL})
    pack = _pack(tmp_path, ["a"])
    record_decision(
        root,
        _validate(root, pack),
        source_image_ids=["a"],
        state=APPROVED,
        reviewed_by="Dio",
    )

    sample = _state(_validate(root, pack), "a")

    assert sample.state == APPROVED
    assert sample.is_eligible is True
    assert sample.reviewed_by == "Dio"


def test_approving_a_sample_with_problems_is_refused(tmp_path):
    """An approval sitting on top of a known problem is the thing to prevent."""
    root = _labelling_dir(
        tmp_path, {"a": "\n".join(COMPLETE_LABEL.splitlines()[:4]) + "\n"}
    )
    pack = _pack(tmp_path, ["a"])

    recorded, refused = record_decision(
        root,
        _validate(root, pack),
        source_image_ids=["a"],
        state=APPROVED,
        reviewed_by="Dio",
    )

    assert recorded == []
    assert PROBLEM_LABEL_INCOMPLETE in refused["a"]
    assert _state(_validate(root, pack), "a").state == NEEDS_REVIEW


def test_approving_an_unlabelled_sample_is_refused(tmp_path):
    root = _labelling_dir(tmp_path, {"a": None})
    pack = _pack(tmp_path, ["a"])

    _, refused = record_decision(
        root,
        _validate(root, pack),
        source_image_ids=["a"],
        state=APPROVED,
        reviewed_by="Dio",
    )

    assert "no label yet" in refused["a"]


def test_editing_a_label_after_approval_revokes_it(tmp_path):
    """Approve-then-edit is the obvious hole; the decision is bound to bytes."""
    root = _labelling_dir(tmp_path, {"a": COMPLETE_LABEL})
    pack = _pack(tmp_path, ["a"])
    record_decision(
        root,
        _validate(root, pack),
        source_image_ids=["a"],
        state=APPROVED,
        reviewed_by="Dio",
    )
    assert _state(_validate(root, pack), "a").state == APPROVED

    (root / "labels" / "a.txt").write_text(
        COMPLETE_LABEL.replace("0.2 0.2", "0.9 0.9"), encoding="utf-8"
    )

    sample = _state(_validate(root, pack), "a")
    assert sample.state == NEEDS_REVIEW
    assert PROBLEM_APPROVAL_STALE in sample.problems
    assert sample.is_eligible is False


def test_rejection_is_allowed_even_when_the_label_is_broken(tmp_path):
    """A person may always discard something."""
    root = _labelling_dir(tmp_path, {"a": "garbage\n"})
    pack = _pack(tmp_path, ["a"])

    recorded, _ = record_decision(
        root,
        _validate(root, pack),
        source_image_ids=["a"],
        state=REJECTED,
        reviewed_by="Dio",
    )

    assert recorded == ["a"]
    sample = _state(_validate(root, pack), "a")
    assert sample.state == REJECTED
    assert sample.is_eligible is False


def test_decisions_survive_a_reload(tmp_path):
    root = _labelling_dir(tmp_path, {"a": COMPLETE_LABEL})
    record_decision(
        root,
        _validate(root, _pack(tmp_path, ["a"])),
        source_image_ids=["a"],
        state=APPROVED,
        reviewed_by="Dio",
        note="checked twice",
    )

    decisions = load_decisions(root)

    assert decisions["a"].state == APPROVED
    assert decisions["a"].note == "checked twice"
    assert decisions["a"].label_sha256


def test_an_unreadable_decision_file_is_not_treated_as_no_decisions(tmp_path):
    root = _labelling_dir(tmp_path, {"a": COMPLETE_LABEL})
    (root / "review_state.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(LabelReviewError, match="silently discard approvals"):
        _validate(root, _pack(tmp_path, ["a"]))


# ---------------------------------------------------------------------------
# Coverage and staging


def _approved_population(tmp_path, counts):
    samples = {}
    groups = {}
    for group, total in counts.items():
        for index in range(total):
            name = f"{group}_{index}"
            samples[name] = COMPLETE_LABEL
            groups[name] = group
    root = _labelling_dir(tmp_path, samples)
    pack = _pack(tmp_path, list(samples), groups)
    record_decision(
        root,
        _validate(root, pack),
        source_image_ids=sorted(samples),
        state=APPROVED,
        reviewed_by="Dio",
    )
    return root, pack


def test_coverage_reports_a_thin_group_as_insufficient(tmp_path):
    root, pack = _approved_population(
        tmp_path, {"red_orange_critical": 12, "hard_case": 3}
    )

    stats = coverage(
        _validate(root, pack).samples,
        groups=["red_orange_critical", "hard_case", "representative"],
        min_group_samples=10,
    )

    assert stats["per_group"]["red_orange_critical"]["status"] == "OK"
    assert stats["per_group"]["hard_case"]["status"] == "INSUFFICIENT"
    assert stats["per_group"]["hard_case"]["shortfall"] == 7
    assert stats["insufficient_groups"] == ["hard_case", "representative"]


def test_coverage_counts_instances_per_class(tmp_path):
    root, pack = _approved_population(tmp_path, {"hard_case": 2})

    stats = coverage(
        _validate(root, pack).samples, groups=["hard_case"], min_group_samples=1
    )

    assert stats["total_images"] == 2
    assert stats["total_instances"] == 12
    assert stats["per_class_instances"] == {
        "Black": 4,
        "Green": 2,
        "Orange": 2,
        "Red": 2,
        "Yellow": 2,
    }


def test_only_approved_samples_are_staged(tmp_path):
    root = _labelling_dir(
        tmp_path, {"good": COMPLETE_LABEL, "unapproved": COMPLETE_LABEL}
    )
    pack = _pack(tmp_path, ["good", "unapproved"])
    record_decision(
        root,
        _validate(root, pack),
        source_image_ids=["good"],
        state=APPROVED,
        reviewed_by="Dio",
    )

    staged, groups = stage_approved(
        _validate(root, pack), tmp_path / "golden", schema=SCHEMA
    )

    names = sorted(p.stem for p in (staged / "images" / "val").iterdir())
    assert names == ["good"]
    assert sorted(p.stem for p in (staged / "labels" / "val").iterdir()) == ["good"]
    assert list(groups.values()) == ["hard_case"]


def test_the_staged_descriptor_is_one_ultralytics_accepts(tmp_path):
    """check_det_dataset raises without both train and val."""
    root, pack = _approved_population(tmp_path, {"hard_case": 2})

    staged, _ = stage_approved(
        _validate(root, pack), tmp_path / "golden", schema=SCHEMA
    )

    payload = yaml.safe_load((staged / "data.yaml").read_text(encoding="utf-8"))
    assert payload["train"] == payload["val"] == "images/val"
    assert payload["names"] == {
        0: "Black",
        1: "Green",
        2: "Orange",
        3: "Red",
        4: "Yellow",
    }


def test_staging_nothing_is_refused(tmp_path):
    root = _labelling_dir(tmp_path, {"a": COMPLETE_LABEL})

    with pytest.raises(LabelReviewError, match="nothing to register"):
        stage_approved(
            _validate(root, _pack(tmp_path, ["a"])),
            tmp_path / "golden",
            schema=SCHEMA,
        )


def test_staging_into_an_occupied_directory_is_refused(tmp_path):
    root, pack = _approved_population(tmp_path, {"hard_case": 1})
    target = tmp_path / "golden"
    target.mkdir()
    (target / "something").write_text("in the way", encoding="utf-8")

    with pytest.raises(LabelReviewError, match="already exists"):
        stage_approved(_validate(root, pack), target, schema=SCHEMA)


def test_a_partial_set_can_be_staged(tmp_path):
    """One group finished, the others not started: still buildable."""
    root, pack = _approved_population(tmp_path, {"red_orange_critical": 4})

    staged, groups = stage_approved(
        _validate(root, pack), tmp_path / "golden", schema=SCHEMA
    )

    assert len(list((staged / "images" / "val").iterdir())) == 4
    assert set(groups.values()) == {"red_orange_critical"}


# ---------------------------------------------------------------------------
# Reading the pack


def test_the_pack_is_indexed_by_content_hash(tmp_path):
    import csv

    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    with open(
        pack_dir / "review_pack.csv", "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_image_id", "group", "image_sha256"])
        writer.writerow(["shot", "hard_case", "a" * 64])

    samples = read_pack_samples(pack_dir)

    assert samples["a" * 64].source_image_id == "shot"
    assert samples["a" * 64].group == "hard_case"


def test_a_pack_without_hashes_cannot_identify_anything(tmp_path):
    import csv

    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    with open(
        pack_dir / "review_pack.csv", "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_image_id", "group", "image_sha256"])
        writer.writerow(["shot", "hard_case", ""])

    with pytest.raises(LabelReviewError, match="no images with a content hash"):
        read_pack_samples(pack_dir)


def test_the_report_serialises(tmp_path):
    root = _labelling_dir(tmp_path, {"a": COMPLETE_LABEL, "b": None})

    payload = json.loads(
        json.dumps(_validate(root, _pack(tmp_path, ["a", "b"])).to_dict())
    )

    assert payload["by_state"][LABELED] == 1
    assert payload["by_state"][NEEDS_LABEL] == 1
    assert payload["eligible"] == 0
    assert payload["class_schema"]["schema_hash"] == SCHEMA.schema_hash
