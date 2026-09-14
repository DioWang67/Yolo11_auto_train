"""The review pack: what a person is actually asked to annotate.

The load-bearing properties are the ones that decide whether a reviewer's
days are wasted: nothing the model has trained on, nothing counted twice
because it exists as several files, and nothing silently promoted to
ground truth.
"""

from __future__ import annotations

import json

import pytest

from picture_tool.autotrain.class_schema import normalize_class_names
from picture_tool.autotrain.golden_candidates import (
    HARD_CASE,
    NEEDS_ANNOTATION,
    NEEDS_LABEL,
    READY_TO_REVIEW,
    REASON_CORRECTION,
    REASON_MISSED_DETECTION,
    REPRESENTATIVE,
    Candidate,
)
from picture_tool.autotrain.review_pack import (
    EXCLUDED_ALREADY_LABELLED,
    EXCLUDED_TRAINED_PERCEPTUAL,
    EXCLUDED_TRAINED_SHA,
    EXCLUDED_TRAINED_SOURCE,
    PACK_GROUPS,
    RED_ORANGE_CRITICAL,
    SELECTED_CLUSTER_DIVERSITY,
    SELECTED_CRITICAL_CORRECTION,
    SELECTED_CRITICAL_COUNT,
    ReviewPackError,
    allocate,
    build_review_pack,
    collapse_to_sources,
    critical_reasons,
    diversity_select,
    exclude_trained,
    expected_class_counts,
    perceptual_matches,
    write_pack,
)

SCHEMA = normalize_class_names(
    ["Black", "Green", "Orange", "Red", "Yellow"], source="test"
)

#: The real Cable1/A multiset: six objects, five classes, Black twice.
EXPECTED_ITEMS = ["Red", "Green", "Orange", "Yellow", "Black", "Black"]
EXPECTED_COUNTS = {"Black": 2, "Green": 1, "Orange": 1, "Red": 1, "Yellow": 1}

#: A complete, correct verified count. Deviating from this is what puts an
#: image in the critical group.
GOOD_COUNTS = dict(EXPECTED_COUNTS)


def _candidate(
    sample_id: str,
    *,
    source: str | None = None,
    sha: str | None = None,
    group: str = REPRESENTATIVE,
    reasons: tuple[str, ...] = (),
    corrections: tuple[str, ...] = (),
    counts: dict[str, int] | None = None,
    status: str = NEEDS_ANNOTATION,
    label_path: str = "",
    duplicate_group: str = "",
    timestamp: str = "2026-07-01T10:00:00",
    min_confidence: float | None = None,
    brightness: float | None = None,
) -> Candidate:
    return Candidate(
        sample_id=sample_id,
        image_path=f"D:/Result/{sample_id}.jpg",
        status=status,
        label_path=label_path,
        group=group,
        source="production:PASS",
        timestamp=timestamp,
        reasons=reasons,
        corrections=corrections,
        class_counts=dict(GOOD_COUNTS if counts is None else counts),
        detection_count=6,
        min_confidence=min_confidence,
        brightness=brightness,
        duplicate_group=duplicate_group or f"single:{sample_id}",
        source_image_id=source or sample_id,
        image_sha256=sha or (sample_id * 8)[:64],
    )


# ---------------------------------------------------------------------------
# Exclusion


def test_expected_counts_come_from_the_station_multiset():
    """expected_items lists Black twice because the station has two."""
    assert expected_class_counts(EXPECTED_ITEMS) == EXPECTED_COUNTS


def test_an_image_the_model_trained_on_is_removed_not_flagged():
    candidates = [
        _candidate("clean"),
        _candidate("trained", sha="a" * 64),
    ]

    kept, excluded = exclude_trained(candidates, trained_sha256={"a" * 64})

    assert [c.sample_id for c in kept] == ["clean"]
    assert excluded[EXCLUDED_TRAINED_SHA] == ["trained"]


def test_a_derivative_of_a_trained_capture_goes_with_its_source():
    """Augmentation runs before the split, so the bytes match nothing."""
    candidates = [
        _candidate("shot_aug_3", source="shot"),
        _candidate("other", source="other"),
    ]

    kept, excluded = exclude_trained(candidates, trained_source_ids={"shot"})

    assert [c.sample_id for c in kept] == ["other"]
    assert excluded[EXCLUDED_TRAINED_SOURCE] == ["shot_aug_3"]


def test_a_re_encoded_renamed_training_copy_is_still_excluded():
    """The hole between the byte check and the lineage check.

    A handoff copy is re-encoded (so its sha matches nothing) and renamed
    to `<uuid>-yolo_Cable1_A_142254.jpg`, which shares no structure with
    production's `yolo_Cable1_A_142252_433429_<hex>.jpg` --- so lineage
    matching between them cannot succeed, not merely fails to. Five images
    in the first real pack were the same photograph as a training image
    while passing both other checks.
    """
    candidates = [
        _candidate("looks_new", source="looks_new", sha="f" * 64),
        _candidate("genuinely_new", source="genuinely_new", sha="e" * 64),
    ]

    kept, excluded = exclude_trained(
        candidates,
        trained_source_ids={"something_else"},
        trained_sha256={"a" * 64},
        perceptually_trained={"looks_new"},
    )

    assert [c.sample_id for c in kept] == ["genuinely_new"]
    assert excluded[EXCLUDED_TRAINED_PERCEPTUAL] == ["looks_new"]


def test_perceptual_matching_uses_the_projects_own_hash():
    candidates = [
        _candidate("same_picture"),
        _candidate("different"),
    ]
    hashes = {"D:/Result/same_picture.jpg": "abc", "D:/Result/different.jpg": "xyz"}

    matched = perceptual_matches(
        candidates, {"abc"}, hasher=lambda path: hashes.get(str(path))
    )

    assert matched == {"same_picture"}


def test_an_unreadable_image_is_not_treated_as_trained():
    """Unreadable is not the same as memorised."""
    matched = perceptual_matches(
        [_candidate("broken")], {"abc"}, hasher=lambda path: None
    )

    assert matched == set()


def test_no_training_hashes_means_no_perceptual_work():
    calls: list[str] = []

    matched = perceptual_matches(
        [_candidate("a")], set(), hasher=lambda path: calls.append(str(path))
    )

    assert matched == set()
    assert calls == []


def test_already_labelled_candidates_are_not_review_pack_material():
    """The pack asks for labels; a labelled image is not asking for one."""
    candidates = [
        _candidate("fresh"),
        _candidate(
            "done", status=READY_TO_REVIEW, label_path="D:/labels/done.txt"
        ),
    ]

    kept, excluded = exclude_trained(candidates)

    assert [c.sample_id for c in kept] == ["fresh"]
    assert excluded[EXCLUDED_ALREADY_LABELLED] == ["done"]


# ---------------------------------------------------------------------------
# Source-level collapse


def test_many_files_of_one_capture_count_as_one_source():
    candidates = [
        _candidate("shot", source="shot"),
        _candidate("shot_aug_1", source="shot"),
        _candidate("shot_aug_2", source="shot"),
        _candidate("elsewhere", source="elsewhere"),
    ]

    kept, dropped = collapse_to_sources(candidates)

    assert sorted(c.source_image_id for c in kept) == ["elsewhere", "shot"]
    assert dropped == 2


def test_the_survivor_of_a_source_is_the_row_that_knows_most():
    candidates = [
        _candidate("shot_aug_1", source="shot"),
        _candidate(
            "shot",
            source="shot",
            reasons=(REASON_CORRECTION, REASON_MISSED_DETECTION),
        ),
    ]

    kept, _ = collapse_to_sources(candidates)

    assert [c.sample_id for c in kept] == ["shot"]


# ---------------------------------------------------------------------------
# The critical group


def test_a_corrected_red_or_orange_call_is_critical():
    """Red->Orange is this station's single largest confusion."""
    candidate = _candidate("x", corrections=("Red->Orange",))

    reasons = critical_reasons(
        candidate,
        critical_classes=("Red", "Orange"),
        expected_counts=EXPECTED_COUNTS,
    )

    assert SELECTED_CRITICAL_CORRECTION in reasons


def test_a_substitution_within_a_complete_image_is_critical():
    """Six objects found, but two Oranges and no Red: a colour swap."""
    candidate = _candidate(
        "x", counts={"Black": 2, "Green": 1, "Orange": 2, "Yellow": 1}
    )

    reasons = critical_reasons(
        candidate,
        critical_classes=("Red", "Orange"),
        expected_counts=EXPECTED_COUNTS,
    )

    assert SELECTED_CRITICAL_COUNT in reasons


def test_an_image_the_model_missed_entirely_is_not_a_colour_confusion():
    """The rule that keeps the critical group critical.

    935 of this station's production rows carry no detections at all, and
    every one of them has zero Reds. Counting those as Red/Orange failures
    put 63% of all candidates in the critical group on the first real run.
    A missed detection is a hard case; it is not a colour swap.
    """
    candidate = _candidate("x", counts={})

    reasons = critical_reasons(
        candidate,
        critical_classes=("Red", "Orange"),
        expected_counts=EXPECTED_COUNTS,
    )

    assert SELECTED_CRITICAL_COUNT not in reasons


def test_a_partial_detection_is_not_a_colour_confusion_either():
    candidate = _candidate("x", counts={"Black": 2, "Green": 1})

    assert (
        SELECTED_CRITICAL_COUNT
        not in critical_reasons(
            candidate,
            critical_classes=("Red", "Orange"),
            expected_counts=EXPECTED_COUNTS,
        )
    )


def test_a_corrected_call_is_critical_even_on_an_incomplete_image():
    """A person actually changed that call; nothing is being inferred."""
    candidate = _candidate("x", counts={}, corrections=("Red->Orange",))

    assert SELECTED_CRITICAL_CORRECTION in critical_reasons(
        candidate,
        critical_classes=("Red", "Orange"),
        expected_counts=EXPECTED_COUNTS,
    )


def test_a_correction_between_two_other_classes_is_not_critical():
    candidate = _candidate("x", corrections=("Green->Black",))

    assert (
        critical_reasons(
            candidate,
            critical_classes=("Red", "Orange"),
            expected_counts=EXPECTED_COUNTS,
        )
        == ()
    )


def test_a_correct_image_is_not_critical():
    assert (
        critical_reasons(
            _candidate("x"),
            critical_classes=("Red", "Orange"),
            expected_counts=EXPECTED_COUNTS,
        )
        == ()
    )


# ---------------------------------------------------------------------------
# Diversity


def test_a_cluster_contributes_several_members_not_one():
    """Collapsing look-alikes to a single sample throws away real variation."""
    members = [
        _candidate(f"c{i}", brightness=float(i * 10), duplicate_group="phash:aa")
        for i in range(8)
    ]

    picked = diversity_select(members, 3)

    assert len(picked) == 3


def test_the_spread_reaches_both_ends_of_the_cluster():
    members = [
        _candidate(f"c{i}", brightness=float(i), duplicate_group="phash:aa")
        for i in range(10)
    ]

    picked = diversity_select(members, 3)
    brightness = sorted(c.brightness for c in picked)

    # Farthest-point must reach the extremes; taking the head would not.
    assert brightness[0] == pytest.approx(0.0)
    assert brightness[-1] == pytest.approx(9.0)


def test_the_first_pick_is_the_most_interesting_member():
    """A cluster cut to one place still surrenders its best sample."""
    members = [
        _candidate("dull", brightness=1.0, duplicate_group="phash:aa"),
        _candidate(
            "interesting",
            brightness=2.0,
            duplicate_group="phash:aa",
            reasons=(REASON_CORRECTION, REASON_MISSED_DETECTION),
        ),
    ]

    assert [c.sample_id for c in diversity_select(members, 1)] == ["interesting"]


def test_a_cluster_smaller_than_the_limit_is_kept_whole():
    members = [_candidate("a"), _candidate("b")]

    assert len(diversity_select(members, 5)) == 2


# ---------------------------------------------------------------------------
# Sizing


def test_a_group_that_cannot_fill_its_share_gives_it_away():
    quota = allocate(
        {RED_ORANGE_CRITICAL: 2, HARD_CASE: 500, REPRESENTATIVE: 500}, 200
    )

    assert quota[RED_ORANGE_CRITICAL] == 2
    assert sum(quota.values()) == 200


def test_no_group_is_padded_past_what_exists():
    quota = allocate(
        {RED_ORANGE_CRITICAL: 1, HARD_CASE: 2, REPRESENTATIVE: 3}, 200
    )

    assert quota == {
        RED_ORANGE_CRITICAL: 1,
        HARD_CASE: 2,
        REPRESENTATIVE: 3,
    }


def test_representative_keeps_a_share_against_a_flood_of_hard_cases():
    """A pack that is all hard cases stops describing the line."""
    quota = allocate(
        {RED_ORANGE_CRITICAL: 1000, HARD_CASE: 1000, REPRESENTATIVE: 1000}, 200
    )

    assert quota[REPRESENTATIVE] >= 40
    assert sum(quota.values()) == 200


# ---------------------------------------------------------------------------
# The pass end to end


def _population():
    candidates = []
    for index in range(40):
        candidates.append(
            _candidate(
                f"crit{index}",
                corrections=("Red->Orange",),
                reasons=(REASON_CORRECTION,),
                timestamp=f"2026-07-{(index % 20) + 1:02d}T10:00:00",
            )
        )
    for index in range(60):
        candidates.append(
            _candidate(
                f"hard{index}",
                group=HARD_CASE,
                reasons=(REASON_MISSED_DETECTION,),
                timestamp=f"2026-07-{(index % 20) + 1:02d}T11:00:00",
            )
        )
    for index in range(60):
        candidates.append(
            _candidate(
                f"plain{index}",
                timestamp=f"2026-07-{(index % 20) + 1:02d}T12:00:00",
            )
        )
    return candidates


def test_the_pack_is_measured_in_distinct_sources():
    candidates = _population() + [
        _candidate(f"crit0_aug_{i}", source="crit0") for i in range(5)
    ]

    entries, summary = build_review_pack(
        candidates, expected_counts=EXPECTED_COUNTS, target_min=10, target_max=100
    )

    sources = [entry.source_image_id for entry in entries]
    assert len(sources) == len(set(sources))
    assert summary["distinct_sources_selected"] == len(entries)


def test_every_entry_leaves_needing_a_label():
    """No prediction is ever promoted to ground truth by this path."""
    entries, summary = build_review_pack(
        _population(), expected_counts=EXPECTED_COUNTS, target_min=10, target_max=100
    )

    assert entries
    assert all(entry.status == NEEDS_LABEL for entry in entries)
    assert summary["all_need_label"] is True


def test_every_entry_says_why_it_was_selected():
    entries, _ = build_review_pack(
        _population(), expected_counts=EXPECTED_COUNTS, target_min=10, target_max=100
    )

    assert all(entry.selected_reasons for entry in entries)
    critical = [e for e in entries if e.group == RED_ORANGE_CRITICAL]
    assert critical
    assert all(
        SELECTED_CRITICAL_CORRECTION in e.selected_reasons for e in critical
    )


def test_trained_sources_never_reach_the_pack():
    candidates = _population()
    trained = {c.source_image_id for c in candidates[:30]}

    entries, summary = build_review_pack(
        candidates,
        trained_source_ids=trained,
        expected_counts=EXPECTED_COUNTS,
        target_min=10,
        target_max=200,
    )

    assert not trained & {entry.source_image_id for entry in entries}
    assert summary["excluded"][EXCLUDED_TRAINED_SOURCE] == 30


def test_all_three_groups_are_present_and_counted():
    entries, summary = build_review_pack(
        _population(), expected_counts=EXPECTED_COUNTS, target_min=10, target_max=100
    )

    assert set(summary["per_group"]) == set(PACK_GROUPS)
    assert sum(summary["per_group"].values()) == len(entries)
    assert summary["per_group"][RED_ORANGE_CRITICAL] > 0
    assert summary["per_group"][REPRESENTATIVE] > 0


def test_a_shortfall_is_reported_rather_than_padded():
    entries, summary = build_review_pack(
        _population()[:20], expected_counts=EXPECTED_COUNTS, target_min=150
    )

    assert summary["meets_target_min"] is False
    assert summary["shortfall"] == 150 - len(entries)


def test_an_impossible_target_is_refused():
    with pytest.raises(ReviewPackError, match="exceeds target_max"):
        build_review_pack([], target_min=300, target_max=100)


def test_near_duplicates_are_spread_not_deleted():
    cluster = [
        _candidate(
            f"look{i}",
            duplicate_group="phash:same",
            brightness=float(i),
            timestamp=f"2026-07-{(i % 20) + 1:02d}T10:00:00",
        )
        for i in range(12)
    ]

    entries, summary = build_review_pack(
        cluster, expected_counts=EXPECTED_COUNTS, target_min=1, per_cluster=3
    )

    assert len(entries) == 3
    assert summary["thinned_by_cluster_diversity"] == 9
    assert all(
        SELECTED_CLUSTER_DIVERSITY in entry.selected_reasons for entry in entries
    )


# ---------------------------------------------------------------------------
# Writing


def test_the_pack_registers_nothing(tmp_path):
    """The load-bearing test. A pack is a reading list, not evidence."""
    entries, summary = build_review_pack(
        _population(), expected_counts=EXPECTED_COUNTS, target_min=5, target_max=20
    )

    write_pack(entries, summary, tmp_path / "pack", schema=SCHEMA, copy_images=False)

    assert not list((tmp_path / "pack").rglob("golden_manifest.json"))
    assert not list((tmp_path / "pack").rglob("*.txt"))


def test_the_written_pack_carries_the_reasons(tmp_path):
    import csv as _csv

    entries, summary = build_review_pack(
        _population(), expected_counts=EXPECTED_COUNTS, target_min=5, target_max=20
    )

    written = write_pack(
        entries, summary, tmp_path / "pack", schema=SCHEMA, copy_images=False
    )

    with open(written["csv"], encoding="utf-8-sig", newline="") as handle:
        rows = list(_csv.DictReader(handle))
    assert rows
    assert all(row["selected_reasons"] for row in rows)
    assert all(row["status"] == NEEDS_LABEL for row in rows)


def test_the_readme_says_in_writing_that_nothing_is_golden(tmp_path):
    entries, summary = build_review_pack(
        _population(), expected_counts=EXPECTED_COUNTS, target_min=5, target_max=20
    )

    written = write_pack(
        entries, summary, tmp_path / "pack", schema=SCHEMA, copy_images=False
    )

    text = written["readme"].read_text(encoding="utf-8")
    assert "not a golden set" in text
    assert "NEEDS_LABEL" in text
    assert "six" in text


def test_the_summary_is_machine_readable(tmp_path):
    entries, summary = build_review_pack(
        _population(), expected_counts=EXPECTED_COUNTS, target_min=5, target_max=20
    )

    written = write_pack(
        entries, summary, tmp_path / "pack", schema=SCHEMA, copy_images=False
    )

    payload = json.loads(written["summary"].read_text(encoding="utf-8"))
    assert payload["distinct_sources_selected"] == len(entries)
    assert payload["images_copied"] is False


def test_copying_images_brings_them_into_the_pack(tmp_path):
    """Production retention deletes passing images after thirty days."""
    source_dir = tmp_path / "production"
    source_dir.mkdir()
    candidates = []
    for index in range(4):
        image = source_dir / f"shot{index}.jpg"
        image.write_bytes(f"image-{index}".encode("utf-8"))
        candidates.append(
            Candidate(
                sample_id=f"shot{index}",
                image_path=str(image),
                status=NEEDS_ANNOTATION,
                group=REPRESENTATIVE,
                source="production:PASS",
                class_counts=dict(GOOD_COUNTS),
                source_image_id=f"shot{index}",
                image_sha256=f"{index}" * 64,
                duplicate_group=f"single:shot{index}",
            )
        )

    entries, summary = build_review_pack(
        candidates, expected_counts=EXPECTED_COUNTS, target_min=1
    )
    write_pack(entries, summary, tmp_path / "pack", schema=SCHEMA, copy_images=True)

    copied = sorted((tmp_path / "pack" / "images").iterdir())
    assert len(copied) == len(entries)
    # Read, never moved: production still has every file.
    assert len(sorted(source_dir.iterdir())) == 4
