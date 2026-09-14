"""The historical replay's analysis, which is the part that produces evidence.

The training and evaluation in that script can only be exercised for real.
What can be pinned down here is the reasoning it reports: what counts as a
duplicate, what counts as a family, and whether a split keeps families
whole. Those numbers get quoted in decisions, so they are worth a test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

replay = pytest.importorskip(
    "autotrain_historical_replay",
    reason="The replay script imports the pipeline, which needs its deps",
)


def _record(name, split, source=None, dhash="x", label=True):
    return {
        "split": split,
        "image": Path(f"D:/ds/{split}/images/{name}.png"),
        "label": Path(f"D:/ds/{split}/labels/{name}.txt") if label else None,
        "sha256": f"sha-{name}",
        "source": source or name,
        "dhash": dhash,
        "augmented": (source or name) != name,
    }


# ---------------------------------------------------------------------------
# Analysis


def test_augmented_siblings_are_one_source_not_many_images():
    records = [
        _record("shot", "train", dhash="a"),
        _record("shot_aug_1", "train", source="shot", dhash="b"),
        _record("shot_aug_2", "train", source="shot", dhash="c"),
        _record("other", "val", source="other", dhash="d"),
    ]

    stats = replay.analyse({"records": records})

    assert stats["physical_images"] == 4
    assert stats["distinct_sources"] == 2
    assert stats["augmented_files"] == 2


def test_a_source_in_two_splits_is_reported():
    records = [
        _record("shot", "train", dhash="a"),
        _record("shot_aug_1", "val", source="shot", dhash="b"),
    ]

    stats = replay.analyse({"records": records})

    assert stats["source_across_splits"] == 1
    assert stats["train_val_source_overlap"] == 1


def test_near_duplicates_across_splits_need_two_sources_to_count():
    """One capture's own augmentations hashing alike is not a leak.

    The leak is two *different* captures a model cannot tell apart landing
    on opposite sides of the boundary. Counting a single source's variants
    would report leakage in every correctly split dataset.
    """
    same_source = [
        _record("shot", "train", dhash="same"),
        _record("shot_aug_1", "train", source="shot", dhash="same"),
    ]
    assert (
        replay.analyse({"records": same_source})[
            "near_duplicate_groups_across_splits"
        ]
        == 0
    )

    two_sources = [
        _record("a", "train", dhash="same"),
        _record("b", "val", dhash="same"),
    ]
    stats = replay.analyse({"records": two_sources})
    assert stats["near_duplicate_groups_across_splits"] == 1
    assert stats["near_duplicate_files_across_splits"] == 2
    assert stats["near_duplicate_sources_across_splits"] == 2


def test_exact_duplicates_are_counted_as_extra_files():
    records = [_record("a", "train", dhash="a"), _record("a2", "val", dhash="b")]
    records[1]["sha256"] = records[0]["sha256"]

    assert replay.analyse({"records": records})["exact_duplicate_files"] == 1


# ---------------------------------------------------------------------------
# Families


def test_a_family_spans_both_shared_source_and_equal_hash():
    """Two captures that hash alike belong together even so."""
    records = [
        _record("a", "train", dhash="same"),
        _record("a_aug_1", "train", source="a", dhash="other"),
        _record("b", "val", dhash="same"),
        _record("lonely", "val", dhash="unique"),
    ]

    membership = replay.families(records)
    groups = {}
    for record in records:
        groups.setdefault(membership[str(record["image"])], []).append(
            record["image"].stem
        )

    families = sorted(sorted(v) for v in groups.values())
    assert families == [["a", "a_aug_1", "b"], ["lonely"]]


def test_no_family_is_split_across_train_and_val():
    """The property the whole re-split exists to guarantee."""
    records = [
        _record(f"s{i}_aug_{j}", "train", source=f"s{i}", dhash=f"h{i}")
        for i in range(10)
        for j in range(4)
    ]

    assignment = replay.family_split(records, val_fraction=0.3)
    membership = replay.families(records)

    per_family: dict[str, set[str]] = {}
    for record in records:
        key = membership[str(record["image"])]
        per_family.setdefault(key, set()).add(assignment[str(record["image"])])
    assert all(len(splits) == 1 for splits in per_family.values())


def test_the_split_is_deterministic():
    records = [
        _record(f"s{i}", "train", source=f"s{i}", dhash=f"h{i}") for i in range(20)
    ]

    assert replay.family_split(records, val_fraction=0.25) == replay.family_split(
        records, val_fraction=0.25
    )


def test_the_replay_decision_cannot_be_a_promotion():
    """Fixed in the source, not decided from a metric."""
    assert replay.NOT_PROMOTABLE == "NOT_PROMOTABLE_NON_INDEPENDENT"
    assert "PROMOTION_CANDIDATE" not in replay.NOT_PROMOTABLE
    assert replay.NON_INDEPENDENT == "NON_INDEPENDENT"
