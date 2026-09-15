"""Draft labels built from boxes already on record, and the notes beside them.

The load-bearing test is the class-contract one. Class ids written in the
wrong order would be wrong in all 76 files, would look entirely plausible in
an annotation tool, and would survive into a golden set that every later
promotion decision is measured against.
"""

from __future__ import annotations

import pytest

from scripts.review_pack_drafts import (
    CLASS_NAMES,
    _overlap,
    draft_lines,
    verified_class_ids,
)


def _box(cx: float, cy: float, name: str, confidence: float = 0.9, agreed: bool = True,
         colour: str | None = None) -> dict:
    return {
        "agreed": agreed,
        "box": {
            "class_name": name,
            "confidence": confidence,
            "cx": cx,
            "cy": cy,
            "width": 0.04,
            "height": 0.11,
        },
        "opinions": [
            {"source": "detector", "class_name": name},
            {"source": "colour", "class_name": colour or name},
        ],
    }


def test_the_class_contract_is_verified_not_assumed() -> None:
    """Wrong ids would be invisible in a viewer and permanent in a golden set."""
    ids = verified_class_ids()

    assert ids == {"Black": 0, "Green": 1, "Orange": 2, "Red": 3, "Yellow": 4}
    assert CLASS_NAMES == ("Black", "Green", "Orange", "Red", "Yellow")


def test_a_draft_line_is_yolo_shaped_with_the_contract_id() -> None:
    lines, _ = draft_lines({"boxes": [_box(0.5, 0.6, "Orange")]}, verified_class_ids())

    assert len(lines) == 1
    fields = lines[0].split()
    assert fields[0] == "2"  # Orange
    assert [float(f) for f in fields[1:]] == [0.5, 0.6, 0.04, 0.11]


def test_a_disputed_box_names_both_opinions() -> None:
    _, notes = draft_lines(
        {"boxes": [_box(0.5, 0.6, "Red", 0.62, agreed=False, colour="Orange")]},
        verified_class_ids(),
    )

    assert any("DISPUTED" in n and "Red" in n and "Orange" in n for n in notes)


def test_two_boxes_on_one_object_are_reported_as_a_pair() -> None:
    """Otherwise a count of seven reads as a missing wire rather than a spare box."""
    _, notes = draft_lines(
        {"boxes": [_box(0.5686, 0.6196, "Black", 0.85), _box(0.5686, 0.6195, "Green", 0.29)]},
        verified_class_ids(),
    )

    overlap_notes = [n for n in notes if "OVERLAPPING" in n]
    assert len(overlap_notes) == 1
    assert "boxes 1 and 2" in overlap_notes[0]
    assert "spurious" in overlap_notes[0]


def test_separate_boxes_are_not_called_duplicates() -> None:
    _, notes = draft_lines(
        {"boxes": [_box(0.20, 0.6, "Black"), _box(0.80, 0.6, "Green")]},
        verified_class_ids(),
    )

    assert not [n for n in notes if "OVERLAPPING" in n]


def test_counts_are_compared_against_what_the_station_expects() -> None:
    """Six objects, five classes, Black twice -- physically."""
    _, notes = draft_lines(
        {"boxes": [_box(0.1 * i, 0.6, name) for i, name in enumerate(CLASS_NAMES)]},
        verified_class_ids(),
    )

    # One of each: Black is short, and the total is five rather than six.
    assert any("Black 1/2" in n for n in notes)
    assert any("5 boxes drafted" in n for n in notes)


def test_a_complete_station_draft_raises_no_count_note() -> None:
    boxes = [
        _box(0.1, 0.6, "Black"), _box(0.2, 0.6, "Black"), _box(0.3, 0.6, "Green"),
        _box(0.4, 0.6, "Orange"), _box(0.5, 0.6, "Red"), _box(0.6, 0.6, "Yellow"),
    ]
    lines, notes = draft_lines({"boxes": boxes}, verified_class_ids())

    assert len(lines) == 6
    assert not [n for n in notes if n.startswith("count:")]


def test_an_unknown_class_is_omitted_rather_than_guessed() -> None:
    lines, notes = draft_lines({"boxes": [_box(0.5, 0.6, "Purple")]}, verified_class_ids())

    assert lines == []
    assert any("unknown class" in n for n in notes)


@pytest.mark.parametrize(
    "second_cx, expected",
    [(0.50, 1.0), (0.52, pytest.approx(0.33, abs=0.05)), (0.90, 0.0)],
)
def test_overlap_measures_what_it_claims(second_cx: float, expected: object) -> None:
    first = {"cx": 0.5, "cy": 0.6, "width": 0.04, "height": 0.11}
    second = {"cx": second_cx, "cy": 0.6, "width": 0.04, "height": 0.11}

    assert _overlap(first, second) == expected
