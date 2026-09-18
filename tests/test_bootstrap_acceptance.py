"""Scoring against boards a person judged.

The number this file exists to produce is ``would_false_reject``. A model
measured on a split of its own training frames reported recall 1.000 on
this station while missing objects on 55 of 173 good boards, so the test
that matters is that a candidate which finds nothing cannot come back
looking fine.
"""

from __future__ import annotations

import csv

import pytest

from picture_tool.bootstrap.acceptance import (
    AcceptanceError,
    Board,
    read_boards,
    score,
)

EXPECTED = {"Black": 2, "Green": 1, "Orange": 1, "Red": 1, "Yellow": 1}
FULL = ["Red", "Green", "Orange", "Yellow", "Black", "Black"]

FIELDS = ["sample_id", "image_path", "expected_verdict", "expected_reasons",
          "review_status"]


def write_manifest(root, rows) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "ground_truth.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def row(sample_id, verdict="OK", reason="", status="confirmed") -> dict:
    return {"sample_id": sample_id, "image_path": f"images/{sample_id}.jpg",
            "expected_verdict": verdict, "expected_reasons": reason,
            "review_status": status}


def boards(n_ok=3, n_missing=0) -> list[Board]:
    out = [Board(f"OK{i}", f"images/OK{i}.jpg", "OK") for i in range(n_ok)]
    out += [Board(f"NG{i}", f"images/NG{i}.jpg", "NG", "MISSING")
            for i in range(n_missing)]
    return out


# -- reading ----------------------------------------------------------------


def test_only_boards_a_person_confirmed_are_ground_truth(tmp_path) -> None:
    write_manifest(tmp_path, [row("A"), row("B", status="pending"), row("C")])

    got = read_boards(tmp_path)

    assert [b.sample_id for b in got] == ["A", "C"]


def test_an_acceptance_set_copied_into_the_training_tree_is_refused(
    tmp_path,
) -> None:
    """Its README: reusing these for training invalidates every later result."""
    inside = tmp_path / "data" / "acceptance"
    write_manifest(inside, [row("A")])

    with pytest.raises(AcceptanceError, match="inside the training tree"):
        read_boards(inside, training_root=tmp_path / "data")

    # Left where it belongs, the same set reads fine.
    assert read_boards(inside, training_root=tmp_path / "elsewhere")


def test_a_set_with_nothing_confirmed_is_an_error_not_an_empty_pass(
    tmp_path,
) -> None:
    write_manifest(tmp_path, [row("A", status="pending")])

    with pytest.raises(AcceptanceError, match="No confirmed rows"):
        read_boards(tmp_path)


# -- scoring ----------------------------------------------------------------


def test_a_candidate_that_finds_everything_rejects_nothing() -> None:
    found = {f"OK{i}": FULL for i in range(3)}

    out = score(boards(3), found, EXPECTED)

    assert out["ok_board_accuracy"] == 1.0
    assert out["would_false_reject"] == 0
    assert out["short_by_class"] == {}


def test_a_candidate_that_finds_nothing_cannot_look_fine() -> None:
    out = score(boards(3), {}, EXPECTED)

    assert out["would_false_reject"] == 3
    assert out["ok_board_accuracy"] == 0.0


def test_missed_objects_are_totalled_by_class_so_they_can_be_photographed() -> None:
    """"Label more" is only actionable if it can say more of what."""
    found = {
        "OK0": ["Red", "Green", "Orange", "Yellow", "Black"],   # one Black short
        "OK1": ["Red", "Green", "Orange", "Black", "Black"],    # no Yellow
        "OK2": FULL,
    }

    out = score(boards(3), found, EXPECTED)

    assert out["would_false_reject"] == 2
    assert out["short_by_class"] == {"Black": 1, "Yellow": 1}
    assert out["detected_count_distribution"] == {5: 2, 6: 1}


def test_a_full_set_on_a_board_that_is_missing_one_is_counted_as_invented() -> None:
    """Generosity on a MISSING board is a hallucinated object, not a pass."""
    found = {"OK0": FULL, "NG0": FULL}

    out = score(boards(1, n_missing=1), found, EXPECTED)

    assert out["missing_boards"] == 1
    assert out["missing_boards_called_complete"] == 1


def test_an_extra_object_fails_the_board_too() -> None:
    found = {"OK0": [*FULL, "Red"], "OK1": FULL, "OK2": FULL}

    out = score(boards(3), found, EXPECTED)

    assert out["would_false_reject"] == 1
    # Nothing was short; the board failed for the opposite reason, and the
    # per-class total must not imply a shortage that is not there.
    assert out["short_by_class"] == {}


def test_the_result_says_it_is_independent() -> None:
    """The whole point is that it is not a split of the training frames."""
    assert score(boards(1), {"OK0": FULL}, EXPECTED)["independent"] is True
