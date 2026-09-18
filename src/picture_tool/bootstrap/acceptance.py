"""Scoring a candidate against boards a person judged, not against itself.

A model trained on 46 frames and measured on a split of those same 46
frames reported mAP50 0.995 and recall 1.000. The same weights, run over
250 boards a person had confirmed, found the right objects on 68% of the
good ones and would have rejected 55 of them; the deployed model rejects
three. Both numbers are real. They differ because one of them was measured
on photographs of the same boards the model learned from.

So this exists to be the other number. It reads the station's acceptance
set --- images with a human OK/NG on each --- and reports what a candidate
does on them, which is the only figure in this project that can be compared
against how the line actually behaves.

**It reads them where they live and never copies them.** The set's own
README says that reusing acceptance images for training invalidates every
future acceptance result, and the way that stays true is that nothing in
this path can move them: :func:`read_boards` refuses a set that has been
copied inside the training tree, because at that point the damage is
already done and the next person to run it deserves to be told.

Scoring is split from running the model on purpose. What counts as a board
the candidate got right is a judgement with a station's contract in it, and
it should be testable without a GPU, a weight file, or ultralytics.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from picture_tool.autotrain import AutoTrainError

GROUND_TRUTH = "ground_truth.csv"

#: The reasons a board was failed that a detector alone can account for.
#: 74 of this station's 77 NG boards are one of these two, which is what
#: makes a detector-only score against this set worth reading at all.
MISSING = "MISSING"
SEQUENCE_MISMATCH = "SEQUENCE_MISMATCH"


class AcceptanceError(AutoTrainError):
    """Raised when the acceptance set cannot be read, or has been misused."""


@dataclass(frozen=True)
class Board:
    """One board, and what a person decided about it."""

    sample_id: str
    image: Path
    verdict: str
    reason: str = ""

    @property
    def is_ok(self) -> bool:
        return self.verdict.upper() == "OK"


def read_boards(root: str | Path, *, training_root: str | Path | None = None
                ) -> list[Board]:
    """The confirmed boards, read in place.

    Only rows a person actually confirmed are returned: a row still awaiting
    review is not ground truth, and counting it as though it were would put
    an unreviewed guess on the same footing as a decision.
    """
    source = Path(root)
    manifest = source / GROUND_TRUTH
    if not manifest.is_file():
        raise AcceptanceError(f"No {GROUND_TRUTH} under {source}")
    if training_root is not None:
        training = Path(training_root).resolve()
        if training == source.resolve() or training in source.resolve().parents:
            raise AcceptanceError(
                f"The acceptance set at {source} sits inside the training "
                f"tree at {training}. Its own README forbids that: images "
                "used for training cannot also measure training, and every "
                "acceptance result taken after the copy is meaningless. "
                "Remove the copy rather than pointing this somewhere else."
            )
    boards: list[Board] = []
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (row.get("review_status") or "").strip() != "confirmed":
                continue
            sample_id = (row.get("sample_id") or "").strip()
            relative = (row.get("image_path") or "").strip()
            if not sample_id or not relative:
                continue
            boards.append(Board(
                sample_id=sample_id,
                image=source / relative,
                verdict=(row.get("expected_verdict") or "").strip(),
                reason=(row.get("expected_reasons") or "").strip(),
            ))
    if not boards:
        raise AcceptanceError(f"No confirmed rows in {manifest}")
    return boards


def score(
    boards: Sequence[Board],
    detected: Mapping[str, Sequence[str]],
    expected: Mapping[str, int],
) -> dict[str, Any]:
    """What a candidate would do to the line, from its detections.

    The headline is ``would_false_reject``: good boards whose objects the
    candidate did not find, each of which is a board the line would stop
    for nothing. It is reported as a count rather than a rate because that
    is the unit the station already thinks in --- the deployed model's
    figure is three.

    ``short_by_class`` totals which classes went missing on those boards,
    so "label more" can name what to photograph instead of asking for more
    of everything.
    """
    wanted = Counter(dict(expected))
    ok = [b for b in boards if b.is_ok]
    missing_boards = [b for b in boards if b.reason == MISSING]

    correct = 0
    short: Counter[str] = Counter()
    counts: Counter[int] = Counter()
    offenders: list[dict[str, Any]] = []
    for board in ok:
        found = detected.get(board.sample_id)
        if found is None:
            counts[-1] += 1
            offenders.append({"sample_id": board.sample_id, "found": None})
            continue
        counts[len(found)] += 1
        seen = Counter(found)
        if seen == wanted:
            correct += 1
            continue
        for name, want in wanted.items():
            gap = want - seen.get(name, 0)
            if gap > 0:
                short[name] += gap
        offenders.append({"sample_id": board.sample_id, "found": list(found)})

    # A board a person failed for MISSING should come back short. A
    # candidate that finds a full set there is not being generous, it is
    # inventing an object that is not on the board.
    invented = sum(
        1 for b in missing_boards
        if Counter(detected.get(b.sample_id) or []) == wanted
    )
    return {
        "independent": True,
        "source": "acceptance set, human-confirmed",
        "boards": len(boards),
        "ok_boards": len(ok),
        "ok_boards_correct": correct,
        "ok_board_accuracy": round(correct / len(ok), 4) if ok else 0.0,
        "would_false_reject": len(ok) - correct,
        "short_by_class": dict(short.most_common()),
        "detected_count_distribution": dict(sorted(counts.items())),
        "missing_boards": len(missing_boards),
        "missing_boards_called_complete": invented,
        "examples": offenders[:5],
    }
