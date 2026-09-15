#!/usr/bin/env python3
"""Turn the boxes already on record into draft labels a person can correct.

Annotating the review pack means six boxes an image, and drawing all of them
from scratch is most of the work. The geometry is not actually unknown: every
NEEDS_REVIEW record already carries the detector's boxes, the independent
colour opinion on each, and which of the two disagreed. This writes that out
in YOLO form so the task becomes correcting boxes rather than drawing them,
and marks where the evidence is in dispute so attention lands there.

**No model is consulted and nothing here is ground truth.** The drafts are
the detector's output, which is exactly what the review pack's own README
forbids promoting --- a production record says what the model *found*, never
what was there. They go to ``labels_draft/`` and never to ``labels/``, so
nothing can be mistaken for finished work, and the registration path cannot
see them.

The cost of a draft is anchoring: a wrong box that looks plausible is easier
to accept than an empty image is to mislabel. That is why every draft image
gets a note saying which boxes the colour measurement disputed and how the
class counts differ from what the station expects --- the places where the
draft is most likely to be wrong are named rather than left to be noticed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picture_tool.autotrain.class_schema import ClassSchema  # noqa: E402

LOGGER = logging.getLogger("review_pack_drafts")

#: The station's contract, as the review pack states it. Verified against the
#: schema hash below rather than trusted: class ids written in the wrong order
#: would be silently wrong in every draft, and wrong in a way that survives
#: into a golden set.
CLASS_NAMES = ("Black", "Green", "Orange", "Red", "Yellow")
CLASS_SCHEMA_HASH = "05f915927011ba63db6d16d535e714c014b53f3f6602391688536aa5b3119df9"

#: Six objects across five classes; Black appears twice, physically.
EXPECTED_COUNTS = {"Black": 2, "Green": 1, "Orange": 1, "Red": 1, "Yellow": 1}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pack", type=Path, default=PROJECT_ROOT / "runs" / "review_pack" / "v1")
    p.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "runs" / "bootstrap_poc" / "report" / "needs_review.json",
    )
    p.add_argument("--group", default="red_orange_critical")
    return p.parse_args(argv)


def verified_class_ids() -> dict[str, int]:
    """Class ids, refused unless the contract still hashes to what we expect."""
    schema = ClassSchema(names=CLASS_NAMES, source="review_pack_contract")
    if schema.schema_hash != CLASS_SCHEMA_HASH:
        raise SystemExit(
            "The class contract no longer hashes to the value the review pack "
            f"records ({CLASS_SCHEMA_HASH[:12]}...). Refusing to write drafts: "
            "class ids in the wrong order would be wrong in every file and "
            "would survive into a golden set."
        )
    return {name: index for index, name in enumerate(schema.names)}


def _overlap(first: dict, second: dict) -> float:
    """Intersection over union of two centre-form boxes."""
    boxes = []
    for box in (first, second):
        cx, cy = float(box.get("cx", 0.0)), float(box.get("cy", 0.0))
        w, h = float(box.get("width", 0.0)), float(box.get("height", 0.0))
        boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = boxes
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def draft_lines(sample: dict, class_ids: dict[str, int]) -> tuple[list[str], list[str]]:
    """One YOLO file's lines, and the notes a reviewer should read first."""
    lines: list[str] = []
    notes: list[str] = []
    for index, item in enumerate(sample.get("boxes", []), start=1):
        box = item.get("box", {})
        name = str(box.get("class_name", ""))
        if name not in class_ids:
            notes.append(f"box {index}: unknown class {name!r}, omitted from the draft")
            continue
        lines.append(
            f"{class_ids[name]} {float(box.get('cx', 0.0)):.6f} "
            f"{float(box.get('cy', 0.0)):.6f} {float(box.get('width', 0.0)):.6f} "
            f"{float(box.get('height', 0.0)):.6f}"
        )
        if not item.get("agreed", True):
            opinions = {
                o.get("source"): o for o in item.get("opinions", []) if isinstance(o, dict)
            }
            colour = (opinions.get("colour") or {}).get("class_name") or "undecided"
            notes.append(
                f"box {index}: DISPUTED --- detector says {name} "
                f"({float(box.get('confidence', 0.0)):.2f}), colour says {colour}"
            )

    # Two boxes on one object is the usual reason a count comes out at seven,
    # and as an arithmetic hint it is easy to misread as a missing wire. Name
    # the pair instead: the reviewer deletes one rather than hunting for the
    # object that was never there.
    boxes = sample.get("boxes", [])
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _overlap(boxes[i].get("box", {}), boxes[j].get("box", {})) > 0.6:
                first = boxes[i].get("box", {})
                second = boxes[j].get("box", {})
                notes.append(
                    f"boxes {i + 1} and {j + 1}: OVERLAPPING --- "
                    f"{first.get('class_name')} ({float(first.get('confidence', 0)):.2f}) "
                    f"and {second.get('class_name')} "
                    f"({float(second.get('confidence', 0)):.2f}) on one object; "
                    "one of them is spurious"
                )

    counts = Counter(
        str(b.get("box", {}).get("class_name", "")) for b in sample.get("boxes", [])
    )
    for name in CLASS_NAMES:
        want, got = EXPECTED_COUNTS[name], counts.get(name, 0)
        if want != got:
            notes.append(f"count: {name} {got}/{want}")
    total = sum(counts.values())
    if total != sum(EXPECTED_COUNTS.values()):
        notes.append(f"count: {total} boxes drafted, the station expects 6")
    return lines, notes


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    class_ids = verified_class_ids()
    rows = list(csv.DictReader((args.pack / "review_pack.csv").open(encoding="utf-8-sig")))
    wanted = {r["sample_id"]: r for r in rows if r["group"] == args.group}
    samples = {
        s["sample_id"]: s
        for s in json.loads(args.report.read_text(encoding="utf-8"))
        if s["sample_id"] in wanted
    }
    LOGGER.info("%d image(s) in group %s, %d with boxes on record",
                len(wanted), args.group, len(samples))

    drafts = args.pack / "labels_draft"
    drafts.mkdir(parents=True, exist_ok=True)
    notes_rows = []
    disputed_total = 0

    for sample_id, row in sorted(wanted.items()):
        sample = samples.get(sample_id)
        stem = Path(row["pack_image"]).stem
        if sample is None:
            notes_rows.append({"pack_image": stem, "boxes": 0,
                               "notes": "no NEEDS_REVIEW record; draw from scratch"})
            continue
        lines, notes = draft_lines(sample, class_ids)
        (drafts / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        disputed_total += sum(1 for n in notes if "DISPUTED" in n)
        notes_rows.append({"pack_image": stem, "boxes": len(lines),
                           "notes": " | ".join(notes) or "draft agrees with the station"})

    with (drafts / "DRAFT_NOTES.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pack_image", "boxes", "notes"])
        writer.writeheader()
        writer.writerows(notes_rows)

    overlapping = sum(1 for r in notes_rows if "OVERLAPPING" in str(r["notes"]))
    needs_attention = sum(
        1
        for r in notes_rows
        if "DISPUTED" in str(r["notes"]) or "count:" in str(r["notes"])
    )
    print(json.dumps({
        "group": args.group,
        "images": len(wanted),
        "drafts_written": sum(1 for r in notes_rows if r["boxes"]),
        "disputed_boxes": disputed_total,
        "images_with_overlapping_boxes": overlapping,
        "images_needing_attention": needs_attention,
        "drafts_dir": str(drafts),
        "ground_truth": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
