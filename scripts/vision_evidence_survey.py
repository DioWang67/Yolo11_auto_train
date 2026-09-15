#!/usr/bin/env python3
"""Ask the vision model about boxes the bootstrapper already weighed.

Turns the third evidence source into a measurement: for every box, what the
detector called it, what the colour measurement called it, and what the
vision model calls it, on the same pixels and without either seeing the
others' answer.

The number worth reading is not the agreement rate. Two sources looking at
the same image can be wrong together, and this station's whole problem is
that its detector is confidently wrong about Red and Orange. The number
worth reading is how many boxes have **independent consensus** --- colour
and vision naming the same class, and not the detector's. Those are the
highest-information samples in the pile: a reviewer confirms rather than
adjudicates, and once confirmed they are ground truth for everything else.

Reads reports and images. Changes no dataset and trains nothing.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picture_tool.autotrain.class_schema import ClassSchema  # noqa: E402
from picture_tool.bootstrap import vision_client as vc  # noqa: E402
from picture_tool.bootstrap.evidence import Box, BoxOpinion  # noqa: E402
from picture_tool.bootstrap.profile import ProductProfile  # noqa: E402
from picture_tool.bootstrap.vision_evidence import (  # noqa: E402
    VisionLLMEvidence,
    VisionPrompt,
)

LOGGER = logging.getLogger("vision_evidence_survey")

DESCRIPTION = (
    "Each object is a wire end where it meets its solder pad. Box only that "
    "short segment, not the wire running away from it."
)
REPORTS = ("needs_review.json", "accepted.json", "rejected.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reports", type=Path,
                   default=PROJECT_ROOT / "runs" / "bootstrap_poc" / "report")
    p.add_argument("--pack", type=Path,
                   default=PROJECT_ROOT / "runs" / "review_pack" / "v1")
    p.add_argument("--group", default="", help="Only this review-pack group.")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "runs" / "vision_evidence_survey")
    p.add_argument("--classes", default="Black,Green,Orange,Red,Yellow")
    p.add_argument("--describe", default=DESCRIPTION)
    p.add_argument("--url-env", default="QWEN_URL")
    p.add_argument("--key-env", default="")
    p.add_argument("--model", default="Qwen3.8-27B-GGUF")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args(argv)


def load_samples(reports: Path, pack: Path, group: str) -> list[dict]:
    """Every sample the bootstrapper decided, with its review-pack group."""
    groups = {}
    csv_path = pack / "review_pack.csv"
    if csv_path.is_file():
        groups = {r["sample_id"]: r["group"]
                  for r in csv.DictReader(csv_path.open(encoding="utf-8-sig"))}
    out = []
    for name in REPORTS:
        path = reports / name
        if not path.is_file():
            continue
        for sample in json.loads(path.read_text(encoding="utf-8")):
            sample["group"] = groups.get(sample["sample_id"], "")
            if not group or sample["group"] == group:
                out.append(sample)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import cv2

    names = tuple(n.strip() for n in args.classes.split(",") if n.strip())
    profile = ProductProfile(
        product="Cable1", area="A",
        class_schema=ClassSchema(names=names, source="station_contract"),
        expected_counts={"Black": 2, "Green": 1, "Orange": 1, "Red": 1, "Yellow": 1},
        confusion_pairs=(("Red", "Orange"),),
    )
    samples = load_samples(args.reports, args.pack, args.group)
    if args.limit:
        samples = samples[: args.limit]
    LOGGER.info("%d sample(s)%s", len(samples),
                f" in group {args.group}" if args.group else "")

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / f"{args.group or 'all'}.jsonl"
    done: set[str] = set()
    if results_path.exists() and not args.no_resume:
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["sample_id"])
        LOGGER.info("%d already surveyed; resuming", len(done))

    cfg = vc.openai_compatible_profile(
        model=args.model, url_env=args.url_env, key_env=args.key_env,
        requires_credential=bool(args.key_env), max_tokens=3000, max_retries=0,
        timeout_seconds=500.0,
    )
    source = VisionLLMEvidence(
        vc.HttpVisionLLMClient(cfg),
        prompt=VisionPrompt(object_description=args.describe),
    )

    for index, sample in enumerate(samples, start=1):
        if sample["sample_id"] in done:
            continue
        image = cv2.imread(sample["image_path"])
        if image is None:
            LOGGER.warning("unreadable: %s", sample["image_path"])
            continue
        boxes, detector, colour = [], [], []
        for item in sample.get("boxes", []):
            raw = item.get("box", {})
            boxes.append(Box(
                class_name=str(raw.get("class_name", "")),
                cx=float(raw.get("cx", 0.0)), cy=float(raw.get("cy", 0.0)),
                width=float(raw.get("width", 0.0)), height=float(raw.get("height", 0.0)),
                confidence=float(raw.get("confidence", 0.0))))
            opinions = {o.get("source"): o for o in item.get("opinions", [])
                        if isinstance(o, dict)}
            detector.append(str((opinions.get("detector") or {}).get("class_name") or ""))
            colour.append(str((opinions.get("colour") or {}).get("class_name") or ""))
        if not boxes:
            continue
        try:
            vision: list[BoxOpinion] = source.read(image, boxes, profile)
        except Exception as exc:  # noqa: BLE001 - one bad frame must not end a survey
            LOGGER.error("%s: %s", sample["sample_id"], str(exc)[:120])
            continue

        rows = []
        for det, col, vis in zip(detector, colour, vision):
            rows.append({
                "detector": det, "colour": col, "vision": vis.class_name,
                "consensus_against_detector": bool(
                    vis.class_name and col and vis.class_name == col != det),
            })
        record = {"sample_id": sample["sample_id"], "group": sample.get("group", ""),
                  "decision": sample.get("decision", ""), "boxes": rows}
        with results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        consensus = sum(1 for r in rows if r["consensus_against_detector"])
        LOGGER.info("%d/%d %s: %d box(es), %d consensus against detector",
                    index, len(samples), sample["sample_id"][-22:], len(rows), consensus)

    # -- summary, rebuilt from the durable records -------------------------
    records = [json.loads(line) for line in
               results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    per_group: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    pairs = collections.Counter()
    for record in records:
        tally = per_group[record["group"] or "ungrouped"]
        for row in record["boxes"]:
            tally["boxes"] += 1
            if not row["vision"]:
                tally["vision_undecided"] += 1
                continue
            tally["vision_detector_agree"] += row["vision"] == row["detector"]
            if row["colour"]:
                tally["colour_compared"] += 1
                tally["vision_colour_agree"] += row["vision"] == row["colour"]
            if row["consensus_against_detector"]:
                tally["consensus_against_detector"] += 1
                pairs[f"{row['detector']}->{row['vision']}"] += 1

    summary = {
        "samples": len(records),
        "per_group": {g: dict(c) for g, c in sorted(per_group.items())},
        "consensus_corrections": dict(pairs.most_common()),
        "note": "agreement is not accuracy; consensus marks what to label first",
    }
    (args.out / f"{args.group or 'all'}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
