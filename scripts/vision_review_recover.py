#!/usr/bin/env python3
"""Rescue the verdicts of an interrupted review run from its console log.

The first full run over the 213 NEEDS_REVIEW samples was interrupted at 81
and wrote its structured report only at the end, so its answers survived
nowhere but the log it happened to be redirected to. Those answers were paid
for. This reads them back into the ``results.jsonl`` that
``vision_review_dryrun.py`` now appends to, so a resumed run asks only for
what is genuinely missing.

**What comes back is not what was sent.** The log carries the verdict, the
class and the first 80 characters of the reason. It does not carry the
confidence, the next action, or the token usage, and the reason it does carry
is truncated. Every recovered record is therefore marked
``fidelity: recovered_from_log`` and ``reason_truncated: true``, and the
fields the log never held are written as ``null`` rather than guessed at. A
recovered record is evidence of a decision, not a replacement for one.

Failures are recovered too, but as failures: a record with no verdict is
unfinished work, and the resume in ``vision_review_dryrun.py`` asks those
samples again. Two of the three that failed in that run failed on a parser
defect that has since been fixed.

This never calls the endpoint and never touches a dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOGGER = logging.getLogger("vision_review_recover")

#: ``"%d/%d %s: %d crop(s)%s"`` as the run emitted it.
HEADER_RE = re.compile(
    r"\b\d+/\d+\s+(?P<sample_id>\S+):\s+(?P<crops>\d+)\s+crop\(s\)"
    r"(?P<frame>\s*\+\s*whole frame)?\s*$"
)

#: ``"   -> %s %s (%s)"``. The reason is taken between the first ``(`` and the
#: final ``)`` because a truncated reason routinely contains an unbalanced
#: bracket of its own.
VERDICT_RE = re.compile(
    r"->\s+(?P<verdict>[A-Z]+)\s+(?P<class_name>\S*)\s*\((?P<reason>.*)\)\s*$"
)

ERROR_RE = re.compile(r"ERROR\s+\S+:\s+(?P<sample_id>\S+?):\s+(?P<message>.+?)\s*$")

VERDICTS = ("ACCEPT", "RETRY", "REJECT")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log", type=Path, default=PROJECT_ROOT / "runs" / "vision_full.log")
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "runs" / "vision_review_full" / "results.jsonl",
        help="results.jsonl to merge into. Samples already present are left "
        "alone; a live record is never overwritten by a recovered one.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be recovered without writing anything.",
    )
    return p.parse_args(argv)


def recover(lines: list[str]) -> list[dict]:
    """Pair each sample header with the outcome line that follows it."""
    records: list[dict] = []
    pending: dict | None = None

    for line in lines:
        header = HEADER_RE.search(line)
        if header:
            if pending is not None:
                # A header with no outcome after it: the run was cut off
                # mid-request. That sample was never answered.
                LOGGER.info("%s was interrupted mid-request", pending["sample_id"])
            pending = {
                "sample_id": header.group("sample_id"),
                "crops_sent": int(header.group("crops")),
                "whole_frame_sent": bool(header.group("frame")),
                "fidelity": "recovered_from_log",
            }
            continue

        if pending is None:
            continue

        verdict = VERDICT_RE.search(line)
        if verdict and verdict.group("verdict") in VERDICTS:
            records.append(
                {
                    **pending,
                    "verdict": verdict.group("verdict"),
                    "class_name": verdict.group("class_name"),
                    "reason": verdict.group("reason"),
                    # The log truncated it; say so rather than let a reader
                    # take a clipped sentence for the whole argument.
                    "reason_truncated": True,
                    # Never recorded in the log. Guessing any of these would
                    # invent evidence.
                    "confidence": None,
                    "next_action": None,
                    "attempts": None,
                    "usage": {},
                }
            )
            pending = None
            continue

        failure = ERROR_RE.search(line)
        if failure:
            records.append(
                {
                    **pending,
                    "verdict": None,
                    "error": failure.group("message"),
                }
            )
            pending = None

    if pending is not None:
        LOGGER.info("%s was interrupted mid-request", pending["sample_id"])
    return records


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not args.log.is_file():
        raise SystemExit(f"No log at {args.log}")

    records = recover(args.log.read_text(encoding="utf-8", errors="replace").splitlines())
    answered = [r for r in records if r.get("verdict")]
    failed = [r for r in records if not r.get("verdict")]
    counts = {v: sum(1 for r in answered if r["verdict"] == v) for v in VERDICTS}
    LOGGER.info(
        "Recovered %d answered and %d failed from %s: %s",
        len(answered),
        len(failed),
        args.log.name,
        counts,
    )

    existing: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                sample_id = json.loads(line).get("sample_id")
            except ValueError:
                continue
            if sample_id:
                existing.add(str(sample_id))

    # A record already on disk was written by a live run, which knows strictly
    # more than the log does. Recovery never overwrites it.
    fresh = [r for r in records if r["sample_id"] not in existing]
    skipped = len(records) - len(fresh)
    if skipped:
        LOGGER.info("%d already present in %s; left untouched", skipped, args.out.name)

    if args.dry_run:
        LOGGER.info("Dry run: %d record(s) would be written to %s", len(fresh), args.out)
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as stream:
        for record in fresh:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    LOGGER.info("Wrote %d record(s) to %s", len(fresh), args.out)
    print(
        json.dumps(
            {
                "recovered_answered": len(answered),
                "recovered_failed": len(failed),
                "verdicts": counts,
                "written": len(fresh),
                "already_present": skipped,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
