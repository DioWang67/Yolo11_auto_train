#!/usr/bin/env python3
"""Send a handful of disputed samples to the company vision endpoint.

A dry run, in the sense that matters: real images, the real endpoint, real
tokens -- on a small enough slice that the per-sample cost can be measured
before deciding whether the other two hundred are worth it.

Token discipline is the point of the payload shape. Only the boxes the
detector and the colour evidence disagreed about are sent, as padded crops;
the whole frame is attached only when the sample's own reasons say the
question is about the scene rather than a single wire. Everything else the
reviewer needs -- the scores, the counts, the station's expectations -- is
text, which costs a fraction of an image.

Nothing here can change a dataset. It writes verdicts to a report and stops.

Every verdict is appended to ``results.jsonl`` the moment it arrives, before
the next request goes out, and a later run skips the samples already recorded
there. The first full run over 213 samples was interrupted at 81 and left
nothing but a log line per sample, so the tokens it had already spent bought
nothing the second run could reuse. A record that is only written at the end
is a record that does not survive the run being stopped.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picture_tool.bootstrap import vision_client as vc  # noqa: E402
from picture_tool.bootstrap.evidence import Box  # noqa: E402

LOGGER = logging.getLogger("vision_review_dryrun")

PROMPT = """You are adjudicating one automatically proposed label from a \
cable-inspection station. The station always shows six objects: two Black \
wires and one each of Green, Orange, Red and Yellow.

An object detector and an independent colour measurement disagreed, or the \
colour measurement could not decide. The images are the disputed crops, in \
the order listed below.

{evidence}

Decide whether the proposed label for this image can be trusted as training \
ground truth.

Reply with only a JSON object:
{{"verdict": "ACCEPT" | "RETRY" | "REJECT",
  "class": "<the correct class for the first disputed crop, or empty>",
  "confidence": <0.0-1.0>,
  "reason": "<one sentence, concrete>",
  "next_action": "<what a person should do next, or empty>"}}

ACCEPT only if the proposed labels are right and complete. RETRY if you \
cannot tell from these crops and a wider view would settle it. REJECT if the \
labels are wrong or the image cannot support a reliable label."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "runs" / "bootstrap_poc" / "report" / "needs_review.json",
    )
    p.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "runs" / "vision_review_dryrun"
    )
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--max-crops", type=int, default=3)
    p.add_argument("--url-env", default="ANTHROPIC_BASE_URL")
    p.add_argument("--key-env", default="ANTHROPIC_API_KEY")
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-ask every sample, even ones already in results.jsonl. Costs "
        "tokens that have already been spent once.",
    )
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Rebuild dryrun.json from the records already on disk and exit "
        "without contacting the endpoint. Spends nothing.",
    )
    return p.parse_args(argv)


def load_recorded(path: Path) -> dict[str, dict]:
    """Verdicts already on disk, keyed by sample id.

    Unreadable lines are skipped rather than fatal: a run killed mid-write
    can leave a partial last line, and that is not a reason to refuse to
    reuse the hundreds of records in front of it.
    """
    if not path.exists():
        return {}
    recorded: dict[str, dict] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            LOGGER.warning("%s line %d is not readable JSON; skipping it", path, number)
            continue
        sample_id = record.get("sample_id")
        if sample_id:
            recorded[str(sample_id)] = record
    return recorded


def append_record(path: Path, record: dict) -> None:
    """Commit one verdict to disk before the next request is made."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def disputed_boxes(sample: dict, limit: int) -> list[dict]:
    """The boxes worth paying to look at: the ones that did not agree."""
    disputed = [b for b in sample.get("boxes", []) if not b.get("agreed", True)]
    return (disputed or sample.get("boxes", []))[:limit]


def evidence_text(sample: dict, boxes: list[dict]) -> str:
    lines = [
        f"Proposed class counts: {json.dumps(sample.get('class_counts', {}))}",
        f"Why this was held back: {', '.join(sample.get('reasons', []))}",
    ]
    for index, item in enumerate(boxes, start=1):
        box = item.get("box", {})
        opinions = {
            o.get("source"): o for o in item.get("opinions", []) if isinstance(o, dict)
        }
        detector = opinions.get("detector", {})
        colour = opinions.get("colour", {})
        lines.append(
            f"Crop {index}: detector says {detector.get('class_name')} "
            f"(confidence {detector.get('confidence')}); colour measurement says "
            f"{colour.get('class_name') or 'undecided'} "
            f"(scores {json.dumps(colour.get('scores', {}))}); "
            f"box size {box.get('width')}x{box.get('height')}"
        )
    return "\n".join(lines)


def needs_whole_frame(sample: dict) -> bool:
    """Only when the question is about the scene, not a single wire."""
    reasons = set(sample.get("reasons", []))
    return bool(
        reasons & {"object_count_mismatch", "class_count_mismatch", "no_detections"}
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    samples = json.loads(args.report.read_text(encoding="utf-8"))
    LOGGER.info("%d NEEDS_REVIEW samples available", len(samples))
    batch = [s for s in samples if s.get("boxes")][: args.limit]
    if not batch:
        raise SystemExit("No reviewable samples with boxes.")

    config = vc.VisionEndpointConfig(
        url_env=args.url_env,
        key_env=args.key_env,
        model=args.model,
        max_retries=args.max_retries,
    )

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"

    recorded = {} if args.no_resume else load_recorded(results_path)
    # Only an answered sample is finished. A sample that errored is
    # unfinished work, and is asked again -- the three that failed in the
    # first full run failed on defects that have since been fixed, and a
    # resume that skipped them would make those fixes unreachable.
    pending = [
        s for s in batch if not (recorded.get(s["sample_id"]) or {}).get("verdict")
    ]
    if recorded:
        retries = sum(1 for s in pending if s["sample_id"] in recorded)
        LOGGER.info(
            "%d of %d already answered in %s; %d to ask (%d of them retries)",
            len(batch) - len(pending),
            len(batch),
            results_path.name,
            len(pending),
            retries,
        )

    if args.summary_only:
        LOGGER.info("Summary only: not contacting the endpoint")
        pending = []

    # Built only when there is something to ask, so a finished run can be
    # re-summarised from its own records without a credential.
    client = None
    if pending:
        client = vc.HttpVisionLLMClient(config)
        LOGGER.info("Endpoint configured from %s / %s", args.url_env, args.key_env)

    interrupted = False
    for index, sample in enumerate(pending, start=1):
        boxes = disputed_boxes(sample, args.max_crops)
        crops = []
        for item in boxes:
            raw = item.get("box", {})
            try:
                crops.append(
                    vc.crop_bytes(
                        sample["image_path"],
                        Box(
                            class_name=str(raw.get("class_name", "")),
                            cx=float(raw.get("cx", 0.0)),
                            cy=float(raw.get("cy", 0.0)),
                            width=float(raw.get("width", 0.0)),
                            height=float(raw.get("height", 0.0)),
                            confidence=float(raw.get("confidence", 0.0)),
                        ),
                    )
                )
            except vc.VisionClientError as exc:
                LOGGER.warning("%s: %s", sample["sample_id"], exc)
        context = (
            vc.whole_image_bytes(sample["image_path"])
            if needs_whole_frame(sample)
            else None
        )
        request = vc.VisionRequest(
            sample_id=sample["sample_id"],
            prompt=PROMPT.format(evidence=evidence_text(sample, boxes)),
            crops=crops,
            context_image=context,
        )
        LOGGER.info(
            "%d/%d %s: %d crop(s)%s",
            index,
            len(pending),
            sample["sample_id"],
            len(crops),
            " + whole frame" if context else "",
        )
        common = {
            "sample_id": sample["sample_id"],
            "crops_sent": len(crops),
            "whole_frame_sent": context is not None,
            "bootstrap_reasons": sample.get("reasons", []),
            # Where this record came from, so a reader can tell a live answer
            # from one reconstructed after the fact.
            "fidelity": "live",
        }
        assert client is not None  # built above whenever `pending` is non-empty
        try:
            # The request is where an interrupt lands in practice --- crop
            # encoding is local and quick --- so catching it here is enough
            # to end the run tidily. Records already appended are on disk
            # either way.
            verdict = client.judge(request)
        except KeyboardInterrupt:
            interrupted = True
            LOGGER.warning("Interrupted before %s was answered", sample["sample_id"])
            break
        except vc.VisionClientError as exc:
            LOGGER.error("%s: %s", sample["sample_id"], exc)
            record = {**common, "error": str(exc), "verdict": None}
            append_record(results_path, record)
            recorded[sample["sample_id"]] = record
            continue
        record = {**common, **verdict.to_dict()}
        append_record(results_path, record)
        recorded[sample["sample_id"]] = record
        LOGGER.info(
            "   -> %s %s%s (%s)",
            verdict.verdict,
            verdict.class_name,
            " [repaired]" if verdict.repaired else "",
            verdict.reason[:80],
        )

    # The summary is derived from the durable records, not from this run's
    # own tally, so it reads the same whether the work took one run or five.
    results = [recorded[s["sample_id"]] for s in batch if s["sample_id"] in recorded]
    answered = [r for r in results if r.get("verdict")]
    counts = {v: sum(1 for r in answered if r["verdict"] == v) for v in vc.VERDICTS}
    totals = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
    for record in answered:
        usage = record.get("usage") or {}
        for key in totals:
            totals[key] += int(usage.get(key, 0) or 0)
    # Mean over the records that actually carry usage: recovered records have
    # none, and dividing by them would understate the real per-sample cost.
    metered = [r for r in answered if r.get("usage")]
    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        # Names only. The key itself is never recorded anywhere.
        "endpoint": config.to_dict(),
        "samples_attempted": len(batch),
        "samples_answered": len(answered),
        "samples_outstanding": len(batch) - len(results),
        "failures": sum(1 for r in results if r.get("error")),
        "interrupted": interrupted,
        "verdicts": counts,
        "repaired_replies": sum(1 for r in answered if r.get("repaired")),
        "fidelity": dict(Counter(r.get("fidelity", "unknown") for r in results)),
        "token_usage_total": totals,
        "token_usage_metered_samples": len(metered),
        "token_usage_mean_per_sample": {
            key: round(value / max(1, len(metered)), 1) for key, value in totals.items()
        },
        "results": results,
    }
    (out / "dryrun.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "results"},
                     ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
