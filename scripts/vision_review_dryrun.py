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
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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
    return p.parse_args(argv)


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
    client = vc.HttpVisionLLMClient(config)
    LOGGER.info("Endpoint configured from %s / %s", args.url_env, args.key_env)

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = []
    totals = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
    failures = 0

    for index, sample in enumerate(batch, start=1):
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
            len(batch),
            sample["sample_id"],
            len(crops),
            " + whole frame" if context else "",
        )
        try:
            verdict = client.judge(request)
        except vc.VisionClientError as exc:
            failures += 1
            LOGGER.error("%s: %s", sample["sample_id"], exc)
            results.append(
                {"sample_id": sample["sample_id"], "error": str(exc), "verdict": None}
            )
            continue
        for key in totals:
            totals[key] += int(verdict.usage.get(key, 0) or 0)
        results.append(
            {
                "sample_id": sample["sample_id"],
                "crops_sent": len(crops),
                "whole_frame_sent": context is not None,
                "bootstrap_reasons": sample.get("reasons", []),
                **verdict.to_dict(),
            }
        )
        LOGGER.info(
            "   -> %s %s (%s)", verdict.verdict, verdict.class_name, verdict.reason[:80]
        )

    answered = [r for r in results if r.get("verdict")]
    counts = {v: sum(1 for r in answered if r["verdict"] == v) for v in vc.VERDICTS}
    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        # Names only. The key itself is never recorded anywhere.
        "endpoint": config.to_dict(),
        "samples_attempted": len(batch),
        "samples_answered": len(answered),
        "failures": failures,
        "verdicts": counts,
        "token_usage_total": totals,
        "token_usage_mean_per_sample": {
            key: round(value / max(1, len(answered)), 1) for key, value in totals.items()
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
