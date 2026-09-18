#!/usr/bin/env python3
"""Measure a vision model's box bias against a station's own records.

The model places box centres well horizontally and draws the boxes too
small, consistently enough that a few constants correct them. This derives
those constants and, because a correction fitted and tested on the same
frames proves nothing, reports what they do on frames it never saw.

It also decomposes the *vertical* error, which is the one the constants are
not expected to fix. Cable1/A's hand labels say the wire ends are collinear
within a frame but that the line they lie on is placed differently in every
frame, so a vertical correction belonging to the station cannot exist. The
``vertical_structure`` block splits the residual into the part that moves
between frames and the part that varies inside one, so that claim is
answered by this run's own numbers rather than inherited from that one.

**The reference is the station's detector, not ground truth.** Production
records say what the detector found, which is what makes them available in
bulk and also what limits them: a calibration derived this way teaches the
model to agree with the current detector, inheriting any systematic offset
the detector has. That is useful --- agreeing with the deployed detector is
what makes the boxes usable for training against it --- and it is not the
same as being right. To measure right, point ``--reference`` at hand-drawn
labels instead.

Nothing here trains, deploys, or writes a dataset. It writes one JSON file.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picture_tool.autotrain.class_schema import ClassSchema  # noqa: E402
from picture_tool.bootstrap.evidence import Box  # noqa: E402
from picture_tool.bootstrap.profile import ProductProfile  # noqa: E402
from picture_tool.bootstrap import vision_client as vc  # noqa: E402
from picture_tool.bootstrap.vision_evidence import (  # noqa: E402
    VisionLLMEvidence,
    VisionPrompt,
    calibrate,
    match_by_centre,
    measure_vertical_structure,
)

LOGGER = logging.getLogger("derive_vision_calibration")

#: Detections are recorded in the preprocessed frame's pixel space, while the
#: same record's image_width/image_height describe the *original*. Anyone
#: normalising by those fields puts every box in the top-left corner.
PREPROCESSED_SIDE = 640.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=None,
                   help="A Result/<date>/<product>/<area> directory. The "
                        "reference is then the station detector, not truth.")
    p.add_argument("--reference-labels", type=Path, default=None,
                   help="A directory with images/ and labels/ holding "
                        "hand-drawn YOLO labels. This is the reference that "
                        "measures accuracy rather than agreement.")
    p.add_argument("--classes", default="Black,Green,Orange,Red,Yellow")
    p.add_argument("--product", default="Cable1")
    p.add_argument("--area", default="A")
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "runs" / "vision_calibration" / "calibration.json")
    p.add_argument("--holdout", type=float, default=0.5,
                   help="Fraction of frames kept out of the fit, for reporting.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--url-env", default="QWEN_URL")
    p.add_argument("--key-env", default="")
    p.add_argument("--model", default="Qwen3.8-27B-GGUF")
    p.add_argument("--max-tokens", type=int, default=6000)
    p.add_argument("--match-radius", type=float, default=0.02,
                   help="How near in cx a proposed centre must be to count "
                        "as the same object, as a fraction of image width.")
    p.add_argument("--vertical-match-radius", type=float, default=None,
                   help="Optional cy gate. Leave unset unless this station "
                        "stacks objects at the same cx: the model's cy is "
                        "the one coordinate known to be wrong, and gating "
                        "on it is what made a 6/6 frame measure 0/6.")
    p.add_argument("--describe", default=(
        "Each object is a wire end where it meets its solder pad. Box only "
        "that short segment, not the wire running away from it."),
        help="What counts as one object. This is the specification; it must "
             "not say where objects sit or how big they are.")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be used and exit without calling anything.")
    return p.parse_args(argv)


def iou(a: Box, b: Box) -> float:
    ax1, ay1 = a.cx - a.width / 2, a.cy - a.height / 2
    ax2, ay2 = a.cx + a.width / 2, a.cy + a.height / 2
    bx1, by1 = b.cx - b.width / 2, b.cy - b.height / 2
    bx2, by2 = b.cx + b.width / 2, b.cy + b.height / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def labelled_frames(root: Path, names: tuple[str, ...]) -> list[tuple[Path, list[Box]]]:
    """Frames whose boxes a person drew.

    The only reference that can answer whether the model is right rather than
    whether it resembles the detector, which is itself the thing under
    suspicion at this station.
    """
    frames: list[tuple[Path, list[Box]]] = []
    for image in sorted((root / "images").iterdir()):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        label = root / "labels" / f"{image.stem}.txt"
        if not label.is_file():
            continue
        boxes = []
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            index, cx, cy, w, h = int(parts[0]), *(float(v) for v in parts[1:])
            if 0 <= index < len(names):
                boxes.append(Box(class_name=names[index], cx=cx, cy=cy,
                                 width=w, height=h, confidence=1.0,
                                 source="human"))
        if boxes:
            frames.append((image, boxes))
    return frames


def reference_frames(root: Path) -> list[tuple[Path, list[Box]]]:
    """Frames that have both a preprocessed image and recorded detections."""
    frames: list[tuple[Path, list[Box]]] = []
    for meta in sorted(glob.glob(str(root / "*" / "metadata" / "*" / "*.json"))):
        stem = os.path.basename(meta).replace("_config_snapshot.json", "")
        verdict_dir = Path(meta).parents[2]
        detector = Path(meta).parent.name
        image = verdict_dir / "preprocessed" / detector / f"{stem}.jpg"
        if not image.is_file():
            continue
        try:
            payload = json.loads(Path(meta).read_text(encoding="utf-8"))
        except ValueError:
            LOGGER.warning("%s is not readable JSON; skipped", meta)
            continue
        boxes = []
        for det in payload.get("detections") or []:
            bbox = det.get("bbox")
            name = det.get("class")
            if not name or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = (float(v) / PREPROCESSED_SIDE for v in bbox)
            boxes.append(Box(class_name=str(name), cx=(x1 + x2) / 2, cy=(y1 + y2) / 2,
                             width=x2 - x1, height=y2 - y1,
                             confidence=float(det.get("confidence", 0.0) or 0.0),
                             source="detector"))
        if boxes:
            frames.append((image, boxes))
    return frames


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    names = tuple(n.strip() for n in args.classes.split(",") if n.strip())
    if args.reference_labels:
        frames = labelled_frames(args.reference_labels, names)
        reference_kind = "hand-drawn labels"
        source_path = args.reference_labels
    elif args.results:
        frames = reference_frames(args.results)
        reference_kind = "station detector"
        source_path = args.results
    else:
        raise SystemExit("Give either --results or --reference-labels.")
    LOGGER.info("reference: %s (%s)", reference_kind, source_path)
    if not frames:
        raise SystemExit(f"No usable reference frames under {source_path}")
    random.Random(args.seed).shuffle(frames)
    cut = max(1, int(len(frames) * (1 - args.holdout)))
    fit, held = frames[:cut], frames[cut:]
    LOGGER.info("%d frame(s): %d to fit, %d held out", len(frames), len(fit), len(held))
    if args.dry_run:
        for image, boxes in frames:
            LOGGER.info("  %s (%d boxes)", image.name, len(boxes))
        return 0

    import cv2

    profile = ProductProfile(
        product=args.product, area=args.area,
        class_schema=ClassSchema(names=names, source="station_contract"),
    )
    LOGGER.info("class contract from records: %s", list(profile.class_schema.names))

    cfg = vc.openai_compatible_profile(
        model=args.model, url_env=args.url_env, key_env=args.key_env,
        requires_credential=bool(args.key_env), max_tokens=args.max_tokens,
        max_retries=0, timeout_seconds=500.0,
    )
    source = VisionLLMEvidence(
        vc.HttpVisionLLMClient(cfg),
        prompt=VisionPrompt(object_description=args.describe),
        match_radius=args.match_radius,
    )
    LOGGER.info("object described as: %s", args.describe)

    def proposals(
        group: list[tuple[Path, list[Box]]]
    ) -> list[tuple[Path, list[tuple[Box, Box]]]]:
        """Matched pairs, kept grouped by frame.

        Grouping is not bookkeeping: whether the vertical error belongs to
        the station or to each frame is answerable only while it is known
        which frame a pair came from, and flattening here is what would
        make that question unanswerable later.
        """
        out: list[tuple[Path, list[tuple[Box, Box]]]] = []
        for image_path, reference in group:
            image = cv2.imread(str(image_path))
            if image is None:
                LOGGER.warning("Could not read %s", image_path)
                continue
            try:
                proposed = source._ask(image, profile)
            except Exception as exc:  # noqa: BLE001 - one frame must not end a survey
                LOGGER.error("  %s: %s", image_path.name, str(exc)[:110])
                continue
            pairs = [
                (match, target)
                for target, match in zip(
                    reference,
                    match_by_centre(
                        proposed, reference,
                        radius=args.match_radius,
                        vertical_radius=args.vertical_match_radius,
                    ),
                )
                if match is not None
            ]
            out.append((image_path, pairs))
            LOGGER.info("  %s: %d/%d matched",
                        image_path.name, len(pairs), len(reference))
        return out

    def flatten(
        per_frame: list[tuple[Path, list[tuple[Box, Box]]]]
    ) -> list[tuple[Box, Box]]:
        return [pair for _, pairs in per_frame for pair in pairs]

    LOGGER.info("Measuring on %d fit frame(s)", len(fit))
    fit_frames = proposals(fit)
    fit_pairs = flatten(fit_frames)
    calibration = calibrate(
        fit_pairs, product=args.product, area=args.area, sample_images=len(fit),
        derived_from=str(source_path),
        notes=f"reference is {reference_kind}",
    )
    LOGGER.info("width x%.3f  height x%.3f  cx %+.4f  cy %+.4f (spread %.4f)",
                calibration.width_scale, calibration.height_scale,
                calibration.cx_shift, calibration.cy_shift,
                calibration.cy_shift_spread)

    held_frames = proposals(held) if held else []
    vertical = measure_vertical_structure(
        [pairs for _, pairs in fit_frames + held_frames]
    )
    report: dict[str, Any] = {
        "calibration": calibration.to_dict(),
        "splits": {},
        "vertical_structure": vertical.to_dict(),
    }
    for label, frames_ in (("fit", fit_frames), ("holdout", held_frames)):
        group = flatten(frames_)
        if not group:
            continue
        raw = [iou(p, r) for p, r in group]
        fixed = [iou(calibration.apply(p), r) for p, r in group]
        reference_boxes = sum(
            len(boxes) for _, boxes in (fit if label == "fit" else held)
        )
        report["splits"][label] = {
            "frames": len(frames_),
            "boxes": len(group),
            # Without this, a calibration derived from the one frame that
            # matched would report a confident mean over almost nothing.
            "match_rate": round(len(group) / reference_boxes, 4)
            if reference_boxes else 0.0,
            "mean_iou_raw": round(sum(raw) / len(raw), 4),
            "mean_iou_calibrated": round(sum(fixed) / len(fixed), 4),
            "usable_raw": sum(1 for v in raw if v >= 0.5),
            "usable_calibrated": sum(1 for v in fixed if v >= 0.5),
        }
    LOGGER.info("vertical: %s", vertical.reading)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    calibration.write(args.out)
    (args.out.parent / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    LOGGER.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
