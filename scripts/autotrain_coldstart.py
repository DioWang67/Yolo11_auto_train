#!/usr/bin/env python3
"""Label a station's frames with a vision model and lay them out for training.

The cold-start half of the autonomous path: no champion exists, so the boxes
come from a vision model corrected by a :class:`BoxCalibration` rather than
from a detector. Produces a raw YOLO dataset --- ``images/``, ``labels/``,
``data.yaml`` --- in the layout the existing pipeline already consumes, so
augmentation, linting, splitting and training stay the code that does them
today rather than a second copy.

It also renders every label it wrote. Boxes measured mean IoU 0.48 against
the station's own detections after correction, which is well short of what
training labels should be, and a dataset nobody looked at is the way that
becomes a model nobody can explain.

Writes a dataset and pictures. It does not train, evaluate, or deploy.
"""

from __future__ import annotations

import argparse
import glob
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
from picture_tool.bootstrap.evidence import Box  # noqa: E402
from picture_tool.bootstrap.profile import ProductProfile  # noqa: E402
from picture_tool.bootstrap.vision_evidence import (  # noqa: E402
    BoxCalibration,
    VisionLLMProposer,
    VisionPrompt,
)

LOGGER = logging.getLogger("autotrain_coldstart")

DESCRIPTION = (
    "Each object is a wire end where it meets its solder pad. Box only that "
    "short segment, not the wire running away from it."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=None,
                   help="A Result/<date>/<product>/<area> directory.")
    p.add_argument("--images", type=Path, default=None,
                   help="A plain directory of frames. What a new station "
                        "has: photographs, and no production records yet.")
    p.add_argument("--seed", type=Path, default=None,
                   help="A directory with images/ and labels/ holding the "
                        "hand-drawn frames the calibration came from. Their "
                        "human labels are copied through in place of asking "
                        "the model about them again.")
    p.add_argument("--max-side", type=int, default=0,
                   help="Longest side to resize to before asking; 0 sends "
                        "the frame as taken. Must match the calibration's.")
    p.add_argument("--calibration", type=Path,
                   default=PROJECT_ROOT / "runs" / "vision_calibration" / "calibration.json")
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "runs" / "coldstart" / "raw")
    p.add_argument("--product", default="Cable1")
    p.add_argument("--area", default="A")
    p.add_argument("--classes", default="Black,Green,Orange,Red,Yellow")
    p.add_argument("--expect", default="Black=2,Green=1,Orange=1,Red=1,Yellow=1")
    p.add_argument("--describe", default=DESCRIPTION)
    p.add_argument("--url-env", default="QWEN_URL")
    p.add_argument("--key-env", default="")
    p.add_argument("--model", default="Qwen3.8-27B-GGUF")
    p.add_argument("--no-preview", action="store_true")
    return p.parse_args(argv)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def frames(root: Path) -> list[Path]:
    """Preprocessed frames, which is the space the detections are recorded in."""
    found = [Path(p) for p in
             sorted(glob.glob(str(root / "*" / "preprocessed" / "*" / "*.jpg")))]
    return found


def plain_frames(root: Path) -> list[Path]:
    """Every image in a directory, which is all a new station has."""
    return sorted(p for p in root.iterdir()
                  if p.suffix.lower() in IMAGE_SUFFIXES)


def seed_labels(root: Path) -> dict[str, Path]:
    """The hand-drawn label for each seed frame, keyed by image name.

    A seed frame's boxes are already known, and a person drew them. Asking
    the model about it again would replace the best labels in the set with
    the ones being corrected *from* them.
    """
    images, labels = root / "images", root / "labels"
    if not images.is_dir() or not labels.is_dir():
        raise SystemExit(f"{root} needs images/ and labels/ side by side.")
    out: dict[str, Path] = {}
    for image in sorted(images.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = labels / f"{image.stem}.txt"
        if label.is_file():
            out[image.name] = label
    return out


def render(image_path: Path, boxes: list[Box], names: tuple[str, ...], out: Path) -> None:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        return
    height, width = image.shape[:2]
    for index, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box.to_pixels(width, height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 235), 2)
        cv2.putText(image, f"{index} {box.class_name}", (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 235), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), image)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    names = tuple(n.strip() for n in args.classes.split(",") if n.strip())
    expected = {}
    for pair in args.expect.split(","):
        if "=" in pair:
            key, value = pair.split("=", 1)
            expected[key.strip()] = int(value)
    profile = ProductProfile(
        product=args.product, area=args.area,
        class_schema=ClassSchema(names=names, source="station_contract"),
        expected_counts=expected,
    )
    class_ids = {name: index for index, name in enumerate(names)}
    LOGGER.info("contract %s  hash %s", list(names),
                profile.class_schema.schema_hash[:12])

    calibration = BoxCalibration.read(args.calibration)
    LOGGER.info("calibration width x%.3f height x%.3f from %d box(es) over %d frame(s)",
                calibration.width_scale, calibration.height_scale,
                calibration.sample_boxes, calibration.sample_images)

    cfg = vc.openai_compatible_profile(
        model=args.model, url_env=args.url_env, key_env=args.key_env,
        requires_credential=bool(args.key_env), max_tokens=3000, max_retries=0,
        timeout_seconds=500.0,
    )
    proposer = VisionLLMProposer(
        vc.HttpVisionLLMClient(cfg), calibration=calibration,
        prompt=VisionPrompt(object_description=args.describe),
        max_side=args.max_side, model=args.model,
    )

    images_dir = args.out / "images"
    labels_dir = args.out / "labels"
    preview_dir = args.out.parent / "preview"
    for directory in (images_dir, labels_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if not args.no_preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    import shutil

    if args.images:
        wanted = plain_frames(args.images)
    elif args.results:
        wanted = frames(args.results)
    else:
        raise SystemExit("Give either --images or --results.")
    seeds = seed_labels(args.seed) if args.seed else {}
    if seeds:
        # A seed that is not among the frames being labelled is dropped by the
        # loop below without a word: hand_drawn_images then under-reports, and
        # the frames the calibration was derived from can be missing from the
        # dataset the calibration is used to seed. Said out loud instead.
        absent = sorted(set(seeds) - {frame.name for frame in wanted})
        if absent:
            raise SystemExit(
                f"{len(absent)} of {len(seeds)} seed frame(s) are not among "
                f"the images being labelled, e.g. {', '.join(absent[:3])}. "
                "Their hand-drawn labels would be silently dropped. Point "
                "--images at the directory the seeds came from, or drop them "
                "from --seed."
            )
        LOGGER.info("%d seed frame(s) keep their hand-drawn labels", len(seeds))

    written = complete = seeded = 0
    report = []
    for frame in wanted:
        if frame.name in seeds:
            shutil.copy2(frame, images_dir / frame.name)
            shutil.copy2(seeds[frame.name], labels_dir / f"{frame.stem}.txt")
            written += 1
            seeded += 1
            report.append({"image": frame.name, "source": "human"})
            LOGGER.info("  %s: hand-drawn, kept", frame.name)
            continue
        boxes = proposer.propose(frame, profile)
        counts: dict[str, int] = {}
        for box in boxes:
            counts[box.class_name] = counts.get(box.class_name, 0) + 1
        matches_station = counts == expected
        complete += matches_station
        lines = [
            f"{class_ids[b.class_name]} {b.cx:.6f} {b.cy:.6f} {b.width:.6f} {b.height:.6f}"
            for b in boxes if b.class_name in class_ids
        ]
        shutil.copy2(frame, images_dir / frame.name)
        (labels_dir / f"{frame.stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        if not args.no_preview:
            render(frame, boxes, names, preview_dir / frame.name)
        written += 1
        report.append({"image": frame.name, "boxes": len(boxes),
                       "counts": counts, "matches_station": matches_station})
        LOGGER.info("  %s: %d box(es) %s", frame.name, len(boxes),
                    "OK" if matches_station else f"counts {counts}")

    (args.out / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\n"
        f"nc: {len(names)}\nnames: [{', '.join(names)}]\n", encoding="utf-8")
    summary = {
        "images": written,
        "hand_drawn_images": seeded,
        "model_labelled_images": written - seeded,
        "matching_station_inventory": complete,
        "calibration": calibration.to_dict(),
        "max_side": args.max_side,
        "described_as": args.describe,
        # True only of the seed frames, and they are counted separately
        # above so nobody has to infer which half of the set is which.
        "labels_are_ground_truth": False,
        "frames": report,
    }
    (args.out.parent / "coldstart_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"},
                     indent=2, sort_keys=True))
    LOGGER.info("dataset at %s, previews at %s", args.out, preview_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
