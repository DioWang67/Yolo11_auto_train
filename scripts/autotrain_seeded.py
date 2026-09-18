#!/usr/bin/env python3
"""Cold-start a station from a handful of hand-drawn frames.

A new station has photographs and nothing else: no detector to propose
boxes, and therefore no reference to measure the vision model's box bias
against. That is a circle, and a small hand-labelled seed set is what cuts
it --- the person draws eight or twelve frames, those become the reference,
and the model labels the rest.

Three steps, in the order their failures get cheaper to fix:

1. **Check the seed.** Every gate here is free and offline, so nothing
   reaches the endpoint until the reference it would be measured against is
   known to be sound. A seed with one unfinished label produces a
   calibration that is wrong in a way no later step can detect.
2. **Measure the bias** from the seed, at the image scale the labelling
   will use. The scale is recorded in the calibration and enforced when it
   is applied: the same station and frames measure x1.497 at max_side 640
   and x1.981 unscaled, and using one for the other is wrong by a third
   with nothing about the boxes looking unusual.
3. **Label the rest.** Seed frames keep their human labels rather than
   being asked about again --- replacing the best labels in the set with
   corrected-from-them ones would be a strange trade.

It stops at a labelled dataset and prints the training command. It does not
train, evaluate, or deploy: ``assert_no_forbidden_tasks`` refuses ``deploy``
in this path by design, and a new station has no golden set, so there is
nothing here that could tell a good model from a bad one yet.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picture_tool.bootstrap.vision_evidence import (  # noqa: E402
    MINIMUM_SEED_IMAGES,
    RECOMMENDED_SEED_IMAGES,
)
from picture_tool.pending_annotations import (  # noqa: E402
    validate_yolo_label_text,
)

import autotrain_coldstart  # noqa: E402
import derive_vision_calibration  # noqa: E402

LOGGER = logging.getLogger("autotrain_seeded")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=Path, required=True,
                   help="Directory with images/ and labels/: the frames a "
                        "person drew.")
    p.add_argument("--images", type=Path, required=True,
                   help="Every frame for the station, seed frames included.")
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "runs" / "seeded" / "raw")
    p.add_argument("--product", default="Cable1")
    p.add_argument("--area", default="A")
    p.add_argument("--classes", default="Black,Green,Orange,Red,Yellow")
    p.add_argument("--expect", default="Black=2,Green=1,Orange=1,Red=1,Yellow=1")
    p.add_argument("--max-side", type=int, default=0,
                   help="0 sends frames as taken, which is what the "
                        "placement measurement supports.")
    p.add_argument("--describe", default=autotrain_coldstart.DESCRIPTION)
    p.add_argument("--url-env", default="QWEN_URL")
    p.add_argument("--key-env", default="")
    p.add_argument("--model", default="Qwen3.8-27B-GGUF")
    p.add_argument("--base-model", default="yolo11n.pt",
                   help="Only used in the command printed at the end; a "
                        "cold start has no champion to continue from.")
    p.add_argument("--check-only", action="store_true",
                   help="Run the seed checks and stop, touching no endpoint.")
    return p.parse_args(argv)


def check_seed(root: Path, class_count: int) -> tuple[int, list[str]]:
    """Frames the seed contributes, and everything wrong with it.

    Returns every problem rather than the first, because the person fixing
    them is going back into a labelling tool and should make one trip.
    """
    problems: list[str] = []
    images, labels = root / "images", root / "labels"
    if not images.is_dir() or not labels.is_dir():
        return 0, [f"{root} needs images/ and labels/ side by side."]

    frames = [p for p in sorted(images.iterdir())
              if p.suffix.lower() in IMAGE_SUFFIXES]
    if not frames:
        return 0, [f"No images under {images}."]

    usable = 0
    for image in frames:
        label = labels / f"{image.stem}.txt"
        if not label.is_file():
            problems.append(f"{image.name}: no label file")
            continue
        text = label.read_text(encoding="utf-8")
        errors = validate_yolo_label_text(text, class_count)
        if errors:
            problems.append(f"{image.name}: {errors[0]}")
            continue
        # An empty label is valid YOLO and means "nothing here", which is a
        # real answer elsewhere in this project. In a seed set it almost
        # always means the annotator did not finish, and the cost of being
        # wrong is not one frame: the whole calibration is measured from
        # these, so a frame contributing no boxes silently shrinks the
        # reference the rest of the station is corrected against.
        if not any(line.strip() for line in text.splitlines()):
            problems.append(f"{image.name}: label is empty")
            continue
        usable += 1

    if usable < MINIMUM_SEED_IMAGES:
        problems.append(
            f"{usable} usable seed frame(s); {MINIMUM_SEED_IMAGES} is the "
            f"floor and {RECOMMENDED_SEED_IMAGES} is worth having. Resampling "
            "this station's hand labels put the height correction's spread at "
            "27% of its own value over 3 frames and 12.5% over 8. What has to "
            "grow is frames, not boxes: the six boxes in one frame share that "
            "frame's board position, so they are one observation, not six."
        )
    return usable, problems


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    names = tuple(n.strip() for n in args.classes.split(",") if n.strip())

    LOGGER.info("1/3 checking the seed at %s", args.seed)
    usable, problems = check_seed(args.seed, len(names))
    for problem in problems:
        LOGGER.error("  %s", problem)
    if problems:
        LOGGER.error("Seed rejected; nothing was asked of the model.")
        return 1
    LOGGER.info("  %d usable seed frame(s)%s", usable,
                "" if usable >= RECOMMENDED_SEED_IMAGES
                else f" (>= {RECOMMENDED_SEED_IMAGES} is worth having)")
    if args.check_only:
        return 0

    calibration_path = args.out.parent / "calibration.json"
    LOGGER.info("2/3 measuring box bias from the seed -> %s", calibration_path)
    derived = derive_vision_calibration.main([
        "--reference-labels", str(args.seed),
        "--classes", args.classes,
        "--product", args.product, "--area", args.area,
        "--out", str(calibration_path),
        # Every seed frame goes into the fit. Holding some back would report
        # a number measured on fewer frames than the floor allows, and the
        # honest test of this calibration is the labels it goes on to
        # produce, not a split of the reference it came from.
        "--holdout", "0",
        "--max-side", str(args.max_side),
        "--model", args.model,
        "--url-env", args.url_env, "--key-env", args.key_env,
        "--describe", args.describe,
    ])
    if derived != 0:
        return derived

    LOGGER.info("3/3 labelling %s", args.images)
    laid_out = autotrain_coldstart.main([
        "--images", str(args.images),
        "--seed", str(args.seed),
        "--calibration", str(calibration_path),
        "--out", str(args.out),
        "--product", args.product, "--area", args.area,
        "--classes", args.classes, "--expect", args.expect,
        "--max-side", str(args.max_side),
        "--model", args.model,
        "--url-env", args.url_env, "--key-env", args.key_env,
        "--describe", args.describe,
    ])
    if laid_out != 0:
        return laid_out

    LOGGER.info("Dataset at %s. Look at %s before training on it.",
                args.out, args.out.parent / "preview")
    print(
        "\nNext, to train on it:\n"
        f"  python scripts/autotrain_smoke.py --raw {args.out} "
        f"--base-model {args.base_model} --product {args.product} "
        f"--area {args.area}\n"
        "\nThe splitter needs at least three independent source groups, so a\n"
        "set this small can refuse to split. Deployment is a separate,\n"
        "gated step: this path cannot run it, and a station with no golden\n"
        "set has nothing to judge the result against yet.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
