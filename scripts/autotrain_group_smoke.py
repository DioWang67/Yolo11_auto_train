#!/usr/bin/env python3
"""Run one real per-group golden evaluation, against real ultralytics.

Every automated test of the golden split injects a fake validator, so the
interface against ultralytics itself --- whether a generated subset
``data.yaml`` is accepted, whether the image list resolves to labels, whether
metrics come back in the shape :mod:`picture_tool.autotrain.metrics` expects
--- is exercised nowhere else. A fake runner will happily "measure" a dataset
ultralytics could never load. This script closes that gap.

It is not a performance check. It answers plumbing questions only:

* does ultralytics accept the subset descriptor this code writes,
* do both groups really validate, for both models,
* does the ground truth actually resolve (a subset whose labels are missing
  scores every object as a miss and still returns a confident-looking number),
* does a group below ``min_group_samples`` stay unmeasured,
* and is anything written into the golden directory while this happens.

**No production data is modified.** The champion weight is copied out of the
inference project and validated as a copy; the golden set is a throwaway
fixture built under the scratch directory and never registered into any
configuration file. The scratch path is checked against the production tree
before anything is written.

    python scripts/autotrain_group_smoke.py --fresh > group_smoke.log 2>&1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from autotrain_smoke import find_job  # noqa: E402
from picture_tool.autotrain import golden as golden_module  # noqa: E402
from picture_tool.autotrain.class_schema import (  # noqa: E402
    resolve_class_schema,
    schema_from_station_config,
)
from picture_tool.autotrain.config import (  # noqa: E402
    DEFAULT_MIN_GROUP_SAMPLES,
)
from picture_tool.autotrain.evaluator import (  # noqa: E402
    GROUP_INSUFFICIENT,
    GROUP_MEASURED,
    evaluate_candidate,
)
from picture_tool.autotrain.golden_candidates import (  # noqa: E402
    HARD_CASE,
    REPRESENTATIVE,
)
from picture_tool.autotrain.paths import AutoTrainPaths  # noqa: E402
from picture_tool.autotrain.registry import (  # noqa: E402
    read_champion,
    read_champion_class_schema,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

#: A third group, deliberately under the floor, to prove the refusal path is
#: not merely untested branch code.
TINY_GROUP = "tiny_smoke"

#: Marks every artefact this script creates. A golden set is evidence, and a
#: throwaway one must never be mistakable for the real thing.
SMOKE_MARKER = "SMOKE/TEST-ONLY -- not a real golden set; do not configure."

LOGGER = logging.getLogger("autotrain_group_smoke")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--product", default="Cable1")
    parser.add_argument("--area", default="A")
    parser.add_argument("--job", default="")
    parser.add_argument(
        "--scratch",
        type=Path,
        default=PROJECT_ROOT / "runs" / "autotrain_group_smoke",
        help="Must not be inside the inference project.",
    )
    parser.add_argument("--hard", type=int, default=12)
    parser.add_argument("--representative", type=int, default=12)
    parser.add_argument(
        "--tiny",
        type=int,
        default=2,
        help="Size of the deliberately-too-small third group.",
    )
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--min-group-samples", type=int, default=DEFAULT_MIN_GROUP_SAMPLES
    )
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_tree(root: Path) -> dict[str, str]:
    """Every file under ``root``, by content --- not just the images."""
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def labelled_pairs(job_raw: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for image in sorted((job_raw / "images").iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = job_raw / "labels" / f"{image.stem}.txt"
        if label.is_file():
            pairs.append((image, label))
    return pairs


def count_boxes(labels: list[Path]) -> int:
    total = 0
    for label in labels:
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.strip():
                total += 1
    return total


def build_dataset(
    root: Path, pairs: list[tuple[Path, Path]], names: dict[int, str]
) -> Path:
    """Lay out a YOLO dataset the way ultralytics expects to find one."""
    images_dir = root / "images" / "val"
    labels_dir = root / "labels" / "val"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for image, label in pairs:
        shutil.copy2(image, images_dir / image.name)
        shutil.copy2(label, labels_dir / f"{image.stem}.txt")
    descriptor = root / "data.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                # Ultralytics requires both keys even when only val is used.
                "train": "images/val",
                "val": "images/val",
                "names": names,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return descriptor


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    paths = AutoTrainPaths.discover()
    scratch = args.scratch.expanduser().resolve()
    # This path must not be able to touch production, so the guard runs
    # before anything is written rather than being assumed.
    paths.assert_not_production(scratch)
    if args.fresh and scratch.exists():
        for path in scratch.rglob("*"):
            if path.is_file():
                path.chmod(0o666)
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    # -- the models -------------------------------------------------------
    model_dir = paths.production_model_dir(args.product, args.area)
    champion = read_champion(model_dir)
    if champion is None:
        raise SystemExit(f"No deployed champion for {args.product}/{args.area}.")
    production_weight = Path(
        champion.training_weight_path or champion.weights_path
    )
    production_sha_before = sha256_file(production_weight)

    challenger_source = PROJECT_ROOT / "runs" / "Cable1" / "train" / "weights" / "best.pt"
    if not challenger_source.is_file():
        raise SystemExit(
            f"No second model to stand in as a challenger at {challenger_source}. "
            "Run scripts/autotrain_smoke.py --fresh first."
        )

    # Copied, never validated in place: the constraint is that production is
    # not modified, and the cheapest way to guarantee that is not to open it.
    weights_dir = scratch / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    champion_copy = weights_dir / "champion.pt"
    challenger_copy = weights_dir / "challenger.pt"
    shutil.copy2(production_weight, champion_copy)
    shutil.copy2(challenger_source, challenger_copy)
    LOGGER.info("Champion   %s (copy of %s)", champion_copy, production_weight.name)
    LOGGER.info("Challenger %s (copy of %s)", challenger_copy, challenger_source.name)

    schema = resolve_class_schema(
        [
            read_champion_class_schema(champion),
            schema_from_station_config(model_dir),
        ],
        context=f"{args.product}/{args.area}",
    )
    names = {index: name for index, name in enumerate(schema.names)}
    LOGGER.info("Class contract %s (hash %s)", names, schema.schema_hash[:12])

    # -- the data ---------------------------------------------------------
    data_root = paths.workspace.training_project / "data"
    job_raw = find_job(data_root, args.product, args.area, args.job)
    pairs = labelled_pairs(job_raw)
    wanted = args.hard + args.representative + args.tiny
    if len(pairs) < wanted + 4:
        raise SystemExit(
            f"{job_raw} has {len(pairs)} labelled images; need at least "
            f"{wanted + 4} to build a golden fixture and a separate main split."
        )

    golden_pairs = pairs[:wanted]
    main_pairs = pairs[wanted:]

    golden_root = scratch / "golden_smoke"
    build_dataset(golden_root, golden_pairs, names)
    (golden_root / "README_SMOKE.md").write_text(
        f"# {SMOKE_MARKER}\n\nBuilt by scripts/autotrain_group_smoke.py to "
        "exercise per-group evaluation against real ultralytics. It is not a "
        "yardstick and must never be put in golden.dataset_path.\n",
        encoding="utf-8",
    )
    main_descriptor = build_dataset(scratch / "main_split", main_pairs, names)

    assignments: dict[str, str] = {}
    group_labels: list[tuple[str, int]] = [
        (HARD_CASE, args.hard),
        (REPRESENTATIVE, args.representative),
        (TINY_GROUP, args.tiny),
    ]
    cursor = 0
    expected_boxes: dict[str, int] = {}
    for label, count in group_labels:
        chunk = golden_pairs[cursor : cursor + count]
        cursor += count
        for image, _ in chunk:
            assignments[sha256_file(image)] = label
        expected_boxes[label] = count_boxes([lbl for _, lbl in chunk])

    dataset = golden_module.register(
        golden_root,
        registered_by="autotrain_group_smoke",
        class_schema=schema,
        description=SMOKE_MARKER,
        groups=assignments,
        overwrite=True,
    )
    status = golden_module.resolve(str(golden_root), dataset.manifest_sha256)
    LOGGER.info(
        "Golden fixture %s: status=%s groups=%s ungrouped=%d",
        golden_root,
        status.status,
        dataset.group_counts(),
        len(dataset.ungrouped_sample_ids),
    )
    if not status.is_passing:
        raise SystemExit(f"Golden fixture did not resolve: {status.detail}")

    golden_before = snapshot_tree(golden_root)

    # -- the real evaluation ---------------------------------------------
    LOGGER.info("Running the real ultralytics evaluation; this is not quick.")
    report = evaluate_candidate(
        champion_weights=champion_copy,
        challenger_weights=challenger_copy,
        data_yaml=main_descriptor,
        golden_status=status,
        split="val",
        confidence=0.4,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        workers=0,
        min_group_samples=args.min_group_samples,
        # No validator: this is the whole point.
    )

    golden_after = snapshot_tree(golden_root)
    production_sha_after = sha256_file(production_weight)

    # -- what happened ----------------------------------------------------
    summary: dict[str, object] = {
        "evaluation_status": report.status,
        "main_map50_delta": report.comparison.overall_delta("map50"),
        "golden_overall_measured": report.golden_comparison is not None,
        "expected_boxes_per_group": expected_boxes,
        "groups": [],
        "production_weight_unchanged": (
            production_sha_before == production_sha_after
        ),
        "golden_directory_unchanged": golden_after == golden_before,
        "golden_directory_new_files": sorted(
            set(golden_after) - set(golden_before)
        ),
        "golden_directory_changed_files": sorted(
            name
            for name in set(golden_after) & set(golden_before)
            if golden_after[name] != golden_before[name]
        ),
    }
    for item in report.golden_groups:
        entry: dict[str, object] = {
            "group": item.group,
            "status": item.status,
            "sample_count": item.sample_count,
            "detail": item.detail,
        }
        if item.comparison is not None:
            entry["champion_map50"] = item.comparison.overall[2].champion
            entry["challenger_map50"] = item.comparison.overall[2].challenger
            entry["map50_delta"] = item.comparison.overall_delta("map50")
            entry["per_class"] = sorted(item.comparison.champion.per_class)
        summary["groups"].append(entry)  # type: ignore[union-attr]

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    (scratch / "group_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # -- the assertions that make this a check, not a demo -----------------
    problems: list[str] = []
    measured = {
        item.group: item
        for item in report.golden_groups
        if item.status == GROUP_MEASURED
    }
    for label in (HARD_CASE, REPRESENTATIVE):
        item = report.group(label)
        if item is None or item.status != GROUP_MEASURED:
            problems.append(
                f"{label} was not measured: "
                f"{item.status if item else 'absent'} {item.detail if item else ''}"
            )
            continue
        # A subset whose labels did not resolve still returns numbers; they
        # are just all zero. That is the failure this check exists for.
        champion_map50 = item.comparison.overall[2].champion  # type: ignore[union-attr]
        if not champion_map50 or champion_map50 <= 0.01:
            problems.append(
                f"{label}: champion mAP50 is {champion_map50}, which is what a "
                "subset with no resolvable ground truth also looks like."
            )
        found = set(item.comparison.champion.per_class)  # type: ignore[union-attr]
        if not found:
            problems.append(f"{label}: no per-class metrics came back.")

    tiny = report.group(TINY_GROUP)
    if tiny is None or tiny.status != GROUP_INSUFFICIENT:
        problems.append(
            f"{TINY_GROUP} ({args.tiny} samples) should have been refused as "
            f"INSUFFICIENT, got {tiny.status if tiny else 'absent'}."
        )

    if production_sha_before != production_sha_after:
        problems.append("THE PRODUCTION CHAMPION WEIGHT CHANGED.")

    if golden_after != golden_before:
        LOGGER.warning(
            "The golden directory was modified by evaluation: new=%s changed=%s",
            summary["golden_directory_new_files"],
            summary["golden_directory_changed_files"],
        )

    if problems:
        for problem in problems:
            LOGGER.error("%s", problem)
        return 1

    LOGGER.info(
        "Real per-group evaluation succeeded: %s measured, %s refused.",
        ", ".join(sorted(measured)),
        TINY_GROUP,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
