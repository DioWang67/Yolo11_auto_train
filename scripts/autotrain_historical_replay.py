#!/usr/bin/env python3
"""Replay a real, already-used training dataset through the AutoTrain path.

The question is not whether the resulting model is good. It is whether
AutoTrain can take over a dataset a person built by hand and carry it all
the way through: import, dataset version, real training, evaluation,
registry, report.

**Every number this produces is NON_INDEPENDENT.** The champion was trained
on this data and has no provenance recorded, so "the challenger scored
better" here means nothing about either model. The decision is fixed at
NOT_PROMOTABLE_NON_INDEPENDENT before any metric is measured, and no
metric can change it. A genuine promotion still needs a held-out golden
set.

The original dataset is opened read-only. Nothing is written into it, into
the production model directory, or into the inference project.

    python scripts/autotrain_historical_replay.py --epochs 1
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picture_tool.autotrain.class_schema import (  # noqa: E402
    normalize_class_names,
    resolve_class_schema,
    schema_from_station_config,
)
from picture_tool.autotrain.dataset_versions import (  # noqa: E402
    DatasetVersionStore,
    LabelledSample,
)
from picture_tool.autotrain.evaluator import evaluate_candidate  # noqa: E402
from picture_tool.autotrain.golden import GoldenStatus, NOT_CONFIGURED  # noqa: E402
from picture_tool.autotrain.golden_candidates import (  # noqa: E402
    difference_hash,
    sha256_file,
    source_image_id,
)
from picture_tool.autotrain.paths import AutoTrainPaths  # noqa: E402
from picture_tool.autotrain.registry import (  # noqa: E402
    TRAINED,
    CandidateRegistry,
    read_champion,
    read_champion_class_schema,
)
from picture_tool.autotrain.orchestrator import (  # noqa: E402
    _load_base_pipeline_config,
)
from picture_tool.autotrain.trainer import train_candidate  # noqa: E402

LOGGER = logging.getLogger("historical_replay")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")

#: The only decision this path may reach, fixed before anything is measured.
NOT_PROMOTABLE = "NOT_PROMOTABLE_NON_INDEPENDENT"
#: Stamped on every metric produced here.
NON_INDEPENDENT = "NON_INDEPENDENT"
REPLAY_MARKER = "historical_replay"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--product", default="Cable1")
    parser.add_argument("--area", default="A")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="The historical split directory. Default: traced from the "
        "champion's dataset_hash through the training run metadata, which "
        "is evidence rather than a guess about which dataset it was.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "runs" / "historical_replay",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--analyse-only",
        action="store_true",
        help="Stop after the leakage analysis and the two imports.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Step 1: find the dataset the champion was actually trained on


def trace_historical_dataset(paths: AutoTrainPaths, product: str, area: str) -> Path:
    """Follow the champion's dataset_hash back to the directory that made it.

    Guessing from dates would probably land on the same place, but "probably"
    is not a provenance. The deployment manifest records a dataset_hash; the
    training run that produced it recorded the same hash beside the directory
    it read. That chain is checkable, so it is the one used.
    """
    manifest = (
        paths.production_model_dir(product, area) / "deployment_manifest.yaml"
    )
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    dataset_hash = str(payload.get("dataset_hash") or "")
    if not dataset_hash:
        raise SystemExit(
            f"{manifest} records no dataset_hash, so the dataset this model "
            "was trained on cannot be identified from here. Pass --dataset."
        )
    for meta in sorted(PROJECT_ROOT.glob("runs/**/last_run_metadata.json")):
        try:
            recorded = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(recorded.get("dataset_hash")) != dataset_hash:
            continue
        directory = Path(str(recorded.get("dataset_dir") or ""))
        if directory.is_dir():
            LOGGER.info(
                "Champion dataset_hash %s traced to %s (trained_at %s)",
                dataset_hash[:12],
                directory,
                recorded.get("trained_at"),
            )
            return directory
    raise SystemExit(
        f"No training run on this machine records dataset_hash {dataset_hash}. "
        "Pass --dataset explicitly rather than guessing."
    )


# ---------------------------------------------------------------------------
# Step 3: what is actually in it


def scan_dataset(root: Path) -> dict:
    """Every physical file, with the three identities that matter."""
    records = []
    for split in SPLITS:
        images = root / split / "images"
        labels = root / split / "labels"
        if not images.is_dir():
            continue
        for image in sorted(images.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = labels / f"{image.stem}.txt"
            records.append(
                {
                    "split": split,
                    "image": image,
                    "label": label if label.is_file() else None,
                    "sha256": sha256_file(image),
                    "source": source_image_id(image),
                    "dhash": difference_hash(image),
                    "augmented": source_image_id(image) != image.stem,
                }
            )
    return {"root": root, "records": records}


def analyse(scan: dict) -> dict:
    records = scan["records"]
    by_split = collections.Counter(r["split"] for r in records)
    sha = [r["sha256"] for r in records]
    src = [r["source"] for r in records]
    dhs = [r["dhash"] for r in records if r["dhash"]]

    splits_of_source: dict[str, set[str]] = collections.defaultdict(set)
    splits_of_sha: dict[str, set[str]] = collections.defaultdict(set)
    splits_of_dhash: dict[str, set[str]] = collections.defaultdict(set)
    sources_of_dhash: dict[str, set[str]] = collections.defaultdict(set)
    for record in records:
        splits_of_source[record["source"]].add(record["split"])
        splits_of_sha[record["sha256"]].add(record["split"])
        if record["dhash"]:
            splits_of_dhash[record["dhash"]].add(record["split"])
            sources_of_dhash[record["dhash"]].add(record["source"])

    cross_dhash = {
        value
        for value, splits in splits_of_dhash.items()
        if len(splits) > 1 and len(sources_of_dhash[value]) > 1
    }
    leaked_files = [r for r in records if r["dhash"] in cross_dhash]
    leaked_sources = {r["source"] for r in leaked_files}

    def pair_overlap(left: str, right: str) -> int:
        return sum(
            1
            for splits in splits_of_source.values()
            if left in splits and right in splits
        )

    return {
        "physical_images": len(records),
        "per_split": dict(by_split),
        "unique_sha256": len(set(sha)),
        "distinct_sources": len(set(src)),
        "unique_dhash": len(set(dhs)),
        "exact_duplicate_files": len(sha) - len(set(sha)),
        "augmented_files": sum(1 for r in records if r["augmented"]),
        "missing_labels": sum(1 for r in records if r["label"] is None),
        "source_across_splits": sum(
            1 for splits in splits_of_source.values() if len(splits) > 1
        ),
        "sha_across_splits": sum(
            1 for splits in splits_of_sha.values() if len(splits) > 1
        ),
        "train_val_source_overlap": pair_overlap("train", "val"),
        "train_test_source_overlap": pair_overlap("train", "test"),
        "val_test_source_overlap": pair_overlap("val", "test"),
        "near_duplicate_groups_across_splits": len(cross_dhash),
        "near_duplicate_files_across_splits": len(leaked_files),
        "near_duplicate_sources_across_splits": len(leaked_sources),
    }


def families(records: list[dict]) -> dict[str, str]:
    """Group images that must not be separated by a split boundary.

    A family is the transitive closure of two relations: sharing a source
    capture, and hashing alike. The first is the splitter's own convention
    and the original split already respects it. The second is what it misses
    --- different captures of the same fixture that a model cannot tell
    apart, which is a val set scoring itself on the training data by another
    route.
    """
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[a] = b

    for record in records:
        union(f"src:{record['source']}", f"img:{record['image']}")
        if record["dhash"]:
            union(f"src:{record['source']}", f"dh:{record['dhash']}")
    return {str(r["image"]): find(f"img:{r['image']}") for r in records}


# ---------------------------------------------------------------------------
# Step 2 and 4: the two imports


def originals(records: list[dict]) -> list[dict]:
    """One record per source capture: the un-augmented original if present."""
    by_source: dict[str, dict] = {}
    for record in records:
        if record["label"] is None:
            continue
        current = by_source.get(record["source"])
        if current is None or (record["augmented"] is False and current["augmented"]):
            by_source[record["source"]] = record
    return list(by_source.values())


def import_version(
    store: DatasetVersionStore,
    records: list[dict],
    *,
    schema,
    source_root: Path,
    description: str,
    extra: dict,
):
    """Import as an immutable dataset version. Reads the original, never writes."""
    samples = [
        LabelledSample(
            sample_id=record["source"],
            image_path=record["image"],
            label_path=record["label"],
            origin=f"{REPLAY_MARKER}:{source_root.name}",
        )
        for record in records
    ]
    return store.create(
        samples,
        source=str(source_root),
        label_source="human_operator_handoff",
        class_schema=schema,
        description=description,
        extra=extra,
    )


def family_split(
    records: list[dict], *, val_fraction: float, seed: int = 0
) -> dict[str, str]:
    """Assign whole families to train or val, never splitting one.

    Deterministic: families are ordered by their key so the same dataset
    always produces the same split, which is what makes a re-run comparable
    with the run before it.
    """
    membership = families(records)
    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for record in records:
        by_family[membership[str(record["image"])]].append(record)

    ordered = sorted(by_family, key=lambda key: (-len(by_family[key]), key))
    total = len(records)
    target = int(total * val_fraction)
    assignment: dict[str, str] = {}
    in_val = 0
    # Largest family first, so one huge family cannot overshoot the target
    # after the small ones have already filled it.
    for index, family in enumerate(ordered):
        members = by_family[family]
        take_val = in_val < target and index % 2 == 1
        for record in members:
            assignment[str(record["image"])] = "val" if take_val else "train"
        if take_val:
            in_val += len(members)
    return assignment


def write_split_dataset(
    records: list[dict],
    assignment: dict[str, str],
    destination: Path,
    schema,
) -> Path:
    """Materialise a YOLO dataset by copying. The source is never touched."""
    for split in ("train", "val"):
        (destination / split / "images").mkdir(parents=True, exist_ok=True)
        (destination / split / "labels").mkdir(parents=True, exist_ok=True)
    for record in records:
        if record["label"] is None:
            continue
        split = assignment[str(record["image"])]
        shutil.copy2(
            record["image"], destination / split / "images" / record["image"].name
        )
        shutil.copy2(
            record["label"],
            destination / split / "labels" / f"{record['image'].stem}.txt",
        )
    (destination / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(destination.resolve()),
                "train": "train/images",
                "val": "val/images",
                "names": {i: n for i, n in enumerate(schema.names)},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return destination / "data.yaml"


def copy_original_val(records: list[dict], destination: Path, schema) -> Path:
    """The historical split exactly as a person left it, copied out.

    Copied rather than pointed at: evaluation makes ultralytics write a
    label cache beside the labels, and the original dataset is the record of
    what the champion was trained on. It does not get written to.

    Uses every physical file, not one per source, because the whole point of
    the comparison is the split a person actually made --- and that split was
    made over the augmented files.
    """
    assignment = {
        str(r["image"]): ("val" if r["split"] == "val" else "train")
        for r in records
        if r["split"] in ("val", "train")
    }
    subset = [r for r in records if r["split"] in ("val", "train")]
    return write_split_dataset(subset, assignment, destination, schema)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    paths = AutoTrainPaths.discover()
    out = args.out.expanduser().resolve()
    paths.assert_not_production(out)
    if args.fresh and out.exists():
        for path in out.rglob("*"):
            if path.is_file():
                path.chmod(0o666)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    model_dir = paths.production_model_dir(args.product, args.area)
    champion = read_champion(model_dir)
    if champion is None:
        raise SystemExit("No deployed champion; there is nothing to compare against.")
    schema = resolve_class_schema(
        [read_champion_class_schema(champion), schema_from_station_config(model_dir)],
        context=f"{args.product}/{args.area}",
    )

    dataset_root = args.dataset or trace_historical_dataset(
        paths, args.product, args.area
    )
    descriptor = dataset_root / "data.yaml"
    recorded_names = (
        (yaml.safe_load(descriptor.read_text(encoding="utf-8")) or {}).get("names")
        if descriptor.is_file()
        else None
    )
    dataset_schema = (
        normalize_class_names(recorded_names, source="historical_data_yaml")
        if recorded_names
        else None
    )
    if dataset_schema and not dataset_schema.agrees_with(schema):
        raise SystemExit(
            "The historical dataset's class order disagrees with the "
            f"champion's.\n  dataset: {dataset_schema.describe()}\n"
            f"  champion: {schema.describe()}"
        )

    LOGGER.info("Historical dataset: %s", dataset_root)
    scan = scan_dataset(dataset_root)
    stats = analyse(scan)
    LOGGER.info("Leakage analysis: %s", json.dumps(stats, ensure_ascii=False))

    summary: dict = {
        "marker": REPLAY_MARKER,
        "independence": NON_INDEPENDENT,
        "decision": NOT_PROMOTABLE,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "historical_dataset": str(dataset_root),
        "class_schema": schema.to_dict(),
        "analysis": stats,
        "champion": {
            "model_version": champion.model_version,
            "weights": champion.training_weight_path or champion.weights_path,
        },
    }

    # -- Step 2 and 4: the two imports ------------------------------------
    store = DatasetVersionStore(
        out / "dataset_versions", product=args.product, area=args.area
    )
    source_records = originals(scan["records"])
    LOGGER.info(
        "%d source captures carry labels, out of %d physical files",
        len(source_records),
        len(scan["records"]),
    )

    faithful = import_version(
        store,
        source_records,
        schema=schema,
        source_root=dataset_root,
        description="Faithful import of the champion's historical dataset.",
        extra={
            "replay": REPLAY_MARKER,
            "independence": NON_INDEPENDENT,
            "original_split": {
                r["source"]: r["split"] for r in sorted(
                    source_records, key=lambda x: x["source"]
                )
            },
            "original_dataset_path": str(dataset_root),
            "imported_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    summary["faithful_version"] = {
        "version": faithful.version,
        "content_id": faithful.content_id,
        "samples": len(source_records),
    }
    LOGGER.info("Faithful import: %s (%s)", faithful.version, faithful.content_id[:12])

    # Both comparison datasets are cut from the same population -- every
    # physical file -- because the question is which *split* of that
    # population is sound. Re-splitting only the 46 originals against a
    # historical val of 98 augmented files would compare two different
    # things and attribute the difference to the split.
    labelled = [r for r in scan["records"] if r["label"] is not None]

    clean_dir = out / "historical_replay_clean"
    assignment = family_split(labelled, val_fraction=args.val_fraction)
    clean_yaml = write_split_dataset(labelled, assignment, clean_dir, schema)
    clean_counts = collections.Counter(assignment.values())
    summary["source_safe_split"] = {
        "path": str(clean_dir),
        "counts": dict(clean_counts),
        "families": len(set(families(labelled).values())),
        "note": (
            "Families are the transitive closure of shared source capture "
            "and equal perceptual hash, so no family crosses the boundary."
        ),
    }
    LOGGER.info("Source-safe split: %s", dict(clean_counts))

    original_dir = out / "historical_original_split"
    original_yaml = copy_original_val(labelled, original_dir, schema)
    summary["original_split_eval"] = {
        "path": str(original_dir),
        "counts": dict(
            collections.Counter(
                r["split"] for r in labelled if r["split"] in ("train", "val")
            )
        ),
    }

    (out / "analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.analyse_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    # -- Step 5: real training -------------------------------------------
    registry = CandidateRegistry(out / "registry", product=args.product, area=args.area)
    model_version = f"{args.product}_{args.area}_historical_replay"
    base_model = champion.training_weight_path or champion.weights_path
    registry.register(
        model_version,
        dataset_version=faithful.version,
        dataset_content_id=faithful.content_id,
        parent_model=champion.model_version,
        cycle_id=REPLAY_MARKER,
        class_schema=schema,
        training_config={
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "base_model": base_model,
            "replay": REPLAY_MARKER,
            "independence": NON_INDEPENDENT,
        },
    )
    LOGGER.info("Training for real from %s; this is not quick.", Path(base_model).name)
    result = train_candidate(
        dataset_version=faithful,
        base_config=_load_base_pipeline_config(paths, args.product),
        candidate_dir=registry.directory_for(model_version),
        work_dir=out / "work",
        model_version=model_version,
        base_model=base_model,
        class_schema=schema,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    registry.update_status(
        model_version,
        TRAINED,
        training_metrics=dict(result.metrics),
        weight_sha256=result.weight_sha256,
        base_model=result.base_model,
        notes=f"{REPLAY_MARKER}: {NON_INDEPENDENT}",
    )
    summary["training"] = {
        "weights": str(result.weights_path),
        "weight_sha256": result.weight_sha256,
        "trained_this_run": result.trained_this_run,
        "dataset_version": result.dataset_version,
        "base_model": result.base_model,
        "run_dir": str(result.run_dir),
    }
    LOGGER.info("Challenger weights: %s", result.weights_path)

    # -- Step 6: evaluate on both splits ---------------------------------
    golden_status = GoldenStatus(
        status=NOT_CONFIGURED,
        detail=(
            "No golden set is configured, and none of this replay could serve "
            "as one: the champion was trained on this data."
        ),
    )
    evaluations = {}
    for name, data_yaml in (
        ("original_historical_val", original_yaml),
        ("source_safe_val", clean_yaml),
    ):
        LOGGER.info("Evaluating on %s", name)
        report = evaluate_candidate(
            champion_weights=base_model,
            challenger_weights=result.weights_path,
            data_yaml=data_yaml,
            golden_status=golden_status,
            split="val",
            confidence=0.4,
            imgsz=args.imgsz,
            device=args.device,
            batch=args.batch,
            workers=0,
        )
        evaluations[name] = {
            "independence": NON_INDEPENDENT,
            "status": report.status,
            "overall": [item.to_dict() for item in report.comparison.overall],
            "per_class_recall": [
                item.to_dict() for item in report.comparison.per_class_recall
            ],
        }
    summary["evaluations"] = evaluations

    # -- Steps 7 and 8 ----------------------------------------------------
    def delta(name: str, metric: str):
        for item in evaluations[name]["overall"]:
            if item["name"] == metric:
                return item
        return None

    summary["split_comparison"] = {
        metric: {
            "original_historical_val": delta("original_historical_val", metric),
            "source_safe_val": delta("source_safe_val", metric),
        }
        for metric in ("precision", "recall", "map50", "map50_95")
    }
    summary["promotion"] = {
        "decision": NOT_PROMOTABLE,
        "reason": (
            "The champion was trained on this data and records no provenance, "
            "so no measurement here is independent of it. Fixed before any "
            "metric was taken; no result can change it."
        ),
        "deployed": False,
    }
    (out / "replay_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    LOGGER.info("Decision: %s (nothing deployed, nothing registered as golden)", NOT_PROMOTABLE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
