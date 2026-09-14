#!/usr/bin/env python3
"""Run the bootstrapper over a real batch and hand the result to AutoTrain.

The batch is the existing review pack. That is not a shortcut around the
selection requirements, it is those requirements already satisfied and
already reproducible: 250 distinct source captures, deduplicated at source
level, spread across capture days and quality, with the Red/Orange hard
cases deliberately over-represented, and a manifest recording every choice.
Selecting a second batch by the same rules would produce the same kind of
list and a second thing to keep in sync.

No accuracy is claimed. The only human labels for this station are images
the champion trained on, and the champion is an evidence source here, so
measuring against them would report memorisation. The reports say so
explicitly rather than leaving a blank a reader fills in optimistically.

    python scripts/bootstrap_poc.py --limit 250
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picture_tool.autotrain.dataset_versions import (  # noqa: E402
    DatasetVersionStore,
    LabelledSample,
)
from picture_tool.autotrain.orchestrator import (  # noqa: E402
    _load_base_pipeline_config,
)
from picture_tool.autotrain.paths import AutoTrainPaths  # noqa: E402
from picture_tool.autotrain.registry import (  # noqa: E402
    TRAINED,
    CandidateRegistry,
    read_champion,
    read_champion_class_schema,
)
from picture_tool.autotrain.trainer import train_candidate  # noqa: E402
from picture_tool.bootstrap import auto_label, export, sample_quality  # noqa: E402
from picture_tool.bootstrap.evidence import (  # noqa: E402
    ColourEvidence,
    DetectorProposer,
    load_colour_ranges,
)
from picture_tool.bootstrap.profile import load_profile  # noqa: E402

LOGGER = logging.getLogger("bootstrap_poc")

#: Fixed in the source: this POC may not claim an accuracy.
NO_ACCURACY = "no_independent_truth_set"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--product", default="Cable1")
    parser.add_argument("--area", default="A")
    parser.add_argument(
        "--batch",
        type=Path,
        default=PROJECT_ROOT / "runs" / "review_pack" / "v1",
        help="Review pack supplying the unlabeled batch and its manifest.",
    )
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "runs" / "bootstrap_poc"
    )
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--detector-confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-imgsz", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    return parser.parse_args(argv)


def read_batch(pack: Path, limit: int) -> list[dict]:
    """The unlabeled batch, in the pack's own recorded order."""
    csv_path = pack / "review_pack.csv"
    if not csv_path.is_file():
        raise SystemExit(f"No review pack at {csv_path}.")
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    batch = []
    for row in rows[:limit]:
        image = pack / row["pack_image"] if row.get("pack_image") else None
        if image is None or not image.is_file():
            continue
        batch.append(
            {
                "sample_id": row["source_image_id"],
                "image": image,
                "group": row.get("group", ""),
                "production_path": row.get("image_path", ""),
            }
        )
    return batch


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
        raise SystemExit("No deployed champion to propose boxes with.")
    profile = load_profile(
        model_dir,
        product=args.product,
        area=args.area,
        champion_schema=read_champion_class_schema(champion),
        # Supplied, not inferred: Red->Orange is 109 of this station's human
        # corrections, but a profile that reshapes itself as data arrives is
        # a moving definition.
        confusion_pairs=[("Red", "Orange")],
    )
    LOGGER.info("Profile: %s", json.dumps(profile.to_dict(), ensure_ascii=False))
    if profile.colour_stats_path is None:
        raise SystemExit(
            "No colour reference statistics. Without an independent source "
            "the detector would be accepting its own output, which this "
            "refuses to do."
        )

    batch = read_batch(args.batch, args.limit)
    LOGGER.info("Batch: %d unlabeled production images", len(batch))
    if not batch:
        raise SystemExit(f"No usable images under {args.batch}.")

    weights = champion.training_weight_path or champion.weights_path
    detector = DetectorProposer(
        weights,
        confidence=args.detector_confidence,
        imgsz=args.imgsz,
        device=args.device,
    )
    colour = ColourEvidence(load_colour_ranges(profile.colour_stats_path))

    qualities = sample_quality.assess_batch(
        [item["image"] for item in batch],
        sample_ids={str(item["image"]): item["sample_id"] for item in batch},
    )
    quality_by_id = {q.sample_id: q for q in qualities}

    import cv2

    decisions = []
    for index, item in enumerate(batch, start=1):
        if index % 25 == 0:
            LOGGER.info("  %d/%d", index, len(batch))
        quality = quality_by_id[item["sample_id"]]
        boxes = []
        opinions = {}
        if quality.is_usable:
            boxes = detector.propose(item["image"], profile)
            image = cv2.imread(str(item["image"]))
            if image is not None and boxes:
                opinions = {
                    "detector": detector.read(image, boxes, profile),
                    "colour": colour.read(image, boxes, profile),
                }
        decisions.append(
            auto_label.decide(
                sample_id=item["sample_id"],
                image_path=str(item["image"]),
                boxes=boxes,
                opinions_by_source=opinions,
                quality=quality,
                profile=profile,
                provenance={
                    "detector_weights": str(weights),
                    "detector_weight_sha256": champion.weight_sha256,
                    "detector_model_version": champion.model_version,
                    "detector_confidence_threshold": args.detector_confidence,
                    "colour_strategy": "picture_tool.color.strategies",
                    "review_pack_group": item["group"],
                    "production_path": item["production_path"],
                },
            )
        )

    stats = auto_label.summarise(decisions)
    LOGGER.info("Decisions: %s", json.dumps(stats["by_decision"]))

    exported = None
    accepted = [d for d in decisions if d.decision == auto_label.AUTO_ACCEPT]
    if accepted:
        exported = export.export_accepted(
            decisions, out / "dataset", profile=profile, overwrite=True
        )
        LOGGER.info(
            "Exported %d image(s), %d instance(s), content %s",
            exported.images,
            exported.instances,
            exported.content_hash[:12],
        )
    else:
        LOGGER.warning(
            "Nothing was accepted. That is a result: the evidence did not "
            "corroborate anything, and an empty dataset is the honest output."
        )

    written = export.write_reports(
        decisions,
        out / "report",
        profile=profile,
        batch_manifest={
            "source": str(args.batch),
            "images": len(batch),
            "selected_at": datetime.now(timezone.utc).isoformat(),
            "reproducible_from": str(args.batch / "review_pack.csv"),
        },
        export=exported,
    )
    for name, path in written.items():
        LOGGER.info("%-13s %s", name, path)

    summary = {
        "accuracy": {"measured": False, "reason": NO_ACCURACY},
        "statistics": stats,
        "export": exported.to_dict() if exported else None,
    }

    if exported and not args.no_train:
        version_store = DatasetVersionStore(
            out / "dataset_versions", product=args.product, area=args.area
        )
        samples = [
            LabelledSample(
                sample_id=item.sample_id,
                image_path=exported.root / "images" / "train"
                / f"{item.sample_id}{Path(item.image_path).suffix}",
                label_path=exported.root / "labels" / "train" / f"{item.sample_id}.txt",
                origin="bootstrap_auto_label",
            )
            for item in accepted
        ]
        version = version_store.create(
            samples,
            source=str(exported.root),
            label_source="bootstrap_auto_label",
            class_schema=profile.class_schema,
            description="Pseudo-labels accepted by the bootstrapper.",
            extra={
                "bootstrap": True,
                "reference_source": "existing_validated_color_stats",
                "accuracy_measured": False,
                "content_hash": exported.content_hash,
            },
        )
        LOGGER.info("Dataset version %s (%s)", version.version, version.content_id[:12])

        registry = CandidateRegistry(
            out / "registry", product=args.product, area=args.area
        )
        model_version = f"{args.product}_{args.area}_bootstrap_poc"
        registry.register(
            model_version,
            dataset_version=version.version,
            dataset_content_id=version.content_id,
            parent_model=champion.model_version,
            cycle_id="bootstrap_poc",
            class_schema=profile.class_schema,
            training_config={
                "epochs": args.epochs,
                "imgsz": args.train_imgsz,
                "batch": args.batch_size,
                "device": args.device,
                "base_model": weights,
                "labels": "bootstrap_pseudo_labels",
            },
        )
        LOGGER.info("Training a challenger on the pseudo-labels; not quick.")
        result = train_candidate(
            dataset_version=version,
            base_config=_load_base_pipeline_config(paths, args.product),
            candidate_dir=registry.directory_for(model_version),
            work_dir=out / "work",
            model_version=model_version,
            base_model=weights,
            class_schema=profile.class_schema,
            epochs=args.epochs,
            imgsz=args.train_imgsz,
            batch=args.batch_size,
            device=args.device,
        )
        registry.update_status(
            model_version,
            TRAINED,
            training_metrics=dict(result.metrics),
            weight_sha256=result.weight_sha256,
            base_model=result.base_model,
            notes="bootstrap POC: pseudo-labels, no independent accuracy",
        )
        summary["dataset_version"] = {
            "version": version.version,
            "content_id": version.content_id,
            "samples": len(samples),
        }
        summary["training"] = {
            "weights": str(result.weights_path),
            "trained_this_run": result.trained_this_run,
            "weight_sha256": result.weight_sha256,
        }
        LOGGER.info("Challenger weights: %s", result.weights_path)

    (out / "poc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
