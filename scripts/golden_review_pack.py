#!/usr/bin/env python3
"""Build a review pack: a few hundred images for a person to annotate.

A candidate pass produces well over a thousand rows, which nobody works
through, so the golden set never gets built. This cuts that to a pile a
reviewer can finish, keyed by distinct source capture rather than by file,
with everything the model has already trained on removed outright.

It creates no golden set, registers nothing, generates no labels, and does
not touch production inference, models or training. Every row leaves with
status NEEDS_LABEL.

    python scripts/golden_review_pack.py --out runs/review_pack/v1

Add ``--trained-provenance`` for every training run whose data must stay
out. Handoff job datasets are excluded automatically: they are what the
deployed models were built from.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picture_tool.autotrain.class_schema import (  # noqa: E402
    resolve_class_schema,
    schema_from_station_config,
)
from picture_tool.autotrain.golden_candidates import (  # noqa: E402
    IMAGE_SUFFIXES,
    build_candidates,
    difference_hash,
    read_training_provenance,
    sha256_file,
    source_image_id,
)
from picture_tool.autotrain.paths import AutoTrainPaths  # noqa: E402
from picture_tool.autotrain.registry import (  # noqa: E402
    read_champion,
    read_champion_class_schema,
)
from picture_tool.autotrain.review_pack import (  # noqa: E402
    build_review_pack,
    expected_class_counts,
    perceptual_matches,
    write_pack,
)

LOGGER = logging.getLogger("golden_review_pack")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--product", default="Cable1")
    parser.add_argument("--area", default="A")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "runs" / "review_pack",
        help="Where the pack is written.",
    )
    parser.add_argument("--expected-boxes", type=int, default=6)
    parser.add_argument("--low-confidence-below", type=float, default=0.55)
    parser.add_argument("--target-min", type=int, default=150)
    parser.add_argument("--target-max", type=int, default=250)
    parser.add_argument(
        "--per-cluster",
        type=int,
        default=3,
        help="How many members a near-duplicate cluster may contribute. "
        "They are chosen to be as unlike each other as the measurements "
        "allow, not taken in order.",
    )
    parser.add_argument(
        "--critical-class",
        action="append",
        default=None,
        help="Class whose confusion gets its own group. Repeatable; "
        "defaults to Red and Orange.",
    )
    parser.add_argument(
        "--trained-provenance",
        type=Path,
        action="append",
        default=None,
        help="A training_provenance.json whose images must stay out. "
        "Repeatable.",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Reference production images where they are instead of copying "
        "them into the pack. Cheaper, but production retention deletes "
        "passing images after thirty days and the pack would rot.",
    )
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="Skip brightness/saturation/blur measurement. Faster, but the "
        "diversity spread then has fewer axes to work with.",
    )
    return parser.parse_args(argv)


def station_expected_items(
    model_dir: Path, product: str, area: str
) -> list[str]:
    """The station's expected object multiset, from its own config.

    Emphatically not a class schema: it lists Black twice because the
    station physically has two black wires. Counting it is the one thing it
    is actually for.
    """
    config = model_dir / "config.yaml"
    try:
        payload = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        LOGGER.warning("Could not read %s: %s", config, exc)
        return []
    items = (payload.get("expected_items") or {}).get(product, {}).get(area)
    return [str(item) for item in items] if isinstance(items, list) else []


def handoff_exclusions(
    labelled_roots: list[Path],
) -> tuple[set[str], set[str], set[str]]:
    """Every image a handoff job has staged for training.

    The deployed champion was built through the operator handoff flow, and
    no provenance file records what it consumed, so the job datasets are
    the best available statement of what the model has seen. Treating them
    all as trained is the fail-closed reading: a job that was prepared and
    never trained costs the pack a few candidates, while the reverse would
    put a memorised image in the yardstick.

    Returns source ids, sha256s and perceptual hashes. The third is not
    redundant: a handoff copy is re-encoded and renamed, so its bytes match
    nothing and its filename --- ``<uuid>-yolo_Cable1_A_142254.jpg`` against
    production's ``yolo_Cable1_A_142252_433429_<hex>.jpg`` --- shares no
    structure with the lineage convention either. Five images passed both
    checks and were still the same photograph.
    """
    sources: set[str] = set()
    hashes: set[str] = set()
    perceptual: set[str] = set()
    for root in labelled_roots:
        images = root / "images"
        if not images.is_dir():
            continue
        for image in sorted(images.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            sources.add(source_image_id(image))
            try:
                hashes.add(sha256_file(image))
            except OSError:
                LOGGER.warning("Could not hash %s", image)
                continue
            digest = difference_hash(image)
            if digest:
                perceptual.add(digest)
    return sources, hashes, perceptual


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    paths = AutoTrainPaths.discover()
    out = args.out.expanduser().resolve()
    # A review pack describes production; it is never written into it.
    paths.assert_not_production(out)

    model_dir = paths.production_model_dir(args.product, args.area)
    champion = read_champion(model_dir)
    schema = resolve_class_schema(
        [
            read_champion_class_schema(champion) if champion else None,
            schema_from_station_config(model_dir),
        ],
        context=f"{args.product}/{args.area}",
    )
    LOGGER.info("Class contract: %s", schema.describe())

    expected_items = station_expected_items(model_dir, args.product, args.area)
    counts = expected_class_counts(expected_items)
    LOGGER.info("Station expects %s", counts or "(unknown)")

    data_root = paths.workspace.training_project / "data"
    manifests = sorted((data_root / ".operator_handoff").rglob("*manifest*.csv"))
    labelled_roots = sorted(
        (data_root / ".operator_handoff" / "jobs").glob(
            f"*/dataset/{args.product}/{args.area}/raw"
        )
    )

    trained_sources, trained_hashes, trained_perceptual = handoff_exclusions(
        labelled_roots
    )
    LOGGER.info(
        "Handoff jobs staged %d image(s) over %d source(s), %d perceptual hash(es)",
        len(trained_hashes),
        len(trained_sources),
        len(trained_perceptual),
    )
    for provenance in args.trained_provenance or []:
        sources, hashes = read_training_provenance(provenance)
        trained_sources |= sources
        trained_hashes |= hashes
        LOGGER.info(
            "%s: %d trained image(s) over %d source(s)",
            provenance,
            len(hashes),
            len(sources),
        )

    # per_duplicate_group=None: exact duplicates still collapse, but the
    # near-duplicate clusters arrive whole so the pack can spread its choice
    # across them instead of inheriting whichever two came first.
    candidates, candidate_summary = build_candidates(
        schema=schema,
        review_manifests=manifests,
        labelled_roots=labelled_roots,
        product=args.product,
        area=args.area,
        expected_boxes=args.expected_boxes,
        low_confidence_below=args.low_confidence_below,
        per_duplicate_group=None,
        measure_quality=not args.no_quality,
        trained_source_ids=trained_sources,
        trained_sha256=trained_hashes,
    )
    LOGGER.info("Candidate pass produced %d row(s)", len(candidates))

    perceptual = perceptual_matches(
        candidates, trained_perceptual, hasher=difference_hash
    )
    LOGGER.info(
        "%d candidate(s) are perceptually identical to a training image "
        "despite differing bytes and filenames",
        len(perceptual),
    )

    entries, summary = build_review_pack(
        candidates,
        trained_source_ids=trained_sources,
        trained_sha256=trained_hashes,
        perceptually_trained=perceptual,
        critical_classes=tuple(args.critical_class or ("Red", "Orange")),
        expected_counts=counts,
        target_min=args.target_min,
        target_max=args.target_max,
        per_cluster=args.per_cluster,
    )
    summary["candidate_pass"] = {
        key: candidate_summary.get(key)
        for key in ("examined_before_thinning", "total", "by_status")
        if key in candidate_summary
    }
    summary["class_schema"] = schema.to_dict()

    written = write_pack(
        entries,
        summary,
        out,
        schema=schema,
        copy_images=not args.no_copy_images,
    )

    LOGGER.info(
        "%d distinct source(s) selected: %s",
        summary["distinct_sources_selected"],
        summary["per_group"],
    )
    if not summary["meets_target_min"]:
        LOGGER.warning(
            "Short of the %d minimum by %d. Reported rather than padded: "
            "there is no honest way to invent distinct captures.",
            args.target_min,
            summary["shortfall"],
        )
    for name, path in written.items():
        LOGGER.info("%-8s %s", name, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
