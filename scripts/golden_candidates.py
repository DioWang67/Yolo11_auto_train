#!/usr/bin/env python3
"""Build a golden *candidate* list for a station, for a person to review.

This does not create a golden set and cannot. It reads production review
manifests and operator handoff datasets, works out which images are worth a
reviewer's attention, and writes a list. Turning some of that list into a
golden set is a human act, done afterwards with
``picture-tool-autotrain golden register``.

Read-only over every input: production images are referenced where they live
and never copied, moved or modified.

    python scripts/golden_candidates.py --out runs/golden_candidates/v1
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

from picture_tool.autotrain.class_schema import (  # noqa: E402
    resolve_class_schema,
    schema_from_station_config,
)
from picture_tool.autotrain.golden_candidates import (  # noqa: E402
    build_candidates,
    read_training_provenance,
    write_report,
)
from picture_tool.autotrain.paths import AutoTrainPaths  # noqa: E402
from picture_tool.autotrain.registry import (  # noqa: E402
    read_champion,
    read_champion_class_schema,
)

LOGGER = logging.getLogger("golden_candidates")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--product", default="Cable1")
    parser.add_argument("--area", default="A")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "runs" / "golden_candidates",
        help="Where the candidate report is written.",
    )
    parser.add_argument(
        "--expected-boxes",
        type=int,
        default=6,
        help="Objects a good image of this station contains. Cable1/A has six "
        "across five classes: two of them are black.",
    )
    parser.add_argument(
        "--low-confidence-below",
        type=float,
        default=0.55,
        help="A detection under this counts as a hard case.",
    )
    parser.add_argument(
        "--per-duplicate-group",
        type=int,
        default=2,
        help="How many candidates to keep from each near-identical group.",
    )
    parser.add_argument(
        "--trained-provenance",
        type=Path,
        action="append",
        default=None,
        help="A training_provenance.json from a run whose data must stay out "
        "of the golden set. Repeatable. Golden has to be held out from every "
        "model it will ever judge, including the smoke runs.",
    )
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="Skip brightness/saturation/blur measurement, which reads every "
        "image and dominates the runtime.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    paths = AutoTrainPaths.discover()
    out = args.out.expanduser().resolve()
    # Candidates describe production; they are never written into it.
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

    data_root = paths.workspace.training_project / "data"
    manifests = sorted(
        (data_root / ".operator_handoff").rglob("*manifest*.csv")
    )
    labelled_roots = sorted(
        (data_root / ".operator_handoff" / "jobs").glob(
            f"*/dataset/{args.product}/{args.area}/raw"
        )
    )
    LOGGER.info(
        "Reading %d review manifest(s) and %d labelled job dataset(s)",
        len(manifests),
        len(labelled_roots),
    )

    trained_sources: set[str] = set()
    trained_hashes: set[str] = set()
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

    candidates, summary = build_candidates(
        schema=schema,
        review_manifests=manifests,
        labelled_roots=labelled_roots,
        product=args.product,
        area=args.area,
        expected_boxes=args.expected_boxes,
        low_confidence_below=args.low_confidence_below,
        per_duplicate_group=args.per_duplicate_group,
        measure_quality=not args.no_quality,
        trained_source_ids=trained_sources,
        trained_sha256=trained_hashes,
    )

    written = write_report(candidates, summary, out)

    LOGGER.info("--- candidate pass ---")
    LOGGER.info("  examined              : %s", summary["examined_before_thinning"])
    LOGGER.info("  thinned as near-dupes : %s", summary["thinned_out_as_near_duplicates"])
    LOGGER.info("  candidates            : %s", summary["image_count"])
    LOGGER.info("  by status             : %s", summary["by_status"])
    LOGGER.info("  by group              : %s", summary["by_group"])
    LOGGER.info("  hard-case reasons     : %s", summary["hard_case_reasons"])
    if summary["coverage_gaps"]:
        LOGGER.warning("  coverage gaps         : %s", summary["coverage_gaps"])
    LOGGER.info("  per class (instances) : %s", summary["per_class"])
    LOGGER.info("  golden eligible       : %s", summary["golden_eligible"])
    LOGGER.info("  by eligibility        : %s", summary["by_eligibility"])
    LOGGER.info("  unique source images  : %s", summary["unique_source_images"])
    LOGGER.info("  unknown lineage       : %s", summary["unknown_lineage"])
    top = list(summary["confusions"].items())[:5]
    LOGGER.info("  top confusions        : %s", dict(top))
    for name, path in written.items():
        LOGGER.info("  %-10s %s", name, path)
    LOGGER.info("Nothing is golden yet. See %s", written["readme"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
