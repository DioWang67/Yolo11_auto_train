#!/usr/bin/env python3
"""Train one real challenger through the autonomous training path.

Every automated test of :mod:`picture_tool.autotrain.trainer` injects a fake
pipeline runner, so the interface against the real
:func:`picture_tool.main_pipeline.run_pipeline` --- and against ultralytics
behind it --- is exercised nowhere else. This script closes that gap: it takes
a real operator handoff job as its labelled data, cuts a dataset version from
it, and trains a challenger for real on CPU.

It never deploys and never writes into the inference project; the task list
forbids publishing tasks and the scratch directory is checked against the
production tree before anything is written.

Output is verbose (ultralytics draws progress bars), so redirect it::

    python scripts/autotrain_smoke.py --fresh > smoke.log 2>&1

Re-running without ``--fresh`` is itself a useful check: the pipeline's skip
cache should recognise the unchanged dataset and config, and the run should
then report ``trained_this_run: False``.

Known flake: the splitter publishes its output by renaming a staging
directory, and on Windows that has been seen to fail once with ``WinError 5``
while something else still held a handle on the freshly written files --- an
editor's file watcher or a virus scanner. It is transient; the same command
succeeded on the next run. ``--scratch`` somewhere outside the working tree
avoids the watchers if it recurs.
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
    ClassSchema,
    normalize_class_names,
    resolve_class_schema,
    schema_from_station_config,
)
from picture_tool.autotrain.dataset_versions import (  # noqa: E402
    DatasetVersionStore,
    LabelledSample,
)
from picture_tool.autotrain.paths import AutoTrainPaths  # noqa: E402
from picture_tool.autotrain.registry import (  # noqa: E402
    read_champion,
    read_champion_class_schema,
)
from picture_tool.autotrain.trainer import train_candidate  # noqa: E402
from picture_tool.config_loader import load_config  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

LOGGER = logging.getLogger("autotrain_smoke")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--product", default="Cable1")
    parser.add_argument("--area", default="A")
    parser.add_argument(
        "--job",
        default="",
        help="Operator handoff job id. Defaults to the newest job that has a "
        "complete labelled raw dataset for this station.",
    )
    parser.add_argument(
        "--scratch",
        type=Path,
        default=PROJECT_ROOT / "runs" / "autotrain_smoke",
        help="Where versions, work files and candidate weights go. Must not "
        "be inside the inference project.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the scratch directory first, so training really runs "
        "instead of being served by the pipeline's skip cache.",
    )
    return parser.parse_args(argv)


def find_job(data_root: Path, product: str, area: str, job_id: str) -> Path:
    """Locate a handoff job's raw dataset for this station."""
    jobs_root = data_root / ".operator_handoff" / "jobs"
    if job_id:
        raw = jobs_root / job_id / "dataset" / product / area / "raw"
        if not raw.is_dir():
            raise SystemExit(f"Job {job_id} has no raw dataset at {raw}")
        return raw

    candidates = sorted(
        (
            job / "dataset" / product / area / "raw"
            for job in jobs_root.iterdir()
            if job.is_dir()
        ),
        key=lambda raw: raw.parent.stat().st_mtime if raw.is_dir() else 0,
    )
    for raw in reversed(candidates):
        if (raw / "images").is_dir() and (raw / "labels").is_dir():
            return raw
    raise SystemExit(
        f"No operator handoff job under {jobs_root} has a labelled raw "
        f"dataset for {product}/{area}."
    )


def read_job_class_schema(job_raw: Path) -> ClassSchema | None:
    """The class order recorded in the job's own data.yaml.

    This is what the job's labels were actually written against, so it is the
    sharpest available cross-check on the contract resolved from the station.
    """
    for data_yaml in sorted(job_raw.parent.rglob("data.yaml")):
        payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        names = (payload or {}).get("names")
        if names:
            return normalize_class_names(names, source=f"job:{data_yaml.name}")
    return None


def collect_samples(job_raw: Path) -> list[LabelledSample]:
    samples = []
    for image in sorted((job_raw / "images").iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = job_raw / "labels" / f"{image.stem}.txt"
        if not label.is_file():
            LOGGER.warning("No label for %s; skipping.", image.name)
            continue
        samples.append(
            LabelledSample(
                sample_id=image.stem,
                image_path=image,
                label_path=label,
                origin="operator_handoff_smoke",
            )
        )
    if not samples:
        raise SystemExit(f"No labelled images found under {job_raw}")
    return samples


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    paths = AutoTrainPaths.discover()
    scratch = args.scratch.expanduser().resolve()
    # The whole point of this path is that it cannot touch production, so the
    # guard runs before anything is written rather than being assumed.
    paths.assert_not_production(scratch)

    if args.fresh and scratch.exists():
        import shutil

        # The version store marks its files read-only; clear that first.
        for path in scratch.rglob("*"):
            if path.is_file():
                path.chmod(0o666)
        shutil.rmtree(scratch)
        LOGGER.info("Cleared %s", scratch)

    data_root = paths.workspace.training_project / "data"
    job_raw = find_job(data_root, args.product, args.area, args.job)
    samples = collect_samples(job_raw)
    LOGGER.info("Using %d labelled samples from %s", len(samples), job_raw)

    champion = read_champion(paths.production_model_dir(args.product, args.area))
    if champion is None:
        raise SystemExit(
            f"No deployed champion for {args.product}/{args.area}; a challenger "
            "continues from the deployed weight, so there is nothing to train "
            "from."
        )
    base_model = champion.training_weight_path or champion.weights_path
    LOGGER.info("Continuing from champion weight %s", base_model)

    # The same resolution a cycle performs, against the same sources, plus
    # the job's own data.yaml -- the order its labels were written against.
    model_dir = paths.production_model_dir(args.product, args.area)
    class_schema = resolve_class_schema(
        [
            read_champion_class_schema(champion),
            schema_from_station_config(model_dir),
            read_job_class_schema(job_raw),
        ],
        context=f"{args.product}/{args.area}",
    )
    LOGGER.info("Class contract: %s", class_schema.describe())

    store = DatasetVersionStore(
        scratch / "datasets", product=args.product, area=args.area
    )
    version = store.create(
        samples,
        source=f"operator_handoff:{job_raw.parents[3].name}",
        label_source="human",
        class_schema=class_schema,
        description="Smoke run of the real training interface",
    )
    problems = store.verify(version.version)
    if problems:
        LOGGER.error("Dataset version %s failed verify: %s", version.version, problems)
        return 1
    LOGGER.info("Built %s (content_id %s)", version.version, version.content_id)

    result = train_candidate(
        dataset_version=version,
        base_config=load_config(str(PROJECT_ROOT / "configs" / "default_pipeline.yaml")),
        candidate_dir=scratch / "candidates" / version.version,
        work_dir=scratch / "work",
        model_version=f"{args.product}_{args.area}_smoke_candidate",
        base_model=str(base_model),
        class_schema=class_schema,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        logger=LOGGER,
    )

    LOGGER.info("--- challenger produced ---")
    for key, value in result.to_dict().items():
        LOGGER.info("  %s: %s", key, value)
    if not result.trained_this_run:
        LOGGER.warning(
            "No training ran: the skip cache served this from an earlier "
            "attempt. Pass --fresh to force a real run."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
