#!/usr/bin/env python3
"""Run one complete seven-step cycle in a sandbox fed by a public dataset.

Every other real-ultralytics script covers one slice of the cycle ---
``autotrain_smoke.py`` trains, ``autotrain_group_smoke.py`` evaluates --- and
the station's own data cannot drive the rest: nothing in the review pack is
labelled, so a real cycle stops at the dataset step and never reaches
evaluate, decide or report. A public, human-labelled detection dataset fills
both gaps at once. Its labels stand in for the annotator, and a split the
cycle never trains on stands in for the golden set.

What the sandbox fabricates, and why each piece is faithful:

* **Champion** --- trained here with ultralytics on a small slice, then placed
  in a station directory with a ``deployment_manifest.yaml`` in the shape
  ``deploy`` writes (bare filenames, which the reader prefixes with
  ``weights/``). Its class contract comes from its own checkpoint.
* **Production records** --- the champion really runs over a second slice and
  each result is written as a ``*_config_snapshot.json`` in the tree layout
  inference uses, so the low-confidence selector sees real confidences.
* **Labels** --- copied from the public dataset into the exported labelling
  request, then imported through the same all-or-nothing validator the
  operator flow uses. Nothing is marked verified behind the pool's back.
* **Golden** --- the dataset's test split, registered and pinned by manifest
  hash. Disjoint by *filename* only: african-wildlife carries 12 byte-identical
  images across train and test under different names, so when one of them is
  selected the golden set comes back CONTAMINATED and the cycle is rejected.
  That is the contamination check working, not a harness fault.

Nothing here touches the real workspace: paths are built directly (the CLI
would discover the real ``workspace.yaml``) and the process runs with the
sandbox as its working directory, because the pipeline writes its lint
output, log file and MLflow store relative to the CWD.

Download the dataset first (100 MB, AGPL-3.0, no account needed)::

    https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip

then::

    python scripts/autotrain_public_e2e.py --dataset D:/tmp/datasets \\
        --sandbox D:/tmp/autotrain_public_sandbox --fresh > e2e.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picture_tool.autotrain import golden  # noqa: E402
from picture_tool.autotrain.candidate_pool import CandidatePool  # noqa: E402
from picture_tool.autotrain.class_schema import ClassSchema  # noqa: E402
from picture_tool.autotrain.config import AutoTrainConfig  # noqa: E402
from picture_tool.autotrain.labeling import export_request, import_request  # noqa: E402
from picture_tool.autotrain.orchestrator import run_training_cycle  # noqa: E402
from picture_tool.autotrain.paths import AutoTrainPaths  # noqa: E402
from picture_tool.workspace_paths import WorkspacePaths  # noqa: E402

PRODUCT = "PublicDemo"
AREA = "A"
CHAMPION_VERSION = "PublicDemo_A_v0.1.0"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

LOGGER = logging.getLogger("autotrain_public_e2e")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Extracted YOLO dataset: images/{train,val,test}, labels/..., "
        "and a data yaml whose 'names' is the class list.",
    )
    parser.add_argument("--sandbox", type=Path, required=True)
    parser.add_argument("--fresh", action="store_true", help="Delete the sandbox first.")
    parser.add_argument("--champion-images", type=int, default=120)
    parser.add_argument("--production-images", type=int, default=300)
    parser.add_argument("--champion-epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3, help="Challenger epochs.")
    parser.add_argument(
        "--augmentations",
        type=int,
        default=3,
        help="Variants per image, beside the originals the challenger always "
        "keeps. The shipped config makes 20, which on CPU turns 230 labelled "
        "images into ~4 minutes per epoch; this run tests the cycle's "
        "mechanics, not the recipe. 0 trains on the originals alone.",
    )
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument(
        "--base-weights",
        type=Path,
        default=PROJECT_ROOT / "models" / "yolo11n.pt",
        help="Pretrained weights the sandbox champion starts from.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Sandbox layout


def build_workspace(sandbox: Path, augmentations: int) -> AutoTrainPaths:
    """The same paired layout the orchestrator tests build, under ``sandbox``."""
    root = sandbox / "workspace"
    training = root / "Yolo11_auto_train"
    inference = root / "yolo11_inference"
    for directory in (
        training / "data",
        training / "configs",
        inference / "models",
        root / "station_data" / "yolo11_inference",
        root / "Result",
        root / "release_artifacts" / "yolo11_inference",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    # The pipeline config a station without its own <product>_pipeline.yaml
    # would actually get, with only the variant count changed.
    pipeline = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "default_pipeline.yaml").read_text(encoding="utf-8")
    )
    pipeline["yolo_augmentation"]["augmentation"]["num_images"] = augmentations
    (training / "configs" / "default_pipeline.yaml").write_text(
        yaml.safe_dump(pipeline, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    workspace = WorkspacePaths(
        workspace_root=root.resolve(),
        training_project=training.resolve(),
        inference_project=inference.resolve(),
        training_data=(training / "data").resolve(),
        inference_models=(inference / "models").resolve(),
        station_data=(root / "station_data" / "yolo11_inference").resolve(),
        inference_results=(root / "Result").resolve(),
        inference_artifacts=(root / "release_artifacts" / "yolo11_inference").resolve(),
        manifest_path=None,
    )
    return AutoTrainPaths.from_workspace(workspace)


def remove_tree(root: Path) -> None:
    """Delete a previous sandbox, including its read-only dataset versions.

    Dataset versions are made read-only on purpose, and on Windows ``rmtree``
    cannot unlink a read-only file.
    """
    import stat

    def clear_readonly_and_retry(function, path, _exc_info) -> None:
        os.chmod(path, stat.S_IWRITE)
        function(path)

    shutil.rmtree(root, onerror=clear_readonly_and_retry)


def assert_outside_real_workspace(sandbox: Path) -> None:
    real_workspace = PROJECT_ROOT.parent.resolve()
    resolved = sandbox.resolve()
    if resolved == real_workspace or resolved.is_relative_to(real_workspace):
        raise SystemExit(
            f"--sandbox {resolved} is inside the real workspace {real_workspace}; "
            "pick a directory outside it."
        )


# ---------------------------------------------------------------------------
# Public dataset


def read_class_names(dataset: Path) -> list[str]:
    for descriptor in sorted(dataset.glob("*.yaml")):
        payload = yaml.safe_load(descriptor.read_text(encoding="utf-8")) or {}
        names = payload.get("names")
        if isinstance(names, dict):
            return [str(names[key]) for key in sorted(names)]
        if isinstance(names, list):
            return [str(name) for name in names]
    raise SystemExit(f"No data yaml with 'names' found in {dataset}")


def list_pairs(dataset: Path, split: str) -> list[tuple[Path, Path]]:
    """Images of one split with their label file; unlabelled images skipped."""
    pairs = []
    for image in sorted((dataset / "images" / split).iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = dataset / "labels" / split / f"{image.stem}.txt"
        if label.is_file():
            pairs.append((image, label))
    return pairs


def write_yolo_dir(
    pairs: list[tuple[Path, Path]], root: Path, split: str, prefix: str
) -> dict[str, Path]:
    """Copy pairs under production-style names; returns new stem -> label."""
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    renamed: dict[str, Path] = {}
    for index, (image, label) in enumerate(pairs):
        stem = f"{prefix}_{index:05d}"
        shutil.copy2(image, root / "images" / split / f"{stem}{image.suffix.lower()}")
        shutil.copy2(label, root / "labels" / split / f"{stem}.txt")
        renamed[stem] = label
    return renamed


def write_data_yaml(root: Path, names: list[str], *, train: str, val: str) -> Path:
    path = root / "data.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(root.resolve()),
                "train": train,
                "val": val,
                "names": dict(enumerate(names)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Champion


def train_champion(
    pairs: list[tuple[Path, Path]],
    val_pairs: list[tuple[Path, Path]],
    names: list[str],
    sandbox: Path,
    args: argparse.Namespace,
) -> Path:
    from ultralytics import YOLO

    root = sandbox / "champion_data"
    write_yolo_dir(pairs, root, "train", "champ")
    write_yolo_dir(val_pairs, root, "val", "champval")
    data_yaml = write_data_yaml(root, names, train="images/train", val="images/val")
    model = YOLO(str(args.base_weights))
    model.train(
        data=str(data_yaml),
        epochs=args.champion_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        project=str(sandbox / "champion_run"),
        name="train",
        exist_ok=True,
        seed=args.seed,
        workers=0,
        plots=False,
        verbose=False,
    )
    best = sandbox / "champion_run" / "train" / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"Champion training produced no {best}")
    return best


def install_champion(paths: AutoTrainPaths, weights: Path) -> Path:
    """Lay the champion out the way ``deploy`` leaves a station directory."""
    import hashlib

    model_dir = paths.production_model_dir(PRODUCT, AREA)
    (model_dir / "weights").mkdir(parents=True, exist_ok=True)
    target = model_dir / "weights" / "best.pt"
    shutil.copy2(weights, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (model_dir / "deployment_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "deployed_version": CHAMPION_VERSION,
                # Bare filenames: the form deploy.py actually writes.
                "deployed_weight_file": "best.pt",
                "training_weight_file": "best.pt",
                "weight_sha256": digest,
                "dataset_id": "",
                "evaluation_metrics": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return target


# ---------------------------------------------------------------------------
# Production records


def write_production_records(
    pairs: list[tuple[Path, Path]],
    champion: Path,
    names: list[str],
    paths: AutoTrainPaths,
    args: argparse.Namespace,
) -> dict[str, Path]:
    """Run the champion over the stream and write inference-shaped snapshots.

    Returns the ground-truth label for each written image, keyed by the image
    path the snapshot records --- the annotator's answer sheet.
    """
    from ultralytics import YOLO

    model = YOLO(str(champion))
    day = datetime.now().strftime("%Y%m%d")
    station = paths.production_results_root() / day / PRODUCT / AREA
    answers: dict[str, Path] = {}
    counts = {"PASS": 0, "FAIL": 0}
    for index, (image, label) in enumerate(pairs):
        inspection_id = f"yolo_{PRODUCT}_{AREA}_{index:06d}"
        prediction = model.predict(
            str(image), imgsz=args.imgsz, conf=0.1, device="cpu", verbose=False
        )[0]
        boxes = prediction.boxes
        detections: list[dict[str, Any]] = []
        if boxes is not None:
            detections = [
                {
                    "class": names[int(cls)],
                    "confidence": round(float(conf), 4),
                    "bbox": [round(float(v), 1) for v in box],
                }
                for box, conf, cls in zip(
                    boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
                )
            ]
        status = "PASS" if any(d["confidence"] >= 0.4 for d in detections) else "FAIL"
        counts[status] += 1
        status_dir = station / status
        original = status_dir / "original" / f"{inspection_id}{image.suffix.lower()}"
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, original)
        metadata = status_dir / "metadata" / "yolo"
        metadata.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "schema_version": 2,
            "inspection_id": inspection_id,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "detector": "yolo",
            "product": PRODUCT,
            "area": AREA,
            "equipment": {"machine_id": "sandbox", "station": AREA, "camera_id": "cam0"},
            "fail_reasons": [] if status == "PASS" else ["NO_CONFIDENT_DETECTION"],
            "model_info": {
                "weights": str(champion),
                "model_version": CHAMPION_VERSION,
                "conf_thres": 0.4,
                "class_names": list(names),
            },
            "inference_time": 0.0,
            "detections": detections,
            "artifacts": {
                "original_path": str(original),
                "preprocessed_path": "",
                "annotated_path": "",
            },
            "config_hash": "sandbox",
            "config": {"exposure_time": "0", "gain": "0", "light_brightness": 0},
        }
        (metadata / f"{inspection_id}_config_snapshot.json").write_text(
            json.dumps(snapshot), encoding="utf-8"
        )
        answers[str(original.resolve())] = label
    LOGGER.info("Wrote %s production record(s): %s", len(pairs), counts)
    return answers


# ---------------------------------------------------------------------------
# Config, golden, labelling


def write_config(
    sandbox: Path, args: argparse.Namespace, golden_root: Path, manifest_sha256: str
) -> AutoTrainConfig:
    path = sandbox / "autonomous_training.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "autonomous_training": {
                    "enabled": True,
                    "station": {"product": PRODUCT, "area": AREA},
                    "collector": {
                        "lookback_days": 1,
                        "max_records": 5000,
                        "compute_image_quality": True,
                    },
                    "selectors": {
                        "low_confidence": {
                            "enabled": True,
                            "conf_below": 0.55,
                            "max_samples": 200,
                        },
                        "review_correction": {"enabled": True, "max_samples": 500},
                        "random_sample": {
                            "enabled": True,
                            "fraction": 0.3,
                            "max_samples": 100,
                            "seed": args.seed,
                        },
                    },
                    "golden": {
                        "dataset_path": str(golden_root),
                        "manifest_sha256": manifest_sha256,
                    },
                    "training": {
                        "base_model": "champion",
                        "epochs": args.epochs,
                        "imgsz": args.imgsz,
                        "device": "cpu",
                        "batch": args.batch,
                    },
                    # Shipped defaults, unchanged: the point is to see which
                    # way the real gate goes, not to tune it into passing.
                    "promotion": {
                        "min_map50_delta": 0.0,
                        "max_overall_regression": 0.02,
                        "critical_classes": [],
                        "max_critical_class_recall_drop": 0.005,
                        "max_false_negatives": 0,
                        "require_golden_pass": True,
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return AutoTrainConfig.load(path)


def build_golden(
    pairs: list[tuple[Path, Path]], names: list[str], sandbox: Path
) -> tuple[Path, str]:
    root = sandbox / "golden" / "public_test"
    # african-wildlife's test split holds byte-identical images under
    # different names, three of them with contradicting labels. Registration
    # refuses those, so every repeated image is left out of the yardstick.
    import hashlib
    from collections import Counter

    digests = [hashlib.sha256(image.read_bytes()).hexdigest() for image, _ in pairs]
    repeated = {digest for digest, count in Counter(digests).items() if count > 1}
    pairs = [pair for pair, digest in zip(pairs, digests) if digest not in repeated]
    write_yolo_dir(pairs, root, "val", "golden")
    # The evaluator measures the golden set on split 'val'.
    write_data_yaml(root, names, train="images/val", val="images/val")
    dataset = golden.register(
        root,
        registered_by="autotrain_public_e2e (public test split)",
        class_schema=ClassSchema(names=tuple(names), source="public-dataset"),
        description="Public dataset test split; disjoint from the train split by filename.",
    )
    LOGGER.info(
        "Registered golden set: %s image(s), manifest %s",
        dataset.image_count,
        dataset.manifest_sha256[:12],
    )
    return root, dataset.manifest_sha256


def annotate_from_ground_truth(
    paths: AutoTrainPaths, names: list[str], answers: dict[str, Path]
) -> dict[str, Any]:
    """Act as the annotator: fill an exported request from the public labels."""
    pool = CandidatePool(paths.pool_dir(PRODUCT, AREA))
    request = export_request(
        pool,
        paths.labeling_root,
        product=PRODUCT,
        area=AREA,
        class_names=names,
    )
    filled = missing = 0
    for sample in pool.load():
        if sample.sample_id not in request.sample_ids:
            continue
        truth = answers.get(str(Path(sample.source_image_path).resolve()))
        if truth is None:
            missing += 1
            continue
        shutil.copy2(truth, request.labels_dir / f"{sample.sample_id}.txt")
        filled += 1
    result = import_request(pool, request.root)
    summary = {"request": str(request.root), "filled": filled, "unmatched": missing}
    summary.update(result.summary())
    LOGGER.info("Labelling: %s", summary)
    return summary


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sandbox = args.sandbox.resolve()
    assert_outside_real_workspace(sandbox)
    if args.fresh and sandbox.exists():
        remove_tree(sandbox)
    sandbox.mkdir(parents=True, exist_ok=True)
    os.chdir(sandbox)

    dataset = args.dataset.resolve()
    names = read_class_names(dataset)
    train_pairs = list_pairs(dataset, "train")
    rng = random.Random(args.seed)
    rng.shuffle(train_pairs)
    needed = args.champion_images + args.production_images
    if len(train_pairs) < needed:
        raise SystemExit(f"Need {needed} labelled train images, found {len(train_pairs)}")
    champion_pairs = train_pairs[: args.champion_images]
    production_pairs = train_pairs[args.champion_images : needed]
    val_pairs = list_pairs(dataset, "val")[:60]
    golden_pairs = list_pairs(dataset, "test")
    LOGGER.info(
        "Classes %s; champion %s, production %s, champion-val %s, golden %s",
        names,
        len(champion_pairs),
        len(production_pairs),
        len(val_pairs),
        len(golden_pairs),
    )

    if args.augmentations < 0:
        raise SystemExit("--augmentations cannot be negative")
    paths = build_workspace(sandbox, args.augmentations)
    champion_weights = train_champion(champion_pairs, val_pairs, names, sandbox, args)
    champion = install_champion(paths, champion_weights)
    answers = write_production_records(production_pairs, champion, names, paths, args)
    golden_root, manifest_sha256 = build_golden(golden_pairs, names, sandbox)
    config = write_config(sandbox, args, golden_root, manifest_sha256)

    cycle_id = datetime.now(timezone.utc).strftime("cycle_%Y%m%dT%H%M%SZ")
    first = run_training_cycle(config, paths, cycle_id=cycle_id)
    LOGGER.info("First pass: %s (%s)", first.status, first.blocked_reason)

    labelling = annotate_from_ground_truth(paths, names, answers)

    second = run_training_cycle(config, paths, cycle_id=cycle_id)
    decision = second.decision.decision if second.decision else None
    summary = {
        "cycle_id": cycle_id,
        "first_pass": first.status,
        "first_pass_blocked_reason": first.blocked_reason,
        "labelling": labelling,
        "second_pass": second.status,
        "decision": decision,
        "report": str(second.report_path),
    }
    (sandbox / "e2e_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("=" * 72)
    print(second.report_text)
    print("=" * 72)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
