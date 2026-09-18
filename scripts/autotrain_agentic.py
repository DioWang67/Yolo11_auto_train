#!/usr/bin/env python3
"""Train from hand-drawn labels, and let the vision model steer the retries.

A person labels the frames; this runs everything after. Between rounds the
vision model reads the measurements and picks the next step, which is the
one job on this station it is measured to be good at --- naming and counting
were right 276 times out of 276 on the same frames where the boxes it drew
scored mean IoU 0.35. It judges; it does not draw, and it does not act.

Every round is written to disk before the next begins. A workflow whose
steps are chosen by something non-deterministic cannot be re-derived by
running it again, so the record of what was decided and what it was decided
on is the only account that will exist.

It stops at a trained candidate and a decision log. It does not deploy:
``assert_no_forbidden_tasks`` refuses that a layer down, and a station new
enough to need this has no golden set, so nothing here could tell a good
model from a bad one well enough to publish it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picture_tool.autotrain.class_schema import ClassSchema  # noqa: E402
from picture_tool.autotrain.dataset_versions import (  # noqa: E402
    DatasetVersionStore,
    LabelledSample,
)
from picture_tool.autotrain.paths import AutoTrainPaths  # noqa: E402
from picture_tool.autotrain.trainer import train_candidate  # noqa: E402
from picture_tool.bootstrap import acceptance  # noqa: E402
from picture_tool.bootstrap import dispatch  # noqa: E402
from picture_tool.bootstrap import vision_client as vc  # noqa: E402
from picture_tool.config_loader import load_config  # noqa: E402

LOGGER = logging.getLogger("autotrain_agentic")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw", type=Path, required=True,
                   help="Directory with images/ and labels/ that a person "
                        "drew. This is the only labelling in the workflow.")
    p.add_argument("--scratch", type=Path,
                   default=PROJECT_ROOT / "runs" / "agentic")
    p.add_argument("--product", default="Cable1")
    p.add_argument("--area", default="A")
    p.add_argument("--classes", default="Black,Green,Orange,Red,Yellow")
    p.add_argument("--expect", default="Black=2,Green=1,Orange=1,Red=1,Yellow=1")
    p.add_argument("--base-model", default="yolo11n.pt",
                   help="A cold start has no champion to continue from.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-rounds", type=int, default=3,
                   help="Hard cap. The decider is non-deterministic, so the "
                        "loop needs an end that does not depend on it "
                        "choosing one.")
    p.add_argument("--acceptance", type=Path, default=None,
                   help="The station's acceptance set: images a person has "
                        "put an OK/NG on. Without it the only measurement "
                        "available is a split of the training frames, which "
                        "on this station read 30 points higher than reality.")
    p.add_argument("--conf", type=float, default=0.4,
                   help="Detection confidence for the acceptance pass.")
    p.add_argument("--url-env", default="QWEN_URL")
    p.add_argument("--key-env", default="")
    p.add_argument("--model", default="Qwen3.8-27B-GGUF")
    p.add_argument("--check-data-only", action="store_true",
                   help="Check the labels and stop: trains nothing, loads "
                        "no model, calls nothing.")
    p.add_argument("--no-decide", action="store_true",
                   help="Train and measure, but do not ask for a decision. "
                        "The numbers are still written down.")
    p.add_argument("--dry-run", action="store_true",
                   help="Train and measure, but decide nothing and call "
                        "nothing. Shows what the model would be shown.")
    return p.parse_args(argv)


def collect_samples(root: Path) -> list[LabelledSample]:
    """Image/label pairs a person drew. Unlabelled frames are not data."""
    images, labels = root / "images", root / "labels"
    if not images.is_dir() or not labels.is_dir():
        raise SystemExit(f"{root} needs images/ and labels/ side by side.")
    samples: list[LabelledSample] = []
    for image in sorted(images.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = labels / f"{image.stem}.txt"
        if not label.is_file():
            LOGGER.warning("  no label for %s; skipped", image.name)
            continue
        if not any(line.strip() for line in
                   label.read_text(encoding="utf-8").splitlines()):
            LOGGER.warning("  %s has an empty label; skipped", image.name)
            continue
        samples.append(LabelledSample(
            sample_id=image.stem, image_path=image, label_path=label,
            origin="human",
        ))
    if not samples:
        raise SystemExit(f"No labelled frames under {root}.")
    return samples


def acceptance_metrics(
    weights: Path, root: Path, expected: dict[str, int], *,
    training_root: Path, conf: float,
) -> dict[str, Any]:
    """Run the candidate over boards a person judged, and score it.

    Detections are sorted left to right because that is the order every
    downstream check on this station reads them in; the scoring itself only
    counts them, but handing on an arbitrary order would make the recorded
    examples useless to whoever reads the report.
    """
    # Imported here, and onnxruntime first: importing anything under core
    # before it leaves onnxruntime unable to load its extension DLL on this
    # workstation.
    import onnxruntime  # noqa: F401
    from ultralytics import YOLO

    boards = acceptance.read_boards(root, training_root=training_root)
    LOGGER.info("  %d confirmed board(s) from %s", len(boards), root)
    # Refused rather than skipped. A board with no detections on file is
    # scored by acceptance.score as one the candidate found nothing on, so
    # skipping a missing image quietly adds it to would_false_reject --- the
    # number this whole run is read by --- and the count would then be
    # describing the filesystem rather than the model.
    absent = [b.sample_id for b in boards if not b.image.is_file()]
    if absent:
        raise SystemExit(
            f"{len(absent)} confirmed board(s) name an image that is not "
            f"there, e.g. {', '.join(absent[:3])}. Scored as they are, each "
            "would be counted as a false reject. Repair the acceptance set "
            "rather than scoring around it."
        )
    model = YOLO(str(weights))
    found: dict[str, list[str]] = {}
    for index, board in enumerate(boards, 1):
        result = model.predict(str(board.image), conf=conf, verbose=False)[0]
        names = result.names
        seen = sorted(
            (float(box.xywh[0][0]), names[int(box.cls)]) for box in result.boxes
        )
        found[board.sample_id] = [name for _, name in seen]
        if index % 50 == 0:
            LOGGER.info("    %d/%d", index, len(boards))
    return acceptance.score(boards, found, expected)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    names = tuple(n.strip() for n in args.classes.split(",") if n.strip())
    schema = ClassSchema(names=names, source="station_contract")
    scratch = args.scratch.expanduser().resolve()

    paths = AutoTrainPaths.discover()
    # This path may not write into the inference tree, and the guard runs
    # before anything is created rather than being assumed.
    paths.assert_not_production(scratch)

    samples = collect_samples(args.raw)
    LOGGER.info("%d hand-labelled frame(s) from %s", len(samples), args.raw)
    LOGGER.info("class contract %s  hash %s",
                list(names), schema.schema_hash[:12])
    if args.check_data_only:
        LOGGER.info("Labels look usable. Nothing was trained.")
        return 0

    store = DatasetVersionStore(scratch / "datasets",
                                product=args.product, area=args.area)
    version = store.create(
        samples,
        source=str(args.raw),
        # The one thing about this dataset worth being able to look up
        # later: a person drew every box in it, and no model proposed any.
        label_source="human",
        class_schema=schema,
        description=f"hand-drawn seed for {args.product}/{args.area}",
    )
    LOGGER.info("dataset version %s", version.version)

    client = None
    if not args.dry_run:
        cfg = vc.openai_compatible_profile(
            model=args.model, url_env=args.url_env, key_env=args.key_env,
            requires_credential=bool(args.key_env), max_tokens=2048,
            max_retries=1, timeout_seconds=180.0,
        )
        client = vc.HttpVisionLLMClient(cfg)

    inventory = ", ".join(
        part.strip() for part in args.expect.split(",") if part.strip()
    )
    expected: dict[str, int] = {}
    for part in args.expect.split(","):
        if "=" in part:
            key, value = part.split("=", 1)
            expected[key.strip()] = int(value)
    base_config = load_config(str(PROJECT_ROOT / "configs" / "default_pipeline.yaml"))
    log_path = scratch / "decisions.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dispatch.Decision] = []
    epochs = args.epochs
    outcome = "cap reached"
    for round_index in range(1, args.max_rounds + 1):
        LOGGER.info("--- round %d/%d, %d epoch(s) ---",
                    round_index, args.max_rounds, epochs)
        result = train_candidate(
            dataset_version=version,
            base_config=base_config,
            candidate_dir=scratch / "candidates" / f"round{round_index}",
            work_dir=scratch / "work",
            model_version=f"{args.product}_{args.area}_agentic_r{round_index}",
            base_model=args.base_model,
            class_schema=schema,
            epochs=epochs, imgsz=args.imgsz, batch=args.batch,
            device=args.device, logger=LOGGER,
        )
        # train_candidate's own metrics are provenance --- dataset ids,
        # hashes, when it ran --- and carry nothing about how well the model
        # did. Asking for a judgement on those would be asking a judgement
        # on no evidence, so the accuracy numbers are read from what the
        # training actually measured.
        metrics = dispatch.metrics_from_results_csv(
            Path(result.run_dir) / "results.csv")
        metrics["epochs"] = epochs
        metrics["labelled_images"] = len(samples)
        metrics["trained_this_run"] = result.trained_this_run
        if not metrics.get("metrics_available"):
            LOGGER.error("  no accuracy numbers: %s",
                         metrics.get("detail", "unknown"))
            outcome = "no metrics to judge"
            break
        # Renamed before anything else sees them. These were measured on a
        # split of the same frames the model learned from, and on this
        # station they read about 30 points higher than the same weights
        # score on boards they have not seen. Leaving them under plain
        # names invites a decision to be made on the flattering half.
        metrics = {f"own_split_{k}" if k in dispatch.RESULT_COLUMNS.values()
                   else k: v for k, v in metrics.items()}
        if args.acceptance:
            LOGGER.info("  scoring against the acceptance set")
            metrics.update(acceptance_metrics(
                Path(result.weights_path), args.acceptance, expected,
                training_root=paths.workspace.training_data,
                conf=args.conf,
            ))
        LOGGER.info("  %s", json.dumps(metrics, sort_keys=True))

        if args.dry_run:
            print(dispatch.render_prompt(
                metrics, product=args.product, area=args.area,
                inventory=inventory, images=len(samples),
                actions=dispatch.DEFAULT_ACTIONS, history=history,
            ))
            return 0

        if args.no_decide:
            # The numbers were still measured and are still on their way to
            # summary.json; what was skipped is asking what to do about them.
            outcome = "measured, no decision requested"
            break

        try:
            decision = dispatch.decide_next(
                client, metrics,
                product=args.product, area=args.area, inventory=inventory,
                images=len(samples), history=history,
            )
        except dispatch.DispatchError as exc:
            # A decider that did not decide must not be papered over with a
            # default; the run stops holding whatever it has.
            LOGGER.error("  no usable decision: %s", exc)
            outcome = f"undecided: {exc}"
            break

        history.append(decision)
        record = {
            "round": round_index,
            "weights": str(result.weights_path),
            "dataset_version": result.dataset_version,
            **decision.to_dict(),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        LOGGER.info("  decided %s: %s", decision.action, decision.reason)

        if decision.action == dispatch.TRAIN_LONGER:
            epochs *= 2
            continue
        outcome = decision.action
        break

    summary: dict[str, Any] = {
        "outcome": outcome,
        "rounds": len(history),
        "labelled_images": len(samples),
        "dataset_version": version.version,
        "decisions": [d.to_dict() for d in history],
        "decisions_log": str(log_path),
        # True of the labels, which a person drew.
        "labels_are_ground_truth": True,
        # Whether anything here judged the model on boards it had not
        # learned from. Without it the reported accuracy is the model
        # marking its own homework, and on this station that read 30
        # points high.
        "scored_on_independent_boards": bool(args.acceptance),
        "acceptance_set": str(args.acceptance) if args.acceptance else None,
    }
    (scratch / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
