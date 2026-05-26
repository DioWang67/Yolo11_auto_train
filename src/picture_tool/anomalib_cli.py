from __future__ import annotations

import argparse
import logging
from pathlib import Path

from picture_tool.train.anomalib_trainer import (
    deploy_anomalib_run,
    supported_anomalib_models,
    train_anomalib_folder,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the lightweight Anomalib CLI parser.

    Returns:
        Configured argparse parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m picture_tool.anomalib_cli",
        description="Anomalib training/deployment helpers that do not require Typer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Model choices:\n"
            "  padim     Fast CPU-friendly baseline; good first choice for PCBA smoke runs.\n"
            "  patchcore Higher-quality memory-bank method; heavier and better for validated runs.\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "models",
        help="Show supported Anomalib model choices and trade-offs.",
    )
    train_folder = subparsers.add_parser(
        "train-folder",
        help="Train Anomalib from a product/area folder with layout detection.",
        description=(
            "Train from a simple folder. The command looks for split/train/images, "
            "train/good, anomalib/train/good, good, or direct images."
        ),
    )
    train_folder.add_argument("--input", required=True, type=Path, dest="input_dir")
    train_folder.add_argument("--product", required=True)
    train_folder.add_argument("--area", required=True)
    train_folder.add_argument("--project", type=Path, default=None)
    train_folder.add_argument("--model", default="padim", choices=["padim", "patchcore"])
    train_folder.add_argument("--image-size", type=int, default=256)
    train_folder.add_argument("--batch-size", type=int, default=8)
    train_folder.add_argument("--max-epochs", type=int, default=1)
    train_folder.add_argument("--accelerator", default="cpu")
    train_folder.add_argument("--devices", default="1")
    train_folder.add_argument("--pre-trained", action="store_true")
    train_folder.add_argument("--require-anomalous-validation", action="store_true")
    train_folder.add_argument("--force", action="store_true", help="Retrain even if a checkpoint exists.")
    train_folder.add_argument("--tmp-dir", type=Path, default=Path("runs/tmp"))
    deploy = subparsers.add_parser(
        "deploy",
        help="Deploy a trained Anomalib run into yolo11_inference models layout.",
    )
    deploy.add_argument("--run", required=True, type=Path, dest="run_dir")
    deploy.add_argument("--inference-root", required=True, type=Path)
    deploy.add_argument("--product", required=True)
    deploy.add_argument("--area", required=True)
    deploy.add_argument("--threshold", type=float, default=0.5)
    deploy.add_argument("--force", action="store_true", help="Overwrite existing deployed files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the lightweight Anomalib CLI.

    Args:
        argv: Optional argument list for tests.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.command == "models":
        _print_model_options()
        return 0

    if args.command == "train-folder":
        try:
            result = train_anomalib_folder(
                input_dir=args.input_dir,
                product=args.product,
                area=args.area,
                project=args.project,
                model=args.model,
                image_size=args.image_size,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                accelerator=args.accelerator,
                devices=_parse_devices(args.devices),
                pre_trained=args.pre_trained,
                require_anomalous_validation=args.require_anomalous_validation,
                force=args.force,
                tmp_dir=args.tmp_dir,
            )
        except (ValueError, RuntimeError, ImportError) as exc:
            parser.exit(1, f"Anomalib training failed: {exc}\n")

        print(f"Run directory: {result.run_dir}")
        print(f"Checkpoint: {result.checkpoint_path or 'not found'}")
        print(f"Report: {result.report_path}")
        print(f"Normal images: {result.normal_image_count}")
        print(f"Abnormal images: {result.abnormal_image_count}")
        if result.baseline_only:
            print("Status: baseline_only=true; threshold is not deployment-grade.")
        else:
            print("Status: validated layout detected.")
        return 0

    if args.command == "deploy":
        try:
            result = deploy_anomalib_run(
                run_dir=args.run_dir,
                inference_root=args.inference_root,
                product=args.product,
                area=args.area,
                threshold=args.threshold,
                force=args.force,
            )
        except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
            parser.exit(1, f"Anomalib deploy failed: {exc}\n")

        print(f"Deploy directory: {result.deploy_dir}")
        print(f"Config: {result.config_path}")
        print(f"Checkpoint: {result.checkpoint_path}")
        print(f"Report: {result.report_path or 'not copied'}")
        print(f"baseline_only: {str(result.baseline_only).lower()}")
        print(f"usable_for_deployment: {str(result.usable_for_deployment).lower()}")
        for warning in result.warnings:
            print(f"warning: {warning}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _parse_devices(value: str) -> str | int:
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def _print_model_options() -> None:
    for option in supported_anomalib_models():
        n_features = (
            "default" if option.default_n_features is None else str(option.default_n_features)
        )
        print(f"{option.name}")
        print(f"  class: {option.class_path}")
        print(f"  summary: {option.summary}")
        print(f"  best_for: {option.best_for}")
        print(f"  trade_off: {option.trade_off}")
        print(
            "  defaults: "
            f"backbone={option.default_backbone}, "
            f"layers={','.join(option.default_layers)}, "
            f"n_features={n_features}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
