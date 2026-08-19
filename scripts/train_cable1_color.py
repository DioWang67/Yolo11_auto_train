#!/usr/bin/env python
"""Train the Cable1 wire-color YOLO detector with color-preserving settings.

Root cause of the production failures (orange<->red confusion, wires dropped
below conf): the shipped model was yolo11n trained for only 5 epochs with
aggressive color augmentation (hsv_s=0.7, hsv_v=0.4, erasing=0.4), which
actively teaches the network to ignore color. This script fixes that:

* trains long enough (default 120 epochs, early-stopped by patience),
* disables hue shift entirely (hsv_h=0) and keeps S/V jitter minimal so
  orange and red stay distinguishable,
* disables random erasing and rotation that mangle thin wires.

Runs on GPU automatically when available (falls back to CPU). On a GPU box
this finishes in ~10-20 min; on CPU it takes hours.

Usage (from the Yolo11_auto_train repo root):
    python scripts/train_cable1_color.py
    python scripts/train_cable1_color.py --model yolo11s.pt --epochs 150 --batch 16

The best weights are exported to ONNX next to best.pt so the inference repo
can consume them directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "Cable1" / "split" / "data.yaml"
DEFAULT_MODEL = REPO_ROOT / "models" / "yolo11n.pt"


def make_portable_data_yaml(data_yaml: Path) -> Path:
    """Rewrite the dataset YAML with an OS-correct absolute ``path``.

    The committed data.yaml hard-codes a Windows ``path:`` which breaks on a
    Linux GPU box. We resolve ``path`` to the directory that actually contains
    the YAML so the same file works on any OS / machine.
    """
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    cfg["path"] = str(data_yaml.parent.resolve())
    out = data_yaml.parent / "data.portable.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[train] portable dataset yaml -> {out} (path={cfg['path']})")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="dataset YAML")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="base weights")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16, help="-1 for auto (GPU)")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--project", default=str(REPO_ROOT / "runs" / "Cable1"))
    parser.add_argument("--name", default="train_color_fixed")
    parser.add_argument(
        "--device",
        default=None,
        help="cuda/cpu/0; default auto-detect",
    )
    parser.add_argument(
        "--allow-missing-onnx",
        action="store_true",
        help="return success with PT only when ONNX export is unavailable",
    )
    return parser


def resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def sha256_file(path: Path) -> str:
    """Return the artifact identity stored in the export contract."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_export_contract(run_dir: Path, best: Path, exported: Path) -> Path:
    """Atomically bind the exported ONNX bytes to their source PT checkpoint."""
    resolved_run_dir = run_dir.resolve()
    resolved_best = best.resolve()
    resolved_exported = exported.resolve()
    if not resolved_best.is_relative_to(resolved_run_dir):
        raise RuntimeError("best.pt is outside the current training run")
    if not resolved_exported.is_relative_to(resolved_run_dir):
        raise RuntimeError("exported ONNX is outside the current training run")
    if resolved_best.suffix.lower() != ".pt" or not resolved_best.is_file():
        raise RuntimeError("export source must be an existing PT checkpoint")
    if resolved_exported.suffix.lower() != ".onnx" or not resolved_exported.is_file():
        raise RuntimeError("export result must be an existing ONNX model")

    payload = {
        "schema_version": 1,
        "runtime_format": "onnx",
        "runtime_file": resolved_exported.relative_to(resolved_run_dir).as_posix(),
        "runtime_sha256": sha256_file(resolved_exported),
        "training_weight_file": resolved_best.relative_to(resolved_run_dir).as_posix(),
        "training_weight_sha256": sha256_file(resolved_best),
    }
    contract_path = resolved_run_dir / "runtime_export_manifest.json"
    temporary = contract_path.with_name(f".{contract_path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(contract_path)
    finally:
        temporary.unlink(missing_ok=True)
    return contract_path


def main() -> int:
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)
    print(f"[train] device={device} (cuda_available={torch.cuda.is_available()})")
    print(f"[train] data={args.data}")
    print(f"[train] base model={args.model}")

    data_yaml = make_portable_data_yaml(Path(args.data))

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        # --- Color-preserving augmentation (the whole point) ---
        hsv_h=0.0,   # NO hue shift: hue is the orange/red signal
        hsv_s=0.2,   # gentle saturation jitter only
        hsv_v=0.2,   # gentle brightness jitter only
        erasing=0.0,  # do not erase parts of thin wires
        degrees=0.0,  # connector is axis-aligned; rotation hurts
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        close_mosaic=15,  # turn mosaic off for the final epochs
        mixup=0.0,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"[train] best weights: {best}")
    if not best.exists():
        print("[train] WARNING: best.pt not found after training")
        return 1
    # Delete an earlier export before starting.  The fixed run directory is
    # intentionally reused, so leaving an old ONNX here could silently pair a
    # new PT checkpoint with a stale runtime.
    expected_onnx = best.with_suffix(".onnx")
    expected_onnx.unlink(missing_ok=True)
    contract_path = Path(results.save_dir) / "runtime_export_manifest.json"
    contract_path.unlink(missing_ok=True)
    try:
        exported = Path(
            str(YOLO(str(best)).export(format="onnx", imgsz=args.imgsz, opset=12))
        )
        contract = write_export_contract(Path(results.save_dir), best, exported)
        print(f"[train] exported ONNX: {exported}")
        print(f"[train] export contract: {contract}")
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
        print(
            f"[train] ERROR: ONNX export/contract failed ({exc}). "
            "best.pt is ready, but it was not marked deployable."
        )
        if not args.allow_missing_onnx:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
