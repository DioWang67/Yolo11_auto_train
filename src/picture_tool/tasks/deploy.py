"""Deploy trained artifacts directly into a yolo11_inference models directory.

After ``yolo_train`` completes, this task copies ``config.yaml`` and weights
into ``{inference_models_dir}/{product}/{area}/yolo/`` so that
``ModelCatalog`` can discover them immediately — no manual unzipping or path
editing required.

Versioning
----------
Each deployment produces a **versioned** weights file following the convention
already used by yolo11_inference::

    {product}_{area}_v{major}.{minor}.{patch}_{YYYYMMDD}.pt

Set ``yolo_training.deploy.version`` to an explicit string such as ``"1.2.0"``
or leave it as ``"auto"`` (the default) to have the patch number incremented
automatically based on what already exists in the destination directory.

Enable by setting ``yolo_training.deploy.enabled: true`` in your pipeline
config and pointing ``inference_models_dir`` at your local inference project.
"""

from __future__ import annotations

import datetime
import logging
import re
import shutil
from pathlib import Path
from typing import Any, List, Tuple

import yaml

from picture_tool.pipeline.utils import detect_existing_weights
from picture_tool.tasks.bundle import rewrite_detection_config

# ---------------------------------------------------------------------------
# Version helpers (mirrors yolo11_inference/core/version_utils.py logic so we
# don't create a cross-project import dependency)
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"_v(\d+)\.(\d+)\.(\d+)_")


def _parse_version(filename: str) -> Tuple[int, int, int] | None:
    m = _VERSION_RE.search(Path(filename).name)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def _version_str(v: Tuple[int, int, int]) -> str:
    return f"{v[0]}.{v[1]}.{v[2]}"


def _resolve_version(
    version_cfg: str,
    weights_dest: Path,
    product: str,
    area: str,
) -> Tuple[int, int, int]:
    """Return the version tuple to use for this deployment.

    * Explicit ``version_cfg`` (e.g. ``"2.1.0"``) → parsed and used as-is.
    * ``"auto"`` or empty → scan *weights_dest* for existing versioned files
      with the same ``{product}_{area}_v*`` prefix and bump the patch number.
      Starts at ``1.0.0`` if no prior versions exist.
    """
    prefix = f"{product}_{area}_v"

    if version_cfg and version_cfg.lower() != "auto":
        parts = version_cfg.split(".")
        if len(parts) == 3:
            try:
                return (int(parts[0]), int(parts[1]), int(parts[2]))
            except ValueError:
                pass
        raise ValueError(
            f"deploy.version '{version_cfg}' is not a valid semantic version "
            "(expected 'major.minor.patch' or 'auto')."
        )

    # Auto-increment: find the highest existing patch and bump it
    existing: list[Tuple[int, int, int]] = []
    if weights_dest.exists():
        for f in weights_dest.glob(f"{prefix}*.pt"):
            ver = _parse_version(f.name)
            if ver:
                existing.append(ver)

    if not existing:
        return (1, 0, 0)

    latest = max(existing)
    return (latest[0], latest[1], latest[2] + 1)


# ---------------------------------------------------------------------------
# Main task
# ---------------------------------------------------------------------------

def run_deploy(config: dict, args: Any) -> None:
    """Copy training artifacts to the inference models directory (versioned)."""
    logger = logging.getLogger(__name__)

    ycfg = config.get("yolo_training", {})
    dcfg = ycfg.get("deploy", {})

    if not dcfg.get("enabled", False):
        logger.info("Deploy task disabled (yolo_training.deploy.enabled is false).")
        return

    # ------------------------------------------------------------------
    # Resolve product / area
    # ------------------------------------------------------------------
    product: str = (
        dcfg.get("product")
        or ycfg.get("position_validation", {}).get("product")
        or ycfg.get("name", "project")
    )
    area: str = (
        dcfg.get("area")
        or ycfg.get("position_validation", {}).get("area")
        or "A"
    )

    # ------------------------------------------------------------------
    # Resolve inference models directory
    # ------------------------------------------------------------------
    raw_dir = dcfg.get("inference_models_dir")
    if not raw_dir:
        logger.error("deploy.inference_models_dir is not set — skipping deploy.")
        return

    inference_models_dir = Path(raw_dir).expanduser()
    if not inference_models_dir.is_absolute():
        inference_models_dir = Path.cwd() / inference_models_dir
    inference_models_dir = inference_models_dir.resolve()

    # ------------------------------------------------------------------
    # Locate training run directory
    # ------------------------------------------------------------------
    _, run_dir = detect_existing_weights(config)
    if not run_dir or not run_dir.exists():
        logger.warning("No training run directory found — deploy skipped.")
        return

    # ------------------------------------------------------------------
    # Prepare destination
    # ------------------------------------------------------------------
    dest_dir = inference_models_dir / product / area / "yolo"
    weights_dest = dest_dir / "weights"
    dest_dir.mkdir(parents=True, exist_ok=True)
    weights_dest.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Resolve version and generate versioned filename
    # ------------------------------------------------------------------
    try:
        version = _resolve_version(
            dcfg.get("version", "auto"), weights_dest, product, area
        )
    except ValueError as exc:
        logger.error("Version resolution failed: %s", exc)
        raise

    date_str = datetime.datetime.now().strftime("%Y%m%d")
    versioned_name = f"{product}_{area}_v{_version_str(version)}_{date_str}.pt"
    versioned_path = weights_dest / versioned_name

    logger.info(
        "Deploying %s/%s/yolo → %s  (version %s)",
        product, area, dest_dir, _version_str(version),
    )

    # ------------------------------------------------------------------
    # Write config.yaml — weights path points to versioned file
    # ------------------------------------------------------------------
    det_cfg_path = run_dir / "detection_config.yaml"
    if not det_cfg_path.exists():
        logger.error(
            "detection_config.yaml not found at %s — deploy aborted.", det_cfg_path
        )
        raise FileNotFoundError(det_cfg_path)

    try:
        det_cfg_data = yaml.safe_load(det_cfg_path.read_text(encoding="utf-8")) or {}
        det_cfg_data = rewrite_detection_config(det_cfg_data, product)
        # Point to the versioned filename so ModelManager knows exactly which
        # file to load — full relative path from the inference project root.
        det_cfg_data["weights"] = (
            f"models/{product}/{area}/yolo/weights/{versioned_name}"
        )
        # Fix color_model_path to use full inference-relative path (same as weights)
        color_model = det_cfg_data.get("color_model_path", "")
        if color_model:
            color_filename = Path(color_model).name
            det_cfg_data["color_model_path"] = (
                f"models/{product}/{area}/yolo/{color_filename}"
            )
        (dest_dir / "config.yaml").write_text(
            yaml.safe_dump(det_cfg_data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        logger.info("config.yaml written (weights → %s).", versioned_name)
    except Exception as exc:
        logger.error("Failed to write config.yaml: %s", exc)
        raise

    # ------------------------------------------------------------------
    # Copy weights — best.pt → versioned name; also keep best.pt alias
    # ------------------------------------------------------------------
    src_best = run_dir / "weights" / "best.pt"
    if src_best.exists():
        shutil.copy2(src_best, versioned_path)
        # Keep a plain best.pt alias so any tool that looks for it still works
        shutil.copy2(src_best, weights_dest / "best.pt")
        logger.info("Copied best.pt → %s (+ best.pt alias).", versioned_name)
    else:
        logger.error("best.pt not found in %s — weights not deployed.", run_dir / "weights")
        return

    for fname in ["last.pt", "best.onnx"]:
        src = run_dir / "weights" / fname
        if src.exists():
            shutil.copy2(src, weights_dest / fname)
            logger.info("Copied %s.", fname)

    # ------------------------------------------------------------------
    # Copy colour statistics — use the filename referenced in config so
    # ModelManager can find it via the color_model_path field.
    # ------------------------------------------------------------------
    color_cfg_name = Path(det_cfg_data.get("color_model_path", "")).name
    for cand in [
        run_dir.parent / "quality" / "color" / "stats.json",
        run_dir / "color_stats.json",
    ]:
        if cand.exists():
            dest_name = color_cfg_name or cand.name
            shutil.copy2(cand, dest_dir / dest_name)
            logger.info("Copied colour stats from %s → %s.", cand, dest_name)
            break

    # ------------------------------------------------------------------
    # Write version_manifest.yaml into the source run directory so the
    # training run can be traced back to its deployed version later.
    # ------------------------------------------------------------------
    manifest = {
        "deployed_version": _version_str(version),
        "deployed_date": date_str,
        "deployed_file": versioned_name,
        "deployed_to": str(dest_dir / "config.yaml"),
        "product": product,
        "area": area,
    }
    try:
        (run_dir / "version_manifest.yaml").write_text(
            yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        logger.info("version_manifest.yaml written to %s.", run_dir)
    except OSError as exc:
        logger.warning("Could not write version_manifest.yaml: %s", exc)

    # ------------------------------------------------------------------
    # Summary with timestamp transparency
    # ------------------------------------------------------------------
    mtime = src_best.stat().st_mtime
    trained_at = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    logger.info(
        "已部署：%s/%s/yolo\n"
        "  ← 版本：v%s\n"
        "  ← 檔案：%s\n"
        "  ← 權重訓練時間：%s\n"
        "  注意：若本次訓練被 cache 略過，此模型並非本次訓練的新產物。\n"
        "  → 在 yolo11_inference 按 F5 重整模型清單即可看到 %s/%s。",
        product, area,
        _version_str(version),
        versioned_name,
        trained_at,
        product, area,
    )


TASKS: List[Any] = []
