import logging
import zipfile
from pathlib import Path
from typing import List, Any
from picture_tool.pipeline.utils import detect_existing_weights


def rewrite_detection_config(det_cfg_data: dict, product: str) -> dict:
    """Return a copy of *det_cfg_data* with paths and keys rewritten for deployment.

    Shared by both :func:`run_artifact_bundle` and the deploy task so the
    two output formats stay in sync automatically.
    """
    data = dict(det_cfg_data)

    # 1. Fix weights path: ensure it lives under 'weights/' sub-folder
    old_weights = data.get("weights", "")
    if old_weights:
        data["weights"] = f"weights/{Path(old_weights).name}"

    # 2. Fix color_model_path: just the filename (sits next to config.yaml)
    color_model = data.get("color_model_path", "")
    if color_model:
        data["color_model_path"] = Path(color_model).name

    # 3. Replace generic product key in expected_items with the real product name
    exp_items = data.get("expected_items", {})
    if exp_items and isinstance(exp_items, dict) and product not in exp_items:
        first_key = next(iter(exp_items), None)
        if first_key:
            exp_items[product] = exp_items.pop(first_key)
    data["current_product"] = product

    # 4. Replace generic product key in position_config
    pos_cfg = data.get("position_config", {})
    if pos_cfg and isinstance(pos_cfg, dict) and product not in pos_cfg:
        pos_keys = [k for k in pos_cfg if k not in ("enabled", "# NOTE")]
        if pos_keys and pos_keys[0] != product:
            pos_cfg[product] = pos_cfg.pop(pos_keys[0])

    return data


def run_artifact_bundle(config, args):
    """Bundle training artifacts into a zip that can be extracted directly
    into ``yolo11_inference/models/`` and be discovered by ModelManager.

    Zip layout mirrors the inference directory convention::

        {product}/{area}/yolo/
        ├── config.yaml
        ├── weights/best.pt
        ├── weights/last.pt       (optional)
        ├── weights/best.onnx     (optional)
        ├── stats.json            (optional)
        ├── results.csv           (optional)
        └── args.yaml             (optional)
    """
    import yaml

    logger = logging.getLogger(__name__)
    ycfg = config.get("yolo_training", {})
    bcfg = ycfg.get("artifact_bundle", {})
    if not bcfg.get("enabled", False):
        logger.info("Artifact bundle disabled.")
        return

    # Determine source run directory
    _, run_dir = detect_existing_weights(config)
    if not run_dir or not run_dir.exists():
        logger.warning("No run directory found for bundling.")
        return

    # Resolve product and area (same logic as deploy task)
    product = ycfg.get("name", "train")
    if getattr(args, "product", None):
        product = args.product

    area: str = (
        bcfg.get("area")
        or ycfg.get("deploy", {}).get("area")
        or ycfg.get("position_validation", {}).get("area")
        or "A"
    )

    # Prefix for all paths inside the zip — mirrors inference models/ layout
    zip_prefix = f"{product}/{area}/yolo"

    out_dir = Path(bcfg.get("base_dir") or run_dir)
    dir_name = bcfg.get("dir_name", "bundle")
    zip_name = f"{product}_{dir_name}.zip"
    zip_path = out_dir / zip_name

    logger.info(f"Bundling deployment-ready artifacts from {run_dir} into {zip_path}...")

    # Load detection config to rewrite it — abort if missing, as the resulting
    # ZIP would be unusable (no config.yaml = ModelCatalog cannot load it).
    det_cfg_path = run_dir / "detection_config.yaml"
    if not det_cfg_path.exists():
        logger.error(
            "detection_config.yaml not found at %s. "
            "Bundle aborted — an incomplete ZIP without config.yaml is not deployable. "
            "Ensure yolo_train ran successfully before artifact_bundle.",
            det_cfg_path,
        )
        return

    try:
        with open(det_cfg_path, "r", encoding="utf-8") as f:
            det_cfg_data = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.error("Failed to read detection_config.yaml: %s — bundle aborted.", exc)
        return

    det_cfg_data = rewrite_detection_config(det_cfg_data, product)

    # Override weights / color_model_path to use full inference-relative paths
    # so ModelManager can resolve them from the project root.
    old_weights = det_cfg_data.get("weights", "")
    if old_weights:
        weights_filename = Path(old_weights).name
        det_cfg_data["weights"] = (
            f"models/{product}/{area}/yolo/weights/{weights_filename}"
        )

    color_model = det_cfg_data.get("color_model_path", "")
    if color_model:
        color_filename = Path(color_model).name
        det_cfg_data["color_model_path"] = (
            f"models/{product}/{area}/yolo/{color_filename}"
        )

    # Prepare files to copy verbatim
    files_to_zip: list[tuple[Path, str]] = []

    # Optional explicitly included files
    inclusion_map = {
        "include_results_csv": [run_dir / "results.csv"],
        "include_args_yaml": [run_dir / "args.yaml"],
    }

    for key, candidates in inclusion_map.items():
        if bcfg.get(key, True):
            for cand in candidates:
                if cand.exists():
                    files_to_zip.append((cand, f"{zip_prefix}/{cand.name}"))

    # Build ZIP
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write rewritten config.yaml
            config_yaml_str = yaml.safe_dump(det_cfg_data, allow_unicode=True, sort_keys=False)
            zf.writestr(f"{zip_prefix}/config.yaml", config_yaml_str)

            # Write weights
            if bcfg.get("include_weights", True):
                weight_cands = ["best.pt", "last.pt", "best.onnx"]
                for wc in weight_cands:
                    wp = run_dir / "weights" / wc
                    if wp.exists():
                        zf.write(wp, arcname=f"{zip_prefix}/weights/{wc}")

            # Write color stats — use the filename referenced in config so
            # ModelManager can find it.  Search same locations as deploy task.
            color_cfg_name = Path(det_cfg_data.get("color_model_path", "")).name
            for cand in [
                run_dir.parent / "quality" / "color" / "stats.json",
                run_dir / "color_stats.json",
            ]:
                if cand.exists():
                    arcname = f"{zip_prefix}/{color_cfg_name}" if color_cfg_name else f"{zip_prefix}/{cand.name}"
                    zf.write(cand, arcname=arcname)
                    break

            # Write verbatim files
            for src, arcname in files_to_zip:
                zf.write(src, arcname=arcname)

        logger.info(
            "Deployment Bundle created: %s\n"
            "  → 解壓到 yolo11_inference/models/ 即可直接使用:\n"
            "    unzip %s -d /path/to/yolo11_inference/models/",
            zip_path, zip_path.name,
        )
    except (FileNotFoundError, PermissionError, OSError) as e:
        logger.warning(f"Failed to create bundle: {e}")


TASKS: List[Any] = []
