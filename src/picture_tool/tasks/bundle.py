import logging
import zipfile
from pathlib import Path
from typing import List, Any
from picture_tool.pipeline.utils import detect_existing_weights


def run_artifact_bundle(config, args):
    """Bundle training artifacts into a zip file for delivery.
    This version dynamically edits the inference config to be 100%
    deployment-ready for yolo11_inference (ModelManager) without manual changes.
    """
    import yaml

    logger = logging.getLogger(__name__)
    bcfg = config.get("yolo_training", {}).get("artifact_bundle", {})
    if not bcfg.get("enabled", False):
        logger.info("Artifact bundle disabled.")
        return

    # Determine source run directory
    _, run_dir = detect_existing_weights(config)
    if not run_dir or not run_dir.exists():
        logger.warning("No run directory found for bundling.")
        return

    # Output setup
    product = config.get("yolo_training", {}).get("name", "train")
    if getattr(args, "product", None):
        product = args.product

    out_dir = Path(bcfg.get("base_dir") or run_dir)
    dir_name = bcfg.get("dir_name", "bundle")
    zip_name = f"{product}_{dir_name}.zip"
    zip_path = out_dir / zip_name

    logger.info(f"Bundling deployment-ready artifacts from {run_dir} into {zip_path}...")

    # Load detection config to rewrite it
    det_cfg_path = run_dir / "detection_config.yaml"
    det_cfg_data = {}
    if det_cfg_path.exists():
        try:
            with open(det_cfg_path, "r", encoding="utf-8") as f:
                det_cfg_data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load detection_config.yaml: {e}")

    # Rewrite config logic for Deployment
    if det_cfg_data:
        # 1. Fix Weights Path: point to 'weights/' sub-folder
        old_weights = det_cfg_data.get("weights", "")
        if old_weights:
            weight_name = Path(old_weights).name
            det_cfg_data["weights"] = f"weights/{weight_name}"

        # 2. Fix color_model_path: just the filename, since it will sit next to config.yaml
        color_model = det_cfg_data.get("color_model_path", "")
        if color_model:
            det_cfg_data["color_model_path"] = Path(color_model).name

        # 3. Fix expected_items schema: replace generic 'project' (or whatever it is) key with actual product
        exp_items = det_cfg_data.get("expected_items", {})
        if exp_items and isinstance(exp_items, dict):
            if product and product not in exp_items:
                first_key = list(exp_items.keys())[0] if exp_items else None
                if first_key and first_key != product:
                    exp_items[product] = exp_items.pop(first_key)
            
            # Also fix 'current_product'
            det_cfg_data["current_product"] = product
        
        # 4. Fix position_config schema (same as expected_items)
        pos_cfg = det_cfg_data.get("position_config", {})
        if pos_cfg and isinstance(pos_cfg, dict):
            if product and product not in pos_cfg:
                # Exclude the "enabled" key when finding the product key
                pos_keys = [k for k in pos_cfg.keys() if k != "enabled" and k != "# NOTE"]
                first_key = pos_keys[0] if pos_keys else None
                if first_key and first_key != product:
                    pos_cfg[product] = pos_cfg.pop(first_key)

    # Prepare files to copy verbatim
    files_to_zip = []
    
    # Optional explicitly included files
    inclusion_map = {
        "include_results_csv": [run_dir / "results.csv"],
        "include_args_yaml": [run_dir / "args.yaml"],
    }
    
    for key, candidates in inclusion_map.items():
        if bcfg.get(key, True):
            for cand in candidates:
                if cand.exists():
                    files_to_zip.append((cand, cand.name)) # (source_path, arcname)

    # Build ZIP
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write rewritten config.yaml
            if det_cfg_data:
                config_yaml_str = yaml.safe_dump(det_cfg_data, allow_unicode=True, sort_keys=False)
                zf.writestr("config.yaml", config_yaml_str)
            else:
                logger.warning("No detection_config.yaml found to convert.")

            # Write weights
            if bcfg.get("include_weights", True):
                weight_cands = ["best.pt", "last.pt", "best.onnx"]
                for wc in weight_cands:
                    wp = run_dir / "weights" / wc
                    if wp.exists():
                        zf.write(wp, arcname=f"weights/{wc}")

            # Write color stats (from quality/color/stats.json) if requested
            # Since bundle config doesn't explicitly mention it, we search for it.
            # Yolo11_auto_train defaults color_stats to runs/project/quality/color/stats.json
            color_stats_file = run_dir.parent / "quality" / "color" / "stats.json"
            if color_stats_file.exists():
                zf.write(color_stats_file, arcname=color_stats_file.name)
            
            # Write verbatim files
            for src, arcname in files_to_zip:
                zf.write(src, arcname=arcname)
                
        logger.info(f"Deployment Bundle created successfully: {zip_path}")
    except (FileNotFoundError, PermissionError, OSError) as e:
        logger.warning(f"Failed to create bundle: {e}")


TASKS: List[Any] = []
