import logging
import zipfile
from pathlib import Path
from picture_tool.pipeline.utils import detect_existing_weights

def run_artifact_bundle(config, args):
    """Bundle training artifacts into a zip file for delivery."""
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

    logger.info(f"Bundling artifacts from {run_dir} into {zip_path}...")

    inclusion_map = {
        "include_weights": [
            run_dir / "weights" / "best.pt",
            run_dir / "weights" / "last.pt",
            run_dir / "weights" / "best.onnx",
        ],
        "include_detection_config": [run_dir / "config.json"],
        "include_position_config": [run_dir / "position_config.yaml", run_dir / "auto_position_config.yaml"],
        "include_results_csv": [run_dir / "results.csv"],
        "include_args_yaml": [run_dir / "args.yaml"],
    }

    files_to_zip = []
    
    # Always include explicit specific files if they exist
    for key, candidates in inclusion_map.items():
        if bcfg.get(key, True):
            for cand in candidates:
                if cand.exists():
                    files_to_zip.append(cand)
    
    # Create Zip
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files_to_zip:
                # Arcname relative to run_dir to keep structure flat-ish or clean
                zf.write(f, arcname=f.name)
        logger.info(f"Bundle created successfully: {zip_path}")
    except Exception as e:
        logger.error(f"Failed to create bundle: {e}")

from typing import List, Any
TASKS: List[Any] = []
