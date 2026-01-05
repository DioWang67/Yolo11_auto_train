import logging
import json
from pathlib import Path
from picture_tool.train.yolo_trainer import train_yolo
from picture_tool.eval.yolo_evaluator import evaluate_yolo
from picture_tool.position import run_position_validation
from picture_tool.pipeline.utils import detect_existing_weights, mtime_latest
from picture_tool.pipeline.core import Task
from picture_tool.utils.onnx_exporter import OnnxExporter
from picture_tool.position.position_config_gen import PositionConfigGenerator
from picture_tool.utils.detection_config import DetectionConfigExporter
from picture_tool.utils.hashing import compute_dir_hash, compute_config_hash
from picture_tool.constants import DEFAULT_RUNS_DIR, DEFAULT_SPLITS_DIR
from picture_tool.tasks.bundle import run_artifact_bundle

def run_yolo_train(config, args):
    logger = logging.getLogger(__name__)
    run_dir = train_yolo(config, args=args, logger=logger)
    
    # Post-training steps
    # 1. Position Config Generation
    try:
        PositionConfigGenerator.generate(config, run_dir, logger)
    except Exception as e:
        logger.warning(f"Position config generation failed: {e}")

    # 2. ONNX Export
    try:
        OnnxExporter.export(config, run_dir, logger)
    except Exception as e:
        logger.warning(f"ONNX export failed: {e}")

    # 3. Detection Config Export
    ycfg = config.get("yolo_training", {})
    pos_cfg = ycfg.get("position_validation", {})
    selected_tasks = getattr(args, "tasks", []) or []
    req_tasks = {str(t) for t in selected_tasks}
    position_requested = not req_tasks or "position_validation" in req_tasks
    position_validation_active = bool(
         isinstance(pos_cfg, dict) and pos_cfg.get("enabled") and position_requested
    )
    
    try:
        DetectionConfigExporter.export(
            config, 
            run_dir, 
            logger, 
            include_position=position_validation_active
        )
    except Exception as e:
        logger.warning(f"Detection config export failed: {e}")


def run_yolo_evaluation(config, args):
    weights_path, run_dir = detect_existing_weights(config, prefer=None)
    if weights_path:
        ecfg = config.setdefault("yolo_evaluation", {})
        ecfg["weights"] = str(weights_path)
        logging.getLogger(__name__).info(
            "Using existing weights for evaluation: %s (run_dir=%s)",
            weights_path,
            run_dir,
        )
    else:
        logging.getLogger(__name__).warning(
            "No existing weights detected before evaluation; will rely on default resolution."
        )
    evaluate_yolo(config)


def run_position_validation_task(config, args):
    """Run offline position validation using trained weights and sample images."""
    ycfg = config.get("yolo_training", {}) if isinstance(config, dict) else {}
    run_root = Path(str(ycfg.get("project", DEFAULT_RUNS_DIR / "detect")))
    run_name = str(ycfg.get("name", "train"))
    default_run_dir = run_root / run_name
    weights_path, detected_run_dir = detect_existing_weights(config, prefer="position")
    run_dir = detected_run_dir or default_run_dir

    if weights_path:
        pv_cfg = ycfg.get("position_validation", {}) or {}
        pv_cfg["weights"] = str(weights_path)
        ycfg["position_validation"] = pv_cfg
        config["yolo_training"] = ycfg
        logging.getLogger(__name__).info(
            "Using existing weights for position validation: %s (run_dir=%s)",
            weights_path,
            run_dir,
        )

    # Fallback: If no explicit config provided, check for auto-generated one in run_dir
    pv_cfg = ycfg.get("position_validation", {})
    
    # DEBUG: Log current config state
    logging.getLogger(__name__).info(
        f"DEBUG: Checking position config. Config keys: {list(pv_cfg.keys())}. "
        f"Run dir: {run_dir}. Exists: {run_dir.exists()}"
    )

    if not pv_cfg.get("config") and not pv_cfg.get("config_path"):
        auto_conf = run_dir / "auto_position_config.yaml"
        if auto_conf.exists():
            pv_cfg["config"] = str(auto_conf)
            ycfg["position_validation"] = pv_cfg
            config["yolo_training"] = ycfg
            logging.getLogger(__name__).info(f"Using auto-generated position config: {auto_conf}")
        else:
            logging.getLogger(__name__).warning(
                "Skipping position_validation: Auto-config not found in run_dir. "
                "This likely means the model detected nothing during training auto-generation."
            )
            return None
    else:
        logging.getLogger(__name__).info(
            f"DEBUG: Position config present in keys. Config: {pv_cfg.get('config')}, Path: {pv_cfg.get('config_path')}"
        )

    if not run_dir.exists():
        raise FileNotFoundError(
            "No trained run found for position_validation. "
            f"Checked {run_dir} (project={run_root}, name prefix={run_name}). "
            "Provide yolo_training.position_validation.weights or run yolo_train manually."
        )

    return run_position_validation(config, run_dir, logger=logging.getLogger(__name__))


def skip_yolo_train(config, args):
    y = config.get("yolo_training", {})
    project = Path(str(y.get("project", DEFAULT_RUNS_DIR / "detect")))
    name = str(y.get("name", "train"))
    run_dir = project / name
    
    metadata_path = run_dir / "last_run_metadata.json"
    
    if getattr(args, "force", False):
        return None
        
    if not run_dir.exists() or not metadata_path.exists():
        return None
        
    # Check hashes
    try:
        with open(metadata_path, "r") as f:
            stored_meta = json.load(f)
            
        dataset_dir = Path(str(y.get("dataset_dir", DEFAULT_SPLITS_DIR))).resolve()
        current_data_hash = compute_dir_hash(dataset_dir)
        current_cfg_hash = compute_config_hash(y)
        
        if (stored_meta.get("dataset_hash") == current_data_hash and
            stored_meta.get("config_hash") == current_cfg_hash and 
            (run_dir / "weights" / "best.pt").exists()):
            return "Skipping training: Source dataset and config match last run."
            
    except Exception:
        # Fallback to mtime if hash check fails or file corrupt
        pass

    # Legacy mtime fallback
    weights = run_dir / "weights" / "best.pt"
    dataset_dir = Path(str(y.get("dataset_dir", DEFAULT_SPLITS_DIR)))
    auto_conf = run_dir / "auto_position_config.yaml"
    if weights.exists() and auto_conf.exists():
        if weights.stat().st_mtime >= mtime_latest([dataset_dir]):
            return "Found latest best.pt (mtime check); skipping training."
            
    return None


TASKS = [
    Task(
        name="yolo_train",
        run=run_yolo_train,
        skip_fn=skip_yolo_train,
        description="Train YOLO model.",
        dependencies=["dataset_splitter"],
    ),
    Task(
        name="yolo_evaluation",
        run=run_yolo_evaluation,
        description="Evaluate YOLO model.",
        dependencies=["yolo_train"],
    ),
    Task(
        name="position_validation",
        run=run_position_validation_task,
        description="Offline position validation.",
        dependencies=[],  # 移除硬依賴,改為運行時檢查權重
    ),
    Task(
        name="artifact_bundle",
        run=run_artifact_bundle,
        description="Bundle training artifacts (Zip).",
        dependencies=[],
    ),
]
