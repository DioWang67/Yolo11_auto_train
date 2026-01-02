import logging
from pathlib import Path
from picture_tool.train.yolo_trainer import train_yolo
from picture_tool.eval.yolo_evaluator import evaluate_yolo
from picture_tool.position import run_position_validation
from picture_tool.pipeline.utils import detect_existing_weights, mtime_latest
from picture_tool.pipeline.core import Task

def run_yolo_train(config, args):
    train_yolo(config, args=args)


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
    run_root = Path(str(ycfg.get("project", "./runs/detect")))
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
    y = config["yolo_training"]
    run_dir = Path(y.get("project", "./runs/detect")) / y.get("name", "train")
    weights = run_dir / "weights" / "best.pt"
    dataset_dir = Path(y.get("dataset_dir", "./data/split"))
    
    # Check if we need to force run to generate auto-config
    auto_conf = run_dir / "auto_position_config.yaml"
    need_auto_conf = not auto_conf.exists()

    if weights.exists() and not need_auto_conf:
        if weights.stat().st_mtime >= mtime_latest([dataset_dir]):
            return "Found latest best.pt; skipping training."
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
]
