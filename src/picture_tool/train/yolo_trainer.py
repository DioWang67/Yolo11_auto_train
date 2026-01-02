import logging
import shutil
import json
from pathlib import Path
from typing import Any, List, Mapping, Optional

import yaml
from picture_tool.utils.experiment import write_experiment
from picture_tool.utils.experiment import _load_metrics_csv  # type: ignore
from picture_tool.utils.hashing import compute_dir_hash, compute_config_hash
from picture_tool.constants import DEFAULT_RUNS_DIR, DEFAULT_SPLITS_DIR
from picture_tool.tracking.experiment_tracker import get_tracker

try:
    from ultralytics import YOLO  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore

def _ensure_data_yaml(
    dataset_dir: Path, names: List[str], out_path: Optional[Path] = None
) -> Path:
    dataset_dir = dataset_dir.resolve()
    data = {
        "path": str(dataset_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": names,
    }
    out_path = out_path or (dataset_dir / "data.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return out_path


def train_yolo(
    config: dict,
    logger: Optional[logging.Logger] = None,
    args: Optional[object] = None,
) -> Path:
    """Train YOLO using Ultralytics and return run directory.

    Expects config['yolo_training'] block with keys:
      - dataset_dir, class_names, model, epochs, imgsz, batch, device, project, name
    """
    logger = logger or logging.getLogger(__name__)
    ycfg = config.get("yolo_training", {})
    
    # Use constant for default instead of hardcoded
    dataset_dir = Path(ycfg.get("dataset_dir", DEFAULT_SPLITS_DIR)).resolve()
    names = ycfg.get("class_names")
    
    # Auto-detect class names if missing OR if set to default placeholder
    can_autodetect = not names or (isinstance(names, list) and len(names) == 1 and names[0] == "object")
    
    if can_autodetect:
        possible_classes = dataset_dir / "classes.txt"
        if possible_classes.exists():
            try:
                content = possible_classes.read_text(encoding="utf-8")
                detected = [line.strip() for line in content.splitlines() if line.strip()]
                if detected:
                    names = detected
                    logger.info(f"Auto-detected class names from {possible_classes}: {names}")
            except Exception as e:
                logger.warning(f"Failed to read classes.txt: {e}")

    if not isinstance(names, list) or not names:
        raise ValueError(
            "yolo_training.class_names must be a non-empty list. "
            "Please add it to config.yaml (e.g., class_names: ['dog', 'cat']) "
            "or ensure classes.txt exists in dataset_dir."
        )

    # Prepare data.yaml
    data_yaml = _ensure_data_yaml(dataset_dir, names)
    logger.info(f"Prepared data.yaml at: {data_yaml}")

    # Resolve model weights
    model_cfg = ycfg.get("model", "yolo11n.pt")
    model_path = Path(str(model_cfg))
    if model_path.exists():
        model_arg = str(model_path)
    else:
        # allow using model name directly (e.g., 'yolo11n.pt')
        model_arg = str(model_cfg)

    epochs = int(ycfg.get("epochs", 100))
    imgsz = int(ycfg.get("imgsz", 640))
    batch = int(ycfg.get("batch", 16))
    device = str(ycfg.get("device", "cpu"))
    project = str(ycfg.get("project", DEFAULT_RUNS_DIR / "detect"))
    name = str(ycfg.get("name", "train"))

    if YOLO is None:
        raise RuntimeError("ultralytics is not available. Please install ultralytics.")

    logger.info(
        f"Starting YOLO training | model={model_arg} epochs={epochs} imgsz={imgsz} batch={batch} device={device}"
    )
    # Initialize Experiment Tracker
    tracker = get_tracker(config)
    tracker.start_run(run_name=name)
    tracker.log_params({
        "model": model_arg,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "dataset": str(dataset_dir)
    })

    model = YOLO(model_arg)
    train_results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
    )
    run_dir: Optional[Path] = None
    if hasattr(train_results, "save_dir"):
        candidate = getattr(train_results, "save_dir")
        if candidate:
            run_dir = Path(str(candidate)).resolve()
    trainer = getattr(model, "trainer", None)
    if run_dir is None and trainer is not None and getattr(trainer, "save_dir", None):
        run_dir = Path(str(trainer.save_dir)).resolve()
    if run_dir is None:
        run_dir = (Path(project) / name).resolve()
    logger.info(f"Training completed. Run directory: {run_dir}")

    # Record Robust Caching Metadata
    try:
        data_hash = compute_dir_hash(dataset_dir)
        cfg_hash = compute_config_hash(ycfg)
        metadata = {
            "dataset_hash": data_hash,
            "config_hash": cfg_hash,
            "dataset_dir": str(dataset_dir)
        }
        with open(run_dir / "last_run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save run metadata: {e}")

    # Basic experiment logging (metrics only, artifacts handled by caller/pipeline now)
    try:
        metrics = _load_metrics_csv(run_dir / "results.csv")
        
        # Log to MLflow/Tracker
        if metrics:
            # Take the last row of metrics
            last_metrics = {k.strip(): float(v) for k, v in metrics[-1].items() if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit())}
            tracker.log_metrics(last_metrics)
        
        tracker.log_artifact(str(run_dir / "results.csv"))
        tracker.log_artifact(str(run_dir / "weights" / "best.pt"))
        if (run_dir / "confusion_matrix.png").exists():
            tracker.log_artifact(str(run_dir / "confusion_matrix.png"))

        # Minimal artifacts dict here, orchestration layer adds more
        artifacts = {
            "weights_best": (run_dir / "weights" / "best.pt"),
            "weights_last": (run_dir / "weights" / "last.pt"),
            "results_csv": (run_dir / "results.csv"),
        }
        write_experiment(
            run_type="train",
            config=config,
            run_dir=run_dir,
            metrics=metrics,
            artifacts=artifacts,
            results_csv=run_dir / "results.csv",
        )
    except Exception as e:
        logger.warning(f"Failed to write experiment log: {e}")
    finally:
        tracker.end_run()

    return run_dir
