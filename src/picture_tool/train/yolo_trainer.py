import logging
import json
import re
from pathlib import Path
from typing import List, Optional

import yaml
from picture_tool.utils.experiment import write_experiment
from picture_tool.utils.experiment import _load_metrics_csv  # type: ignore
from picture_tool.utils.hashing import compute_dir_hash, compute_config_hash
from picture_tool.constants import DEFAULT_RUNS_DIR, DEFAULT_SPLITS_DIR
from picture_tool.tracking.experiment_tracker import get_tracker


# Define a stop exception for cleaner handling
class TrainingInterrupted(Exception):
    pass


try:
    from ultralytics import YOLO  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
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


def _parse_yolo_config(config: dict) -> dict:
    """Extract and validate basic YOLO configurations."""
    ycfg = config.get("yolo_training", {})
    return {
        "dataset_dir": Path(ycfg.get("dataset_dir", DEFAULT_SPLITS_DIR)).resolve(),
        "names": ycfg.get("class_names"),
        "model_cfg": ycfg.get("model", "yolo11n.pt"),
        "epochs": int(ycfg.get("epochs", 100)),
        "imgsz": int(ycfg.get("imgsz", 640)),
        "batch": int(ycfg.get("batch", 16)),
        "device": str(ycfg.get("device", "cpu")),
        "project": str(ycfg.get("project", DEFAULT_RUNS_DIR / "detect")),
        "name": str(ycfg.get("name", "train")),
    }


def _prepare_dataset_names(dataset_dir: Path, names: Optional[List[str]], logger: logging.Logger) -> List[str]:
    """Auto-detect class names or validate provided ones."""
    can_autodetect = not names or (isinstance(names, list) and len(names) == 1 and names[0] == "object")
    if can_autodetect:
        possible_classes = dataset_dir / "classes.txt"
        if possible_classes.exists():
            try:
                content = possible_classes.read_text(encoding="utf-8")
                detected = [line.strip() for line in content.splitlines() if line.strip()]
                if detected:
                    logger.info(f"Auto-detected class names from {possible_classes}: {detected}")
                    return detected
            except (UnicodeDecodeError, OSError) as e:
                logger.warning(f"Failed to read classes.txt: {e}")

    if not isinstance(names, list) or not names:
        raise ValueError(
            "yolo_training.class_names must be a non-empty list. "
            "Please add it to config.yaml (e.g., class_names: ['dog', 'cat']) "
            "or ensure classes.txt exists in dataset_dir."
        )
    return names


def _build_yolo_model(model_cfg: str) -> str:
    """Resolve correct YOLO model weights path."""
    model_path = Path(str(model_cfg))
    if model_path.exists():
        return str(model_path.resolve())
    
    fallback = Path("models") / model_cfg
    if fallback.exists():
         return str(fallback.resolve())
    return str(model_cfg)


def _attach_yolo_callbacks(model: Any, logger: logging.Logger, stop_event: Any) -> None:
    """Attach required lifecycle callbacks for logging and cancellation to YOLO."""
    tb_state = {"batch": 0}

    def on_train_start(trainer):
        logger.info("  [YOLO Lifecycle] Training successfully started.")

    def on_train_epoch_start(trainer):
        epoch = getattr(trainer, "epoch", 0) + 1
        logger.info(f"  [YOLO Lifecycle] Starting Epoch {epoch}/{getattr(trainer, 'epochs', '?')}")

    def on_train_epoch_end(trainer):
        try:
            current_epoch = getattr(trainer, "epoch", 0) + 1
            total_epochs = getattr(trainer, "epochs", "?")
            logger.info(f"  [YOLO Lifecycle] Finished Epoch {current_epoch}/{total_epochs}")
        except AttributeError as e:
            logger.error(f"Error in on_train_epoch_end (Missing attributes): {e}")

        if stop_event and stop_event.is_set():
            logger.info("Stop event detected. Stopping YOLO training gracefully.")
            trainer.stop = True
            
    def on_train_batch_end(trainer):
        try:
            tb_state["batch"] += 1
            if tb_state["batch"] % 10 == 0:
                epoch = getattr(trainer, "epoch", 0) + 1
                loss_str = ""
                if hasattr(trainer, "loss_items") and hasattr(trainer, "loss_names"):
                    try:
                        lnames = [n.strip() for n in trainer.loss_names]
                        lvals = [v.item() if hasattr(v, 'item') else float(v) for v in trainer.loss_items]
                        losses = [f"{n}: {v:.3f}" for n, v in zip(lnames, lvals)]
                        loss_str = f" | {' | '.join(losses)}"
                    except (ValueError, AttributeError, TypeError) as e:
                        loss_str = f" | (Loss parse error: {e})"
                logger.info(f"  [YOLO Progress] Epoch {epoch} - Batch {tb_state['batch']}{loss_str}")
                
            if stop_event and stop_event.is_set():
                trainer.stop = True
                
        except (AttributeError, TypeError) as e:
            logger.debug(f"Ignoring batch callback attribute error: {e}")

    try:
        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_train_batch_end", on_train_batch_end)
    except AttributeError as e:
        logger.warning(f"Could not attach progress callbacks to YOLO model (not supported): {e}")


def train_yolo(
    config: dict,
    logger: Optional[logging.Logger] = None,
    args: Optional[object] = None,
) -> Path:
    """Train YOLO using Ultralytics and return run directory.

    Expects config['yolo_training'] block.
    """
    logger = logger or logging.getLogger(__name__)
    
    # 1. Parse and Validate Configurations
    params = _parse_yolo_config(config)
    names = _prepare_dataset_names(params["dataset_dir"], params["names"], logger)
    model_arg = _build_yolo_model(params["model_cfg"])
    
    # 2. Prepare data.yaml
    data_yaml = _ensure_data_yaml(params["dataset_dir"], names)
    logger.info(f"Prepared data.yaml at: {data_yaml}")

    if YOLO is None:
        raise RuntimeError("ultralytics is not available. Please install ultralytics.")

    logger.info(
        f"Starting YOLO training | model={model_arg} epochs={params['epochs']} imgsz={params['imgsz']} batch={params['batch']} device={params['device']}"
    )
    # 3. Initialize Experiment Tracker
    tracker = get_tracker(config)
    tracker.start_run(run_name=params["name"])
    tracker.log_params(
        {
            "model": model_arg,
            "epochs": params["epochs"],
            "imgsz": params["imgsz"],
            "batch": params["batch"],
            "device": params["device"],
            "dataset": str(params["dataset_dir"]),
        }
    )

    # 4. Initialize YOLO and Attach Callbacks (DI Supported)
    # Allows injecting a mock model class or pre-initialized instance via args for testing
    model_factory = getattr(args, "model_factory", YOLO)
    yolo_instance = getattr(args, "yolo_instance", None)
    
    model = yolo_instance if yolo_instance else model_factory(model_arg)
    
    stop_event = getattr(args, "stop_event", None)
    _attach_yolo_callbacks(model, logger, stop_event)

    # 5. Execute Training
    try:
        train_results = model.train(
            data=str(data_yaml),
            epochs=params["epochs"],
            imgsz=params["imgsz"],
            batch=params["batch"],
            device=params["device"],
            project=params["project"],
            name=params["name"],
            exist_ok=True,
        )
    except RuntimeError as e:
        # Pass up specific runtime errors from Ultralytics (e.g., CUDA OOM)
        raise e
    except Exception as e:
        logger.error(f"Unexpected training failure: {e}")
        raise RuntimeError(f"YOLO training failed due to unforeseen error: {e}") from e

    # 6. Finalize Outputs
    if stop_event and stop_event.is_set():
        logger.info("Training interrupted by user (graceful shutdown).")
        return Path(params["project"]) / params["name"]

    run_dir: Optional[Path] = None
    if hasattr(train_results, "save_dir"):
        candidate = getattr(train_results, "save_dir")
        if candidate:
            run_dir = Path(str(candidate)).resolve()
    trainer_obj = getattr(model, "trainer", None)
    if run_dir is None and trainer_obj is not None and getattr(trainer_obj, "save_dir", None):
        run_dir = Path(str(trainer_obj.save_dir)).resolve()
    if run_dir is None:
        run_dir = (Path(params["project"]) / params["name"]).resolve()
    logger.info(f"Training completed. Run directory: {run_dir}")

    # Record Robust Caching Metadata
    try:
        data_hash = compute_dir_hash(params["dataset_dir"])
        cfg_hash = compute_config_hash(config.get("yolo_training", {}))
        metadata = {
            "dataset_hash": data_hash,
            "config_hash": cfg_hash,
            "dataset_dir": str(params["dataset_dir"]),
        }
        with open(run_dir / "last_run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    except (FileNotFoundError, OSError, ValueError) as e:
        logger.warning(f"Failed to save run metadata: {e}")

    # Basic experiment logging (metrics only, artifacts handled by caller/pipeline now)
    try:
        metrics = _load_metrics_csv(run_dir / "results.csv")

        # Log to MLflow/Tracker
        if metrics:
            # Take the last row of metrics, sanitize keys for MLflow
            last_metrics = {
                re.sub(r'[^a-zA-Z0-9_\-\.\ /]', '_', k.strip()): float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float))
                or (isinstance(v, str) and v.lstrip("-").replace(".", "", 1).isdigit())
            }
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
    except (FileNotFoundError, KeyError, ValueError, OSError) as e:
        logger.warning(f"Failed to write experiment log: {e}")
    finally:
        tracker.end_run()

    return run_dir
