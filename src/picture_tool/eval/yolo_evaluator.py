import logging
from pathlib import Path
from typing import Optional, List

from picture_tool.utils.experiment import write_experiment
from picture_tool.utils.experiment import _load_metrics_csv  # type: ignore
from picture_tool.pipeline.utils import detect_existing_weights

import os

try:
    if os.environ.get("PYTEST_IS_RUNNING") == "1":
        raise ImportError("Bypass ultralytics during pytest to avoid Windows PyTorch DLL crashes")
    from ultralytics import YOLO  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


def _resolve_weights(config: dict) -> Path:
    # Prefer explicit weights via standard detection util instead of DRY violation
    weights_path, _ = detect_existing_weights(config, prefer=None)
    if not weights_path:
        raise FileNotFoundError("Could not detect any existing model weights to evaluate.")
    return Path(str(weights_path)).resolve()


def evaluate_yolo(config: dict, logger: Optional[logging.Logger] = None) -> None:
    """Evaluate a trained YOLO model on the dataset specified in training config."""
    logger = logger or logging.getLogger(__name__)
    if YOLO is None and os.environ.get("PYTEST_IS_RUNNING") != "1":
        raise RuntimeError("ultralytics is not available. Please install ultralytics.")

    ycfg = config.get("yolo_training", {})
    ecfg = config.get("yolo_evaluation", {})
    imgsz = int(ecfg.get("imgsz", ycfg.get("imgsz", 640)))
    device = str(ecfg.get("device", ycfg.get("device", "cpu")))
    # OOM Prevention: Limit default workers and batch size
    workers = int(ecfg.get("workers", ycfg.get("workers", 1)))
    batch = int(ecfg.get("batch", 4))

    dataset_dir = Path(str(ycfg.get("dataset_dir", "./datasets/split_dataset")))
    data_yaml = (dataset_dir / "data.yaml").resolve()
    weights_path = _resolve_weights(config)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    logger.info(
        f"Evaluating model: {weights_path} | data={data_yaml} imgsz={imgsz} device={device} workers={workers} batch={batch}"
    )
    model = YOLO(str(weights_path))
    results = model.val(data=str(data_yaml), imgsz=imgsz, device=device, workers=workers, batch=batch)
    logger.info("Evaluation completed.")
    run_dir = weights_path.parent.parent
    artifacts = {
        "weights": weights_path,
        "data_yaml": data_yaml,
    }
    metrics = {}
    if hasattr(results, "results_file"):
        metrics_path = Path(str(results.results_file))
        metrics.update(_load_metrics_csv(metrics_path))
    write_experiment(
        run_type="eval",
        config=config,
        run_dir=run_dir,
        metrics=metrics,
        artifacts=artifacts,
        extra={"imgsz": imgsz, "device": device},
        results_csv=metrics_path if "metrics_path" in locals() else None,
    )
