import logging
from pathlib import Path
from typing import List, Optional

import yaml

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


def _ensure_data_yaml(dataset_dir: Path, names: List[str], out_path: Optional[Path] = None) -> Path:
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


def train_yolo(config: dict, logger: Optional[logging.Logger] = None) -> Path:
    """Train YOLO using Ultralytics and return run directory.

    Expects config['yolo_training'] block with keys:
      - dataset_dir, class_names, model, epochs, imgsz, batch, device, project, name
    """
    logger = logger or logging.getLogger(__name__)
    ycfg = config.get("yolo_training", {})

    dataset_dir = Path(ycfg.get("dataset_dir", "./datasets/split_dataset")).resolve()
    names = ycfg.get("class_names") or []
    if not isinstance(names, list) or not names:
        raise ValueError("yolo_training.class_names must be a non-empty list")

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
    project = str(ycfg.get("project", "./runs/detect"))
    name = str(ycfg.get("name", "train"))

    if YOLO is None:
        raise RuntimeError("ultralytics is not available. Please install ultralytics.")

    logger.info(
        f"Starting YOLO training | model={model_arg} epochs={epochs} imgsz={imgsz} batch={batch} device={device}"
    )
    model = YOLO(model_arg)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
    )
    # Ultralytics returns a Results object; run dir is typically project/name
    run_dir = Path(project) / name
    logger.info(f"Training completed. Run directory: {run_dir}")
    return run_dir

