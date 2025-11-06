import logging
from pathlib import Path
from typing import Optional, List

try:
    from ultralytics import YOLO  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


def _candidate_runs(project: Path, name: str) -> List[Path]:
    if not project.exists():
        return []
    # Match 'train', 'train2', 'train3', ... or custom names with numeric suffix
    runs = [p for p in project.iterdir() if p.is_dir() and p.name.startswith(name)]
    # Prefer those that actually have a best.pt
    runs = [p for p in runs if (p / "weights" / "best.pt").exists()]
    # Sort by latest modification time of best.pt (fallback to dir mtime)
    runs.sort(
        key=lambda p: (
            (p / "weights" / "best.pt").stat().st_mtime
            if (p / "weights" / "best.pt").exists()
            else p.stat().st_mtime
        ),
        reverse=True,
    )
    return runs


def _resolve_weights(config: dict) -> Path:
    ycfg = config.get("yolo_training", {})
    ecfg = config.get("yolo_evaluation", {})
    # Prefer explicit weights
    weights = ecfg.get("weights")
    if weights:
        p = Path(str(weights)).resolve()
        return p
    project = Path(str(ycfg.get("project", "./runs/detect")))
    name = str(ycfg.get("name", "train"))
    # Try latest matching run first
    candidates = _candidate_runs(project, name)
    if candidates:
        return (candidates[0] / "weights" / "best.pt").resolve()
    # Fallback to exact name
    return (project / name / "weights" / "best.pt").resolve()


def evaluate_yolo(config: dict, logger: Optional[logging.Logger] = None) -> None:
    """Evaluate a trained YOLO model on the dataset specified in training config."""
    logger = logger or logging.getLogger(__name__)
    if YOLO is None:
        raise RuntimeError("ultralytics is not available. Please install ultralytics.")

    ycfg = config.get("yolo_training", {})
    ecfg = config.get("yolo_evaluation", {})
    imgsz = int(ecfg.get("imgsz", ycfg.get("imgsz", 640)))
    device = str(ecfg.get("device", ycfg.get("device", "cpu")))

    dataset_dir = Path(str(ycfg.get("dataset_dir", "./datasets/split_dataset")))
    data_yaml = (dataset_dir / "data.yaml").resolve()
    weights_path = _resolve_weights(config)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    logger.info(
        f"Evaluating model: {weights_path} | data={data_yaml} imgsz={imgsz} device={device}"
    )
    model = YOLO(str(weights_path))
    _ = model.val(data=str(data_yaml), imgsz=imgsz, device=device)
    logger.info("Evaluation completed.")
