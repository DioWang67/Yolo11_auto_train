import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

from tqdm import tqdm  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    # Imported for annotations alone. ultralytics is loaded guardedly below
    # and is absent under pytest, so this must never run at import time.
    from ultralytics.engine.results import Results

import os

try:
    if os.environ.get("PYTEST_IS_RUNNING") == "1":
        raise ImportError("Bypass ultralytics during pytest")
    from ultralytics import YOLO  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore

from picture_tool.eval.yolo_evaluator import _resolve_weights  # reuse weight resolution


def run_batch_inference(config: dict, logger: Optional[logging.Logger] = None) -> Path:
    logger = logger or logging.getLogger(__name__)

    icfg = config.get("batch_inference", {})
    input_dir = Path(str(icfg.get("input_dir", "./data/project/raw/images"))).resolve()
    output_dir = Path(str(icfg.get("output_dir", "./runs/project/infer"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    imgsz = int(icfg.get("imgsz", config.get("yolo_training", {}).get("imgsz", 640)))
    device = str(
        icfg.get("device", config.get("yolo_training", {}).get("device", "cpu"))
    )
    conf = float(icfg.get("conf", 0.25))

    if YOLO is None:
        raise RuntimeError("ultralytics is not available. Please install ultralytics.")
    if not input_dir.exists():
        raise FileNotFoundError(f"Inference input_dir not found: {input_dir}")

    # Resolve weights: use explicit if provided; fallback to latest training run
    weights = icfg.get("weights") or None
    if not weights:
        weights = str(_resolve_weights(config))
    weights_path = Path(str(weights)).resolve()
    logger.info(
        f"Batch infer using weights={weights_path} imgsz={imgsz} conf={conf} device={device} input={input_dir}"
    )
    model = YOLO(str(weights_path))

    # Collect images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = [
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    ]
    logger.info(f"Found {len(images)} images for inference in {input_dir}")
    if not images:
        raise FileNotFoundError(f"No images in {input_dir}")

    csv_path = output_dir / "predictions.csv"
    
    # Calculate progress logging interval
    total_images = len(images)
    progress_interval = max(1, min(10, total_images // 10))
    processed_count = 0
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["file", "class_id", "class_name", "conf", "x1", "y1", "x2", "y2"]
        )
        for img_path in tqdm(images, desc="Batch inference", unit="img"):
            # ultralytics 8.4 annotates predict() as possibly yielding
            # tensors and as possibly not a list. Without stream=True it
            # returns a list of Results; the annotation is wider than the
            # runtime contract. Narrowing here also settles results[0]
            # below. Nothing about this changes at runtime.
            results = cast(
                "list[Results]",
                model(str(img_path), imgsz=imgsz, device=device, conf=conf),
            )
            for res in results:
                names = res.names
                if res.boxes is None:
                    logger.info(f"{img_path.name}: 0 detections (no boxes)")
                    continue
                # 8.4 types the Boxes fields as Tensor | ndarray | Any. Only
                # the tensor has .cpu(), and a detect model produces tensors.
                boxes = cast(Any, res.boxes)
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy().tolist()
                clss = boxes.cls.cpu().numpy().tolist()
                logger.info(
                    f"{img_path.name}: {len(confs)} detections (min_conf={min(confs) if confs else 'NA'})"
                )
                for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, clss):
                    cid_int = int(cid)
                    cname = (
                        names.get(cid_int, str(cid_int))
                        if isinstance(names, dict)
                        else str(cid_int)
                    )
                    writer.writerow(
                        [
                            img_path.name,
                            cid_int,
                            cname,
                            float(c),
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                        ]
                    )
            # Save visualized image (Ultralytics returns BGR array from plot())
            try:
                res = results[0]
                vis = res.plot()
                out_img = output_dir / img_path.name
                import cv2

                cv2.imwrite(str(out_img), vis)
            except (OSError, RuntimeError, AttributeError):
                continue
            
            # Log progress at intervals
            processed_count += 1
            if processed_count % progress_interval == 0 or processed_count == total_images:
                percentage = int((processed_count / total_images) * 100)
                logger.info(f"進度: {processed_count}/{total_images} 張圖片 ({percentage}%)")

    logger.info(f"✅ Batch inference complete. Results: {csv_path}")
    return output_dir
