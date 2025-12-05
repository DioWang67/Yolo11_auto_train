import csv
import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

try:
    from ultralytics import YOLO  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore

from picture_tool.eval.yolo_evaluator import _resolve_weights  # reuse weight resolution


def run_batch_inference(config: dict, logger: Optional[logging.Logger] = None) -> Path:
    logger = logger or logging.getLogger(__name__)

    icfg = config.get("batch_inference", {})
    input_dir = Path(str(icfg.get("input_dir", "./data/raw/images"))).resolve()
    output_dir = Path(str(icfg.get("output_dir", "./reports/infer"))).resolve()
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
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["file", "class_id", "class_name", "conf", "x1", "y1", "x2", "y2"]
        )
        for img_path in tqdm(images, desc="Batch inference", unit="img"):
            results = model(str(img_path), imgsz=imgsz, device=device, conf=conf)
            for res in results:
                names = res.names
                if res.boxes is None:
                    logger.info(f"{img_path.name}: 0 detections (no boxes)")
                    continue
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy().tolist()
                clss = res.boxes.cls.cpu().numpy().tolist()
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
            except Exception:
                pass

    logger.info(f"Batch inference completed. CSV: {csv_path}")
    return csv_path
