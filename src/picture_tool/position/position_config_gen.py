import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, MutableMapping

try:
    from ultralytics import YOLO # type: ignore
except ImportError:
    YOLO = None

from picture_tool.utils.normalization import normalize_imgsz

def _resolve_sample_images(directory: Path, suffixes: set) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Position validation sample_dir not found: {directory}"
        )
    images = [
        p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in suffixes
    ]
    if not images:
        raise FileNotFoundError(f"No images available under {directory}")
    return images

class PositionConfigGenerator:
    @staticmethod
    def generate(config: MutableMapping[str, Any], run_dir: Path, logger: logging.Logger) -> Optional[Path]:
        """Automatically derive a position config from latest training results."""
        if YOLO is None:
            logger.info(
                "Skipping auto position config generation: ultralytics not available."
            )
            return None
            
        ycfg = config.get("yolo_training")
        if not isinstance(ycfg, MutableMapping):
            return None
        pos_cfg = ycfg.get("position_validation")
        if not isinstance(pos_cfg, MutableMapping):
            return None
        if not pos_cfg.get("enabled"):
            return None
        auto_generate = pos_cfg.get("auto_generate", True)
        if not auto_generate:
            logger.info("Auto position config generation disabled via config.")
            return None

        product = pos_cfg.get("product")
        area = pos_cfg.get("area")
        if not product or not area:
            logger.warning(
                "Auto position config generation skipped: product/area must be specified when position validation is enabled."
            )
            return None

        imgsz_value = pos_cfg.get("imgsz") or ycfg.get("imgsz") or 640
        imgsz_norm = normalize_imgsz(imgsz_value) or [640, 640]
        # normalize_imgsz guarantees list[int] of len 2 if not None
        imgsz_int = imgsz_norm[0]

        dataset_dir_val = ycfg.get("dataset_dir")
        if dataset_dir_val:
            dataset_dir = Path(str(dataset_dir_val))
        else:
             # Fallback if key missing, though it usually has defaults
             dataset_dir = Path("data/split") # Ideally use constant, but we are inside logic
             
        sample_dir_value = pos_cfg.get("sample_dir") or (dataset_dir / "val" / "images")
        sample_dir = Path(str(sample_dir_value)).resolve()
        
        # Hardcoded suffixes for now or pass in
        ImageSuffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        
        try:
            images = _resolve_sample_images(sample_dir, ImageSuffixes)
        except Exception as exc:
            logger.warning("Auto position config generation skipped: %s", exc)
            return None

        weights_path = run_dir / "weights" / "best.pt"
        if not weights_path.exists():
            weights_path = run_dir / "weights" / "last.pt"
        if not weights_path.exists():
            logger.warning(
                "Auto position config generation skipped: unable to locate weights under %s",
                run_dir,
            )
            return None

        try:
            # Import specific logic from position module
            from picture_tool.position.yolo_position_validator import (
                convert_results_to_detections,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Auto position config generation skipped: %s", exc)
            return None

        device_value = pos_cfg.get("device") or ycfg.get("device") or "cpu"
        conf_value = pos_cfg.get("conf")
        if conf_value is None:
            conf_value = 0.25
        try:
            model = YOLO(str(weights_path))
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Auto position config generation skipped: failed to load weights (%s)", exc
            )
            return None

        boxes_by_class: Dict[str, List[List[int]]] = {}
        for img_path in images:
            try:
                results = model(
                    str(img_path),
                    imgsz=imgsz_norm[0], # Model inference usually implies square or stride
                    device=str(device_value),
                    conf=float(conf_value),
                    verbose=False
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Auto position config: inference failed for %s (%s)", img_path.name, exc
                )
                continue
            for res in results:
                detections = convert_results_to_detections(res, imgsz_int)
                for det in detections:
                    cls = det.get("class")
                    bbox = det.get("bbox")
                    if not isinstance(cls, str):
                        cls = str(cls)
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    try:
                        box_vals = [int(round(float(v))) for v in bbox]
                    except (TypeError, ValueError):
                        continue
                    boxes_by_class.setdefault(cls, []).append(box_vals)

        if not boxes_by_class:
            logger.warning(
                "Auto position config generation skipped: no detections collected from sample images."
            )
            return None

        def aggregate(boxes: List[List[int]]) -> Dict[str, int]:
            x1 = min(b[0] for b in boxes)
            y1 = min(b[1] for b in boxes)
            x2 = max(b[2] for b in boxes)
            y2 = max(b[3] for b in boxes)
            return {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}

        expected_boxes = {cls: aggregate(bxs) for cls, bxs in boxes_by_class.items()}
        if not expected_boxes:
            logger.warning(
                "Auto position config generation skipped: expected boxes could not be computed."
            )
            return None

        tolerance_value = pos_cfg.get("tolerance_override")
        if tolerance_value is None:
            tolerance_value = (
                float(pos_cfg.get("tolerance", 0.0)) if "tolerance" in pos_cfg else 0.0
            )

        area_block: Dict[str, Any] = {
            "enabled": True,
            "mode": str(pos_cfg.get("mode", "bbox")),
            "tolerance": float(tolerance_value),
            "expected_boxes": expected_boxes,
        }
        area_block["imgsz"] = imgsz_norm[0]

        position_config = {str(product): {str(area): area_block}}
        out_path = (run_dir / "auto_position_config.yaml").resolve()
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(position_config, fh, allow_unicode=True, sort_keys=False)
        except Exception as exc:  # pragma: no cover
            logger.warning("Auto position config generation failed: %s", exc)
            return None

        previous_path = pos_cfg.get("config_path")
        if previous_path and previous_path != str(out_path):
            pos_cfg["previous_config_path"] = previous_path
        pos_cfg["config_path"] = str(out_path)
        if pos_cfg.get("config"):
            pos_cfg.pop("config", None)
        if not pos_cfg.get("sample_dir"):
            pos_cfg["sample_dir"] = str(sample_dir)
        if not pos_cfg.get("weights"):
            pos_cfg["weights"] = str(weights_path)
        logger.info("Auto-generated position config at %s", out_path)
        return out_path
