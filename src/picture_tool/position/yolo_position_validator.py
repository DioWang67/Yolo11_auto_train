"""Utilities for optional YOLO position validation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import yaml

try:
    from ultralytics import YOLO  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


Number = Union[int, float]
BBox = Tuple[float, float, float, float]


@dataclass
class ExpectedBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def as_tuple(self) -> BBox:
        return (self.x1, self.y1, self.x2, self.y2)

    def expand(self, tolerance_px: float, limit: float) -> BBox:
        return (
            max(0.0, self.x1 - tolerance_px),
            max(0.0, self.y1 - tolerance_px),
            min(limit, self.x2 + tolerance_px),
            min(limit, self.y2 + tolerance_px),
        )

    def contains(self, cx: float, cy: float, tolerance_px: float, limit: float) -> bool:
        x1, y1, x2, y2 = self.expand(tolerance_px, limit)
        return x1 <= cx <= x2 and y1 <= cy <= y2


@dataclass
class PositionAreaConfig:
    enabled: bool = True
    mode: str = "bbox"
    tolerance: float = 0.0
    expected_boxes: Mapping[str, ExpectedBox] = field(default_factory=dict)


PositionConfig = Dict[str, Dict[str, PositionAreaConfig]]


@dataclass
class PositionValidationResult:
    product: str
    area: str
    tolerance_percent: float
    imgsz: int
    status: str
    results: List[Dict[str, Any]]
    missing: List[str]
    unexpected: List[str]
    wrong: List[str]
    unknown: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "product": self.product,
            "area": self.area,
            "tolerance_percent": self.tolerance_percent,
            "imgsz": self.imgsz,
            "status": self.status,
            "results": self.results,
            "missing": self.missing,
            "unexpected": self.unexpected,
            "wrong": self.wrong,
            "unknown": self.unknown,
        }


def load_position_config(
    source: Optional[Union[str, Path, Mapping[str, Any]]],
) -> PositionConfig:
    if source is None:
        return {}
    if isinstance(source, (str, Path)):
        data = yaml.safe_load(Path(source).read_text(encoding="utf-8")) or {}
    else:
        data = source
    if not isinstance(data, Mapping):
        raise TypeError("position_config must be a mapping")
    parsed: PositionConfig = {}
    for product, areas in data.items():
        if not isinstance(areas, Mapping):
            continue
        product_map: Dict[str, PositionAreaConfig] = {}
        for area, cfg in areas.items():
            if not isinstance(cfg, Mapping):
                continue
            expected_raw = cfg.get("expected_boxes", {})
            boxes: Dict[str, ExpectedBox] = {}
            if isinstance(expected_raw, Mapping):
                for class_name, box in expected_raw.items():
                    if not isinstance(box, Mapping):
                        continue
                    try:
                        boxes[class_name] = ExpectedBox(
                            float(box.get("x1", 0.0)),
                            float(box.get("y1", 0.0)),
                            float(box.get("x2", 0.0)),
                            float(box.get("y2", 0.0)),
                        )
                    except (TypeError, ValueError):
                        continue
            product_map[str(area)] = PositionAreaConfig(
                enabled=bool(cfg.get("enabled", True)),
                mode=str(cfg.get("mode", "bbox")),
                tolerance=float(cfg.get("tolerance", 0.0)),
                expected_boxes=boxes,
            )
        if product_map:
            parsed[str(product)] = product_map
    return parsed


def _imgsz_value(imgsz: Any) -> int:
    if imgsz in (None, "", 0):
        raise ValueError("imgsz must be a positive integer or sequence of integers")
    if isinstance(imgsz, Sequence) and not isinstance(imgsz, (str, bytes, bytearray)):
        for item in imgsz:
            try:
                return int(item)
            except (TypeError, ValueError):
                continue
        raise ValueError("imgsz sequence does not contain numeric values")
    return int(imgsz)


def _letterbox_transform(
    bbox: Sequence[Number], orig_w: int, orig_h: int, imgsz: int
) -> BBox:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    scale = min(imgsz / float(orig_w), imgsz / float(orig_h))
    new_w = float(orig_w) * scale
    new_h = float(orig_h) * scale
    pad_x = (imgsz - new_w) / 2.0
    pad_y = (imgsz - new_h) / 2.0
    lx1 = x1 * scale + pad_x
    ly1 = y1 * scale + pad_y
    lx2 = x2 * scale + pad_x
    ly2 = y2 * scale + pad_y
    return (
        max(0.0, min(float(imgsz), lx1)),
        max(0.0, min(float(imgsz), ly1)),
        max(0.0, min(float(imgsz), lx2)),
        max(0.0, min(float(imgsz), ly2)),
    )


def _resolve_class_name(names: Any, class_id: int) -> str:
    if isinstance(names, Mapping):
        name = names.get(class_id)
        if name is not None:
            return str(name)
    if isinstance(names, Sequence) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def convert_results_to_detections(
    result: Any, imgsz: Union[int, Sequence[int]]
) -> List[Dict[str, Any]]:
    imgsz_int = _imgsz_value(imgsz)
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = getattr(boxes, "xyxy", None)
    confs = getattr(boxes, "conf", None)
    clss = getattr(boxes, "cls", None)
    if xyxy is None or confs is None or clss is None:
        return []
    try:
        xyxy_list = xyxy.cpu().tolist()
        conf_list = confs.cpu().tolist()
        cls_list = [int(c) for c in clss.cpu().tolist()]
    except AttributeError:
        xyxy_list = xyxy.tolist()
        conf_list = confs.tolist()
        cls_list = [int(c) for c in clss.tolist()]
    orig_h, orig_w = getattr(result, "orig_shape", (imgsz_int, imgsz_int))
    detections: List[Dict[str, Any]] = []
    for bbox, conf, cls_id in zip(xyxy_list, conf_list, cls_list):
        lb_box = _letterbox_transform(bbox, int(orig_w), int(orig_h), imgsz_int)
        cx = (lb_box[0] + lb_box[2]) / 2.0
        cy = (lb_box[1] + lb_box[3]) / 2.0
        detections.append(
            {
                "class": _resolve_class_name(getattr(result, "names", {}), cls_id),
                "class_id": int(cls_id),
                "confidence": float(conf),
                "bbox": [int(round(v)) for v in lb_box],
                "bbox_letterbox": [float(v) for v in lb_box],
                "bbox_original": [float(v) for v in bbox],
                "image_width": int(orig_w),
                "image_height": int(orig_h),
                "cx": float(cx),
                "cy": float(cy),
            }
        )
    return detections


def _select_best_detection(
    detections: Iterable[Dict[str, Any]], class_name: str
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for det in detections:
        if det.get("class") != class_name:
            continue
        if best is None or float(det.get("confidence", 0.0)) > float(
            best.get("confidence", 0.0)
        ):
            best = det
    return best


def validate_detections_against_area(
    detections: Sequence[Dict[str, Any]],
    area_cfg: PositionAreaConfig,
    imgsz: Union[int, Sequence[int]],
    product: str,
    area: str,
    tolerance_override: Optional[float] = None,
) -> PositionValidationResult:
    if not area_cfg.enabled:
        return PositionValidationResult(
            product=product,
            area=area,
            tolerance_percent=float(tolerance_override)
            if tolerance_override is not None
            else area_cfg.tolerance,
            imgsz=_imgsz_value(imgsz),
            status="SKIPPED",
            results=[],
            missing=[],
            unexpected=[],
            wrong=[],
            unknown=[],
        )
    if area_cfg.mode not in {"bbox", "box"}:
        raise ValueError(f"Unsupported position validation mode: {area_cfg.mode}")
    imgsz_int = _imgsz_value(imgsz)
    tol_percent = (
        float(tolerance_override)
        if tolerance_override is not None
        else float(area_cfg.tolerance)
    )
    tolerance_px = imgsz_int * (tol_percent / 100.0)
    expected_items = list(area_cfg.expected_boxes.items())
    expected_names = {name for name, _ in expected_items}
    results: List[Dict[str, Any]] = []
    missing: List[str] = []
    unexpected: List[str] = []
    wrong: List[str] = []
    unknown: List[str] = []

    for class_name, expected_box in expected_items:
        det = _select_best_detection(detections, class_name)
        if det is None:
            missing.append(class_name)
            results.append(
                {
                    "class": class_name,
                    "status": "MISSING",
                    "expected_box": list(expected_box.as_tuple()),
                }
            )
            continue
        entry = {
            "class": class_name,
            "class_id": det.get("class_id"),
            "confidence": det.get("confidence"),
            "bbox": det.get("bbox"),
            "cx": det.get("cx"),
            "cy": det.get("cy"),
        }
        cx = det.get("cx")
        cy = det.get("cy")
        if cx is None or cy is None:
            entry["status"] = "UNKNOWN"
            unknown.append(class_name)
        else:
            if expected_box.contains(
                float(cx), float(cy), tolerance_px, float(imgsz_int)
            ):
                entry["status"] = "CORRECT"
            else:
                entry["status"] = "WRONG"
                entry["expected_box"] = list(expected_box.as_tuple())
                entry["allowed_box"] = list(
                    expected_box.expand(tolerance_px, float(imgsz_int))
                )
                wrong.append(class_name)
        results.append(entry)

    for det in detections:
        class_name = str(det.get("class"))
        if class_name in expected_names:
            continue
        entry = {
            "class": class_name,
            "class_id": det.get("class_id"),
            "confidence": det.get("confidence"),
            "bbox": det.get("bbox"),
            "cx": det.get("cx"),
            "cy": det.get("cy"),
            "status": "UNEXPECTED",
        }
        results.append(entry)
        unexpected.append(class_name)

    status = "PASS"
    if missing or wrong or unexpected or unknown:
        status = "FAIL"

    return PositionValidationResult(
        product=product,
        area=area,
        tolerance_percent=tol_percent,
        imgsz=imgsz_int,
        status=status,
        results=results,
        missing=missing,
        unexpected=unexpected,
        wrong=wrong,
        unknown=unknown,
    )


def _resolve_sample_images(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Position validation sample_dir not found: {directory}"
        )
    images = [
        p
        for p in sorted(directory.iterdir())
        if p.is_file()
        and p.suffix.lower()
        in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    ]
    if not images:
        raise FileNotFoundError(f"No images available under {directory}")
    return images


def _resolve_weights(run_dir: Path, override: Optional[Union[str, Path]]) -> Path:
    if override:
        candidate = Path(str(override)).expanduser().resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Position validation weights not found: {candidate}")
    weights_dir = run_dir / "weights"
    for name in ("best.pt", "last.pt"):
        candidate = weights_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate trained weights under {weights_dir}")


def run_position_validation(
    config: MutableMapping[str, Any],
    run_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    logger = logger or logging.getLogger(__name__)
    ycfg = config.get("yolo_training", {}) if isinstance(config, Mapping) else {}
    pv_cfg = ycfg.get("position_validation", {}) if isinstance(ycfg, Mapping) else {}
    if not pv_cfg or not pv_cfg.get("enabled"):
        return None
    if YOLO is None:  # pragma: no cover
        raise RuntimeError(
            "ultralytics is required for position validation but is not available"
        )
    product = pv_cfg.get("product")
    area = pv_cfg.get("area")
    if not product or not area:
        raise ValueError(
            "position_validation requires product and area to be specified"
        )
    imgsz_value = pv_cfg.get("imgsz")
    if imgsz_value in (None, "", 0):
        imgsz_value = ycfg.get("imgsz", 640)
    if imgsz_value in (None, "", 0):
        imgsz_value = 640
    imgsz_int = _imgsz_value(imgsz_value)
    dataset_dir = Path(str(ycfg.get("dataset_dir", "./data/split")))
    default_sample_dir = dataset_dir / "val" / "images"
    sample_dir_value = pv_cfg.get("sample_dir")
    if not sample_dir_value:
        sample_dir_value = default_sample_dir
    sample_dir = Path(str(sample_dir_value))
    images = _resolve_sample_images(sample_dir)
    config_source: Optional[Union[str, Path, Mapping[str, Any]]] = pv_cfg.get("config")
    if not config_source:
        config_source = pv_cfg.get("config_path")
    if not config_source:
        config_source = (
            config.get("position_config") if isinstance(config, Mapping) else None
        )
    if not config_source:
        raise ValueError("position_validation requires a position config source")
    position_config = load_position_config(config_source)
    product_cfg = position_config.get(str(product))
    if not product_cfg:
        raise KeyError(f"Product '{product}' not defined in position_config")
    area_cfg = product_cfg.get(str(area))
    if not area_cfg:
        raise KeyError(
            f"Area '{area}' not defined in position_config for product '{product}'"
        )
    tolerance_override = pv_cfg.get("tolerance_override")
    output_dir_value = pv_cfg.get("output_dir")
    if not output_dir_value:
        output_dir_value = run_dir / "position_validation"
    output_dir = Path(str(output_dir_value))
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = _resolve_weights(run_dir, pv_cfg.get("weights") or None)
    device_value = pv_cfg.get("device")
    if not device_value:
        device_value = ycfg.get("device", "cpu")
    device = str(device_value)
    conf_value = pv_cfg.get("conf")
    if conf_value is None:
        conf_value = 0.25
    conf = float(conf_value)
    logger.info(
        "Running position validation | product=%s area=%s imgsz=%s device=%s conf=%.3f samples=%s",
        product,
        area,
        imgsz_int,
        device,
        conf,
        sample_dir,
    )
    model = YOLO(str(weights_path))
    records: List[Dict[str, Any]] = []
    status_counter: Dict[str, int] = {"PASS": 0, "FAIL": 0, "SKIPPED": 0}
    for img_path in images:
        results = model(str(img_path), imgsz=imgsz_int, device=device, conf=conf)
        detections: List[Dict[str, Any]] = []
        for res in results:
            detections.extend(convert_results_to_detections(res, imgsz_int))
        validation = validate_detections_against_area(
            detections,
            area_cfg,
            imgsz_int,
            str(product),
            str(area),
            tolerance_override=tolerance_override,
        )
        status_counter[validation.status] = status_counter.get(validation.status, 0) + 1
        records.append(
            {
                "image": img_path.name,
                "path": str(img_path),
                "detections": detections,
                "validation": validation.as_dict(),
            }
        )
    summary = {
        "product": str(product),
        "area": str(area),
        "imgsz": imgsz_int,
        "samples": len(images),
        "status_counts": status_counter,
    }
    out_path = output_dir / "position_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "records": records,
                "config_source": str(config_source),
                "tolerance_override": tolerance_override,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Position validation results saved to %s", out_path)
    return out_path
