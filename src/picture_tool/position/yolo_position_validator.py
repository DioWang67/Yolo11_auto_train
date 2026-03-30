"""Utilities for optional YOLO position validation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
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
import os

try:
    if os.environ.get("PYTEST_IS_RUNNING") == "1":
        raise ImportError("Bypass ultralytics during pytest")
    from ultralytics import YOLO  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore


Number = Union[int, float]
BBox = Tuple[float, float, float, float]


@dataclass
class ExpectedBox:
    x1: float
    y1: float
    x2: float
    y2: float
    cx: Optional[float] = None
    cy: Optional[float] = None
    sigma_cx: float = 0.0
    sigma_cy: float = 0.0
    count: int = 0
    tolerance: Optional[float] = None  # per-class tolerance override (percent)

    def center(self) -> Tuple[float, float]:
        """Return the expected center (precomputed or derived from bbox)."""
        ecx = self.cx if self.cx is not None else (self.x1 + self.x2) / 2.0
        ecy = self.cy if self.cy is not None else (self.y1 + self.y2) / 2.0
        return ecx, ecy

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


def _base_class_name(key: str) -> str:
    """Strip ``#N`` instance index suffix from a class key."""
    idx = key.rfind("#")
    if idx > 0:
        suffix = key[idx + 1:]
        if suffix.isdigit():
            return key[:idx]
    return key


@dataclass
class PositionAreaConfig:
    enabled: bool = True
    mode: str = "center"
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
                        eb = ExpectedBox(
                            x1=float(box.get("x1", 0.0)),
                            y1=float(box.get("y1", 0.0)),
                            x2=float(box.get("x2", 0.0)),
                            y2=float(box.get("y2", 0.0)),
                        )
                        # Parse optional statistical / override fields
                        if "cx" in box:
                            eb.cx = float(box["cx"])
                        if "cy" in box:
                            eb.cy = float(box["cy"])
                        if "sigma_cx" in box:
                            eb.sigma_cx = float(box["sigma_cx"])
                        if "sigma_cy" in box:
                            eb.sigma_cy = float(box["sigma_cy"])
                        if "count" in box:
                            eb.count = int(box["count"])
                        if "tolerance" in box:
                            eb.tolerance = float(box["tolerance"])
                        boxes[class_name] = eb
                    except (TypeError, ValueError):
                        continue
            # Accept both legacy "bbox" and new "center" mode names
            mode = str(cfg.get("mode", "center"))
            product_map[str(area)] = PositionAreaConfig(
                enabled=bool(cfg.get("enabled", True)),
                mode=mode,
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
    scale, pad_x, pad_y = _letterbox_params(orig_w, orig_h, imgsz)
    x1, y1, x2, y2 = [float(v) for v in bbox]
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


@lru_cache(maxsize=128)
def _letterbox_params(
    orig_w: int, orig_h: int, imgsz: int
) -> Tuple[float, float, float]:
    scale = min(imgsz / float(orig_w), imgsz / float(orig_h))
    new_w = float(orig_w) * scale
    new_h = float(orig_h) * scale
    pad_x = (imgsz - new_w) / 2.0
    pad_y = (imgsz - new_h) / 2.0
    return scale, pad_x, pad_y


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


def _greedy_match(
    expected_entries: List[Tuple[str, ExpectedBox]],
    det_candidates: List[Dict[str, Any]],
    imgsz_int: int,
    default_tolerance_px: float,
) -> Tuple[
    List[Tuple[str, ExpectedBox, Dict[str, Any]]],  # matched
    List[str],                                        # unmatched expected keys
    List[Dict[str, Any]],                             # unmatched detections
]:
    """Greedy nearest-neighbor matching by Euclidean center distance.

    For each (expected, detection) pair, compute distance and greedily assign
    the closest pair first. O(N*M) which is fine for small N, M (< 20).
    """
    if not expected_entries or not det_candidates:
        unmatched_exp = [key for key, _ in expected_entries]
        return [], unmatched_exp, list(det_candidates)

    # Build all candidate pairs with distances
    pairs: List[Tuple[float, int, int]] = []
    for ei, (_, ebox) in enumerate(expected_entries):
        ecx, ecy = ebox.center()
        for di, det in enumerate(det_candidates):
            dcx = det.get("cx")
            dcy = det.get("cy")
            if dcx is None or dcy is None:
                continue
            dist = ((float(dcx) - ecx) ** 2 + (float(dcy) - ecy) ** 2) ** 0.5
            pairs.append((dist, ei, di))

    pairs.sort(key=lambda t: t[0])

    matched_exp: set = set()
    matched_det: set = set()
    matched: List[Tuple[str, ExpectedBox, Dict[str, Any]]] = []

    for _, ei, di in pairs:
        if ei in matched_exp or di in matched_det:
            continue
        matched_exp.add(ei)
        matched_det.add(di)
        key, ebox = expected_entries[ei]
        matched.append((key, ebox, det_candidates[di]))

    unmatched_exp_keys = [
        expected_entries[ei][0] for ei in range(len(expected_entries)) if ei not in matched_exp
    ]
    unmatched_dets = [
        det_candidates[di] for di in range(len(det_candidates)) if di not in matched_det
    ]
    return matched, unmatched_exp_keys, unmatched_dets


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
    if area_cfg.mode not in {"bbox", "box", "center"}:
        raise ValueError(f"Unsupported position validation mode: {area_cfg.mode}")
    imgsz_int = _imgsz_value(imgsz)
    tol_percent = (
        float(tolerance_override)
        if tolerance_override is not None
        else float(area_cfg.tolerance)
    )
    default_tolerance_px = imgsz_int * (tol_percent / 100.0)

    # Group expected boxes by base class name for multi-instance matching
    from collections import defaultdict
    base_class_groups: Dict[str, List[Tuple[str, ExpectedBox]]] = defaultdict(list)
    for key, ebox in area_cfg.expected_boxes.items():
        base = _base_class_name(key)
        base_class_groups[base].append((key, ebox))

    # Group detections by class name
    det_by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for det in detections:
        cls = str(det.get("class", ""))
        det_by_class[cls].append(det)

    expected_base_names = set(base_class_groups.keys())
    results: List[Dict[str, Any]] = []
    missing: List[str] = []
    unexpected: List[str] = []
    wrong: List[str] = []
    unknown: List[str] = []

    for base_class, expected_entries in base_class_groups.items():
        candidates = det_by_class.pop(base_class, [])

        if len(expected_entries) == 1 and not any("#" in k for k, _ in expected_entries):
            # Single-instance path: keep backward-compatible behavior
            key, ebox = expected_entries[0]
            det = _select_best_detection(candidates, base_class) if candidates else None
            if det is None:
                missing.append(key)
                results.append({
                    "class": key, "status": "MISSING",
                    "expected_box": list(ebox.as_tuple()),
                })
                continue
            entry = {
                "class": key,
                "class_id": det.get("class_id"),
                "confidence": det.get("confidence"),
                "bbox": det.get("bbox"),
                "cx": det.get("cx"), "cy": det.get("cy"),
            }
            cx, cy = det.get("cx"), det.get("cy")
            if cx is None or cy is None:
                entry["status"] = "UNKNOWN"
                unknown.append(key)
            else:
                class_tol_px = (
                    imgsz_int * (ebox.tolerance / 100.0)
                    if ebox.tolerance is not None
                    else default_tolerance_px
                )
                if ebox.contains(float(cx), float(cy), class_tol_px, float(imgsz_int)):
                    entry["status"] = "CORRECT"
                else:
                    entry["status"] = "WRONG"
                    entry["expected_box"] = list(ebox.as_tuple())
                    entry["allowed_box"] = list(ebox.expand(class_tol_px, float(imgsz_int)))
                    wrong.append(key)
            results.append(entry)
        else:
            # Multi-instance path: greedy matching
            matched, unmatched_keys, unmatched_dets = _greedy_match(
                expected_entries, candidates, imgsz_int, default_tolerance_px
            )
            for key in unmatched_keys:
                ebox_miss = dict(area_cfg.expected_boxes).get(key)
                missing.append(key)
                results.append({
                    "class": key, "status": "MISSING",
                    "expected_box": list(ebox_miss.as_tuple()) if ebox_miss else [],
                })

            for key, ebox, det in matched:
                entry = {
                    "class": key,
                    "class_id": det.get("class_id"),
                    "confidence": det.get("confidence"),
                    "bbox": det.get("bbox"),
                    "cx": det.get("cx"), "cy": det.get("cy"),
                }
                cx, cy = det.get("cx"), det.get("cy")
                if cx is None or cy is None:
                    entry["status"] = "UNKNOWN"
                    unknown.append(key)
                else:
                    class_tol_px = (
                        imgsz_int * (ebox.tolerance / 100.0)
                        if ebox.tolerance is not None
                        else default_tolerance_px
                    )
                    if ebox.contains(float(cx), float(cy), class_tol_px, float(imgsz_int)):
                        entry["status"] = "CORRECT"
                    else:
                        entry["status"] = "WRONG"
                        entry["expected_box"] = list(ebox.as_tuple())
                        entry["allowed_box"] = list(ebox.expand(class_tol_px, float(imgsz_int)))
                        wrong.append(key)
                results.append(entry)

            # Unmatched detections from this base class are not unexpected —
            # they are extra instances beyond the expected count
            for det in unmatched_dets:
                entry = {
                    "class": str(det.get("class")),
                    "class_id": det.get("class_id"),
                    "confidence": det.get("confidence"),
                    "bbox": det.get("bbox"),
                    "cx": det.get("cx"), "cy": det.get("cy"),
                    "status": "UNEXPECTED",
                }
                results.append(entry)
                unexpected.append(str(det.get("class")))

    # Any remaining detection classes not in expected_base_names are unexpected
    for cls, dets in det_by_class.items():
        if cls in expected_base_names:
            continue
        for det in dets:
            entry = {
                "class": cls,
                "class_id": det.get("class_id"),
                "confidence": det.get("confidence"),
                "bbox": det.get("bbox"),
                "cx": det.get("cx"), "cy": det.get("cy"),
                "status": "UNEXPECTED",
            }
            results.append(entry)
            unexpected.append(cls)

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


def _resolve_sample_images(directory: Path, suffixes: Optional[set] = None) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Position validation sample_dir not found: {directory}"
        )
    if suffixes is None:
        suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    images = [
        p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in suffixes
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
