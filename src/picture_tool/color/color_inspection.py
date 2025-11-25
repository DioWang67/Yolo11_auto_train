#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM-assisted color sampling workflow.

This module replaces the previous LED-specific heuristics with an interactive
Segment Anything (SAM) powered annotator. Users can open a target directory,
draw bounding boxes (or click) on each image, and the resulting mask is
summarised into HSV/LAB ranges per color category; outputs are stored in JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

try:  # pragma: no cover - optional dependency resolution
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency resolution
    from segment_anything import SamPredictor, sam_model_registry  # type: ignore
except ImportError:  # pragma: no cover
    SamPredictor = None  # type: ignore
    sam_model_registry = {}  # type: ignore


SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif")


# ---------------------------------------------------------------------------
# Legacy helpers kept for backwards compatibility with CLI/tests
# ---------------------------------------------------------------------------


def safe_percentile(
    samples: np.ndarray | Sequence[float],
    q: float,
    default: float = float("nan"),
) -> float:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return float(default)
    return float(np.percentile(arr, q))


def safe_ratio(
    numerator: np.ndarray | Sequence[float] | float,
    denominator: float,
    default: float = float("nan"),
) -> float:
    if abs(float(denominator)) < 1e-12:
        return float(default)
    num_arr = np.asarray(numerator, dtype=np.float64)
    if num_arr.size == 0:
        return float(default)
    return float(num_arr.mean() / float(denominator))


@dataclass
class ColorPalette:
    """Utility structure used by legacy command-line tooling."""

    names: Tuple[str, ...]
    aliases: Dict[str, str]
    hue_ranges: Dict[str, Tuple[float, float]]

    def __init__(
        self,
        names: Sequence[str],
        aliases: Optional[Dict[str, str]] = None,
        hue_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        unique: List[str] = []
        seen = set()
        for name in names:
            normalized = (name or "").strip()
            if not normalized or normalized in seen:
                continue
            unique.append(normalized)
            seen.add(normalized)
        self.names = tuple(unique)
        canonical_aliases: Dict[str, str] = {}
        for name in self.names:
            canonical_aliases[name.lower()] = name
        if aliases:
            for alias, target in aliases.items():
                if target:
                    canonical_aliases[alias.lower()] = target
        self.aliases = canonical_aliases
        self.hue_ranges = hue_ranges.copy() if hue_ranges else {}


@dataclass
class EnhancedColorModel:
    avg_color_hist: Sequence[float]
    avg_rotation_hist: Sequence[float]
    hist_thr: float
    rotation_hist_thr: float
    mean_v_mu: float
    mean_v_std: float
    std_v_mu: float
    std_v_std: float
    uniformity_mu: float
    uniformity_std: float
    area_ratio_mu: float
    area_ratio_std: float
    hole_ratio_mu: float
    hole_ratio_std: float
    aspect_ratio_mu: float
    aspect_ratio_std: float
    compactness_mu: float
    compactness_std: float
    regularity_mu: float
    regularity_std: float
    texture_energy_mu: float
    texture_energy_std: float
    samples: int
    avg_confidence: float
    last_updated: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class EnhancedReferenceModel:
    version: int
    config: Dict[str, object]
    colors: Dict[str, EnhancedColorModel] = field(default_factory=dict)
    creation_time: str = ""
    total_samples: int = 0

    @classmethod
    def from_json(cls, path: str | Path) -> "EnhancedReferenceModel":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle) or {}
        color_payload = payload.get("colors", {}) or {}
        colors = {
            name: EnhancedColorModel(**values)
            for name, values in color_payload.items()
            if isinstance(values, dict)
        }
        total_samples = payload.get("total_samples")
        if total_samples is None:
            total_samples = sum(model.samples for model in colors.values())
        return cls(
            version=int(payload.get("version", 1)),
            config=payload.get("config", {}),
            colors=colors,
            creation_time=payload.get("creation_time", ""),
            total_samples=int(total_samples),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.version,
            "config": self.config,
            "creation_time": self.creation_time,
            "total_samples": self.total_samples,
            "colors": {name: model.to_dict() for name, model in self.colors.items()},
        }


def cmd_info(args) -> None:
    """CLI helper mirroring the legacy color inspection behavior."""
    model = EnhancedReferenceModel.from_json(args.model)
    summary = {
        "version": model.version,
        "total_samples": model.total_samples,
        "colors": sorted(model.colors.keys()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_analyze(args) -> None:
    """Dispatch analysis call to the appropriate helper."""
    model = EnhancedReferenceModel.from_json(args.model)
    if getattr(args, "image", None):
        _analyze_single(args, model)
    else:
        _analyze_directory(args, model)


def _analyze_directory(args, model) -> None:  # pragma: no cover - placeholder
    raise NotImplementedError("Directory analysis is not implemented in this build.")


def _analyze_single(args, model) -> None:  # pragma: no cover - placeholder
    raise NotImplementedError("Single-image analysis is not implemented in this build.")


# ---------------------------------------------------------------------------
# Configuration / dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SamSettings:
    checkpoint: Path
    model_type: str = "vit_b"
    device: str = "auto"

    def resolved_device(self) -> str:
        if self.device and self.device.lower() != "auto":
            return self.device
        if torch is None:
            return "cpu"
        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover - CUDA env issues
            return "cpu"


@dataclass
class SessionConfig:
    input_dir: Path
    output_json: Path
    colors: List[str]
    sam: SamSettings
    sam_infer_max_side: int = 4096
    image_suffixes: Tuple[str, ...] = SUPPORTED_FORMATS

    @staticmethod
    def from_namespace(ns) -> "SessionConfig":
        sam_cfg = getattr(ns, "sam", None) or {}
        checkpoint = Path(sam_cfg.get("checkpoint", "./models/sam/sam_vit_b.pth"))
        model_type = sam_cfg.get("model_type", "vit_b")
        device = sam_cfg.get("device", "auto")
        max_side = (
            getattr(ns, "max_side", None)
            or sam_cfg.get("max_side")
            or 2048
        )
        colors = list(getattr(ns, "colors", []) or ["Default"])
        return SessionConfig(
            input_dir=Path(getattr(ns, "input_dir", "./data/led_qc/samples")),
            output_json=Path(getattr(ns, "output_json", "./reports/led_qc/color_stats.json")),
            colors=colors,
            sam=SamSettings(checkpoint=checkpoint, model_type=model_type, device=device),
            sam_infer_max_side=int(max_side),
        )


# ---------------------------------------------------------------------------
# Aggregation utilities
# ---------------------------------------------------------------------------


def _ensure_package_available() -> None:
    if SamPredictor is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "segment-anything is not installed. Please pip install segment-anything."
        )
    if torch is None:  # pragma: no cover - runtime guard
        raise RuntimeError("PyTorch is required for SAM. Please install torch first.")


def _to_numpy(arr: Iterable[float]) -> np.ndarray:
    return np.asarray(list(arr), dtype=np.float64)


@dataclass
class RunningStats:
    count: int = 0
    hsv_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    lab_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    hsv_p10_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    hsv_p90_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    lab_p10_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    lab_p90_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    coverage_sum: float = 0.0
    hsv_min: np.ndarray = field(
        default_factory=lambda: np.full(3, np.inf, dtype=np.float64)
    )
    hsv_max: np.ndarray = field(
        default_factory=lambda: np.full(3, -np.inf, dtype=np.float64)
    )
    lab_min: np.ndarray = field(
        default_factory=lambda: np.full(3, np.inf, dtype=np.float64)
    )
    lab_max: np.ndarray = field(
        default_factory=lambda: np.full(3, -np.inf, dtype=np.float64)
    )

    def add(
        self,
        hsv_mean,
        hsv_min,
        hsv_max,
        lab_mean,
        lab_min,
        lab_max,
        hsv_p10,
        hsv_p90,
        lab_p10,
        lab_p90,
        coverage: float,
    ) -> None:
        hsv_mean_arr = _to_numpy(hsv_mean)
        lab_mean_arr = _to_numpy(lab_mean)
        self.count += 1
        self.hsv_sum += hsv_mean_arr
        self.lab_sum += lab_mean_arr
        self.hsv_p10_sum += _to_numpy(hsv_p10)
        self.hsv_p90_sum += _to_numpy(hsv_p90)
        self.lab_p10_sum += _to_numpy(lab_p10)
        self.lab_p90_sum += _to_numpy(lab_p90)
        self.coverage_sum += float(coverage)
        self.hsv_min = np.minimum(self.hsv_min, _to_numpy(hsv_min))
        self.hsv_max = np.maximum(self.hsv_max, _to_numpy(hsv_max))
        self.lab_min = np.minimum(self.lab_min, _to_numpy(lab_min))
        self.lab_max = np.maximum(self.lab_max, _to_numpy(lab_max))

    def as_dict(self) -> Dict[str, object]:
        if self.count == 0:
            raise ValueError("No samples recorded")  # pragma: no cover
        hsv_mean = (self.hsv_sum / self.count).tolist()
        lab_mean = (self.lab_sum / self.count).tolist()
        hsv_p10 = (self.hsv_p10_sum / self.count).tolist()
        hsv_p90 = (self.hsv_p90_sum / self.count).tolist()
        lab_p10 = (self.lab_p10_sum / self.count).tolist()
        lab_p90 = (self.lab_p90_sum / self.count).tolist()
        coverage_mean = self.coverage_sum / self.count
        return {
            "count": self.count,
            "hsv_mean": hsv_mean,
            "hsv_min": self.hsv_min.tolist(),
            "hsv_max": self.hsv_max.tolist(),
            "lab_mean": lab_mean,
            "lab_min": self.lab_min.tolist(),
            "lab_max": self.lab_max.tolist(),
            "hsv_p10": hsv_p10,
            "hsv_p90": hsv_p90,
            "lab_p10": lab_p10,
            "lab_p90": lab_p90,
            "coverage_mean": coverage_mean,
        }


class ColorStatsRecorder:
    """Aggregate per-color statistics and expose JSON friendly payloads."""

    def __init__(self) -> None:
        self.summary: Dict[str, RunningStats] = {}

    def record(
        self,
        image_path: Path,
        color_name: str,
        hsv_mean,
        hsv_min,
        hsv_max,
        lab_mean,
        lab_min,
        lab_max,
        hsv_p10,
        hsv_p90,
        lab_p10,
        lab_p90,
        coverage: float,
    ) -> None:
        if color_name not in self.summary:
            self.summary[color_name] = RunningStats()
        self.summary[color_name].add(
            hsv_mean,
            hsv_min,
            hsv_max,
            lab_mean,
            lab_min,
            lab_max,
            hsv_p10,
            hsv_p90,
            lab_p10,
            lab_p90,
            coverage,
        )

    def to_json(self) -> Dict[str, object]:
        return {"summary": {color: stats.as_dict() for color, stats in self.summary.items()}}


# ---------------------------------------------------------------------------
# SAM predictor wrapper
# ---------------------------------------------------------------------------


class SamPredictorWrapper:
    """Thin wrapper around Segment Anything's predictor."""

    def __init__(self, cfg: SamSettings) -> None:
        _ensure_package_available()
        registry = sam_model_registry.get(cfg.model_type)
        if registry is None:
            raise ValueError(
                f"Unknown SAM model_type '{cfg.model_type}'. "
                f"Available: {', '.join(sam_model_registry.keys())}"
            )
        checkpoint = Path(cfg.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
        logging.info("Loading SAM model (%s) from %s", cfg.model_type, checkpoint)
        sam_model = registry(checkpoint=None)
        map_location = "cpu"
        try:
            state_dict = torch.load(  # type: ignore[arg-type]
                str(checkpoint), map_location=map_location, weights_only=True
            )
        except TypeError:
            state_dict = torch.load(str(checkpoint), map_location=map_location)  # type: ignore[arg-type]
        sam_model.load_state_dict(state_dict)
        device = cfg.resolved_device()
        sam_model.to(device)
        self.predictor = SamPredictor(sam_model)
        self.device = device
        self._last_image_id: Optional[int] = None

    def _ensure_image(self, image_bgr: np.ndarray) -> None:
        img_id = id(image_bgr)
        if img_id == self._last_image_id:
            return
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        self._last_image_id = img_id

    def predict(
        self,
        image_bgr: np.ndarray,
        *,
        box: Optional[Tuple[int, int, int, int]] = None,
        point_coords: Optional[List[Tuple[float, float]]] = None,
        point_labels: Optional[List[int]] = None,
    ) -> np.ndarray:
        self._ensure_image(image_bgr)
        box_arr = np.array(box, dtype=np.float32) if box else None
        pts = (
            np.array(point_coords, dtype=np.float32)
            if point_coords and len(point_coords) > 0
            else None
        )
        labels = (
            np.array(point_labels, dtype=np.int32)
            if point_labels and len(point_labels) > 0
            else None
        )
        masks, scores, _ = self.predictor.predict(
            point_coords=pts,
            point_labels=labels,
            box=box_arr,
            multimask_output=True,
        )
        idx = self._choose_mask_index(masks, scores, pts, labels, box_arr)
        mask = masks[idx].astype(np.uint8)
        return mask

    def _choose_mask_index(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        pts: Optional[np.ndarray],
        labels: Optional[np.ndarray],
        box: Optional[np.ndarray],
    ) -> int:
        best_idx = 0
        best_metric = -float("inf")
        # Heuristic scoring so that thin / dark structures (like wires) win over
        # large PCB background regions returned by SAM's default ranking.
        for i, mask in enumerate(masks):
            metric = float(scores[i])
            metric += self._score_points(mask, pts, labels)
            metric += self._score_box(mask, box)
            metric += self._score_area(mask)
            if metric > best_metric:
                best_metric = metric
                best_idx = i
        return best_idx

    @staticmethod
    def _score_points(
        mask: np.ndarray,
        pts: Optional[np.ndarray],
        labels: Optional[np.ndarray],
    ) -> float:
        if pts is None or labels is None or len(pts) == 0:
            return 0.0
        h, w = mask.shape[-2:]
        mask_vals = mask.astype(bool)
        pos_score = 0.0
        neg_score = 0.0
        for (x, y), lbl in zip(pts, labels):
            xi = int(np.clip(round(float(x)), 0, w - 1))
            yi = int(np.clip(round(float(y)), 0, h - 1))
            inside = bool(mask_vals[yi, xi])
            if lbl == 1:
                pos_score += 1.0 if inside else -0.8
            else:
                neg_score += 1.0 if inside else 0.0
        return pos_score * 0.35 - neg_score * 0.45

    @staticmethod
    def _score_box(mask: np.ndarray, box: Optional[np.ndarray]) -> float:
        if box is None:
            return 0.0
        x0, y0, x1, y1 = [int(round(float(v))) for v in box]
        h, w = mask.shape[-2:]
        x0 = int(np.clip(x0, 0, w - 1))
        x1 = int(np.clip(x1, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        crop = mask[y0:y1, x0:x1]
        inside = float(np.count_nonzero(crop))
        total = float(np.count_nonzero(mask)) + 1e-6
        inside_ratio = inside / total
        coverage = inside / float((x1 - x0) * (y1 - y0))
        return inside_ratio * 0.4 + min(coverage, 1.0) * 0.25

    @staticmethod
    def _score_area(mask: np.ndarray) -> float:
        area = float(np.count_nonzero(mask))
        if area <= 0:
            return -1.0
        total = mask.size
        ratio = area / float(total)
        penalty = 0.0
        if ratio > 0.4:
            penalty -= (ratio - 0.4) * 2.5
        if ratio < 0.0005:
            penalty -= 0.5
        return penalty


# ---------------------------------------------------------------------------
# Image statistics helpers
# ---------------------------------------------------------------------------


def _compute_channel_stats(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    masked = image[mask > 0]
    if masked.size == 0:
        raise ValueError("Empty mask; no pixels selected.")
    mean = masked.mean(axis=0)
    min_vals = masked.min(axis=0)
    max_vals = masked.max(axis=0)
    return mean, min_vals, max_vals


def compute_hsv_lab_stats(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    hsv_image: Optional[np.ndarray] = None,
    lab_image: Optional[np.ndarray] = None,
) -> Dict[str, List[float] | float]:
    hsv = hsv_image if hsv_image is not None else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = lab_image if lab_image is not None else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    hsv_mean, hsv_min, hsv_max = _compute_channel_stats(hsv, mask)
    lab_mean, lab_min, lab_max = _compute_channel_stats(lab, mask)
    mask_bool = mask > 0
    coverage = float(np.count_nonzero(mask_bool)) / max(mask.size, 1)
    hsv_p10 = np.percentile(hsv[mask_bool], 10, axis=0)
    hsv_p90 = np.percentile(hsv[mask_bool], 90, axis=0)
    lab_p10 = np.percentile(lab[mask_bool], 10, axis=0)
    lab_p90 = np.percentile(lab[mask_bool], 90, axis=0)
    return {
        "hsv_mean": hsv_mean.tolist(),
        "hsv_min": hsv_min.tolist(),
        "hsv_max": hsv_max.tolist(),
        "lab_mean": lab_mean.tolist(),
        "lab_min": lab_min.tolist(),
        "lab_max": lab_max.tolist(),
        "hsv_p10": hsv_p10.tolist(),
        "hsv_p90": hsv_p90.tolist(),
        "lab_p10": lab_p10.tolist(),
        "lab_p90": lab_p90.tolist(),
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# PyQt widgets
# ---------------------------------------------------------------------------


class _SamInferenceTask(QtCore.QRunnable):
    def __init__(
        self,
        request_id: int,
        predictor: SamPredictorWrapper,
        sam_image: np.ndarray,
        box: Optional[Tuple[int, int, int, int]],
        point_coords: Optional[List[Tuple[float, float]]],
        point_labels: Optional[List[int]],
        callback,
    ) -> None:
        super().__init__()
        self.request_id = request_id
        self.predictor = predictor
        self.sam_image = sam_image
        self.box = box
        self.point_coords = point_coords
        self.point_labels = point_labels
        self.callback = callback

    def run(self) -> None:  # pragma: no cover - background thread
        t0 = time.time()
        try:
            mask = self.predictor.predict(
                self.sam_image,
                box=self.box,
                point_coords=self.point_coords,
                point_labels=self.point_labels,
            )
            inference_ms = (time.time() - t0) * 1000.0
            self.callback(self.request_id, mask, inference_ms, None)
        except Exception as exc:
            self.callback(self.request_id, None, 0.0, exc)


class ImageCanvas(QtWidgets.QLabel):
    boxDrawn = QtCore.pyqtSignal(QtCore.QRect)
    pointPlaced = QtCore.pyqtSignal(QtCore.QPointF, int)

    def __init__(self) -> None:
        super().__init__()
        self._qimage: Optional[QtGui.QImage] = None
        self._overlay_mask: Optional[np.ndarray] = None
        self._drawing = False
        self._box_start = QtCore.QPoint()
        self._current_box = QtCore.QRect()
        self.scale_factor = 1.0
        self._min_scale = 0.2
        self._max_scale = 5.0
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)

    def set_image(self, image: np.ndarray) -> None:
        # 確保連續記憶體，避免 stride 被 OpenCV 影響
        image = np.ascontiguousarray(image)
        h, w, ch = image.shape
        assert ch == 3, "Expect BGR 3-channel image"

        # 每列的位元組數 (stride)
        bytes_per_line = image.strides[0]  # 等同於 w * ch

        # 用正確的 stride 來建 QImage
        qimg = QtGui.QImage(
            image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_BGR888
        ).copy()  # copy 以擁有資料的生命週期

        self._qimage = qimg
        self.scale_factor = 1.0
        self._update_scaled_pixmap()
        self._overlay_mask = None
        self.update()


    def set_overlay_mask(self, mask: Optional[np.ndarray]) -> None:
        self._overlay_mask = mask
        self.update()

    def _update_scaled_pixmap(self) -> None:
        if self._qimage is None:
            return
        new_w = max(1, int(self._qimage.width() * self.scale_factor))
        new_h = max(1, int(self._qimage.height() * self.scale_factor))
        scaled = self._qimage.scaled(
            new_w,
            new_h,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(QtGui.QPixmap.fromImage(scaled))
        self.setFixedSize(self.pixmap().size())

    def set_scale_factor(self, factor: float) -> None:
        factor = max(self._min_scale, min(self._max_scale, factor))
        if abs(factor - self.scale_factor) < 1e-3:
            return
        self.scale_factor = factor
        self._update_scaled_pixmap()
        self.update()

    def zoom_in(self) -> None:
        self.set_scale_factor(self.scale_factor * 1.2)

    def zoom_out(self) -> None:
        self.set_scale_factor(self.scale_factor / 1.2)

    def reset_zoom(self) -> None:
        self.set_scale_factor(1.0)

    def _display_to_image(self, point: QtCore.QPoint) -> QtCore.QPointF:
        if self._qimage is None:
            return QtCore.QPointF(0.0, 0.0)
        x = point.x() / self.scale_factor
        y = point.y() / self.scale_factor
        x = max(0.0, min(x, self._qimage.width() - 1))
        y = max(0.0, min(y, self._qimage.height() - 1))
        return QtCore.QPointF(x, y)

    def rect_to_image(self, rect: QtCore.QRect) -> Tuple[int, int, int, int]:
        top_left = self._display_to_image(rect.topLeft())
        bottom_right = self._display_to_image(rect.bottomRight())
        return (
            int(round(top_left.x())),
            int(round(top_left.y())),
            int(round(bottom_right.x())),
            int(round(bottom_right.y())),
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.pixmap():
            self._drawing = True
            self._box_start = event.pos()
            self._current_box = QtCore.QRect(self._box_start, self._box_start)
            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drawing:
            self._current_box = QtCore.QRect(self._box_start, event.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self.pixmap():
            return
        if self._drawing and event.button() == QtCore.Qt.MouseButton.LeftButton:
            rect = QtCore.QRect(self._box_start, event.pos()).normalized()
            self._drawing = False
            self._current_box = rect
            self.update()
            if rect.width() > 5 and rect.height() > 5:
                self.boxDrawn.emit(rect)
            else:
                img_point = self._display_to_image(event.pos())
                self.pointPlaced.emit(img_point, 1)
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            img_point = self._display_to_image(event.pos())
            self.pointPlaced.emit(img_point, 0)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        if self.pixmap() is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        if self._overlay_mask is not None:
            mask = self._overlay_mask
            display_w = self.width()
            display_h = self.height()
            resized = cv2.resize(
                mask,
                (display_w, display_h),
                interpolation=cv2.INTER_NEAREST,
            )
            alpha = np.zeros((display_h, display_w, 4), dtype=np.uint8)
            alpha[..., 0] = 30
            alpha[..., 1] = 190
            alpha[..., 2] = 255
            alpha[..., 3] = (resized * 200).astype(np.uint8)
            overlay_img = QtGui.QImage(
                alpha.data, display_w, display_h, QtGui.QImage.Format.Format_RGBA8888
            )
            painter.drawImage(0, 0, overlay_img)

        if self._drawing or (not self._current_box.isNull() and self._current_box.isValid()):
            pen = QtGui.QPen(QtGui.QColor(255, 165, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(self._current_box)
        painter.end()


class SamSelectionWindow(QtWidgets.QWidget):
    samResultReady = QtCore.pyqtSignal(int, object, float, object)
    """Main window orchestrating per-image SAM selections."""

    def __init__(self, cfg: SessionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("SAM Color Selection")
        self.recorder = ColorStatsRecorder()
        self.sam = SamPredictorWrapper(cfg.sam)
        self.sam_image_bgr: Optional[np.ndarray] = None
        self.sam_scale: float = 1.0
        self.sam_max_side = max(1, int(cfg.sam_infer_max_side))
        self._sam_pool = QtCore.QThreadPool()
        self._sam_pool.setMaxThreadCount(1)
        self._sam_pending = False
        self._sam_request_id = 0
        self._queued_sam_args: Optional[
            Tuple[Optional[Tuple[int, int, int, int]], Optional[List[Tuple[float, float]]], Optional[List[int]]]
        ] = None
        self.samResultReady.connect(self._on_sam_result)
        self.image_paths = sorted(
            [
                p
                for p in cfg.input_dir.rglob("*")
                if p.suffix.lower() in cfg.image_suffixes
            ]
        )
        if not self.image_paths:
            raise RuntimeError(f"No images found under {cfg.input_dir}")
        self.current_index = 0
        self.current_image_bgr: Optional[np.ndarray] = None
        self.current_hsv: Optional[np.ndarray] = None
        self.current_lab: Optional[np.ndarray] = None
        self.current_mask: Optional[np.ndarray] = None
        self.current_box: Optional[Tuple[int, int, int, int]] = None
        self.point_coords: List[Tuple[float, float]] = []
        self.point_labels: List[int] = []
        self.canvas = ImageCanvas()
        self.canvas.boxDrawn.connect(self._handle_box_drawn)
        self.canvas.pointPlaced.connect(self._handle_point)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.canvas)

        self.color_combo = QtWidgets.QComboBox()
        self.color_combo.addItems(cfg.colors)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setStyleSheet("font-weight: 600;")
        self.file_label = QtWidgets.QLabel("File: -")
        self.file_label.setWordWrap(True)
        self.stats_label = QtWidgets.QLabel("HSV/LAB stats: -")
        self.stats_label.setWordWrap(True)
        self.sam_status_label = QtWidgets.QLabel("SAM: Idle")
        self.sam_status_label.setStyleSheet("color: #888;")

        self.info_label = QtWidgets.QLabel(
            "Drag a loose rectangle to initialize the mask. Left click adds positive points, right click adds negative points."
        )
        self.info_label.setWordWrap(True)

        self.prompt_label = QtWidgets.QLabel("Points: 0(+)/0(-)")

        btn_prev = QtWidgets.QPushButton("Previous")
        btn_prev.clicked.connect(self._prev_image)
        btn_next = QtWidgets.QPushButton("Next (Skip)")
        btn_next.clicked.connect(self._next_image)
        btn_save = QtWidgets.QPushButton("Save Selection")
        btn_save.clicked.connect(self._save_current_selection)
        btn_finish = QtWidgets.QPushButton("Finish & Export")
        btn_finish.clicked.connect(self._finish_session)
        btn_undo_point = QtWidgets.QPushButton("Undo Point")
        btn_undo_point.clicked.connect(self._undo_last_point)
        btn_clear_points = QtWidgets.QPushButton("Clear Points")
        btn_clear_points.clicked.connect(self._clear_points)
        btn_clear_mask = QtWidgets.QPushButton("Clear Mask")
        btn_clear_mask.clicked.connect(self._clear_mask)

        zoom_in_btn = QtWidgets.QPushButton("Zoom +")
        zoom_in_btn.clicked.connect(self.canvas.zoom_in)
        zoom_out_btn = QtWidgets.QPushButton("Zoom -")
        zoom_out_btn.clicked.connect(self.canvas.zoom_out)
        zoom_reset_btn = QtWidgets.QPushButton("Reset Zoom")
        zoom_reset_btn.clicked.connect(self.canvas.reset_zoom)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(btn_prev)
        btn_layout.addWidget(btn_next)
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_finish)

        prompt_layout = QtWidgets.QHBoxLayout()
        prompt_layout.addWidget(self.prompt_label)
        prompt_layout.addWidget(btn_undo_point)
        prompt_layout.addWidget(btn_clear_points)
        prompt_layout.addWidget(btn_clear_mask)

        zoom_layout = QtWidgets.QHBoxLayout()
        zoom_layout.addWidget(QtWidgets.QLabel("Zoom:"))
        zoom_layout.addWidget(zoom_out_btn)
        zoom_layout.addWidget(zoom_in_btn)
        zoom_layout.addWidget(zoom_reset_btn)
        zoom_layout.addStretch()

        color_layout = QtWidgets.QHBoxLayout()
        color_layout.addWidget(QtWidgets.QLabel("Color label:"))
        color_layout.addWidget(self.color_combo, 1)
        color_layout.addWidget(self.status_label)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(self.scroll_area, 1)
        main_layout.addLayout(zoom_layout)
        main_layout.addLayout(color_layout)
        main_layout.addWidget(self.file_label)
        main_layout.addWidget(self.info_label)
        main_layout.addWidget(self.stats_label)
        main_layout.addWidget(self.sam_status_label)
        main_layout.addLayout(prompt_layout)
        main_layout.addLayout(btn_layout)

        self._load_current_image()

    # UI helpers ---------------------------------------------------------

    def _update_status(self) -> None:
        self.status_label.setText(f"{self.current_index + 1}/{len(self.image_paths)}")

    def _load_current_image(self) -> None:
        path = self.image_paths[self.current_index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        self.current_image_bgr = image
        self.current_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.current_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        self.current_mask = None
        self.current_box = None
        self.sam_image_bgr, self.sam_scale = self._prepare_sam_inputs(image)
        self._clear_points(update_info=False)
        self.canvas.set_image(image)
        self.canvas.set_overlay_mask(None)
        rel = path.relative_to(self.cfg.input_dir) if path.is_relative_to(self.cfg.input_dir) else path
        self.file_label.setText(f"File: {rel}")
        self.stats_label.setText("HSV/LAB stats: -")
        self.info_label.setText(
            "Draw a box or click a few points to guide SAM, then adjust until the mask looks correct."
        )
        self._update_status()

    def _prepare_sam_inputs(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize oversized frames for SAM while tracking the scale factor."""
        h, w = image.shape[:2]
        longest = max(h, w)
        max_side = self.sam_max_side
        if longest <= max_side:
            return np.ascontiguousarray(image), 1.0
        scale = max_side / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(resized), scale

    def _scaled_box_for_sam(self) -> Optional[Tuple[int, int, int, int]]:
        if self.current_box is None:
            return None
        if abs(self.sam_scale - 1.0) < 1e-6:
            return self.current_box
        return tuple(int(round(coord * self.sam_scale)) for coord in self.current_box)

    def _scaled_points_for_sam(self) -> Optional[List[Tuple[float, float]]]:
        if not self.point_coords:
            return None
        if abs(self.sam_scale - 1.0) < 1e-6:
            return list(self.point_coords)
        return [(x * self.sam_scale, y * self.sam_scale) for x, y in self.point_coords]

    def _handle_box_drawn(self, rect: QtCore.QRect) -> None:
        if self.current_image_bgr is None:
            return
        x0, y0, x1, y1 = self.canvas.rect_to_image(rect)
        x0 = max(0, min(x0, self.current_image_bgr.shape[1] - 1))
        y0 = max(0, min(y0, self.current_image_bgr.shape[0] - 1))
        x1 = max(0, min(x1, self.current_image_bgr.shape[1] - 1))
        y1 = max(0, min(y1, self.current_image_bgr.shape[0] - 1))
        if x1 - x0 < 5 or y1 - y0 < 5:
            self.info_label.setText("Bounding box too small. Try again.")
            return
        self.current_box = (x0, y0, x1, y1)
        self._run_sam_prediction()

    def _handle_point(self, point: QtCore.QPointF, label: int) -> None:
        if self.current_image_bgr is None:
            return
        self.point_coords.append((point.x(), point.y()))
        self.point_labels.append(label)
        self._update_prompt_label()
        self._run_sam_prediction()

    def _save_current_selection(self) -> None:
        if self.current_image_bgr is None or self.current_mask is None:
            self.info_label.setText("Please draw a box and generate a mask first.")
            return
        try:
            stats = compute_hsv_lab_stats(
                self.current_image_bgr,
                self.current_mask,
                hsv_image=self.current_hsv,
                lab_image=self.current_lab,
            )
        except ValueError as exc:
            self.info_label.setText(str(exc))
            return
        color = self.color_combo.currentText() or "Unknown"
        image_path = self.image_paths[self.current_index]
        self.recorder.record(
            image_path,
            color,
            stats["hsv_mean"],
            stats["hsv_min"],
            stats["hsv_max"],
            stats["lab_mean"],
            stats["lab_min"],
            stats["lab_max"],
            stats["hsv_p10"],
            stats["hsv_p90"],
            stats["lab_p10"],
            stats["lab_p90"],
            float(stats["coverage"]),
        )
        self.info_label.setText(f"Saved selection for {color}. Moving to next image.")
        self._next_image()

    def _next_image(self) -> None:
        if self.current_index + 1 >= len(self.image_paths):
            self.info_label.setText("Reached final image. You can Finish & Export now.")
            return
        self.current_index += 1
        self._load_current_image()

    def _prev_image(self) -> None:
        if self.current_index == 0:
            return
        self.current_index -= 1
        self._load_current_image()

    def _finish_session(self) -> None:
        if not self.recorder.summary:
            QtWidgets.QMessageBox.warning(self, "No data", "No selections have been saved yet.")
            return
        payload = self.recorder.to_json()
        self.cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
        with self.cfg.output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        QtWidgets.QMessageBox.information(
            self,
            "Export complete",
            f"Color statistics saved to:\n{self.cfg.output_json}",
        )
        self.close()

    def _run_sam_prediction(self) -> None:
        if self.current_image_bgr is None:
            return
        if self.current_box is None and not self.point_coords:
            self.info_label.setText("Add a bounding box or click points to guide SAM.")
            self.canvas.set_overlay_mask(None)
            self.current_mask = None
            self.stats_label.setText("HSV/LAB stats: -")
            return
        if self.sam_image_bgr is None:
            self.info_label.setText("SAM input not ready yet. Reload the image and try again.")
            return
        scaled_points = self._scaled_points_for_sam()
        point_labels = self.point_labels if scaled_points else None
        params = (self._scaled_box_for_sam(), scaled_points, point_labels)
        if self._sam_pending:
            self._queued_sam_args = params
            self.info_label.setText("SAM running... queued latest prompts.")
            self.sam_status_label.setText("SAM: Queued")
            return
        self._queued_sam_args = None
        self._launch_sam_task(params)

    def _launch_sam_task(
        self,
        params: Tuple[
            Optional[Tuple[int, int, int, int]], Optional[List[Tuple[float, float]]], Optional[List[int]]
        ],
    ) -> None:
        if self.sam_image_bgr is None:
            return
        box, scaled_points, point_labels = params
        self._sam_request_id += 1
        request_id = self._sam_request_id
        self._sam_pending = True
        self.info_label.setText("Running SAM...")
        self.sam_status_label.setText("SAM: Running")
        task = _SamInferenceTask(
            request_id=request_id,
            predictor=self.sam,
            sam_image=self.sam_image_bgr,
            box=box,
            point_coords=scaled_points,
            point_labels=point_labels,
            callback=self.samResultReady.emit,
        )
        self._sam_pool.start(task)

    @QtCore.pyqtSlot(int, object, float, object)
    def _on_sam_result(
        self,
        request_id: int,
        mask: Optional[np.ndarray],
        inference_ms: float,
        error: Optional[BaseException],
    ) -> None:
        if request_id != self._sam_request_id:
            return
        self._sam_pending = False
        if error is not None or mask is None:
            if error is not None:
                logging.exception("SAM prediction failed", exc_info=error)
                error_msg = str(error)
            else:
                logging.error("SAM prediction returned empty mask.")
                error_msg = "Unknown SAM error"
            self.current_mask = None
            self.canvas.set_overlay_mask(None)
            self.stats_label.setText("HSV/LAB stats: -")
            self.info_label.setText(f"SAM failed: {error_msg}")
        else:
            if self.current_image_bgr is None:
                return
            resized_mask = cv2.resize(
                mask,
                (self.current_image_bgr.shape[1], self.current_image_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            self.current_mask = resized_mask
            self.canvas.set_overlay_mask(resized_mask)
            coverage = float(np.count_nonzero(resized_mask)) / resized_mask.size * 100.0
            self._refresh_stats_label()
            self.info_label.setText(
                f"Mask updated in {inference_ms:.0f} ms (coverage {coverage:.2f}%). Save when satisfied."
            )
        if self._queued_sam_args:
            pending = self._queued_sam_args
            self._queued_sam_args = None
            self._launch_sam_task(pending)
            self.sam_status_label.setText("SAM: Running (queued)")
        else:
            self.sam_status_label.setText("SAM: Idle")

    def _update_prompt_label(self) -> None:
        pos = sum(1 for lbl in self.point_labels if lbl == 1)
        neg = sum(1 for lbl in self.point_labels if lbl == 0)
        self.prompt_label.setText(f"Points: {pos}(+)/{neg}(-)")

    def _refresh_stats_label(self) -> None:
        if (
            self.current_image_bgr is None
            or self.current_mask is None
            or np.count_nonzero(self.current_mask) == 0
        ):
            self.stats_label.setText("HSV/LAB stats: -")
            return
        try:
            stats = compute_hsv_lab_stats(
                self.current_image_bgr,
                self.current_mask,
                hsv_image=self.current_hsv,
                lab_image=self.current_lab,
            )
        except ValueError:
            self.stats_label.setText("HSV/LAB stats: empty mask")
            return
        hsv_mean = stats.get("hsv_mean", [float("nan")] * 3)
        lab_mean = stats.get("lab_mean", [float("nan")] * 3)
        hsv_text = f"H= {hsv_mean[0]:.1f}, S= {hsv_mean[1]:.1f}, V= {hsv_mean[2]:.1f}"
        lab_text = f"L= {lab_mean[0]:.1f}, a= {lab_mean[1]:.1f}, b= {lab_mean[2]:.1f}"
        self.stats_label.setText(f"HSV/LAB stats: {hsv_text} | {lab_text}")

    def _clear_points(self, update_info: bool = True) -> None:
        self.point_coords.clear()
        self.point_labels.clear()
        self._update_prompt_label()
        if update_info:
            self.info_label.setText("Points cleared.")
        self._run_sam_prediction()

    def _undo_last_point(self) -> None:
        if not self.point_coords:
            self.info_label.setText("No points to undo.")
            return
        self.point_coords.pop()
        self.point_labels.pop()
        self._update_prompt_label()
        self._run_sam_prediction()

    def _clear_mask(self) -> None:
        self.current_mask = None
        self.current_box = None
        self._clear_points(update_info=False)
        self.canvas.set_overlay_mask(None)
        self.stats_label.setText("HSV/LAB stats: -")
        self.info_label.setText(
            "Draw a box or click a few points to guide SAM, then adjust until the mask looks correct."
        )
# ---------------------------------------------------------------------------
# CLI / entrypoints
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("SAM Color Inspection")
    sub = parser.add_subparsers(dest="cmd")

    collect = sub.add_parser("collect", help="Open GUI to collect color stats via SAM")
    collect.add_argument("--input-dir", required=True)
    collect.add_argument("--output-json", required=True)
    collect.add_argument("--sam-checkpoint", required=True)
    collect.add_argument("--sam-model", default="vit_b")
    collect.add_argument("--device", default="auto")
    collect.add_argument("--colors", nargs="+", default=["Default"])
    collect.add_argument("--max-side", type=int, default=2048)
    collect.set_defaults(cmd="collect")
    return parser.parse_args(argv)


def run_gui_session(cfg: SessionConfig) -> None:
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication([])
    window = SamSelectionWindow(cfg)
    window.show()
    if owns_app:
        app.exec_()


# Backwards-compatible API for pipeline --------------------------------------


def cmd_collect(args: argparse.Namespace) -> None:
    session_cfg = SessionConfig.from_namespace(args)
    run_gui_session(session_cfg)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.cmd != "collect":
        print("Please specify the 'collect' command. Use -h for help.")
        return 1
    ns = argparse.Namespace(
        input_dir=args.input_dir,
        output_json=args.output_json,
        colors=args.colors,
        max_side=args.max_side,
        sam={"checkpoint": args.sam_checkpoint, "model_type": args.sam_model, "device": args.device},
    )
    cmd_collect(ns)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
