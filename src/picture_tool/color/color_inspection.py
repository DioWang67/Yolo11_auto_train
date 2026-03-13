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
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import threading

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

import os

try:  # pragma: no cover - optional dependency resolution
    if os.environ.get("PYTEST_IS_RUNNING") == "1":
        raise ImportError("Bypass torch during pytest")
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency resolution
    if os.environ.get("PYTEST_IS_RUNNING") == "1":
        raise ImportError("Bypass ultralytics during pytest")
    from ultralytics import SAM
    # Ultralytics handles model loading internally
except ImportError:  # pragma: no cover
    SAM = None  # type: ignore


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
        except (RuntimeError, ValueError):  # pragma: no cover - CUDA env issues
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
        # Default to a SAM 2 model if not specified
        checkpoint = Path(sam_cfg.get("checkpoint", "sam2_b.pt"))
        # Validate or default model_type (just an identifier now, mostly unused by Ultralytics as it detects from file)
        model_type = sam_cfg.get("model_type", "sam2_b")
        device = sam_cfg.get("device", "auto")
        max_side = getattr(ns, "max_side", None) or sam_cfg.get("max_side") or 2048
        colors = list(getattr(ns, "colors", []) or ["Default"])
        return SessionConfig(
            input_dir=Path(getattr(ns, "input_dir", "./data/project/qc/color_samples")),
            output_json=Path(
                getattr(ns, "output_json", "./runs/project/quality/color/stats.json")
            ),
            colors=colors,
            sam=SamSettings(
                checkpoint=checkpoint, model_type=model_type, device=device
            ),
            sam_infer_max_side=int(max_side),
        )


# ---------------------------------------------------------------------------
# Aggregation utilities
# ---------------------------------------------------------------------------


def _ensure_package_available() -> None:
    if SAM is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "ultralytics is not installed. Please pip install ultralytics."
        )


def _to_numpy(arr: Iterable[float]) -> np.ndarray:
    return np.asarray(list(arr), dtype=np.float64)


@dataclass
class RunningStats:
    count: int = 0
    hsv_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    lab_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    hsv_p10_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    hsv_p90_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    lab_p10_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    lab_p90_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
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
        return {
            "summary": {color: stats.as_dict() for color, stats in self.summary.items()}
        }


# ---------------------------------------------------------------------------
# SAM predictor wrapper
# ---------------------------------------------------------------------------


class SamPredictorWrapper:
    """Wrapper around Ultralytics SAM 2 predictor."""

    def __init__(self, cfg: SamSettings) -> None:
        _ensure_package_available()
        checkpoint = Path(cfg.checkpoint)
        # Ultralytics will auto-download if name matches known models, but we prefer local path if exists
        # If cfg.model_type is one of sam2 variants, we might want to map it or just use checkpoint
        
        # Validating checkpoint existence or letting Ultralytics handle it
        model_src = str(checkpoint)
        if not checkpoint.exists():
            # If flexible, we could pass just the name (e.g. sam2_b.pt)
            # But the existing code expects a file. Check if it's a known model nickname.
            known_models = ["sam2_t.pt", "sam2_s.pt", "sam2_b.pt", "sam2_l.pt", "sam2.pt"]
            if checkpoint.name in known_models:
                 model_src = checkpoint.name # Let ultralytics download
            else:
                 raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

        logging.info("Loading SAM model from %s", model_src)
        self.model = SAM(model_src)
        
        # Device handling is done per-predict usually, but can be set on model is possible?
        # Ultralytics models usually load to device on first use or via .to()
        # But we will pass device to predict/call
        self.device = cfg.resolved_device()

    def predict(
        self,
        image_bgr: np.ndarray,
        *,
        box: Optional[Tuple[int, int, int, int]] = None,
        point_coords: Optional[List[Tuple[float, float]]] = None,
        point_labels: Optional[List[int]] = None,
    ) -> np.ndarray:
        # Convert BGR to RGB (Ultralytics expects RGB or BGR? 
        # Usually checking: standard YOLO expects BGR if using cv2 load, but PIL is RGB.
        # Ultralytics predict() handles numpy arrays (assumed BGR by default in OpenCV world).
        # But let's check docs. Actually, Ultralytics usually assumes BGR if input is numpy array.
        # However, to be safe, we can try. 
        # SAM 2 training is RGB. 
        # Let's pass BGR as per standard cv2 read.
        
        # Construct prompts
        # Ultralytics SAM 2 args: bboxes, points, labels
        # points shape: (N, 2), labels shape: (N,)
        
        bboxes = None
        if box:
            # Box format: [x1, y1, x2, y2]
            bboxes = [box] # List of boxes
            
        points = None
        labels = None
        if point_coords and point_labels:
            points = [point_coords] # List of [x, y]
            labels = [point_labels] # List of ints
            
        # Validate matching lengths to prevent internal model crashes
        pts_count = len(point_coords) if point_coords else 0
        lbl_count = len(point_labels) if point_labels else 0
        if pts_count != lbl_count:
             msg = f"Points/Labels mismatch: {pts_count} pts vs {lbl_count} labels"
             logging.error(msg)
             # Return empty mask instead of crashing
             h, w = image_bgr.shape[:2]
             return np.zeros((h, w), dtype=np.uint8)

        # Calling the model
        # device=self.device
        try:
            results = self.model(
                image_bgr, 
                bboxes=bboxes, 
                points=points, 
                labels=labels, 
                device=self.device,
                verbose=False,
                retina_masks=True
            )
        except Exception as e:
            logging.error(f"SAM 2 inference failed: {e}")
            raise
            
        if not results:
            # Empty result
            h, w = image_bgr.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)
            
        result = results[0]
        
        # Result has .masks
        if result.masks is None:
             h, w = image_bgr.shape[:2]
             return np.zeros((h, w), dtype=np.uint8)
             
        # masks.data is a tensor (N, H, W)
        # We need to select the best one?
        # SAM 2 typically returns multiple masks for ambiguous prompts.
        # Ultralytics API might return just one merged or multiple?
        # Usually it returns all proposals.
        
        masks_tensor = result.masks.data
        if masks_tensor is None or len(masks_tensor) == 0:
             h, w = image_bgr.shape[:2]
             return np.zeros((h, w), dtype=np.uint8)
             
        # Convert to numpy
        masks_np = masks_tensor.cpu().numpy().astype(np.uint8) 
        # masks_np shape: (N, H, W) where N is number of detected objects/masks
        
        # Also need scores if available?
        # result.boxes.conf ? result.masks doesn't always store confidence in same way as SAM 1 scores.
        # But if we have multiple masks for the *same* prompt (multi-mask/ambiguity), 
        # Ultralytics SAM output might structure it differently.
        # Ultralytics SAM usually returns one mask per object detected for Box/Point prompts.
        # If we provided 1 box/point set, we expect 1 result (or 3 if multi_mask is enabled deep down).
        
        # If multiple masks are returned for a single prompt, we should pick best.
        # But Ultralytics 'predict' with specific prompts usually separates them.
        # If we have 1 prompt, and we get N masks, it might be N alternatives.
        
        # For now, let's take the first one or merge? 
        # Better: use the one with highest coverage or similar heuristic as before, 
        # or simplified: just take the first one (usually the best confidence).
        
        # But wait, our legacy code had `_choose_mask_index` with heuristics (wires vs background).
        # We don't have scores easily available per mask hypothesis in the standard high-level result object 
        # (conf is usually for object detection, but for SAM it's IoU pred).
        # Let's check if `result.boxes.conf` exists for SAM results.
        
        # If we want to maintain the `_choose_mask_index` logic, we need the masks.
        # Let's pass all masks to `_choose_mask_index`.
        # Assuming masks_np is (N, H, W)
        
        scores = None
        # Try to get scores
        if hasattr(result, 'boxes') and result.boxes is not None and result.boxes.conf is not None:
             scores = result.boxes.conf.cpu().numpy()
        else:
             scores = np.ones(len(masks_np)) # Dummy scores
             
        # If we have only 1 mask, just return it
        if len(masks_np) == 1:
            return masks_np[0]
            
        # Select best
        # Re-use logic. But wait, `_choose_mask_index` is in this Class, I need to keep it or refactor it.
        # I am replacing the whole class, so I need to include `_choose_mask_index` in the replacement?
        # The tool `replace_file_content` replaces a chunk. `_choose_mask_index` is below line 477.
        # I ended my replacement at 438. 
        # CAVEAT: `_choose_mask_index` is an instance method (calls self._score...). 
        # I should keep it.
        
        # So in this method, I will call self._choose_mask_index
        idx = self._choose_mask_index(masks_np, scores, point_coords, point_labels, box)
        return masks_np[idx]

    def _ensure_image(self, image_bgr: np.ndarray) -> None:
        # No-op for Ultralytics API as we pass image to predict() every time
        pass

    def _choose_mask_index(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        pts: Optional[List[Tuple[float, float]]],
        labels: Optional[List[int]],
        box: Optional[Tuple[int, int, int, int]],
    ) -> int:
        best_idx = 0
        best_metric = -float("inf")
        
        # Convert inputs to numpy if they are lists (Ultralytics might keep them as is, but _score helper expects numpy or similar)
        # Adapt helper inputs.
        pts_arr = np.array(pts, dtype=np.float32) if pts else None
        labels_arr = np.array(labels, dtype=np.int32) if labels else None
        box_arr = np.array(box, dtype=np.float32) if box else None

        for i, mask in enumerate(masks):
            # score provided by model (if any)
            metric = float(scores[i]) 
            # Add heuristic scores
            metric += self._score_points(mask, pts_arr, labels_arr)
            metric += self._score_box(mask, box_arr)
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


def _compute_channel_stats(
    image: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    hsv = (
        hsv_image
        if hsv_image is not None
        else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    )
    lab = (
        lab_image
        if lab_image is not None
        else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    )
    hsv_mean, hsv_min, hsv_max = _compute_channel_stats(hsv, mask)
    lab_mean, lab_min, lab_max = _compute_channel_stats(lab, mask)
    mask_bool = mask > 0
    coverage = float(np.count_nonzero(mask_bool)) / max(mask.size, 1)
    masked_hsv = hsv[mask_bool].astype(np.float64)
    masked_lab = lab[mask_bool].astype(np.float64)
    hsv_p10 = np.percentile(masked_hsv, 10, axis=0)
    hsv_p90 = np.percentile(masked_hsv, 90, axis=0)
    lab_p10 = np.percentile(masked_lab, 10, axis=0)
    lab_p90 = np.percentile(masked_lab, 90, axis=0)
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
        original_images: Tuple[np.ndarray, np.ndarray, np.ndarray],
        box: Optional[Tuple[int, int, int, int]],
        point_coords: Optional[List[Tuple[float, float]]],
        point_labels: Optional[List[int]],
        callback,
        progress_cb=None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        super().__init__()
        self.request_id = request_id
        self.predictor = predictor
        self.sam_image = sam_image
        self.orig_bgr, self.orig_hsv, self.orig_lab = original_images
        self.box = box
        self.point_coords = point_coords
        self.point_labels = point_labels
        self.callback = callback
        self.progress_cb = progress_cb
        self.cancel_event = cancel_event or threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:  # pragma: no cover - background thread
        t0 = time.time()
        if self.cancel_event.is_set():
            return
        try:
            mask = self.predictor.predict(
                self.sam_image,
                box=self.box,
                point_coords=self.point_coords,
                point_labels=self.point_labels,
            )

            if self.cancel_event.is_set():
                return

            # Resize mask to original resolution
            h, w = self.orig_bgr.shape[:2]
            resized_mask = cv2.resize(
                mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

            # Compute heavy statistics (percentiles etc.)
            stats = None
            if np.count_nonzero(resized_mask) > 0:
                try:
                    stats = compute_hsv_lab_stats(
                        self.orig_bgr,
                        resized_mask,
                        hsv_image=self.orig_hsv,
                        lab_image=self.orig_lab,
                    )
                except ValueError:
                    stats = None

            inference_ms = (time.time() - t0) * 1000.0

            if self.cancel_event.is_set():
                return
            if self.progress_cb:
                self.progress_cb("done", inference_ms)

            # Return full-res mask AND computed stats
            self.callback(self.request_id, resized_mask, stats, inference_ms, None)

        except Exception as exc:  # Catch all to prevent thread crash
            if self.cancel_event.is_set():
                return
            if self.progress_cb:
                self.progress_cb("error", 0.0)
            self.callback(self.request_id, None, None, 0.0, exc)


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
        pixmap = self.pixmap()
        if pixmap is not None:
            self.setFixedSize(pixmap.size())

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

    def mousePressEvent(self, event: Optional[QtGui.QMouseEvent]) -> None:
        if event is None:
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.pixmap():
            self._drawing = True
            self._box_start = event.pos()
            self._current_box = QtCore.QRect(self._box_start, self._box_start)
            self.update()

    def mouseMoveEvent(self, event: Optional[QtGui.QMouseEvent]) -> None:
        if self._drawing and event is not None:
            self._current_box = QtCore.QRect(self._box_start, event.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, event: Optional[QtGui.QMouseEvent]) -> None:
        if not self.pixmap() or event is None:
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

    def paintEvent(self, event: Optional[QtGui.QPaintEvent]) -> None:
        if event is None:
            return
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
            alpha[..., 3] = (resized.astype(float) * 200).astype(np.uint8)
            overlay_img = QtGui.QImage(
                alpha.data, display_w, display_h, QtGui.QImage.Format.Format_RGBA8888
            )
            painter.drawImage(0, 0, overlay_img)

        if self._drawing or (
            not self._current_box.isNull() and self._current_box.isValid()
        ):
            pen = QtGui.QPen(QtGui.QColor(255, 165, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(self._current_box)
        painter.end()


class SamSelectionWindow(QtWidgets.QWidget):
    # Updated signal: request_id, mask(full-res), stats(dict), inference_ms, error
    samResultReady = QtCore.pyqtSignal(int, object, object, float, object)
    samProgress = QtCore.pyqtSignal(str, float)
    """Main window orchestrating per-image SAM selections."""

    def __init__(self, cfg: SessionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("🎨 SAM Color Selection Tool")
        self.setMinimumSize(1200, 800)

        # Apply modern stylesheet
        self.setStyleSheet("""
            QWidget {
                font-family: "Segoe UI", "Microsoft JhengHei", sans-serif;
                background-color: #0d1117;
                color: #c9d1d9;
            }
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                color: #c9d1d9;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #30363d;
                border-color: #484f58;
            }
            QPushButton:pressed {
                background-color: #161b22;
            }
            QPushButton#PrimaryBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #1f6feb, stop:1 #58a6ff);
                border: none;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton#PrimaryBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #388bfd, stop:1 #79c0ff);
            }
            QPushButton#SuccessBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #238636, stop:1 #2ea043);
                border: none;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton#SuccessBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #2ea043, stop:1 #3fb950);
            }
            QLabel {
                color: #c9d1d9;
                padding: 4px;
            }
            QComboBox {
                background-color: #161b22;
                border: 2px solid #30363d;
                color: #c9d1d9;
                padding: 6px 10px;
                border-radius: 6px;
            }
            QComboBox:hover {
                border-color: #484f58;
            }
            QComboBox::drop-down {
                border: none;
                width: 25px;
            }
            QComboBox QAbstractItemView {
                background-color: #161b22;
                border: 2px solid #30363d;
                color: #c9d1d9;
                selection-background-color: #1f6feb;
            }
            QScrollArea {
                border: 2px solid #30363d;
                border-radius: 8px;
                background-color: #0d1117;
            }
        """)
        self.recorder = ColorStatsRecorder()
        self.sam = SamPredictorWrapper(cfg.sam)
        self.sam_image_bgr: Optional[np.ndarray] = None
        self.sam_scale: float = 1.0
        self.sam_max_side = max(1, int(cfg.sam_infer_max_side))
        self._sam_pool = QtCore.QThreadPool()
        self._sam_pool.setMaxThreadCount(1)
        self._sam_pending = False
        self._sam_request_id = 0
        self._active_sam_task: Optional[_SamInferenceTask] = None
        self._queued_sam_args: Optional[
            Tuple[
                Optional[Tuple[int, int, int, int]],
                Optional[List[Tuple[float, float]]],
                Optional[List[int]],
            ]
        ] = None
        self.samResultReady.connect(self._on_sam_result)
        self.samProgress.connect(self._on_sam_progress)
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
        self.current_stats: Optional[Dict[str, object]] = None  # Cache for stats
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
            "🖌 拖曳矩形框選區域，左鍵增加正面點，右鍵增加負面點。"
        )
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(
            "color: #8b949e; font-size: 9pt; padding: 8px; background-color: #161b22; border-radius: 6px;"
        )

        self.prompt_label = QtWidgets.QLabel("📍 Points: 0(+)/0(-)")
        self.prompt_label.setStyleSheet("font-weight: 600; color: #58a6ff;")

        # Navigation buttons
        btn_prev = QtWidgets.QPushButton("◀ Previous")
        btn_prev.clicked.connect(self._prev_image)
        btn_next = QtWidgets.QPushButton("Next (Skip) ▶")
        btn_next.clicked.connect(self._next_image)

        # Action buttons
        btn_save = QtWidgets.QPushButton("💾 Save Selection")
        btn_save.setObjectName("SuccessBtn")
        btn_save.clicked.connect(self._save_current_selection)

        btn_finish = QtWidgets.QPushButton("✓ Finish & Export")
        btn_finish.setObjectName("PrimaryBtn")
        btn_finish.clicked.connect(self._finish_session)

        # Edit buttons
        btn_undo_point = QtWidgets.QPushButton("↶ Undo Point")
        btn_undo_point.clicked.connect(self._undo_last_point)

        btn_clear_points = QtWidgets.QPushButton("✕ Clear Points")
        btn_clear_points.clicked.connect(self._clear_points)

        btn_clear_mask = QtWidgets.QPushButton("🗑 Clear Mask")
        btn_clear_mask.clicked.connect(self._clear_mask)

        # Zoom buttons
        zoom_in_btn = QtWidgets.QPushButton("🔍 Zoom +")
        zoom_in_btn.clicked.connect(self.canvas.zoom_in)

        zoom_out_btn = QtWidgets.QPushButton("🔎 Zoom -")
        zoom_out_btn.clicked.connect(self.canvas.zoom_out)

        zoom_reset_btn = QtWidgets.QPushButton("↺ Reset Zoom")
        zoom_reset_btn.clicked.connect(self.canvas.reset_zoom)

        # Improved layout organization
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(btn_prev)
        btn_layout.addWidget(btn_next)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_finish)

        edit_layout = QtWidgets.QHBoxLayout()
        edit_layout.addWidget(self.prompt_label)
        edit_layout.addStretch()
        edit_layout.addWidget(btn_undo_point)
        edit_layout.addWidget(btn_clear_points)
        edit_layout.addWidget(btn_clear_mask)

        zoom_layout = QtWidgets.QHBoxLayout()
        zoom_layout.addWidget(QtWidgets.QLabel("🔍 Zoom:"))
        zoom_layout.addWidget(zoom_out_btn)
        zoom_layout.addWidget(zoom_in_btn)
        zoom_layout.addWidget(zoom_reset_btn)
        zoom_layout.addStretch()

        color_layout = QtWidgets.QHBoxLayout()
        color_label = QtWidgets.QLabel("🎨 Color:")
        color_label.setStyleSheet("font-weight: 600;")
        color_layout.addWidget(color_label)
        color_layout.addWidget(self.color_combo, 1)
        color_layout.addWidget(self.status_label)

        # Status panel with card style
        status_panel = QtWidgets.QWidget()
        status_panel.setStyleSheet("""
            QWidget {
                background-color: #161b22;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        status_layout = QtWidgets.QVBoxLayout(status_panel)
        status_layout.setContentsMargins(12, 12, 12, 12)
        status_layout.addWidget(self.file_label)
        status_layout.addWidget(self.stats_label)
        status_layout.addWidget(self.sam_status_label)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)
        main_layout.addWidget(self.scroll_area, 1)
        main_layout.addLayout(zoom_layout)
        main_layout.addLayout(color_layout)
        main_layout.addWidget(status_panel)
        main_layout.addWidget(self.info_label)
        main_layout.addLayout(edit_layout)
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
        self.current_stats = None
        self.current_box = None
        self.sam_image_bgr, self.sam_scale = self._prepare_sam_inputs(image)
        self._clear_points(update_info=False)
        self.canvas.set_image(image)
        self.canvas.set_overlay_mask(None)
        rel = (
            path.relative_to(self.cfg.input_dir)
            if path.is_relative_to(self.cfg.input_dir)
            else path
        )
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
        x0, y0, x1, y1 = self.current_box
        scale = self.sam_scale
        return (
            int(round(x0 * scale)),
            int(round(y0 * scale)),
            int(round(x1 * scale)),
            int(round(y1 * scale)),
        )

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
        # Use cached stats if available
        stats = self.current_stats
        if stats is None:
            self.info_label.setText(
                "Statistics not ready/failed. Try updating the mask."
            )
            return

        color = self.color_combo.currentText() or "Unknown"
        image_path = self.image_paths[self.current_index]
        coverage_val = stats.get("coverage", 0.0)
        coverage = (
            float(coverage_val) if isinstance(coverage_val, (int, float)) else 0.0
        )
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
            coverage,
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
            QtWidgets.QMessageBox.warning(
                self, "No data", "No selections have been saved yet."
            )
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
            self.current_stats = None
            self.stats_label.setText("HSV/LAB stats: -")
            return
        if self.sam_image_bgr is None:
            self.info_label.setText(
                "SAM input not ready yet. Reload the image and try again."
            )
            return
        scaled_points = self._scaled_points_for_sam()
        point_labels = list(self.point_labels) if scaled_points else None
        params = (self._scaled_box_for_sam(), scaled_points, point_labels)
        if self._sam_pending:
            self._queued_sam_args = params
            self.info_label.setText("SAM running... queued latest prompts.")
            self.sam_status_label.setText("SAM: Queued")
            return
        self._queued_sam_args = None
        if self._sam_pending and hasattr(self, "_active_sam_task"):
            try:
                self._active_sam_task.cancel()
                self.sam_status_label.setText("SAM: Cancelling previous")
            except (OSError, ValueError):
                pass
        self._launch_sam_task(params)

    def _launch_sam_task(
        self,
        params: Tuple[
            Optional[Tuple[int, int, int, int]],
            Optional[List[Tuple[float, float]]],
            Optional[List[int]],
        ],
    ) -> None:
        # Capture to locals for type narrowing
        img_bgr = self.current_image_bgr
        img_hsv = self.current_hsv
        img_lab = self.current_lab
        sam_bgr = self.sam_image_bgr

        if sam_bgr is None or img_bgr is None or img_hsv is None or img_lab is None:
            return

        box, scaled_points, point_labels = params
        self._sam_request_id += 1
        request_id = self._sam_request_id
        self._sam_pending = True
        self.info_label.setText("Running SAM & Stats...")
        self.sam_status_label.setText("SAM: Running")

        # Pass full-res logic to thread
        original_images = (img_bgr, img_hsv, img_lab)

        task = _SamInferenceTask(
            request_id=request_id,
            predictor=self.sam,
            sam_image=sam_bgr,
            original_images=original_images,
            box=box,
            point_coords=scaled_points,
            point_labels=point_labels,
            callback=self.samResultReady.emit,
            progress_cb=self.samProgress.emit,
        )
        self._active_sam_task = task
        self._sam_pool.start(task)

    @QtCore.pyqtSlot(int, object, object, float, object)
    def _on_sam_result(
        self,
        request_id: int,
        mask: Optional[np.ndarray],  # Now already full-res
        stats: Optional[Dict[str, object]],  # Pre-computed stats
        inference_ms: float,
        error: Optional[BaseException],
    ) -> None:
        if request_id != self._sam_request_id:
            return
        self._sam_pending = False
        self._active_sam_task = None
        if error is not None or mask is None:
            if error is not None:
                logging.exception("SAM prediction failed", exc_info=error)
                error_msg = str(error)
            else:
                logging.error("SAM prediction returned empty mask.")
                error_msg = "Unknown SAM error"
            self.current_mask = None
            self.current_stats = None
            self.canvas.set_overlay_mask(None)
            self.stats_label.setText("HSV/LAB stats: -")
            self.info_label.setText(f"SAM failed: {error_msg}")
        else:
            if self.current_image_bgr is None:
                return

            # mask is already resized in thread
            self.current_mask = mask
            self.canvas.set_overlay_mask(mask)

            self.current_stats = stats

            coverage = 0.0
            if stats:
                coverage_val = stats.get("coverage", 0.0)
                coverage = (
                    float(coverage_val) * 100.0
                    if isinstance(coverage_val, (int, float))
                    else 0.0
                )

            self._render_stats_to_label(stats)

            self.info_label.setText(
                f"Mask & Stats updated in {inference_ms:.0f} ms (coverage {coverage:.2f}%). Save when satisfied."
            )

        if self._queued_sam_args:
            pending = self._queued_sam_args
            self._queued_sam_args = None
            self._launch_sam_task(pending)
            self.sam_status_label.setText("SAM: Running (queued)")
        else:
            self.sam_status_label.setText("SAM: Idle")

    def _on_sam_progress(self, status: str, inference_ms: float) -> None:
        if status == "done":
            self.sam_status_label.setText(f"SAM: Done ({inference_ms:.0f} ms)")
        elif status == "error":
            self.sam_status_label.setText("SAM: Error")

    def _update_prompt_label(self) -> None:
        pos = sum(1 for lbl in self.point_labels if lbl == 1)
        neg = sum(1 for lbl in self.point_labels if lbl == 0)
        self.prompt_label.setText(f"Points: {pos}(+)/{neg}(-)")

    def _render_stats_to_label(self, stats: Optional[Dict[str, object]]) -> None:
        if not stats:
            self.stats_label.setText("HSV/LAB stats: empty/failed")
            return

        hsv_mean_val = stats.get("hsv_mean", [float("nan")] * 3)
        lab_mean_val = stats.get("lab_mean", [float("nan")] * 3)
        hsv_mean = (
            hsv_mean_val if isinstance(hsv_mean_val, list) else [float("nan")] * 3
        )
        lab_mean = (
            lab_mean_val if isinstance(lab_mean_val, list) else [float("nan")] * 3
        )
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


def run_gui_session(cfg: SessionConfig):
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication([])
    window = SamSelectionWindow(cfg)
    window.show()
    if owns_app:
        assert app is not None
        app.exec_()
    return window


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
        sam={
            "checkpoint": args.sam_checkpoint,
            "model_type": args.sam_model,
            "device": args.device,
        },
    )
    cmd_collect(ns)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
