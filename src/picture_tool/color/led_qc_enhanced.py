#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LED 品質檢測（增強版，批次友善）
Python 3.10+
依賴：opencv-python-headless, numpy

重點：
- 修正 lru_cache 導致 ndarray 不可雜湊問題（改手寫 LRU 快取 + 鎖）
- 建模/批次預設不使用快取；單張檢測使用快取
- 黑洞偵測改為 Black-hat（亮底局部暗點）
"""

from __future__ import annotations
import argparse
from datetime import datetime
import json
import logging
import math
import os
import re
import sys
import time
import warnings
import hashlib
from collections import OrderedDict
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from matplotlib.font_manager import FontProperties

# ----------------------------
# 基本設定
# ----------------------------


def safe_percentile(a: np.ndarray, q: float, default=np.nan):
    a = np.asarray(a)
    return np.percentile(a, q) if a.size > 0 else default


def safe_ratio(numer_mask: np.ndarray, denom_count: int, default=np.nan):
    return (
        (float(np.count_nonzero(numer_mask)) / denom_count * 100.0)
        if denom_count > 0
        else default
    )


warnings.filterwarnings("ignore", category=UserWarning)
# ---- JSON 安全轉換器 ----


def _json_default(o):
    import numpy as _np

    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    # dataclass/其他無法序列化的型別，最後一招轉成字串
    return str(o)


SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif")


DEFAULT_COLOR_NAMES = ("Red", "Green", "Blue", "White")
DEFAULT_COLOR_ALIASES = {name.lower(): name for name in DEFAULT_COLOR_NAMES}
DEFAULT_COLOR_HUE_RANGES = {
    "Red": (0.0, 20.0),
    "Green": (35.0, 85.0),
    "Blue": (85.0, 130.0),
    "White": (0.0, 360.0),
}


class ColorPalette:
    """Encapsulates color names, aliases, and hue ranges used by LED QC."""

    def __init__(
        self,
        names: Optional[Sequence[str]] = None,
        aliases: Optional[Mapping[str, str]] = None,
        hue_ranges: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> None:
        clean_names: List[str] = []
        for raw in names or DEFAULT_COLOR_NAMES:
            if raw is None:
                continue
            name = str(raw).strip()
            if name and name not in clean_names:
                clean_names.append(name)
        if not clean_names:
            clean_names = list(DEFAULT_COLOR_NAMES)
        self.names: Tuple[str, ...] = tuple(clean_names)

        alias_map: Dict[str, str] = {name.lower(): name for name in self.names}
        if aliases is not None:
            for alias, target in aliases.items():
                alias_key = str(alias or "").strip().lower()
                target_name = str(target or "").strip()
                if alias_key and target_name in self.names:
                    alias_map[alias_key] = target_name
        self.aliases: Dict[str, str] = alias_map

        normalized_ranges: Dict[str, Tuple[float, float]] = {}
        base_ranges: Mapping[str, Sequence[float]]
        base_ranges = hue_ranges if hue_ranges is not None else {}
        for name in self.names:
            rng_obj = (
                base_ranges.get(name)
                or DEFAULT_COLOR_HUE_RANGES.get(name)
                or (0.0, 360.0)
            )
            try:
                start_val = float(rng_obj[0])
                end_val = float(rng_obj[1])
            except (TypeError, ValueError, IndexError):
                start_val, end_val = DEFAULT_COLOR_HUE_RANGES.get(name, (0.0, 360.0))
            normalized_ranges[name] = (start_val, end_val)
        self.hue_ranges: Dict[str, Tuple[float, float]] = normalized_ranges

        self.regex: re.Pattern = _build_color_regex(self.aliases)

    @classmethod
    def default(cls) -> "ColorPalette":
        return cls(
            list(DEFAULT_COLOR_NAMES), DEFAULT_COLOR_ALIASES, DEFAULT_COLOR_HUE_RANGES
        )

    def canonical(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return self.aliases.get(str(value).strip().lower())

    def hue_range(self, color: str) -> Tuple[float, float]:
        return self.hue_ranges.get(
            color, DEFAULT_COLOR_HUE_RANGES.get(color, (0.0, 360.0))
        )

    def to_config(self) -> Dict[str, object]:
        return {
            "colors": list(self.names),
            "color_aliases": {alias: target for alias, target in self.aliases.items()},
            "color_hue_ranges": {
                name: [start, end] for name, (start, end) in self.hue_ranges.items()
            },
        }


def _build_color_regex(alias_map: Mapping[str, str]) -> re.Pattern:
    if not alias_map:
        return re.compile(r"(red|green|blue|white)", re.IGNORECASE)
    pattern = "|".join(sorted(alias_map.keys(), key=len, reverse=True))
    return re.compile(f"({pattern})", re.IGNORECASE)


_ACTIVE_PALETTE: ColorPalette = ColorPalette.default()
COLORS: Tuple[str, ...] = _ACTIVE_PALETTE.names
_COLOR_CANONICAL_MAP: Dict[str, str] = dict(_ACTIVE_PALETTE.aliases)
FNAME_COLOR_RE = _ACTIVE_PALETTE.regex


def get_active_palette() -> ColorPalette:
    return _ACTIVE_PALETTE


def set_active_palette(palette: ColorPalette) -> Tuple[Tuple[str, ...], Dict[str, str]]:
    global _ACTIVE_PALETTE, COLORS, _COLOR_CANONICAL_MAP, FNAME_COLOR_RE
    _ACTIVE_PALETTE = palette
    COLORS = palette.names
    _COLOR_CANONICAL_MAP = dict(palette.aliases)
    FNAME_COLOR_RE = palette.regex
    return COLORS, _COLOR_CANONICAL_MAP


def create_palette_from_config(cfg: Dict[str, object]) -> ColorPalette:
    colors_obj = cfg.get("colors")
    colors_seq: Optional[List[str]]
    if isinstance(colors_obj, (list, tuple)):
        colors_seq = [str(item).strip() for item in colors_obj if str(item).strip()]
    else:
        colors_seq = None

    aliases_obj = cfg.get("color_aliases")
    aliases_map: Optional[Dict[str, str]]
    if isinstance(aliases_obj, dict):
        alias_temp: Dict[str, str] = {}
        for raw_key, raw_value in aliases_obj.items():
            key = str(raw_key or "").strip().lower()
            value = str(raw_value or "").strip()
            if key and value:
                alias_temp[key] = value
        aliases_map = alias_temp
    else:
        aliases_map = None

    ranges_obj = cfg.get("color_hue_ranges")
    ranges_map: Optional[Dict[str, List[float]]]
    if isinstance(ranges_obj, dict):
        temp: Dict[str, List[float]] = {}
        for raw_key, raw_value in ranges_obj.items():
            key = str(raw_key or "").strip()
            if not key:
                continue
            if isinstance(raw_value, (list, tuple)) and len(raw_value) >= 2:
                try:
                    start_val = float(raw_value[0])
                    end_val = float(raw_value[1])
                except (TypeError, ValueError):
                    continue
                temp[key] = [start_val, end_val]
        ranges_map = temp
    else:
        ranges_map = None

    return ColorPalette(colors_seq, aliases_map, ranges_map)


@contextmanager
def use_palette(palette: ColorPalette):
    previous = get_active_palette()
    if palette is previous:
        yield
        return
    set_active_palette(palette)
    try:
        yield
    finally:
        set_active_palette(previous)


def set_active_colors(
    colors: Optional[Sequence[str]] = None,
    aliases: Optional[Dict[str, str]] = None,
    hue_ranges: Optional[Dict[str, Sequence[float]]] = None,
) -> Tuple[Tuple[str, ...], Dict[str, str]]:
    config_seed: Dict[str, object] = {}
    if colors is not None:
        config_seed["colors"] = list(colors)
    if aliases is not None:
        config_seed["color_aliases"] = dict(aliases)
    if hue_ranges is not None:
        config_seed["color_hue_ranges"] = dict(hue_ranges)
    _auto_fill_color_config(config_seed)
    palette = create_palette_from_config(config_seed)
    return set_active_palette(palette)


def _auto_fill_color_config(cfg: dict) -> None:
    colors_obj = cfg.get("colors")
    if isinstance(colors_obj, (list, tuple)):
        candidate = [str(item).strip() for item in colors_obj]
    else:
        candidate = list(DEFAULT_COLOR_NAMES)
    candidate = [name for name in candidate if name]
    if not candidate:
        candidate = list(DEFAULT_COLOR_NAMES)

    seen: set[str] = set()
    norm_colors: List[str] = []
    for name in candidate:
        if name not in seen:
            seen.add(name)
            norm_colors.append(name)
    cfg["colors"] = norm_colors

    aliases_obj = cfg.get("color_aliases")
    aliases: Dict[str, str] = {}
    if isinstance(aliases_obj, dict):
        for raw_key, raw_value in aliases_obj.items():
            key = str(raw_key or "").strip().lower()
            value = str(raw_value or "").strip()
            if key and value:
                aliases[key] = value
    for name in norm_colors:
        aliases.setdefault(name.lower(), name)
    default_order = list(DEFAULT_COLOR_NAMES)
    for idx, default_name in enumerate(default_order):
        key = default_name.lower()
        if key in aliases:
            continue
        if not norm_colors:
            continue
        target_name = norm_colors[idx] if idx < len(norm_colors) else norm_colors[-1]
        aliases[key] = target_name
    cfg["color_aliases"] = aliases

    ranges_obj = cfg.get("color_hue_ranges")
    ranges: Dict[str, List[float]] = {}
    if isinstance(ranges_obj, dict):
        for raw_key, raw_value in ranges_obj.items():
            key = str(raw_key or "").strip()
            if not key:
                continue
            if isinstance(raw_value, (list, tuple)) and len(raw_value) >= 2:
                try:
                    start_val = float(raw_value[0])
                    end_val = float(raw_value[1])
                except (TypeError, ValueError):
                    continue
                ranges[key] = [start_val, end_val]
    default_ranges = DEFAULT_COLOR_HUE_RANGES
    for idx, name in enumerate(norm_colors):
        if name in ranges:
            continue
        if name in default_ranges:
            start_val, end_val = default_ranges[name]
        elif idx < len(default_order):
            start_val, end_val = default_ranges.get(default_order[idx], (0.0, 360.0))
        else:
            start_val, end_val = (0.0, 360.0)
        ranges[name] = [float(start_val), float(end_val)]
    cfg["color_hue_ranges"] = ranges


def normalize_color_config(cfg: dict) -> dict:
    _auto_fill_color_config(cfg)
    palette = create_palette_from_config(cfg)
    set_active_palette(palette)
    cfg["colors"] = list(palette.names)
    cfg["color_aliases"] = dict(palette.aliases)
    cfg["color_hue_ranges"] = {
        name: [start, end] for name, (start, end) in palette.hue_ranges.items()
    }

    per_color_min = cfg.get("color_conf_min_per_color")
    normalized_per_color = {}
    if isinstance(per_color_min, dict):
        for key, value in per_color_min.items():
            canonical = palette.canonical(key) or (
                str(key).strip() if str(key).strip() in palette.names else None
            )
            if not canonical:
                continue
            try:
                normalized_per_color[canonical] = float(value)
            except (TypeError, ValueError):
                continue
    cfg["color_conf_min_per_color"] = normalized_per_color

    def _safe_float(name: str, default: float) -> float:
        try:
            return float(cfg.get(name, default))
        except (TypeError, ValueError):
            return default

    cfg["color_hue_range_margin"] = _safe_float("color_hue_range_margin", 8.0)
    cfg["color_hue_range_weight"] = max(
        0.0, _safe_float("color_hue_range_weight", 0.35)
    )
    cfg["color_conf_hue_balance"] = min(
        1.0, max(0.0, _safe_float("color_conf_hue_balance", 0.4))
    )
    return cfg


def get_color_hue_range(cfg: dict, color: str) -> Tuple[float, float]:
    ranges_obj = cfg.get("color_hue_ranges")
    ranges_map = ranges_obj if isinstance(ranges_obj, dict) else {}
    value = ranges_map.get(color)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            start = float(value[0])
            end = float(value[1])
            return start, end
        except (TypeError, ValueError):
            pass
    return DEFAULT_COLOR_HUE_RANGES.get(color, (0.0, 360.0))


DEFAULT_CONFIG = {
    "hist_bins": [12, 12, 12],
    "min_led_area_ratio": 0.01,
    "max_led_area_ratio": 0.99,
    "default_hist_thr": 0.25,
    "sigma_multiplier": 2,
    "min_std_fallback": 8.0,
    "morphology_kernel_size": 5,
    "rotation_angles": [0, 90, 180, 270],
    "max_workers": 4,
    "cache_size": 256,
    "v_headroom": 5.0,  # 亮度上界緩衝，避免 8bit 飽和邊界誤判
    "color_conf_min": 0.20,
    "color_conf_min_white": 0.30,  # 白建議更嚴
    "color_conf_min_per_color": {},
    "color_hue_range_margin": 8.0,
    "color_hue_range_weight": 0.35,
    "color_conf_hue_balance": 0.4,
    "color_hue_auto_sigma": 2.5,
    "color_hue_auto_min_width": 20.0,
    "color_hue_auto_std_fallback": 6.0,
    # 顏色判定相關
    # black-hat 洞洞偵測
    "blackhat_kernel": 7,  # 橢圓核尺寸（奇數）
    "hole_k": 2.5,  # 門檻 = mean + k*std on blackhat
    "hole_min_area": 15,  # 最小暗洞面積
    # 顏色判定相關
    "color_hist_min_conf": 0.35,  # 提高顏色直方圖最小置信度
    "det_conf_weights": {
        "brightness": 0.25,
        "uniformity": 0.20,
        "area": 0.30,
        "holes": 0.25,
    },
    "det_conf_color_alpha": 0.05,  # 降低顏色對總分的影響
    "colors": list(DEFAULT_COLOR_NAMES),
    "color_aliases": dict(DEFAULT_COLOR_ALIASES),
    "color_hue_ranges": {
        name: list(range_vals) for name, range_vals in DEFAULT_COLOR_HUE_RANGES.items()
    },
    # 白光判定參數
    "white_s_p90_max": 80,  # 降低飽和度上限，更嚴格
    "white_v_p50_min": 200,  # 提高亮度要求
    "white_conf_min": 0.45,  # 調整白光置信度門檻
    "white_weights": [0.45, 0.35, 0.20],  # [S項, V項, RGB方差項] 權重
    "white_ring_erode_px": 6,
    # 遮罩處理
    "mask_shrink_when_large": True,
    "mask_shrink_target": 0.85,  # 期望收縮到 ~85%
    "mask_shrink_iter_max": 2,  # 最多收縮 2 次，避免過頭
    "build_min_feat_conf": 0.20,
    # 環形區域相關
    "ring_target_ratio": 0.25,  # 環形面積占整個 LED mask 的目標比例
    "ring_max_iter": 4,  # 為達成目標比例最多腐蝕幾次
    # 白平衡與色彩空間
    "awb_clip_percent": 1.0,  # AWB 計算時排除最亮/最暗各百分比像素
    "lab_white_c_p90_thr": 15.0,  # Lab 色度 C 的 p90 小於此閾值，越像白
    "lab_white_weight": 0.55,  # Lab 白度分數在總白度融合中的權重
}


def apply_high_conf_preset(cfg: dict) -> dict:
    """
    以 DEFAULT_CONFIG 為基礎在執行時補齊缺漏，
    確保回傳的 dict 與內部流程預期一致。
    """
    cfg["sigma_multiplier"] = float(cfg.get("sigma_multiplier", 2.0))
    cfg["det_conf_color_alpha"] = float(cfg.get("det_conf_color_alpha", 0.10))
    cfg["color_hist_min_conf"] = float(cfg.get("color_hist_min_conf", 0.30))
    cfg["det_conf_weights"] = cfg.get(
        "det_conf_weights",
        {"brightness": 0.25, "uniformity": 0.20, "area": 0.30, "holes": 0.25},
    )

    if not isinstance(cfg.get("color_conf_min_per_color"), dict):
        cfg["color_conf_min_per_color"] = {}

    cfg.setdefault("color_hue_range_margin", 8.0)
    cfg.setdefault("color_hue_range_weight", 0.35)
    cfg.setdefault("color_conf_hue_balance", 0.4)
    cfg.setdefault("color_hue_auto_sigma", cfg.get("sigma_multiplier", 2.0))
    cfg.setdefault("color_hue_auto_min_width", 20.0)
    cfg.setdefault("color_hue_auto_std_fallback", 6.0)

    cfg.setdefault("white_s_p90_max", 100)
    cfg.setdefault("white_v_p50_min", 200)
    cfg.setdefault("white_conf_min", 0.55)
    cfg.setdefault("white_weights", [0.45, 0.35, 0.20])

    cfg.setdefault("colors", list(COLORS))
    cfg.setdefault("color_aliases", dict(DEFAULT_COLOR_ALIASES))
    cfg.setdefault(
        "color_hue_ranges",
        {
            name: list(range_vals)
            for name, range_vals in DEFAULT_COLOR_HUE_RANGES.items()
        },
    )

    return normalize_color_config(cfg)


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("led_qc")
    logger.setLevel(getattr(logging, level.upper()))
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(h)
    return logger


logger = setup_logging(os.environ.get("LED_QC_LOG_LEVEL", "INFO"))

# ----------------------------
# 小工具
# ----------------------------


def load_config(config_path: Optional[Path] = None) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if config_path and config_path.exists():
        try:
            user_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            cfg.update(user_cfg)
            logger.info(f"已載入設定檔：{config_path}")
        except Exception as e:
            logger.warning(f"讀取設定檔失敗：{e}")
    cfg = apply_high_conf_preset(cfg)
    return cfg


def save_default_config(path: Path) -> None:
    path.write_text(
        json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"已儲存預設配置檔：{path}")


def robust_imread(path: str) -> Optional[np.ndarray]:
    try:
        if not os.path.exists(path):
            logger.error(f"檔案不存在：{path}")
            return None
        size = os.path.getsize(path)
        if size == 0:
            logger.error(f"檔案為空：{path}")
            return None
        if size > 50 * 1024 * 1024:
            logger.error(f"檔案過大：{size / 1024 / 1024:.1f}MB {path}")
            return None
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            logger.error(f"OpenCV 無法讀取：{path}")
            return None
        h, w = img.shape[:2]
        if min(h, w) < 5 or max(h, w) > 10000:
            logger.error(f"圖像尺寸不合理 {w}x{h}：{path}")
            return None
        return img
    except Exception as e:
        logger.error(f"讀圖錯誤 {path}：{e}")
        return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def bhattacharyya_dist(h1: np.ndarray, h2: np.ndarray) -> float:
    if h1.shape != h2.shape:
        return float("inf")
    return float(
        cv2.compareHist(
            h1.astype("float32"), h2.astype("float32"), cv2.HISTCMP_BHATTACHARYYA
        )
    )


def safe_div(a: float, b: float, eps: float = 1e-6) -> float:
    return float(a) / (float(b) + eps)


# 快取輔助


def compute_image_md5(img: np.ndarray) -> str:
    return hashlib.md5(img.tobytes()).hexdigest()


# 形態學核心（這個 cache 不會用 ndarray 當 key）


def get_morph_kernel(size: int, shape=cv2.MORPH_ELLIPSE) -> np.ndarray:
    return cv2.getStructuringElement(shape, (size, size))


# ----------------------------
# 特徵計算（含 Black-hat 洞洞）
# ----------------------------


@dataclass
class EnhancedImageFeatures:
    color_hist: np.ndarray
    rotation_invariant_hist: np.ndarray
    mean_v: float
    std_v: float
    uniformity: float
    area_ratio: float
    hole_ratio: float
    aspect_ratio: float
    compactness: float
    contour_regularity: float
    texture_energy: float
    mask_area: int
    valid_mask: bool
    confidence_score: float
    processing_time: float
    # 遮罩內平均色彩（供訓練統計使用）
    mean_hsv: Tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )  # H(0-180), S(0-255), V(0-255)
    mean_bgr: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # B,G,R(0-255)


def _make_adaptive_led_mask(v: np.ndarray, config: dict) -> Tuple[np.ndarray, float]:
    """Adaptive LED mask generator returning (mask, used_threshold)."""
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_equalized = clahe.apply(v)
        methods = [
            (cv2.THRESH_BINARY + cv2.THRESH_OTSU, "Otsu"),
            (cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE, "Triangle"),
        ]
        best_mask: Optional[np.ndarray] = None
        best_score = 0.0
        used_thr = 0.0

        for method, _ in methods:
            thr_val, mask_arr = cv2.threshold(v_equalized, 0, 255, method)
            mask_u8 = np.asarray(mask_arr, dtype=np.uint8)

            area_ratio = float(np.count_nonzero(mask_u8)) / float(mask_u8.size)
            if (
                config["min_led_area_ratio"]
                <= area_ratio
                <= config["max_led_area_ratio"]
            ):
                score = 1.0 - abs(area_ratio - 0.3)
                if score > best_score:
                    best_mask = mask_u8
                    best_score = score
                    used_thr = float(thr_val)

        if best_mask is None:
            fallback_thr, fallback_mask = cv2.threshold(
                v_equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            best_mask = np.asarray(fallback_mask, dtype=np.uint8)
            used_thr = float(fallback_thr)

        k = get_morph_kernel(int(config.get("morphology_kernel_size", 3)))
        best_mask = np.asarray(
            cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, k, iterations=1),
            dtype=np.uint8,
        )
        best_mask = np.asarray(
            cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, k, iterations=1),
            dtype=np.uint8,
        )

        best_mask = _keep_largest_component(
            best_mask, min_area=int(config.get("hole_min_area", 20))
        )

        if bool(config.get("mask_shrink_when_large", True)):
            max_ratio = float(config.get("max_led_area_ratio", 0.98))
            target = float(config.get("mask_shrink_target", 0.85))
            iter_max = int(config.get("mask_shrink_iter_max", 2))

            area_ratio = float(np.count_nonzero(best_mask)) / float(best_mask.size)
            it = 0
            while area_ratio > max_ratio and it < iter_max:
                best_mask = np.asarray(
                    cv2.erode(best_mask, k, iterations=1),
                    dtype=np.uint8,
                )
                area_ratio = float(np.count_nonzero(best_mask)) / float(best_mask.size)
                it += 1

            if area_ratio > target and it < iter_max:
                best_mask = np.asarray(
                    cv2.erode(best_mask, k, iterations=1),
                    dtype=np.uint8,
                )

        return best_mask, used_thr

    except Exception as e:
        logger.error(f"LED mask generation failed: {e}")
        fallback = np.asarray(np.ones_like(v, dtype=np.uint8) * 255, dtype=np.uint8)
        return fallback, 128.0


def _keep_largest_component(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """保留最大連通區，過小則回傳原 mask。"""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:  # 只有背景
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask)
    if stats[idx, cv2.CC_STAT_AREA] >= int(min_area):
        out[labels == idx] = 255
    return out


def _rotation_invariant_hist(
    hsv: np.ndarray, mask: np.ndarray, cfg: dict
) -> np.ndarray:
    angles = cfg.get("rotation_angles", [0, 90, 180, 270])
    bins = cfg.get("hist_bins", [8, 8, 8])
    acc = np.zeros((bins[0], bins[1], bins[2]), dtype=np.float32)
    h, w = hsv.shape[:2]
    center = (w // 2, h // 2)
    for ang in angles:
        if ang == 0:
            hsv_r, mask_r = hsv, mask
        else:
            M = cv2.getRotationMatrix2D(center, ang, 1.0)
            hsv_r = cv2.warpAffine(hsv, M, (w, h))
            mask_r = cv2.warpAffine(mask, M, (w, h))
        hist = cv2.calcHist([hsv_r], [0, 1, 2], mask_r, bins, [0, 180, 0, 256, 0, 256])
        hist_arr = np.asarray(hist, dtype=np.float32)
        acc = np.add(acc, hist_arr, out=acc)
    normed = cv2.normalize(acc, dst=np.empty_like(acc))
    if normed is None:
        normed = acc
    return np.asarray(normed, dtype=np.float32).flatten()


def _compute_geometric(mask: np.ndarray) -> Tuple[float, float, float]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 1.0, 0.0, 0.0
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    aspect = float(max(w, h)) / float(min(w, h) + 1e-6)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    compact = 4 * math.pi * area / (peri * peri + 1e-6)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    regularity = area / (hull_area + 1e-6)
    return aspect, compact, regularity


def _texture_energy(v: np.ndarray, mask: np.ndarray) -> float:
    if np.count_nonzero(mask) == 0:
        return 0.0
    gx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.asarray(cv2.magnitude(gx, gy), dtype=np.float32)
    active = mag[mask > 0]
    if active.size == 0:
        return 0.0
    return float(np.std(active))


def _blackhat_holes(v: np.ndarray, mask: np.ndarray, cfg: dict) -> np.ndarray:
    ksize = int(cfg.get("blackhat_kernel", 9))
    k = get_morph_kernel(ksize)
    closed = np.asarray(cv2.morphologyEx(v, cv2.MORPH_CLOSE, k), dtype=np.uint8)
    v_u8 = np.asarray(v, dtype=np.uint8)
    bh = np.asarray(cv2.subtract(closed, v_u8), dtype=np.float32)
    mask_active = np.asarray(mask > 0, dtype=bool)
    bh_in = bh[mask_active]
    if bh_in.size == 0:
        return np.zeros_like(v_u8, dtype=np.uint8)
    hole_k = float(cfg.get("hole_k", 3.0))
    thr = float(np.mean(bh_in) + hole_k * np.std(bh_in))
    hole_bool = (bh > thr) & mask_active
    hole = np.zeros_like(v_u8, dtype=np.uint8)
    hole[hole_bool] = 255
    hole = np.asarray(
        cv2.morphologyEx(hole, cv2.MORPH_OPEN, k, iterations=1), dtype=np.uint8
    )
    hole = np.asarray(
        cv2.morphologyEx(hole, cv2.MORPH_CLOSE, k, iterations=1), dtype=np.uint8
    )
    return hole


# ---- 手寫 LRU 快取（鍵：image_md5 + config_str） ----
_FEATURE_CACHE: "OrderedDict[tuple[str,str], EnhancedImageFeatures]" = OrderedDict()
_CACHE_LOCK = Lock()


def _features_impl(img_bgr: np.ndarray, cfg: dict) -> EnhancedImageFeatures:
    t0 = time.time()
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mask, _ = _make_adaptive_led_mask(v, cfg)
    total = img_bgr.shape[0] * img_bgr.shape[1]
    mcount = int(np.count_nonzero(mask))
    area_ratio = mcount / float(total)
    valid_mask = cfg["min_led_area_ratio"] <= area_ratio <= cfg["max_led_area_ratio"]

    bins = cfg.get("hist_bins", [8, 8, 8])
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    rot_hist = _rotation_invariant_hist(hsv, mask, cfg)

    v_led = v[mask > 0].astype(np.float32)
    mean_v = float(np.mean(v_led)) if v_led.size else 0.0
    std_v = float(np.std(v_led)) if v_led.size else 0.0
    uniformity = 1.0 - safe_div(std_v, mean_v)

    # 遮罩內平均 HSV/BGR
    if np.count_nonzero(mask) > 0:
        # Hue 使用圓形平均
        h_led = hsv[:, :, 0][mask > 0].astype(np.float64)
        s_led = hsv[:, :, 1][mask > 0].astype(np.float64)
        if h_led.size > 0:
            angles = h_led * (2.0 * math.pi / 180.0)
            sin_m = float(np.sin(angles).mean())
            cos_m = float(np.cos(angles).mean())
            ang = math.atan2(sin_m, cos_m)
            if ang < 0:
                ang += 2.0 * math.pi
            mean_h = ang * (180.0 / (2.0 * math.pi))
        else:
            mean_h = 0.0
        mean_s = float(np.mean(s_led)) if s_led.size else 0.0
        mean_hsv = (float(mean_h), float(mean_s), float(mean_v))

        bgr_led = img_bgr[mask > 0].astype(np.float64)
        if bgr_led.size:
            mb, mg, mr = bgr_led.mean(axis=0)
            mean_bgr = (float(mb), float(mg), float(mr))
        else:
            mean_bgr = (0.0, 0.0, 0.0)
    else:
        mean_hsv = (0.0, 0.0, float(mean_v))
        mean_bgr = (0.0, 0.0, 0.0)

    hole_mask = _blackhat_holes(v, mask, cfg)
    hole_ratio = float(np.count_nonzero(hole_mask)) / float(mcount + 1e-6)

    aspect, compact, regularity = _compute_geometric(mask)
    tex = _texture_energy(v, mask)

    # 簡單可信度
    conf = 1.0
    if not valid_mask:
        conf *= 0.5
    if area_ratio < 0.1 or area_ratio > 0.8:
        conf *= 0.7
    if std_v < 5 or std_v > 100:
        conf *= 0.8
    if mcount < 100:
        conf *= 0.6

    return EnhancedImageFeatures(
        color_hist=hist,
        rotation_invariant_hist=rot_hist,
        mean_v=mean_v,
        std_v=std_v,
        uniformity=uniformity,
        area_ratio=area_ratio,
        hole_ratio=hole_ratio,
        aspect_ratio=aspect,
        compactness=compact,
        contour_regularity=regularity,
        texture_energy=tex,
        mask_area=mcount,
        valid_mask=valid_mask,
        confidence_score=max(0.0, min(1.0, conf)),
        processing_time=time.time() - t0,
        mean_hsv=mean_hsv,
        mean_bgr=mean_bgr,
    )


def compute_enhanced_features(
    img_bgr: np.ndarray, cfg: dict, use_cache: bool = True
) -> EnhancedImageFeatures:
    if not use_cache:
        return _features_impl(img_bgr, cfg)

    key = (compute_image_md5(img_bgr), json.dumps(cfg, sort_keys=True))
    with _CACHE_LOCK:
        feat = _FEATURE_CACHE.get(key)
        if feat is not None:
            _FEATURE_CACHE.move_to_end(key)
            return feat

    feat = _features_impl(img_bgr, cfg)

    with _CACHE_LOCK:
        _FEATURE_CACHE[key] = feat
        maxsize = int(cfg.get("cache_size", DEFAULT_CONFIG["cache_size"]))
        while len(_FEATURE_CACHE) > maxsize:
            _FEATURE_CACHE.popitem(last=False)
    return feat


# ----------------------------
# 模型
# ----------------------------


@dataclass
class EnhancedColorModel:
    avg_color_hist: List[float]
    avg_rotation_hist: List[float]
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
    # 訓練統計：遮罩內平均 HSV/BGR
    mean_hsv_mu: List[float] = field(default_factory=list)
    mean_hsv_std: List[float] = field(default_factory=list)
    mean_bgr_mu: List[float] = field(default_factory=list)
    mean_bgr_std: List[float] = field(default_factory=list)


@dataclass
class EnhancedReferenceModel:
    version: int
    config: dict
    colors: Dict[str, EnhancedColorModel]
    creation_time: str
    total_samples: int

    def to_json(self, path: Path) -> None:
        payload = {
            "version": self.version,
            "config": self.config,
            "colors": {k: asdict(v) for k, v in self.colors.items()},
            "creation_time": self.creation_time,
            "total_samples": self.total_samples,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"模型已儲存：{path}")

    @staticmethod
    def from_json(path: Path) -> "EnhancedReferenceModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        # 套用 presets，避免舊 config 值把新邏輯卡死
        cfg = apply_high_conf_preset(payload.get("config", DEFAULT_CONFIG))
        colors = {
            k: EnhancedColorModel(**v) for k, v in payload.get("colors", {}).items()
        }
        return EnhancedReferenceModel(
            version=payload.get("version", 2),
            config=cfg,
            colors=colors,
            creation_time=payload.get("creation_time", "unknown"),
            total_samples=payload.get("total_samples", 0),
        )


# ----------------------------
# 顏色識別與建模
# ----------------------------


def extract_color_from_name(name: str) -> Optional[str]:
    if not name:
        return None
    match = FNAME_COLOR_RE.search(name)
    if not match:
        return None
    return _COLOR_CANONICAL_MAP.get(match.group(1).lower())


def _make_ring_mask(mask: np.ndarray, erode_px: int) -> np.ndarray:
    """
    從原始 mask 取出「環帶」：ring = mask - erode(mask, erode_px)
    目的：避開中心過曝與邊緣色散，只在穩定區域評分。
    """
    erode_px = max(int(erode_px), 0)
    if erode_px == 0:
        return mask
    inner = cv2.erode(mask, get_morph_kernel(erode_px * 2 + 1), iterations=1)
    ring = cv2.subtract(mask, inner)
    # 如果環帶太薄（像素太少），退回原 mask
    return ring if np.count_nonzero(ring) > max(200, mask.size * 0.005) else mask


def _gray_world_awb(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    在 mask 內做 Gray-World White Balance：
    令三通道均值相等（簡單、穩定、成本低），回傳校正後的 bgr。
    只在判別白光時計算一次，不動用到特徵抽取與模型，以免破壞閾值。
    """
    m = mask > 0
    if not np.any(m):
        return bgr
    b, g, r = [float(np.mean(bgr[..., i][m])) for i in range(3)]
    mean = (b + g + r) / 3.0 + 1e-6
    gains = np.array([mean / b, mean / g, mean / r], dtype=np.float32)
    out = (bgr.astype(np.float32) * gains).clip(0, 255).astype(np.uint8)
    return out


def _expand_hue_range(
    start: float, end: float, margin: float
) -> List[Tuple[float, float]]:
    margin = max(0.0, float(margin))
    start = float(start) % 360.0
    end = float(end) % 360.0
    if margin:
        start = (start - margin) % 360.0
        end = (end + margin) % 360.0
    if start <= end:
        return [(start, end)]
    return [(start, 360.0), (0.0, end)]


def _hue_coverage(
    h_values_deg: np.ndarray, segments: List[Tuple[float, float]]
) -> float:
    if h_values_deg.size == 0 or not segments:
        return 0.0
    mask = np.zeros(h_values_deg.shape, dtype=bool)
    for start, end in segments:
        start = float(start)
        end = float(end)
        if start <= end:
            mask |= (h_values_deg >= start) & (h_values_deg <= end)
        else:
            mask |= (h_values_deg >= start) | (h_values_deg <= end)
    return float(np.count_nonzero(mask)) / float(h_values_deg.size)


def _decide_color_robust(
    img_bgr: np.ndarray, config: dict, model: "EnhancedReferenceModel"
) -> Tuple[str, float, Dict[str, float], Dict[str, float]]:
    """
    綜合統計遮罩、顏色模型與直方圖距離，輸出 (顏色, 信度, 距離映射, 色相覆蓋率)。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    mask, _ = _make_adaptive_led_mask(v, config)
    palette = create_palette_from_config(config)

    color_names = list(model.colors.keys())
    if not color_names:
        return "White", 0.0, {}, {}

    if np.count_nonzero(mask) == 0:
        empty_coverage = {color: 0.0 for color in color_names}
        distances = {color: 1.0 for color in color_names}
        fallback = "White" if "White" in model.colors else color_names[0]
        return fallback, 0.0, distances, empty_coverage

    led_region = mask > 0
    h_vals = hsv[..., 0][led_region].astype(np.float32)
    s_vals = hsv[..., 1][led_region].astype(np.float32)
    v_vals = hsv[..., 2][led_region].astype(np.float32)
    h_vals_deg = h_vals * 2.0  # OpenCV Hue ∈ [0,180)，轉換成 0~360 度

    hue_margin = float(config.get("color_hue_range_margin", 8.0))
    hue_weight = max(0.0, float(config.get("color_hue_range_weight", 0.35)))
    hue_balance = min(1.0, max(0.0, float(config.get("color_conf_hue_balance", 0.4))))

    coverage: Dict[str, float] = {}
    for color in color_names:
        start, end = palette.hue_ranges.get(color, (0.0, 360.0))
        segments = _expand_hue_range(start, end, hue_margin)
        coverage[color] = _hue_coverage(h_vals_deg, segments)

    s_p90 = float(np.percentile(s_vals, 90)) if s_vals.size else 0.0
    v_p50 = float(np.percentile(v_vals, 50)) if v_vals.size else 0.0
    s_mean = float(np.mean(s_vals)) if s_vals.size else 0.0
    v_mean = float(np.mean(v_vals)) if v_vals.size else 0.0

    white_score = None
    white_threshold = float(config.get("white_conf_min", 0.45))
    if (
        "White" in model.colors
        and s_p90 < float(config.get("white_s_p90_max", 80))
        and v_p50 > float(config.get("white_v_p50_min", 200))
    ):
        bgr_means = np.mean(img_bgr[led_region], axis=0)
        rgb_std = float(np.std(bgr_means))
        rgb_balance = 1.0 - min(1.0, rgb_std / 20.0)

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        a = lab[..., 1][led_region].astype(np.float32)
        b = lab[..., 2][led_region].astype(np.float32)
        chroma = np.sqrt(a * a + b * b)
        c_p90 = float(np.percentile(chroma, 90)) if chroma.size else 0.0

        white_weights = config.get("white_weights", [0.45, 0.35, 0.20])
        if not isinstance(white_weights, (list, tuple)) or len(white_weights) < 3:
            white_weights = [0.45, 0.35, 0.20]
        w0, w1, w2 = (float(white_weights[i]) for i in range(3))
        white_mix = (
            w0 * (1.0 - s_mean / 255.0) + w1 * (v_mean / 255.0) + w2 * rgb_balance
        )

        lab_weight = float(config.get("lab_white_weight", 0.55))
        lab_score = (
            1.0 - min(1.0, c_p90 / float(config.get("lab_white_c_p90_thr", 15.0)))
            if c_p90
            else 1.0
        )
        white_score = lab_weight * lab_score + (1.0 - lab_weight) * white_mix

    bins = config.get("hist_bins", [12, 12, 12])
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])
    hist_norm = cv2.normalize(hist, dst=np.empty_like(hist))
    hist_vec = np.asarray(
        hist_norm if hist_norm is not None else hist, dtype=np.float32
    ).flatten()

    distances = {
        color: bhattacharyya_dist(
            hist_vec, np.array(cm.avg_color_hist, dtype=np.float32)
        )
        for color, cm in model.colors.items()
    }
    if not distances:
        fallback = "White" if "White" in model.colors else color_names[0]
        return fallback, 0.0, {}, coverage

    scored = {
        color: float(dist) + (1.0 - coverage.get(color, 0.0)) * hue_weight
        for color, dist in distances.items()
    }
    best_candidate = min(scored, key=lambda color: scored[color])
    best_color = best_candidate

    if white_score is not None and white_score > white_threshold:
        best_color = "White" if "White" in model.colors else best_candidate

    chosen_dist = float(distances.get(best_color, 1.0))
    dist_conf = max(0.0, min(1.0, 1.0 - chosen_dist))
    hue_conf = coverage.get(best_color, 0.0)
    color_conf = (1.0 - hue_balance) * dist_conf + hue_balance * hue_conf
    if best_color == "White" and white_score is not None:
        color_conf = max(color_conf, min(1.0, float(white_score)))

    return best_color, color_conf, distances, coverage


def _build_color_model_statistics(
    feats: List[EnhancedImageFeatures], cfg: dict
) -> dict:
    n = len(feats)

    # 降低sigma倍數，從3.0降到2.0

    # 色彩直方圖
    ch = np.stack([f.color_hist for f in feats])
    rh = np.stack([f.rotation_invariant_hist for f in feats])
    avg_ch = np.mean(ch, axis=0)
    avg_rh = np.mean(rh, axis=0)

    # 計算Bhattacharyya距離
    col_distances = []
    rot_distances = []
    for f in feats:
        col_d = cv2.compareHist(
            f.color_hist.astype("float32"),
            avg_ch.astype("float32"),
            cv2.HISTCMP_BHATTACHARYYA,
        )
        rot_d = cv2.compareHist(
            f.rotation_invariant_hist.astype("float32"),
            avg_rh.astype("float32"),
            cv2.HISTCMP_BHATTACHARYYA,
        )
        col_distances.append(float(col_d))
        rot_distances.append(float(rot_d))

    # 使用95百分位數而非mean+std，更寬鬆
    hist_thr = float(np.percentile(col_distances, 95))
    rot_thr = float(np.percentile(rot_distances, 95))
    # 確保最小閾值
    min_thr = cfg.get("default_hist_thr", 0.15)
    hist_thr = max(hist_thr, min_thr)
    rot_thr = max(rot_thr, min_thr)

    def safe_stats(vals):
        """計算統計量，確保標準差不會太小"""
        arr = np.array(vals, dtype=np.float32)
        mu = float(np.mean(arr))
        std = float(np.std(arr))

        # 使用更大的最小標準差保障
        min_std = cfg.get("min_std_fallback", 2.0)  # 原本1.0提高到2.0
        if std < min_std:
            std = min_std

        return mu, std

    def safe_stats_vec(vectors: List[Tuple[float, float, float]]):
        """對 3 維向量做逐維統計，並套用最小標準差保護"""
        if not vectors:
            return [0.0, 0.0, 0.0], [cfg.get("min_std_fallback", 2.0)] * 3
        arr = np.array(vectors, dtype=np.float32)  # (N,3)
        mu = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        min_std = float(cfg.get("min_std_fallback", 2.0))
        std = np.maximum(std, min_std)
        return mu.astype(float).tolist(), std.astype(float).tolist()

    # 計算各特徵統計量
    mean_v_mu, mean_v_std = safe_stats([f.mean_v for f in feats])
    std_v_mu, std_v_std = safe_stats([f.std_v for f in feats])
    uni_mu, uni_std = safe_stats([f.uniformity for f in feats])
    area_mu, area_std = safe_stats([f.area_ratio for f in feats])
    hole_mu, hole_std = safe_stats([f.hole_ratio for f in feats])
    ar_mu, ar_std = safe_stats([f.aspect_ratio for f in feats])
    comp_mu, comp_std = safe_stats([f.compactness for f in feats])
    reg_mu, reg_std = safe_stats([f.contour_regularity for f in feats])
    tex_mu, tex_std = safe_stats([f.texture_energy for f in feats])

    # 遮罩內平均 HSV / BGR 統計
    hsv_mu, hsv_std = safe_stats_vec([f.mean_hsv for f in feats])
    bgr_mu, bgr_std = safe_stats_vec([f.mean_bgr for f in feats])

    return {
        "avg_color_hist": avg_ch.tolist(),
        "avg_rotation_hist": avg_rh.tolist(),
        "hist_thr": hist_thr,
        "rotation_hist_thr": rot_thr,
        "mean_v_mu": mean_v_mu,
        "mean_v_std": mean_v_std,
        "std_v_mu": std_v_mu,
        "std_v_std": std_v_std,
        "uniformity_mu": uni_mu,
        "uniformity_std": uni_std,
        "area_ratio_mu": area_mu,
        "area_ratio_std": area_std,
        "hole_ratio_mu": hole_mu,
        "hole_ratio_std": hole_std,
        "aspect_ratio_mu": ar_mu,
        "aspect_ratio_std": ar_std,
        "compactness_mu": comp_mu,
        "compactness_std": comp_std,
        "regularity_mu": reg_mu,
        "regularity_std": reg_std,
        "texture_energy_mu": tex_mu,
        "texture_energy_std": tex_std,
        "samples": n,
        "avg_confidence": float(np.mean([f.confidence_score for f in feats])),
        "last_updated": "optimized_version",
        "mean_hsv_mu": hsv_mu,
        "mean_hsv_std": hsv_std,
        "mean_bgr_mu": bgr_mu,
        "mean_bgr_std": bgr_std,
    }


def _maybe_update_auto_hue_ranges(
    cfg: dict, color_models: Dict[str, "EnhancedColorModel"]
) -> List[str]:
    """Derive hue ranges for colors that do not have explicit config."""
    ranges = dict(cfg.get("color_hue_ranges") or {})
    updated: List[str] = []
    auto_sigma = float(
        cfg.get("color_hue_auto_sigma", cfg.get("sigma_multiplier", 2.0))
    )
    min_width = float(cfg.get("color_hue_auto_min_width", 20.0))
    std_fallback = float(cfg.get("color_hue_auto_std_fallback", 6.0))
    min_width = max(min_width, 1.0)
    std_fallback = max(std_fallback, 1.0)

    for name, model in color_models.items():
        existing = ranges.get(name)
        need_auto = False
        if existing is None:
            need_auto = True
        else:
            try:
                start, end = float(existing[0]), float(existing[1])
            except Exception:
                need_auto = True
            else:
                if name not in DEFAULT_COLOR_HUE_RANGES:
                    if abs(start) < 1e-4 and abs(end - 360.0) < 1e-4:
                        need_auto = True
        if not need_auto:
            continue
        if not getattr(model, "mean_hsv_mu", None):
            continue
        try:
            hue_mean = float(model.mean_hsv_mu[0]) * 2.0
        except (IndexError, TypeError, ValueError):
            continue
        try:
            hue_std = float(model.mean_hsv_std[0]) * 2.0
        except (IndexError, TypeError, ValueError):
            hue_std = 0.0
        hue_std = max(hue_std, std_fallback)
        half = max(auto_sigma * hue_std, min_width / 2.0)
        if half >= 180.0:
            start, end = 0.0, 360.0
        else:
            start = (hue_mean - half) % 360.0
            end = (hue_mean + half) % 360.0
        ranges[name] = [round(start, 2), round(end, 2)]
        updated.append(name)

    if updated:
        cfg["color_hue_ranges"] = ranges
    return updated


def build_enhanced_reference(ref_dir: Path, cfg: dict) -> EnhancedReferenceModel:
    cfg = normalize_color_config(cfg)
    palette = create_palette_from_config(cfg)

    with use_palette(palette):
        valid_suffixes = {s.lower() for s in SUPPORTED_FORMATS}
        paths = sorted(
            {
                p.resolve()
                for p in ref_dir.rglob("*")
                if p.suffix.lower() in valid_suffixes
            }
        )
        if not paths:
            raise RuntimeError(f"No reference images found in {ref_dir}")
        logger.info(f"Discovered {len(paths)} reference images (deduplicated)")

        buckets: Dict[str, List[EnhancedImageFeatures]] = {c: [] for c in palette.names}
        failed: List[str] = []

        def process(path: Path):
            try:
                img = robust_imread(str(path))
                if img is None:
                    return (None, None, f"{path.name}: failed to read image")

                color = extract_color_from_name(path.name) or extract_color_from_name(
                    path.parent.name
                )
                if color is None:
                    joined = "/".join(palette.names) or "configured colors"
                    return (
                        None,
                        None,
                        f"{path.name}: unable to infer color label (expect directory/file name containing {joined})",
                    )

                feat = compute_enhanced_features(img, cfg, use_cache=False)
                min_conf = float(cfg.get("build_min_feat_conf", 0.20))
                if feat.confidence_score < min_conf:
                    return (
                        None,
                        None,
                        f"{path.name}: feature confidence too low (conf={feat.confidence_score:.2f} < {min_conf})",
                    )

                return (color, feat, "")
            except Exception as e:
                return (None, None, f"{path.name}: {e}")

        with ThreadPoolExecutor(max_workers=cfg.get("max_workers", 4)) as ex:
            futures = [ex.submit(process, p) for p in paths]
            for fut in as_completed(futures):
                color, feat, err = fut.result()
                if err:
                    failed.append(err)
                elif color and feat:
                    buckets[color].append(feat)

        if failed:
            logger.warning(
                f"Reference extraction failed for {len(failed)} files (showing first 5)"
            )
            for m in failed[:5]:
                logger.warning(f"- {m}")

        colors: Dict[str, EnhancedColorModel] = {}
        total = 0
        for c in palette.names:
            feats = buckets.get(c, [])
            if not feats:
                logger.warning(f"{c} has no samples")
                continue
            colors[c] = EnhancedColorModel(**_build_color_model_statistics(feats, cfg))
            total += len(feats)

        auto_adjusted = _maybe_update_auto_hue_ranges(cfg, colors)
        if auto_adjusted:
            logger.info("Auto-derived hue ranges: %s", ", ".join(auto_adjusted))

        if not colors:
            joined = "/".join(palette.names) or "configured colors"
            raise RuntimeError(
                f"No color models were built; ensure folders/files include {joined}"
            )

    return EnhancedReferenceModel(
        version=2,
        config=cfg,
        colors=colors,
        creation_time=time.strftime("%Y-%m-%d %H:%M:%S"),
        total_samples=total,
    )


@dataclass
class EnhancedDetectionResult:
    is_anomaly: bool
    confidence: float
    color_used: str
    color_confidence: float
    reasons: List[str]
    severity_score: float
    feature_anomalies: Dict[str, float]
    processing_time: float
    features: EnhancedImageFeatures
    anomaly_regions: List[Tuple[int, int, int, int]]
    recommendations: List[str]
    color_distances: Dict[str, float] = field(default_factory=dict)
    color_hue_coverage: Dict[str, float] = field(default_factory=dict)


def _detect_anomalies(
    features: EnhancedImageFeatures,
    model: EnhancedColorModel,
    config: dict,
    sensitivity: float,
    color_confidence: float = 1.0,
) -> dict:
    """
    依參考模型偵測異常；當 color_confidence 過低時，跳過色彩相關 NG，
    並對亮度上界加 headroom（避免 8-bit 飽和邊界誤判）。
    回傳: {"reasons": List[str], "feature_scores": Dict[str, float]}
    """
    reasons = []
    feature_scores = {}
    sigma_mult = float(config.get("sigma_multiplier", 3.0)) * float(sensitivity)

    # ---- 色彩直方圖 / 旋轉不變（僅在顏色信度足夠時啟用）----
    min_conf = float(config.get("color_hist_min_conf", 0.60))
    if color_confidence < min_conf:
        pass  # 不做顏色直方圖與旋轉直方圖檢查
    else:
        color_dist = bhattacharyya_dist(
            features.color_hist, np.array(model.avg_color_hist)
        )
        if color_dist > model.hist_thr * sensitivity:
            reasons.append(
                f"色彩分布異常 (dist={color_dist:.3f} > {model.hist_thr:.3f})"
            )
            feature_scores["color_histogram"] = color_dist / (model.hist_thr + 1e-6)

        rotation_dist = bhattacharyya_dist(
            features.rotation_invariant_hist, np.array(model.avg_rotation_hist)
        )
        if rotation_dist > model.rotation_hist_thr * sensitivity:
            reasons.append(
                f"旋轉不變特徵異常 (dist={rotation_dist:.3f} > {model.rotation_hist_thr:.3f})"
            )
            feature_scores["rotation_invariant"] = rotation_dist / (
                model.rotation_hist_thr + 1e-6
            )

    # ---- 亮度（上界加 headroom）----
    v_guard = float(config.get("v_headroom", 0.0))
    mean_v_low = model.mean_v_mu - sigma_mult * model.mean_v_std
    mean_v_high = model.mean_v_mu + sigma_mult * model.mean_v_std + v_guard
    if not (mean_v_low <= features.mean_v <= mean_v_high):
        reasons.append(
            f"平均亮度異常 (V={features.mean_v:.1f}, 正常範圍=[{mean_v_low:.1f}, {mean_v_high:.1f}])"
        )
        feature_scores["brightness"] = abs(features.mean_v - model.mean_v_mu) / (
            model.mean_v_std + 1e-6
        )

    # ---- 均勻度 ----
    uniformity_low = model.uniformity_mu - sigma_mult * model.uniformity_std
    if features.uniformity < uniformity_low:
        reasons.append(
            f"亮度不均勻 (均勻度={features.uniformity:.3f}, 下限={uniformity_low:.3f})"
        )
        feature_scores["uniformity"] = (uniformity_low - features.uniformity) / (
            model.uniformity_std + 1e-6
        )

    # ---- 洞洞比例 ----
    hole_high = model.hole_ratio_mu + sigma_mult * model.hole_ratio_std
    if features.hole_ratio > hole_high:
        reasons.append(
            f"黑洞過多 (比例={features.hole_ratio:.3f}, 上限={hole_high:.3f})"
        )
        feature_scores["holes"] = (features.hole_ratio - model.hole_ratio_mu) / (
            model.hole_ratio_std + 1e-6
        )

    # ---- 幾何：長寬比 ----
    aspect_high = model.aspect_ratio_mu + sigma_mult * model.aspect_ratio_std
    if features.aspect_ratio > aspect_high:
        reasons.append(
            f"形狀異常 (長寬比={features.aspect_ratio:.2f}, 上限={aspect_high:.2f})"
        )
        feature_scores["aspect_ratio"] = (
            features.aspect_ratio - model.aspect_ratio_mu
        ) / (model.aspect_ratio_std + 1e-6)

    # ---- 幾何：緊密度 ----
    compactness_low = model.compactness_mu - sigma_mult * model.compactness_std
    if features.compactness < compactness_low:
        reasons.append(
            f"形狀不規則 (緊密度={features.compactness:.3f}, 下限={compactness_low:.3f})"
        )
        feature_scores["compactness"] = (compactness_low - features.compactness) / (
            model.compactness_std + 1e-6
        )

    return {"reasons": reasons, "feature_scores": feature_scores}


def _detection_confidence(
    ft: EnhancedImageFeatures, cm: EnhancedColorModel, color_conf: float, cfg: dict
) -> float:
    """
    檢測信度：
    - 有效遮罩給較高的基礎分 base（0.74）
    - Sigmoid 中心左移：1/(1+exp(z-0.8))，正常樣本更容易 >0.75
    - 顏色信度權重以 cfg['det_conf_color_alpha'] 控制（建議 <=0.05）
    """

    def _z(val: float, mu: float, std: float) -> float:
        std = max(float(std), 1e-6)
        return abs(float(val) - float(mu)) / std

    z_brightness = _z(ft.mean_v, cm.mean_v_mu, cm.mean_v_std)
    z_uniformity = max(
        0.0, (cm.uniformity_mu - ft.uniformity) / max(cm.uniformity_std, 1e-6)
    )
    z_area = _z(ft.area_ratio, cm.area_ratio_mu, cm.area_ratio_std)
    z_holes = max(
        0.0, (ft.hole_ratio - cm.hole_ratio_mu) / max(cm.hole_ratio_std, 1e-6)
    )

    w = cfg.get(
        "det_conf_weights",
        {"brightness": 0.25, "uniformity": 0.20, "area": 0.30, "holes": 0.25},
    )
    denom = max(1e-6, sum(w.values()))
    z_weighted = (
        z_brightness * w.get("brightness", 0.0)
        + z_uniformity * w.get("uniformity", 0.0)
        + z_area * w.get("area", 0.0)
        + z_holes * w.get("holes", 0.0)
    ) / denom

    raw = 1.0 / (1.0 + np.exp(z_weighted - 0.8))  # 左移
    base = 0.74 if ft.valid_mask else 0.48
    raw = max(raw, base)

    alpha = float(cfg.get("det_conf_color_alpha", 0.05))
    final_conf = (1.0 - alpha) * raw + alpha * max(0.0, min(1.0, color_conf))
    return float(max(0.0, min(1.0, final_conf)))


def _severity_score(details: dict) -> float:
    if not details["feature_scores"]:
        return 0.0
    w = {
        "color_histogram": 0.3,
        "brightness": 0.25,
        "uniformity": 0.2,
        "holes": 0.15,
        "aspect_ratio": 0.05,
        "compactness": 0.05,
    }
    s = sum(
        details["feature_scores"].get(k, 0.0) * w.get(k, 0.1)
        for k in set(w) | set(details["feature_scores"])
    )
    return float(min(1.0, s / max(1e-6, sum(w.values()))))


def _recommend(details: dict, ft: EnhancedImageFeatures) -> List[str]:
    out = []
    if "color_histogram" in details["feature_scores"]:
        out.append("檢查 LED 發光波長/螢光粉一致性與製程穩定度")
    if "brightness" in details["feature_scores"]:
        out.append("亮度偏差：檢查驅動電流、曝光與供電")
    if "uniformity" in details["feature_scores"]:
        out.append("亮度不均：檢查晶片/封裝與光學散射")
    if "holes" in details["feature_scores"]:
        out.append("暗點缺陷：檢查清潔度與材料均質性")
    if not ft.valid_mask:
        out.append("影像分割品質差：改善拍攝/遮光/固定治具")
    return out


def _locate_regions(
    img_bgr: np.ndarray, ft: EnhancedImageFeatures, cfg: dict, details: dict
) -> List[Tuple[int, int, int, int]]:
    if "holes" not in details["feature_scores"]:
        return []
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mask, _ = _make_adaptive_led_mask(v, cfg)
    hole = _blackhat_holes(v, mask, cfg)
    cnts, _ = cv2.findContours(hole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    amin = int(cfg.get("hole_min_area", 20))
    for c in cnts:
        if cv2.contourArea(c) < amin:
            continue
        x, y, w, h = cv2.boundingRect(c)
        regions.append((x, y, w, h))
    return regions


def enhanced_detect_one(
    img_bgr: np.ndarray,
    model: EnhancedReferenceModel,
    color_hint: Optional[str] = None,
    sensitivity: float = 1.0,
) -> EnhancedDetectionResult:
    """
    推理流程（含 White 的動態覆蓋）：
      - White 時提高直方圖啟用門檻、縮黑帽核、放寬亮度上界
      - 壓低顏色信度對總分影響
    """
    t0 = time.time()

    # 1) 特徵
    ft = compute_enhanced_features(img_bgr, model.config, use_cache=True)

    # 2) 顏色決策
    auto_color, auto_conf, dist_by_color, hue_coverage = _decide_color_robust(
        img_bgr, model.config, model
    )
    dist_by_color = {k: float(v) for k, v in dist_by_color.items()}
    hue_coverage = {k: float(v) for k, v in hue_coverage.items()}

    chosen_color = auto_color
    color_conf = auto_conf
    manual_override = False
    if color_hint and color_hint in model.colors:
        chosen_color = color_hint
        color_conf = 1.0
        manual_override = True

    if chosen_color not in model.colors:
        if dist_by_color:
            chosen_color = min(dist_by_color, key=lambda name: dist_by_color[name])
            if not manual_override:
                color_conf = auto_conf
        else:
            chosen_color = next(iter(model.colors.keys()))
            if not manual_override:
                color_conf = auto_conf

    color_conf = max(0.0, min(1.0, float(color_conf)))
    hue_cov_used = float(hue_coverage.get(chosen_color, 0.0))

    cm = model.colors[chosen_color]

    # 3) 依顏色覆蓋局部參數（不污染原 config）
    local_cfg = dict(model.config)

    # 壓低顏色信度對總分影響（全色）
    local_cfg["det_conf_color_alpha"] = min(
        float(local_cfg.get("det_conf_color_alpha", 0.05)), 0.05
    )

    if chosen_color == "White":
        # White：只有在 color_conf 夠高時才啟用色彩直方圖/旋轉直方圖
        local_cfg["color_hist_min_conf"] = max(
            float(local_cfg.get("color_hist_min_conf", 0.30)), 0.70
        )
        # White：黑帽核略小，避免把高光邊緣視為洞
        local_cfg["blackhat_kernel"] = 5
        # White：亮度上界更寬，降低 8-bit 飽和邊界誤判
        local_cfg["v_headroom"] = max(float(local_cfg.get("v_headroom", 5.0)), 10.0)

    per_color_min = model.config.get("color_conf_min_per_color", {})
    min_cc = float(
        model.config.get(
            "color_conf_min_white" if chosen_color == "White" else "color_conf_min",
            0.20,
        )
    )
    if isinstance(per_color_min, dict):
        override_min = per_color_min.get(chosen_color)
        if override_min is not None:
            try:
                min_cc = float(override_min)
            except (TypeError, ValueError):
                pass

    low_conf_reason = None
    if color_conf < min_cc:
        low_conf_reason = f"顏色置信度過低 (color_conf={color_conf:.2f} < {min_cc:.2f}, hue_cov={hue_cov_used:.2f})"

    # 4) 異常偵測
    details = _detect_anomalies(
        ft, cm, local_cfg, sensitivity, color_confidence=color_conf
    )
    if low_conf_reason:
        details["reasons"].append(low_conf_reason)
        details["feature_scores"]["color_conf"] = (min_cc - color_conf) / max(
            1e-6, min_cc
        )
    # 5) 後處理
    severity = _severity_score(details)
    recs = _recommend(details, ft)
    regions = _locate_regions(img_bgr, ft, local_cfg, details)

    # 6) 信度（用 local_cfg，使前述權重/門檻生效）
    det_conf = _detection_confidence(ft, cm, color_conf, local_cfg)

    return EnhancedDetectionResult(
        is_anomaly=len(details["reasons"]) > 0,
        confidence=det_conf,
        color_used=chosen_color,
        color_confidence=color_conf,
        reasons=details["reasons"],
        severity_score=severity,
        feature_anomalies=details["feature_scores"],
        processing_time=time.time() - t0,
        features=ft,
        anomaly_regions=regions,
        recommendations=recs,
        color_distances=dist_by_color,
        color_hue_coverage=hue_coverage,
    )


def enhanced_annotate(img: np.ndarray, res: EnhancedDetectionResult) -> np.ndarray:
    """簡化標註函數，只標記異常區域，不添加文字"""
    out = img.copy()

    # 只標記異常區域的矩形框
    for x, y, w, h in res.anomaly_regions:
        # 根據嚴重程度決定顏色
        color = (0, 0, 255) if res.severity_score > 0.7 else (0, 165, 255)
        # 繪製矩形框
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

    return out


# ----------------------------
# CLI 指令
# ----------------------------


def cmd_build(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config) if args.config else None)
    if args.save_config:
        save_default_config(Path(args.save_config))
        return
    model = build_enhanced_reference(Path(args.ref_dir), cfg)
    out = Path(args.model_out)
    ensure_dir(out.parent)
    model.to_json(out)
    # 顯示各顏色的平均 HSV/BGR（遮罩內）
    try:
        print("\n各色平均值 (遮罩區域內):")
        for c, cm in model.colors.items():
            hsv = getattr(cm, "mean_hsv_mu", [])
            bgr = getattr(cm, "mean_bgr_mu", [])
            line = f"[{c}]"
            if isinstance(hsv, list) and len(hsv) == 3:
                line += f"  HSV_mean(H,S,V)=({hsv[0]:.1f}, {hsv[1]:.1f}, {hsv[2]:.1f})"
            if isinstance(bgr, list) and len(bgr) == 3:
                line += f"  BGR_mean(B,G,R)=({bgr[0]:.1f}, {bgr[1]:.1f}, {bgr[2]:.1f})"
            print(line)
    except Exception:
        pass
    print(f"✅ 模型已建立：{out}")
    print(f"📊 總樣本：{model.total_samples}；顏色：{', '.join(model.colors.keys())}")


def cmd_detect(args: argparse.Namespace) -> None:
    model = EnhancedReferenceModel.from_json(Path(args.model))
    img = robust_imread(args.image)
    if img is None:
        print(f"❌ 無法讀圖：{args.image}")
        return
    res = enhanced_detect_one(img, model, args.label, args.sensitivity)
    status = (
        "✅ 正常" if not res.is_anomaly else f"⚠️ 異常(嚴重度:{res.severity_score:.2f})"
    )
    print(status)
    print(
        f"🎨 顏色：{res.color_used} (conf={res.color_confidence:.2f}, hue_cov={res.color_hue_coverage.get(res.color_used, 0.0):.2f})"
    )
    print(f"⚡ 時間：{res.processing_time * 1000:.1f}ms")
    if res.reasons:
        print("📋 原因：")
        [print("  -", r) for r in res.reasons]
    if res.recommendations:
        print("💡 建議：")
        [print("  -", r) for r in res.recommendations]
    if args.out_dir:
        ensure_dir(Path(args.out_dir))
        if args.save_annotated:
            ann = enhanced_annotate(img, res)
            ap = Path(args.out_dir) / f"{Path(args.image).stem}_annotated.png"
            cv2.imwrite(str(ap), ann)
            print(f"🖼 已存標註圖：{ap}")
        # 詳細 JSON
        payload = {
            "file": args.image,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "result": {
                "is_anomaly": res.is_anomaly,
                "severity_score": res.severity_score,
                "reasons": res.reasons,
                "recommendations": res.recommendations,
                "color_used": res.color_used,
                "color_confidence": res.color_confidence,
                "feature_anomalies": res.feature_anomalies,
                "processing_time": res.processing_time,
                "anomaly_regions": res.anomaly_regions,
                "color_diagnostics": {
                    "distances": res.color_distances,
                    "hue_coverage": res.color_hue_coverage,
                },
            },
            "features": asdict(res.features),
        }
        jp = Path(args.out_dir) / f"{Path(args.image).stem}_result.json"
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)

        print(f"📄 已存 JSON：{jp}")


def cmd_detect_dir(args: argparse.Namespace) -> None:
    model = EnhancedReferenceModel.from_json(Path(args.model))
    in_dir, out_dir = Path(args.dir), Path(args.out_dir)
    ensure_dir(out_dir)

    # ---- 檔案蒐集：單次 rglob + 大小寫無關副檔名 + 去重 ----
    import os

    valid_suffixes = {s.lower() for s in SUPPORTED_FORMATS}
    raw_paths = [p for p in in_dir.rglob("*") if p.suffix.lower() in valid_suffixes]

    def _normkey(p: Path) -> str:
        # Windows/NTFS 大小寫不敏感；用絕對路徑正規化做 key
        try:
            return os.path.normcase(os.path.abspath(str(p)))
        except Exception:
            return os.path.normcase(str(p))

    dedup_map = {}
    for p in raw_paths:
        k = _normkey(p)
        if k not in dedup_map:
            dedup_map[k] = p

    paths: List[Path] = sorted(dedup_map.values(), key=lambda x: _normkey(x))

    if not paths:
        print(f"❌ 目錄中沒有圖像：{in_dir}")
        return

    print(f"📁 開始批次檢測（去重後）{len(paths)} 個檔案...")

    import csv
    import hashlib

    rows = []
    t0 = time.time()

    def run_one(p: Path) -> dict:
        try:
            img = robust_imread(str(p))
            if img is None:
                return {"file": str(p), "error": "read_fail"}

            # 批次：大量不同圖片 → 交給 enhanced_detect_one 內部做特徵（避免重算）
            res = enhanced_detect_one(img, model, None, args.sensitivity)

            # 儲存標註圖（用路徑雜湊避免不同資料夾同名衝突）
            if args.save_annotated:
                h = hashlib.md5(_normkey(p).encode("utf-8")).hexdigest()[:8]
                ann_name = f"{p.stem}_{h}_annotated.png"
                ann = enhanced_annotate(img, res)
                cv2.imwrite(str(out_dir / ann_name), ann)

            return {
                "file": str(p),
                "is_anomaly": int(res.is_anomaly),
                "severity_score": float(res.severity_score),
                "color": res.color_used,
                "color_conf": float(res.color_confidence),
                "conf": float(res.confidence),
                "time_ms": float(res.processing_time * 1000.0),
                "anomaly_boxes": ";".join(
                    [f"{x},{y},{w},{h}" for x, y, w, h in res.anomaly_regions]
                ),
                "reasons": "; ".join(res.reasons),
                # ---- 新增診斷欄位 ----
                "area_ratio": float(res.features.area_ratio),
                "valid_mask": int(res.features.valid_mask),
                "mean_v": float(res.features.mean_v),
                "uniformity": float(res.features.uniformity),
                "hole_ratio": float(res.features.hole_ratio),
                "color_dist": float(res.color_distances.get(res.color_used, 0.0)),
                "color_hue_cov": float(res.color_hue_coverage.get(res.color_used, 0.0)),
            }

        except Exception as e:
            return {"file": str(p), "error": str(e)}

    # ---- 並行處理 ----
    with ThreadPoolExecutor(max_workers=model.config.get("max_workers", 4)) as ex:
        futures = [ex.submit(run_one, p) for p in paths]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 10 == 0 or i == len(paths):
                print(f"⏳ 已處理 {i}/{len(paths)}")

    # ---- 結果再去重（保險）----
    unique_rows = []
    seen = set()
    for r in rows:
        k = os.path.normcase(os.path.abspath(r.get("file", "")))
        if k in seen:
            continue
        seen.add(k)
        unique_rows.append(r)
    rows = unique_rows

    # ---- 儲存 CSV（穩健表頭）----
    csv_path = out_dir / "enhanced_summary.csv"
    header = [
        "file",
        "is_anomaly",
        "severity_score",
        "color",
        "color_conf",
        "conf",
        "time_ms",
        "anomaly_boxes",
        "reasons",
        "error",
        "area_ratio",
        "valid_mask",
        "mean_v",
        "uniformity",
        "hole_ratio",
        "color_dist",
        "color_hue_cov",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # ---- 統計 ----
    dt = time.time() - t0
    total = len(rows)
    err = sum(1 for r in rows if r.get("error"))
    ano = sum(1 for r in rows if r.get("is_anomaly", 0) == 1)
    per_img = (dt / total * 1000.0) if total > 0 else 0.0
    print(
        f"✅ 完成：{total} 張；異常 {ano}；錯誤 {err}；總耗時 {dt:.1f}s  (~{per_img:.1f}ms/張)"
    )
    print(f"📈 CSV：{csv_path}")


def _get_visual_font() -> "FontProperties":
    from matplotlib.font_manager import FontProperties

    try:
        return FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    except Exception:
        return FontProperties()


def _visualize_color_analysis(
    image_path: Path,
    img_bgr: np.ndarray,
    model: "EnhancedReferenceModel",
    res: "EnhancedDetectionResult",
    font: "FontProperties",
    save_path: Optional[Path],
    show_plot: bool,
) -> None:
    import matplotlib.pyplot as plt

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask, _ = _make_adaptive_led_mask(hsv[..., 2], model.config)
    led_mask = mask > 0

    h_values = np.asarray(hsv[..., 0][led_mask], dtype=np.float32)
    s_values = np.asarray(hsv[..., 1][led_mask], dtype=np.float32)
    v_values = np.asarray(hsv[..., 2][led_mask], dtype=np.float32)

    fig = plt.figure(figsize=(15, 10))
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    # Panel 1: original image
    plt.subplot(231)
    rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_img)
    plt.title("Original image", fontproperties=font, fontsize=12)

    # Panel 2: hue histogram with ranges
    plt.subplot(232)
    plt.hist(h_values, bins=180, range=(0, 180), density=True, alpha=0.6, color="blue")

    hue_regions = [
        ((0, 20), (1, 0, 0, 0.2), "Red range"),
        ((35, 85), (0, 1, 0, 0.2), "Green range"),
        ((85, 130), (0, 0, 1, 0.2), "Blue range"),
    ]
    ymax = max(plt.ylim()[1], 1e-3)
    for (start, end), rgba, label in hue_regions:
        plt.axvspan(start, end, color=rgba, label=label)
        region_percent = safe_ratio(
            (h_values >= start) & (h_values <= end), len(h_values)
        )
        rp = float(region_percent) if region_percent is not None else float("nan")
        if not math.isnan(rp) and rp > 5:
            text_y = ymax * 0.9 if ymax > 0 else 0.1
            plt.text(
                (start + end) / 2,
                text_y,
                f"{rp:.1f}%",
                horizontalalignment="center",
                fontproperties=font,
            )

    plt.title("Hue distribution", fontproperties=font, fontsize=12)
    plt.xlabel("Hue (H)", fontproperties=font)
    plt.ylabel("Density", fontproperties=font)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(prop=font, loc="upper right")

    # Panel 3: brightness histogram
    plt.subplot(233)
    plt.hist(v_values, bins=50, range=(0, 255), alpha=0.7, color="gray")
    brightness_ranges = [
        (0, 84, "Dark"),
        (85, 170, "Mid"),
        (171, 255, "Bright"),
    ]
    for start, end, label in brightness_ranges:
        range_percent = safe_ratio(
            (v_values >= start) & (v_values <= end), len(v_values)
        )
        rp = float(range_percent) if range_percent is not None else float("nan")
        if not math.isnan(rp) and rp > 5:
            plt.axvspan(start, end, alpha=0.2, label=f"{label}: {rp:.1f}%")

    plt.title("Brightness distribution", fontproperties=font, fontsize=12)
    plt.xlabel("Value (V)", fontproperties=font)
    plt.ylabel("Count", fontproperties=font)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(prop=font, loc="upper right")

    # Panel 4: mask visualization
    plt.subplot(234)
    plt.imshow(mask, cmap="gray")
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        area = cv2.contourArea(max_contour)
        perimeter = cv2.arcLength(max_contour, True)
        circularity = (
            4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
        )

        plt.text(
            0.02,
            0.98,
            f"Shape metrics\nCircularity: {circularity:.2f}\nArea: {area:.0f}px",
            transform=plt.gca().transAxes,
            fontproperties=font,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8),
        )
        plt.plot(
            [x, x + w, x + w, x, x],
            [y, y, y + h, y + h, y],
            "r-",
            linewidth=2,
        )

    plt.title("LED mask", fontproperties=font, fontsize=12)

    # Panel 5: saturation histogram
    plt.subplot(235)
    plt.hist(s_values, bins=50, range=(0, 255), alpha=0.7, color="orange")
    s_low = safe_percentile(s_values, 25)
    s_high = safe_percentile(s_values, 75)
    s_median = float(np.median(s_values)) if s_values.size else float("nan")
    if s_values.size:
        plt.axvline(
            s_median, color="r", linestyle="--", label=f"Median: {s_median:.1f}"
        )
        plt.axvspan(
            s_low, s_high, alpha=0.2, color="y", label=f"IQR: {s_high - s_low:.1f}"
        )

    plt.title("Saturation distribution", fontproperties=font, fontsize=12)
    plt.xlabel("Saturation (S)", fontproperties=font)
    plt.ylabel("Count", fontproperties=font)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(prop=font, loc="upper right")

    # Panel 6: textual summary
    plt.subplot(236)
    color_hue_cov = float(res.color_hue_coverage.get(res.color_used, 0.0))
    h_mean = float(np.mean(h_values)) if h_values.size else 0.0
    h_std = float(np.std(h_values)) if h_values.size else 0.0
    s_mean = float(np.mean(s_values)) if s_values.size else 0.0
    s_median_display = s_median if not math.isnan(s_median) else 0.0
    v_mean = float(np.mean(v_values)) if v_values.size else 0.0
    v_uniformity = (
        1 - (float(np.std(v_values)) / 255.0) if v_values.size else float("nan")
    )
    color_purity = s_mean * (v_mean / 255.0) if v_values.size else 0.0

    main_hue_range = {"Red": (0, 20), "Green": (35, 85), "Blue": (85, 130)}
    main_hue_percent = 0.0
    if res.color_used in main_hue_range and h_values.size:
        start, end = main_hue_range[res.color_used]
        mh = safe_ratio((h_values >= start) & (h_values <= end), len(h_values))
        mh = float(mh) if mh is not None else float("nan")
        if not math.isnan(mh):
            main_hue_percent = mh

    ft = res.features
    area_ratio = float(getattr(ft, "area_ratio", float("nan")))
    uniformity = float(getattr(ft, "uniformity", float("nan")))
    hole_ratio = float(getattr(ft, "hole_ratio", float("nan")))

    result_lines = [
        "Detection summary",
        "----------------",
        f"Predicted color: {res.color_used}",
        f"Color confidence: {res.color_confidence:.3f}",
        f"Feature confidence: {res.confidence:.3f}",
        f"Hue coverage: {color_hue_cov:.3f}",
        f"Severity score: {res.severity_score:.3f}",
        f"Anomaly: {'Yes' if res.is_anomaly else 'No'}",
        "",
        "Hue statistics",
        "----------------",
        f"Main hue coverage: {main_hue_percent:.1f}%",
        f"Mean hue: {h_mean:.1f} deg",
        f"Hue std: {h_std:.1f} deg",
        "",
        "Brightness & saturation",
        "----------------",
        f"Mean saturation: {s_mean:.1f}",
        f"Median saturation: {s_median_display:.1f}",
        f"Mean brightness: {v_mean:.1f}",
        f"Brightness uniformity: {v_uniformity:.2f}",
        f"Color purity: {color_purity:.1f}",
        "",
        "Mask features",
        "----------------",
        f"Area ratio: {area_ratio:.3f}",
        f"Uniformity: {uniformity:.3f}",
        f"Hole ratio: {hole_ratio:.3f}",
    ]
    result_text = "\n".join(result_lines)

    plt.text(
        0.05,
        0.95,
        result_text,
        fontproperties=font,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=1", facecolor="white", alpha=0.8),
    )
    plt.axis("off")

    plt.tight_layout()

    if save_path:
        ensure_dir(save_path.parent)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[analyze] saved figure: {save_path}")
    if show_plot:
        plt.show()
    plt.close(fig)


def _analyze_single(args: argparse.Namespace, model: "EnhancedReferenceModel") -> None:
    image_arg = getattr(args, "image", None)
    if not image_arg:
        print("[analyze] --image is required when directory is not provided")
        return

    image_path = Path(image_arg)
    img = robust_imread(str(image_path))
    if img is None:
        print(f"[analyze] failed to read image: {image_path}")
        return

    res = enhanced_detect_one(img, model)
    hue_cov = float(res.color_hue_coverage.get(res.color_used, 0.0))
    status = (
        "PASS" if not res.is_anomaly else f"FAIL (severity={res.severity_score:.2f})"
    )
    print(f"[analyze] {status}")
    print(
        f"[analyze] color={res.color_used} (conf={res.color_confidence:.3f}, "
        f"hue_cov={hue_cov:.3f})"
    )
    print(
        f"[analyze] feature_conf={res.confidence:.3f} "
        f"time={res.processing_time * 1000.0:.1f}ms"
    )
    if res.reasons:
        print("[analyze] reasons:")
        for r in res.reasons:
            print(f"  - {r}")
    if res.recommendations:
        print("[analyze] recommendations:")
        for r in res.recommendations:
            print(f"  - {r}")

    if getattr(args, "visualize", False):
        font = _get_visual_font()
        out_dir_arg = getattr(args, "out_dir", None)
        show_plot = not out_dir_arg
        save_path = None
        if out_dir_arg:
            out_dir = Path(out_dir_arg)
            ensure_dir(out_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = out_dir / f"{image_path.stem}_analysis_{timestamp}.png"
        _visualize_color_analysis(
            image_path, img, model, res, font, save_path, show_plot
        )


def _analyze_directory(
    args: argparse.Namespace, model: "EnhancedReferenceModel"
) -> None:
    dir_arg = getattr(args, "dir", None)
    if not dir_arg:
        print("[analyze] --dir is required when image is not provided")
        return

    root = Path(dir_arg)
    if not root.is_dir():
        print(f"[analyze] directory not found: {root}")
        return

    valid_suffixes = {s.lower() for s in SUPPORTED_FORMATS}
    image_paths = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in valid_suffixes
    )
    if not image_paths:
        print(f"[analyze] no images found under {root}")
        return

    print(f"[analyze] found {len(image_paths)} images under {root}")

    visualize = bool(getattr(args, "visualize", False))
    out_dir_arg = getattr(args, "out_dir", None)
    font = _get_visual_font() if visualize else None
    out_root: Optional[Path]
    if visualize:
        out_root = Path(out_dir_arg) if out_dir_arg else root / "analysis"
        ensure_dir(out_root)
    else:
        out_root = Path(out_dir_arg) if out_dir_arg else None
        if out_root:
            ensure_dir(out_root)

    rows: List[Dict[str, object]] = []
    failed = 0
    total = len(image_paths)

    for idx, img_path in enumerate(image_paths, 1):
        img = robust_imread(str(img_path))
        if img is None:
            print(f"[analyze] #{idx} failed to read: {img_path}")
            failed += 1
            continue

        res = enhanced_detect_one(img, model)
        hue_cov = float(res.color_hue_coverage.get(res.color_used, 0.0))
        print(
            f"[analyze] #{idx}/{total} {img_path.name}: "
            f"color={res.color_used} conf={res.color_confidence:.3f} "
            f"hue_cov={hue_cov:.3f}"
        )

        if visualize and out_root is not None and font is not None:
            rel = img_path.relative_to(root)
            hash_suffix = hashlib.md5(str(rel).encode("utf-8")).hexdigest()[:8]
            target_dir = out_root / rel.parent
            ensure_dir(target_dir)
            save_name = f"{img_path.stem}_{hash_suffix}_analysis.png"
            save_path = target_dir / save_name
            _visualize_color_analysis(
                img_path, img, model, res, font, save_path, show_plot=False
            )

        rows.append(
            {
                "file": str(img_path),
                "is_anomaly": int(res.is_anomaly),
                "severity_score": float(res.severity_score),
                "color": res.color_used,
                "color_conf": float(res.color_confidence),
                "conf": float(res.confidence),
                "time_ms": float(res.processing_time * 1000.0),
                "anomaly_boxes": ";".join(
                    f"{x},{y},{w},{h}" for x, y, w, h in res.anomaly_regions
                ),
                "reasons": "; ".join(res.reasons),
                "recommendations": "; ".join(res.recommendations),
                "area_ratio": float(getattr(res.features, "area_ratio", float("nan"))),
                "valid_mask": int(bool(getattr(res.features, "valid_mask", 0))),
                "mean_v": float(getattr(res.features, "mean_v", float("nan"))),
                "uniformity": float(getattr(res.features, "uniformity", float("nan"))),
                "hole_ratio": float(getattr(res.features, "hole_ratio", float("nan"))),
                "color_dist": float(res.color_distances.get(res.color_used, 0.0)),
                "color_hue_cov": hue_cov,
            }
        )

    processed = len(rows)
    print(f"[analyze] completed {processed} images (failed: {failed})")

    if rows and out_root:
        import csv

        csv_path = out_root / "analysis_summary.csv"
        fieldnames = [
            "file",
            "is_anomaly",
            "severity_score",
            "color",
            "color_conf",
            "conf",
            "time_ms",
            "anomaly_boxes",
            "reasons",
            "recommendations",
            "area_ratio",
            "valid_mask",
            "mean_v",
            "uniformity",
            "hole_ratio",
            "color_dist",
            "color_hue_cov",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"[analyze] saved summary: {csv_path}")


def cmd_analyze(args: argparse.Namespace) -> None:
    model_arg = getattr(args, "model", None)
    if not isinstance(model_arg, (str, os.PathLike)):
        raise TypeError("model path must be a string or PathLike")
    model = EnhancedReferenceModel.from_json(Path(model_arg))

    image_arg = getattr(args, "image", None)
    dir_arg = getattr(args, "dir", None)

    has_image = bool(image_arg)
    has_dir = bool(dir_arg)

    if has_image:
        image_path: Optional[Path]
        if isinstance(image_arg, (str, os.PathLike)):
            image_path = Path(image_arg)
        else:
            image_path = None
        if image_path is not None and image_path.is_dir():
            # allow --image pointing to a directory
            dir_arg = str(image_path)
            has_dir = True
            has_image = False
            setattr(args, "dir", dir_arg)
            setattr(args, "image", None)

    if has_image and has_dir:
        print("[analyze] please provide either --image or --dir, not both")
        return
    if not has_image and not has_dir:
        print("[analyze] please provide --image or --dir")
        return

    if has_dir:
        _analyze_directory(args, model)
    else:
        _analyze_single(args, model)


def cmd_info(args: argparse.Namespace) -> None:
    model_arg = getattr(args, "model", None)
    if not isinstance(model_arg, (str, os.PathLike)):
        raise TypeError("model path must be a string or PathLike")
    m = EnhancedReferenceModel.from_json(Path(model_arg))
    print("=" * 50)
    print("模型資訊")
    print("=" * 50)
    print(f"版本：{m.version} 建立時間：{m.creation_time} 總樣本：{m.total_samples}")
    print(f"顏色：{', '.join(m.colors.keys())}")
    for c, cm in m.colors.items():
        print(f"\n[{c}] 樣本:{cm.samples}  avg_conf:{cm.avg_confidence:.3f}")
        print(f"  hist_thr:{cm.hist_thr:.3f}  rot_thr:{cm.rotation_hist_thr:.3f}")
        print(
            f"  meanV:{cm.mean_v_mu:.1f}±{cm.mean_v_std:.1f}  uniformity:{cm.uniformity_mu:.3f}±{cm.uniformity_std:.3f}"
        )
        # 顯示每色的平均 HSV/BGR（若有）
        if (
            hasattr(cm, "mean_hsv_mu")
            and isinstance(cm.mean_hsv_mu, list)
            and len(cm.mean_hsv_mu) == 3
        ):
            h, s, v = cm.mean_hsv_mu
            print(f"  HSV_mean: H={h:.1f}  S={s:.1f}  V={v:.1f}")
            # 以 mean ± sigma_multiplier*std 顯示 HSV 上下限
            if (
                hasattr(cm, "mean_hsv_std")
                and isinstance(cm.mean_hsv_std, list)
                and len(cm.mean_hsv_std) == 3
            ):
                sh, ss, sv = cm.mean_hsv_std
                sigma_multiplier = float(m.config.get("sigma_multiplier", 2.0))
                h_low = max(0.0, h - sigma_multiplier * sh)
                h_high = min(180.0, h + sigma_multiplier * sh)
                s_low = max(0.0, s - sigma_multiplier * ss)
                s_high = min(255.0, s + sigma_multiplier * ss)
                v_low = max(0.0, v - sigma_multiplier * sv)
                v_high = min(255.0, v + sigma_multiplier * sv)
                print(
                    f"  HSV_range(K={sigma_multiplier:.1f}): H=[{h_low:.1f}, {h_high:.1f}]  S=[{s_low:.1f}, {s_high:.1f}]  V=[{v_low:.1f}, {v_high:.1f}]"
                )
        if (
            hasattr(cm, "mean_bgr_mu")
            and isinstance(cm.mean_bgr_mu, list)
            and len(cm.mean_bgr_mu) == 3
        ):
            b, g, r = cm.mean_bgr_mu
            print(f"  BGR_mean: B={b:.1f}  G={g:.1f}  R={r:.1f}")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("LED QC (Enhanced, batch-friendly)")
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("build", help="建立參考模型")
    b.add_argument("--ref-dir", required=True)
    b.add_argument("--model-out", required=True)
    b.add_argument("--config")
    b.add_argument("--save-config")
    b.set_defaults(func=cmd_build)

    d = sub.add_parser("detect", help="檢測單張")
    d.add_argument("--model", required=True)
    d.add_argument("--image", required=True)
    d.add_argument("--label", choices=COLORS)
    d.add_argument("--sensitivity", type=float, default=1.0)
    d.add_argument("--out-dir")
    d.add_argument("--save-annotated", action="store_true")
    d.set_defaults(func=cmd_detect)

    dd = sub.add_parser("detect-dir", help="批次檢測資料夾")
    dd.add_argument("--model", required=True)
    dd.add_argument("--dir", required=True)
    dd.add_argument("--sensitivity", type=float, default=0.85)
    dd.add_argument("--out-dir", required=True)
    dd.add_argument("--save-annotated", action="store_true")
    dd.set_defaults(func=cmd_detect_dir)

    a = sub.add_parser("analyze", help="分析工具")
    a.add_argument("--model", required=True, help="模型檔案路徑")
    a.add_argument("--image", help="Single image path")
    a.add_argument(
        "--dir", help="Root directory containing paired folders for analysis"
    )
    a.add_argument("--visualize", action="store_true", help="執行可視化分析")
    a.add_argument("--stability", action="store_true", help="執行穩定性測試")
    a.add_argument("--out-dir", help="輸出目錄")
    a.set_defaults(func=cmd_analyze)

    i = sub.add_parser("info", help="模型資訊")
    i.add_argument("--model", required=True)
    i.set_defaults(func=cmd_info)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    try:
        start = time.time()
        args.func(args)
        dt = time.time() - start
        if dt > 1.0:
            logger.info(f"命令耗時 {dt:.2f}s")
        return 0
    except Exception as e:
        logger.error(f"執行失敗：{e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
