import numpy as np
from typing import Any, Dict, Tuple

from picture_tool.color.strategies.base import ColorRange
from picture_tool.color.strategies.generic import GenericStrategy

YELLOW_H_RANGE = (20, 35)
YELLOW_S_MIN = 80
YELLOW_V_MIN = 150

from picture_tool.color.strategies.registry import ColorStrategyRegistry  # noqa: E402

@ColorStrategyRegistry.register("Yellow")
class YellowStrategy(GenericStrategy):
    """Specific logic for verifying Yellow colors."""

    def match_ratio(
        self,
        hsv_vals: np.ndarray,
        lab_vals: np.ndarray,
        color_range: ColorRange,
    ) -> Tuple[float, Dict[str, Any]]:
        debug: Dict[str, float] = {}
        if hsv_vals.size == 0 or lab_vals.size == 0:
            return 0.0, debug
            
        h_vals = hsv_vals[:, 0]
        s_vals = hsv_vals[:, 1]
        v_vals = hsv_vals[:, 2]

        h_mask = (h_vals >= YELLOW_H_RANGE[0]) & (h_vals <= YELLOW_H_RANGE[1]) & (s_vals >= YELLOW_S_MIN) & (v_vals >= YELLOW_V_MIN)
        hsv_ratio = float(np.count_nonzero(h_mask)) / len(hsv_vals)
        debug["hsv_ratio"] = hsv_ratio

        # Delegate the rest to generic (LAB matching, hue similarity)
        generic_score, generic_debug = super().match_ratio(hsv_vals, lab_vals, color_range)
        
        # Yellow weights favor HSV heavily
        weights = {"hsv": 0.5, "lab": 0.2, "hue_sim": 0.3}
        final_score = (
            hsv_ratio * weights["hsv"]
            + generic_debug.get("lab_ratio", 0.0) * weights["lab"]
            + generic_debug.get("hue_similarity", 1.0) * weights["hue_sim"]
        )
        
        debug.update(generic_debug)
        debug["final_score"] = float(final_score)
        return float(final_score), debug

    def fast_detect(
        self,
        hsv_img: np.ndarray,
        lab_img: np.ndarray,
        color_range: ColorRange
    ) -> Tuple[bool, float]:
        """Rapidly detects yellow majorities."""
        h, w = hsv_img.shape[:2]
        margin = int(min(h, w) * 0.15)
        center = hsv_img[margin : h - margin, margin : w - margin]

        if center.size == 0:
            center = hsv_img

        h_vals = center[:, :, 0]
        s_vals = center[:, :, 1]
        v_vals = center[:, :, 2]

        yellow_mask_primary = (
            (h_vals >= YELLOW_H_RANGE[0])
            & (h_vals <= YELLOW_H_RANGE[1])
            & (s_vals >= YELLOW_S_MIN)
            & (v_vals >= YELLOW_V_MIN)
        )

        yellow_mask_secondary = (
            (h_vals >= 18) & (h_vals <= 38) & (s_vals >= 60) & (v_vals >= 180)
        )

        yellow_mask = yellow_mask_primary | yellow_mask_secondary
        yellow_ratio = float(np.count_nonzero(yellow_mask)) / yellow_mask.size

        # Rule from original code: compare with orange-like pixels
        orange_like_mask = (h_vals < 20) & (h_vals > 5) & (s_vals > 100)
        orange_ratio = float(np.count_nonzero(orange_like_mask)) / orange_like_mask.size

        is_yellow = (yellow_ratio > 0.25) and (yellow_ratio > orange_ratio * 1.3)
        return is_yellow, yellow_ratio
