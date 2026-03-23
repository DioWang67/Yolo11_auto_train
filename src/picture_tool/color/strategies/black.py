import numpy as np
from typing import Any, Dict, Tuple

from picture_tool.color.strategies.base import ColorRange
from picture_tool.color.strategies.generic import GenericStrategy

BLACK_S_THRESHOLD = 50.0
BLACK_V_THRESHOLD = 80.0
BLACK_MIN_COVERAGE = 0.6

from picture_tool.color.strategies.registry import ColorStrategyRegistry  # noqa: E402

@ColorStrategyRegistry.register("Black")
class BlackStrategy(GenericStrategy):
    """Specific logic for verifying Black colors."""

    def match_ratio(
        self,
        hsv_vals: np.ndarray,
        lab_vals: np.ndarray,
        color_range: ColorRange,
    ) -> Tuple[float, Dict[str, Any]]:
        # For match ratio, delegate to super mostly, but override mask logic
        debug: Dict[str, float] = {}
        if hsv_vals.size == 0 or lab_vals.size == 0:
            return 0.0, debug
            
        s_vals = hsv_vals[:, 1]
        v_vals = hsv_vals[:, 2]

        h_mask = (s_vals < BLACK_S_THRESHOLD) & (v_vals < BLACK_V_THRESHOLD)
        hsv_ratio = float(np.count_nonzero(h_mask)) / len(hsv_vals)
        debug["hsv_ratio"] = hsv_ratio

        # Lab match
        lab_mask = (
            (lab_vals[:, 0] >= color_range.lab_min[0])
            & (lab_vals[:, 0] <= color_range.lab_max[0])
            & (lab_vals[:, 1] >= color_range.lab_min[1])
            & (lab_vals[:, 1] <= color_range.lab_max[1])
            & (lab_vals[:, 2] >= color_range.lab_min[2])
            & (lab_vals[:, 2] <= color_range.lab_max[2])
        )
        lab_ratio = float(np.count_nonzero(lab_mask)) / len(lab_vals)
        debug["lab_ratio"] = lab_ratio

        weights = {"hsv": 0.5, "lab": 0.3, "hue_sim": 0.2, "lab_chroma": 0.0}
        
        # Black ignores hue distance similarity since its hue is undefined
        hue_similarity = 1.0  
        final_score = (
            hsv_ratio * weights["hsv"]
            + lab_ratio * weights["lab"]
            + hue_similarity * weights["hue_sim"]
        )
        debug["final_score"] = float(final_score)
        return float(final_score), debug

    def fast_detect(
        self,
        hsv_img: np.ndarray,
        lab_img: np.ndarray,
        color_range: ColorRange
    ) -> Tuple[bool, float]:
        """Detects if it's black early-on for short-circuiting."""
        h, w = hsv_img.shape[:2]
        margin_y = int(h * 0.15)
        margin_x = int(w * 0.15)
        center_region = hsv_img[margin_y : h - margin_y, margin_x : w - margin_x]

        if center_region.size == 0:
            center_region = hsv_img

        mean_s = float(np.mean(center_region[:, :, 1]))
        mean_v = float(np.mean(center_region[:, :, 2]))
        median_s = float(np.median(center_region[:, :, 1]))
        median_v = float(np.median(center_region[:, :, 2]))

        black_mask = (center_region[:, :, 1] < BLACK_S_THRESHOLD) & (
            center_region[:, :, 2] < BLACK_V_THRESHOLD
        )
        black_coverage = float(np.count_nonzero(black_mask)) / black_mask.size

        is_black = (
            (mean_s < BLACK_S_THRESHOLD and mean_v < BLACK_V_THRESHOLD)
            or (median_s < BLACK_S_THRESHOLD * 0.8 and median_v < BLACK_V_THRESHOLD * 0.8)
            or (black_coverage > BLACK_MIN_COVERAGE)
        )

        confidence = black_coverage if is_black else 0.0
        return is_black, confidence
