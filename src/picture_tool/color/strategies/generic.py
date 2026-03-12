import numpy as np
from typing import Any, Dict, Tuple

from picture_tool.color.strategies.base import ColorStrategy, ColorRange


def circular_hue_distance(h1: float, h2: float) -> float:
    """Calculate circular distance for hue (0-180 degrees)."""
    diff = abs(h1 - h2)
    return float(min(diff, 180 - diff))


class GenericStrategy(ColorStrategy):
    """Fallback strategy for colors that don't need special overrides."""

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

        h_mask = (
            (h_vals >= color_range.hsv_min[0])
            & (h_vals <= color_range.hsv_max[0])
            & (s_vals >= color_range.hsv_min[1])
            & (s_vals <= color_range.hsv_max[1])
            & (v_vals >= color_range.hsv_min[2])
            & (v_vals <= color_range.hsv_max[2])
        )

        hsv_ratio = float(np.count_nonzero(h_mask)) / len(hsv_vals)
        debug["hsv_ratio"] = hsv_ratio

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

        mean_h = float(np.mean(h_vals))
        debug["mean_hue"] = mean_h

        hue_similarity = 1.0
        if color_range.hsv_mean is not None:
            expected_h = float(color_range.hsv_mean[0])
            hue_dist = circular_hue_distance(mean_h, expected_h)
            hue_similarity = float(np.exp(-hue_dist / 15.0))
            debug["hue_distance"] = hue_dist
            debug["hue_similarity"] = hue_similarity

        weights = {"hsv": 0.5, "lab": 0.3, "hue_sim": 0.2, "lab_chroma": 0.0}

        final_score = (
            hsv_ratio * weights["hsv"]
            + lab_ratio * weights["lab"]
            + hue_similarity * weights["hue_sim"]
        )

        debug["final_score"] = float(final_score)
        return float(final_score), debug

    def build_mask(
        self,
        hsv_img: np.ndarray,
        lab_img: np.ndarray,
        color_range: ColorRange,
        global_sat_mask: np.ndarray,
    ) -> np.ndarray:
        hsv_cond = (
            (hsv_img[:, :, 0] >= color_range.hsv_min[0])
            & (hsv_img[:, :, 0] <= color_range.hsv_max[0])
            & (hsv_img[:, :, 1] >= color_range.hsv_min[1])
            & (hsv_img[:, :, 1] <= color_range.hsv_max[1])
            & (hsv_img[:, :, 2] >= color_range.hsv_min[2])
            & (hsv_img[:, :, 2] <= color_range.hsv_max[2])
        )
        lab_cond = (
            (lab_img[:, :, 0] >= color_range.lab_min[0])
            & (lab_img[:, :, 0] <= color_range.lab_max[0])
            & (lab_img[:, :, 1] >= color_range.lab_min[1])
            & (lab_img[:, :, 1] <= color_range.lab_max[1])
            & (lab_img[:, :, 2] >= color_range.lab_min[2])
            & (lab_img[:, :, 2] <= color_range.lab_max[2])
        )
        mask = hsv_cond & lab_cond & global_sat_mask
        return mask
