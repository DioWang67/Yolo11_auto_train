import numpy as np
from typing import Any, Dict, Tuple

from picture_tool.color.strategies.base import (
    ColorRange,
    ColorStrategy,
    circular_hue_distance,
    circular_hue_mean,
    hue_in_range,
    measure_color_region,
    safe_ratio,
    weighted_score,
)


from picture_tool.color.strategies.registry import ColorStrategyRegistry  # noqa: E402

@ColorStrategyRegistry.register_fallback()
class GenericStrategy(ColorStrategy):
    """Fallback strategy for colors that don't need special overrides."""

    def match_ratio(
        self,
        hsv_img: np.ndarray,
        lab_img: np.ndarray,
        color_range: ColorRange,
    ) -> Tuple[float, Dict[str, Any]]:
        debug: Dict[str, float] = {}
        if hsv_img.size == 0 or lab_img.size == 0:
            return 0.0, debug

        h_vals = hsv_img[:, :, 0]
        s_vals = hsv_img[:, :, 1]
        v_vals = hsv_img[:, :, 2]

        h_mask = (
            hue_in_range(h_vals, color_range.hsv_min[0], color_range.hsv_max[0])
            & (s_vals >= color_range.hsv_min[1])
            & (s_vals <= color_range.hsv_max[1])
            & (v_vals >= color_range.hsv_min[2])
            & (v_vals <= color_range.hsv_max[2])
        )

        hsv_ratio, blob_hsv, blob_lab, candidate_pixels = measure_color_region(
            hsv_img, lab_img, h_mask
        )
        debug["hsv_ratio"] = hsv_ratio
        if candidate_pixels == 0:
            debug["final_score"] = 0.0
            return 0.0, debug

        lab_mask = (
            (blob_lab[:, 0] >= color_range.lab_min[0])
            & (blob_lab[:, 0] <= color_range.lab_max[0])
            & (blob_lab[:, 1] >= color_range.lab_min[1])
            & (blob_lab[:, 1] <= color_range.lab_max[1])
            & (blob_lab[:, 2] >= color_range.lab_min[2])
            & (blob_lab[:, 2] <= color_range.lab_max[2])
        )
        # Same denominator as hsv_ratio, not the selection's own size: see
        # ``measure_color_region``'s docstring and, in the inference
        # repository, ``core/stats_color_checker.py``'s ``_improved_match_ratio``.
        lab_ratio = safe_ratio(int(np.count_nonzero(lab_mask)), candidate_pixels)
        debug["lab_ratio"] = lab_ratio

        mean_h = circular_hue_mean(blob_hsv[:, 0])
        debug["mean_hue"] = mean_h

        # Left as None when the baseline carries no hue statistic, so the term
        # is dropped rather than awarded a perfect score it did not earn.
        hue_similarity = None
        if color_range.hsv_mean is not None:
            expected_h = float(color_range.hsv_mean[0])
            hue_dist = circular_hue_distance(mean_h, expected_h)
            hue_similarity = float(np.exp(-hue_dist / 15.0))
            debug["hue_distance"] = hue_dist
            debug["hue_similarity"] = hue_similarity

        weights = {"hsv": 0.5, "lab": 0.3, "hue_sim": 0.2, "lab_chroma": 0.0}

        final_score = weighted_score(
            {"hsv": hsv_ratio, "lab": lab_ratio, "hue_sim": hue_similarity},
            weights,
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
            hue_in_range(hsv_img[:, :, 0], color_range.hsv_min[0], color_range.hsv_max[0])
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
