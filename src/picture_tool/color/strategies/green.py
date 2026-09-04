import numpy as np
from typing import Any, Dict, Tuple

from picture_tool.color.strategies.base import (
    ColorRange,
    circular_hue_distance,
    circular_hue_mean,
    measure_color_region,
    safe_ratio,
    weighted_score,
)
from picture_tool.color.strategies.generic import GenericStrategy
from picture_tool.color.strategies.registry import ColorStrategyRegistry

@ColorStrategyRegistry.register("Green")
class GreenStrategy(GenericStrategy):
    """Specific logic for verifying Green colors."""

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
            (h_vals >= 70)
            & (h_vals <= 100)
            & (s_vals >= 75)
            & (v_vals >= 30)
            & (v_vals <= 100)
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
        lab_ratio = safe_ratio(int(np.count_nonzero(lab_mask)), candidate_pixels)
        debug["lab_ratio"] = lab_ratio

        mean_h = circular_hue_mean(blob_hsv[:, 0])
        hue_similarity = None
        if color_range.hsv_mean is not None:
            expected_h = float(color_range.hsv_mean[0])
            hue_dist = circular_hue_distance(mean_h, expected_h)
            hue_similarity = float(np.exp(-hue_dist / 15.0))
            debug["hue_similarity"] = hue_similarity

        # Green weights
        weights = {"hsv": 0.6, "lab": 0.2, "hue_sim": 0.2}
        final_score = weighted_score(
            {
                "hsv": hsv_ratio,
                "lab": lab_ratio,
                "hue_sim": hue_similarity,
            },
            weights,
        )
        debug["final_score"] = float(final_score)
        return float(final_score), debug

    # No ``post_correction`` here, deliberately.
    #
    # There used to be a green-dominance override: when Red won, it counted
    # pixels with hue in [70, 100] over the whole center crop -- no saturation
    # or value filter, so unlit and washed-out pixels counted -- and above 0.3
    # replaced the winner with ("Green", that raw ratio). It flipped a
    # red-majority region to its green minority, and the confidence it reported
    # was a hue count, not a score comparable with any other color's.
    #
    # That is what the orange/red tie-breaker in ``red_orange.py`` was already
    # fixed not to do: a step that decides *which* color it is measures nothing
    # new about how strongly the region matches, so nothing may be created. The
    # inference runtime has no counterpart to this override, so keeping it here
    # meant the gate and the line disagreed on the same board. Green now
    # competes on ``match_ratio`` like every other color.
