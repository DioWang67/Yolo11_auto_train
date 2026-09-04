import numpy as np
from typing import Any, Dict, Tuple

from picture_tool.color.strategies.base import (
    ColorRange,
    center_crop,
    circular_hue_distance,
    circular_hue_mean,
    measure_color_region,
    safe_ratio,
    weighted_score,
)
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

        h_mask = (h_vals >= YELLOW_H_RANGE[0]) & (h_vals <= YELLOW_H_RANGE[1]) & (s_vals >= YELLOW_S_MIN) & (v_vals >= YELLOW_V_MIN)
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

        # Yellow weights favor HSV heavily
        weights = {"hsv": 0.5, "lab": 0.2, "hue_sim": 0.3}
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

    def fast_detect(
        self,
        hsv_img: np.ndarray,
        lab_img: np.ndarray,
        color_range: ColorRange
    ) -> Tuple[bool, float]:
        """Rapidly detects yellow majorities."""
        center = center_crop(hsv_img)
        if center.size == 0:
            return False, 0.0

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
        yellow_ratio = safe_ratio(np.count_nonzero(yellow_mask), yellow_mask.size)

        # Rule from original code: compare with orange-like pixels
        orange_like_mask = (h_vals < 20) & (h_vals > 5) & (s_vals > 100)
        orange_ratio = safe_ratio(
            np.count_nonzero(orange_like_mask), orange_like_mask.size
        )

        is_yellow = (yellow_ratio > 0.25) and (yellow_ratio > orange_ratio * 1.3)
        return is_yellow, yellow_ratio
