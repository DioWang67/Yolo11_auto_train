import numpy as np
from typing import Any, Dict, Optional, Tuple

from picture_tool.color.strategies.base import ColorRange
from picture_tool.color.strategies.generic import GenericStrategy

ORANGE_RED_TIE_MARGIN = 0.15

from picture_tool.color.strategies.registry import ColorStrategyRegistry  # noqa: E402

@ColorStrategyRegistry.register("Red", "Orange")
class RedOrangeStrategy(GenericStrategy):
    """Specific logic for Red and Orange, which have high similarity and cross-dependency."""

    def match_ratio(
        self,
        hsv_vals: np.ndarray,
        lab_vals: np.ndarray,
        color_range: ColorRange,
    ) -> Tuple[float, Dict[str, Any]]:
        debug: Dict[str, Any] = {}
        if hsv_vals.size == 0 or lab_vals.size == 0:
            return 0.0, debug
            
        h_vals = hsv_vals[:, 0]
        s_vals = hsv_vals[:, 1]
        v_vals = hsv_vals[:, 2]

        if color_range.name == "Red":
            h_mask = (
                ((h_vals <= 10) | (h_vals >= 170))
                & (s_vals >= max(color_range.hsv_min[1], 130))
                & (v_vals >= max(color_range.hsv_min[2], 80))
            )
        else:  # Orange
            h_mask = (
                (h_vals >= 5)
                & (h_vals <= 20)
                & (s_vals >= max(color_range.hsv_min[1], 130))
                & (v_vals >= max(color_range.hsv_min[2], 100))
            )
            
        hsv_ratio = float(np.count_nonzero(h_mask)) / len(hsv_vals)
        debug["hsv_ratio"] = hsv_ratio

        # Reuse generic LAB and Hue Similarity
        generic_score, generic_debug = super().match_ratio(hsv_vals, lab_vals, color_range)
        debug.update(generic_debug)

        # LAB Chroma similarity is specific to O/R
        lab_chroma_similarity = 1.0
        if color_range.lab_mean is not None:
            mean_a = float(np.mean(lab_vals[:, 1]))
            mean_b = float(np.mean(lab_vals[:, 2]))
            expected_a = float(color_range.lab_mean[1])
            expected_b = float(color_range.lab_mean[2])

            lab_chroma_dist = np.sqrt(
                (mean_a - expected_a) ** 2 + (mean_b - expected_b) ** 2
            )
            lab_chroma_similarity = float(np.exp(-lab_chroma_dist / 20.0))

            debug["lab_chroma_dist"] = float(lab_chroma_dist)
            debug["lab_chroma_similarity"] = lab_chroma_similarity

        weights = {"hsv": 0.35, "lab": 0.25, "hue_sim": 0.25, "lab_chroma": 0.15}
        final_score = (
            hsv_ratio * weights["hsv"]
            + debug.get("lab_ratio", 0.0) * weights["lab"]
            + debug.get("hue_similarity", 1.0) * weights["hue_sim"]
            + lab_chroma_similarity * weights["lab_chroma"]
        )
        debug["final_score"] = float(final_score)
        return float(final_score), debug

    def post_correction(
        self,
        predicted_color: str,
        confidence: float,
        ratios: Dict[str, float],
        center_hsv: np.ndarray,
        center_lab: np.ndarray
    ) -> Optional[Tuple[str, float]]:
        """Implements the Orange/Red tiebreak logic."""
        if (
            "Orange" not in ratios
            or "Red" not in ratios
            or predicted_color not in {"Orange", "Red"}
            or abs(ratios["Orange"] - ratios["Red"]) >= ORANGE_RED_TIE_MARGIN
        ):
            return None

        if center_hsv.size == 0 or center_lab.size == 0:
            return None

        flat_hsv = center_hsv.reshape(-1, 3)
        flat_lab = center_lab.reshape(-1, 3)
        
        # Constant from original code
        DEFAULT_SAT_THRESHOLD = 20.0
        sat_mask = flat_hsv[:, 1] >= DEFAULT_SAT_THRESHOLD
        valid_hsv = flat_hsv[sat_mask]
        valid_lab = flat_lab[sat_mask]

        if len(valid_hsv) == 0:
            return None

        # Disambiguation logic based on hue distribution and AB ratio
        hue_vals = valid_hsv[:, 0]
        orange_core = np.sum((hue_vals >= 8) & (hue_vals <= 16))
        red_core = np.sum((hue_vals <= 5) | (hue_vals >= 175))

        orange_hue_ratio = orange_core / len(hue_vals)
        red_hue_ratio = red_core / len(hue_vals)

        mean_a = float(np.mean(valid_lab[:, 1]))
        mean_b = float(np.mean(valid_lab[:, 2]))
        ab_ratio = mean_b / max(mean_a, 1.0)

        if ab_ratio > 1.05:
            lab_vote = "Orange"
        elif ab_ratio < 0.90:
            lab_vote = "Red"
        else:
            lab_vote = "Unclear"

        hue_vote = (
            "Orange"
            if orange_hue_ratio > red_hue_ratio * 1.2
            else "Red"
            if red_hue_ratio > orange_hue_ratio * 1.2
            else "Unclear"
        )

        if hue_vote == lab_vote and hue_vote != "Unclear":
            predicted = hue_vote
            confidence = max(ratios["Orange"], ratios["Red"]) * 1.3
        elif hue_vote != "Unclear":
            predicted = hue_vote
            confidence = ratios["Orange"] if hue_vote == "Orange" else ratios["Red"]
            confidence *= 1.1
        elif lab_vote != "Unclear":
            predicted = lab_vote
            confidence = ratios["Orange"] if lab_vote == "Orange" else ratios["Red"]
            confidence *= 1.1
        else:
            predicted = "Orange" if ratios["Orange"] > ratios["Red"] else "Red"
            confidence = max(ratios["Orange"], ratios["Red"]) * 0.9

        return predicted, float(confidence)
