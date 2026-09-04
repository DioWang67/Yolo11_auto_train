import numpy as np
from typing import Any, Dict, Optional, Tuple

from picture_tool.color.strategies.base import (
    ColorRange,
    DEFAULT_SAT_THRESHOLD,
    circular_hue_distance,
    circular_hue_mean,
    measure_color_region,
    safe_ratio,
    weighted_score,
)
from picture_tool.color.strategies.generic import GenericStrategy

ORANGE_RED_TIE_MARGIN = 0.15

from picture_tool.color.strategies.registry import ColorStrategyRegistry  # noqa: E402

@ColorStrategyRegistry.register("Red", "Orange")
class RedOrangeStrategy(GenericStrategy):
    """Specific logic for Red and Orange, which have high similarity and cross-dependency."""

    def match_ratio(
        self,
        hsv_img: np.ndarray,
        lab_img: np.ndarray,
        color_range: ColorRange,
    ) -> Tuple[float, Dict[str, Any]]:
        debug: Dict[str, Any] = {}
        if hsv_img.size == 0 or lab_img.size == 0:
            return 0.0, debug

        h_vals = hsv_img[:, :, 0]
        s_vals = hsv_img[:, :, 1]
        v_vals = hsv_img[:, :, 2]

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

        # LAB Chroma similarity is specific to O/R. None when the baseline
        # carries no Lab statistic, so the term is dropped instead of scoring a
        # perfect match against a baseline that does not exist.
        lab_chroma_similarity = None
        if color_range.lab_mean is not None:
            mean_a = float(np.mean(blob_lab[:, 1]))
            mean_b = float(np.mean(blob_lab[:, 2]))
            expected_a = float(color_range.lab_mean[1])
            expected_b = float(color_range.lab_mean[2])

            lab_chroma_dist = np.sqrt(
                (mean_a - expected_a) ** 2 + (mean_b - expected_b) ** 2
            )
            lab_chroma_similarity = float(np.exp(-lab_chroma_dist / 20.0))

            debug["lab_chroma_dist"] = float(lab_chroma_dist)
            debug["lab_chroma_similarity"] = lab_chroma_similarity

        weights = {"hsv": 0.35, "lab": 0.25, "hue_sim": 0.25, "lab_chroma": 0.15}
        final_score = weighted_score(
            {
                "hsv": hsv_ratio,
                "lab": lab_ratio,
                "hue_sim": hue_similarity,
                "lab_chroma": lab_chroma_similarity,
            },
            weights,
        )
        debug["final_score"] = float(final_score)
        return float(final_score), debug

    def post_correction(
        self,
        predicted_color: str,
        confidence: float,
        ratios: Dict[str, float],
        hsv_img: np.ndarray,
        lab_img: np.ndarray
    ) -> Optional[Tuple[str, float]]:
        """Implements the Orange/Red tiebreak logic.

        ``hsv_img``/``lab_img`` are the whole detection box, not a fixed
        geometric center-crop -- matching what ``match_ratio`` above measures
        against, since a tie-break drawn from a different pixel pool than the
        scores it is breaking a tie between would not be resolving the same
        disagreement.
        """
        if (
            "Orange" not in ratios
            or "Red" not in ratios
            or predicted_color not in {"Orange", "Red"}
            or abs(ratios["Orange"] - ratios["Red"]) >= ORANGE_RED_TIE_MARGIN
        ):
            return None

        if hsv_img.size == 0 or lab_img.size == 0:
            return None

        flat_hsv = hsv_img.reshape(-1, 3)
        flat_lab = lab_img.reshape(-1, 3)
        pair_score = max(ratios["Orange"], ratios["Red"])

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

        # The tie-breaker decides *which* of Orange and Red the pair is; it
        # measures nothing new about how strongly the region matches. The
        # multipliers below used to be 1.3 / 1.1, which manufactured confidence
        # out of a disambiguation step and let the winner overtake an unrelated
        # color that had legitimately scored higher -- the Black-reported-as-
        # Orange failure. The pair's own best score transfers to the winner and
        # nothing is created.
        if hue_vote == lab_vote and hue_vote != "Unclear":
            predicted = hue_vote
        elif hue_vote != "Unclear":
            predicted = hue_vote
        elif lab_vote != "Unclear":
            predicted = lab_vote
        else:
            predicted = "Orange" if ratios["Orange"] > ratios["Red"] else "Red"

        return predicted, float(pair_score)
