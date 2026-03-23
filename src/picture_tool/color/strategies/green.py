import numpy as np
from typing import Any, Dict, Optional, Tuple

from picture_tool.color.strategies.base import ColorRange
from picture_tool.color.strategies.generic import GenericStrategy

GREEN_DOMINANCE_RATIO = 0.3

from picture_tool.color.strategies.registry import ColorStrategyRegistry  # noqa: E402

@ColorStrategyRegistry.register("Green")
class GreenStrategy(GenericStrategy):
    """Specific logic for verifying Green colors."""

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
            (h_vals >= 70)
            & (h_vals <= 100)
            & (s_vals >= 75)
            & (v_vals >= 30)
            & (v_vals <= 100)
        )
        hsv_ratio = float(np.count_nonzero(h_mask)) / len(hsv_vals)
        debug["hsv_ratio"] = hsv_ratio

        # Delegate matching to generic
        generic_score, generic_debug = super().match_ratio(hsv_vals, lab_vals, color_range)
        debug.update(generic_debug)

        # Green weights
        weights = {"hsv": 0.6, "lab": 0.2, "hue_sim": 0.2}
        final_score = (
            hsv_ratio * weights["hsv"]
            + debug.get("lab_ratio", 0.0) * weights["lab"]
            + debug.get("hue_similarity", 1.0) * weights["hue_sim"]
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
        """Implements green dominance correction (override red)."""
        if predicted_color != "Red":
            return None

        if center_hsv.size == 0:
            return None

        h_vals = center_hsv[:, :, 0]
        total_pixels = h_vals.size
        if total_pixels == 0:
            return None

        green_pixels = np.sum((h_vals >= 70) & (h_vals <= 100))
        green_ratio = green_pixels / total_pixels

        if green_ratio > GREEN_DOMINANCE_RATIO:
            return "Green", float(green_ratio)
            
        return None
