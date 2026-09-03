import numpy as np
from typing import Any, Dict, Tuple

from picture_tool.color.strategies.base import (
    ColorRange,
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
        hsv_ratio = safe_ratio(np.count_nonzero(h_mask), len(hsv_vals))
        debug["hsv_ratio"] = hsv_ratio

        # Delegate matching to generic
        generic_score, generic_debug = super().match_ratio(hsv_vals, lab_vals, color_range)
        debug.update(generic_debug)
        debug["hsv_ratio"] = hsv_ratio

        # Green weights
        weights = {"hsv": 0.6, "lab": 0.2, "hue_sim": 0.2}
        final_score = weighted_score(
            {
                "hsv": hsv_ratio,
                "lab": debug.get("lab_ratio", 0.0),
                "hue_sim": debug.get("hue_similarity"),
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
