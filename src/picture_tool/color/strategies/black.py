import numpy as np
from typing import Any, Dict, Tuple

from picture_tool.color.strategies.base import (
    ColorRange,
    safe_ratio,
)
from picture_tool.color.strategies.generic import GenericStrategy

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
        debug: Dict[str, float] = {}
        if hsv_vals.size == 0 or lab_vals.size == 0:
            return 0.0, debug

        # Hue is undefined for achromatic pixels. Score the learned S/V and
        # LAB envelope jointly, then normalize by the crop coverage recorded
        # when this baseline was built. This is the same contract as the line
        # inference implementation.
        sv_mask = (
            (hsv_vals[:, 1] >= color_range.hsv_min[1])
            & (hsv_vals[:, 1] <= color_range.hsv_max[1])
            & (hsv_vals[:, 2] >= color_range.hsv_min[2])
            & (hsv_vals[:, 2] <= color_range.hsv_max[2])
        )
        lab_mask = (
            (lab_vals[:, 0] >= color_range.lab_min[0])
            & (lab_vals[:, 0] <= color_range.lab_max[0])
            & (lab_vals[:, 1] >= color_range.lab_min[1])
            & (lab_vals[:, 1] <= color_range.lab_max[1])
            & (lab_vals[:, 2] >= color_range.lab_min[2])
            & (lab_vals[:, 2] <= color_range.lab_max[2])
        )
        raw_ratio = safe_ratio(np.count_nonzero(sv_mask & lab_mask), len(hsv_vals))
        reference_coverage = color_range.coverage_mean
        if (
            reference_coverage is None
            or not np.isfinite(reference_coverage)
            or not 0.0 < reference_coverage <= 1.0
        ):
            debug.update(
                {
                    "raw_ratio": raw_ratio,
                    "reference_coverage": 0.0,
                    "final_score": 0.0,
                    "invalid_reference_coverage": 1.0,
                }
            )
            return 0.0, debug
        score = min(1.0, raw_ratio / max(reference_coverage, 1e-6))
        debug.update(
            {
                "raw_ratio": raw_ratio,
                "reference_coverage": float(reference_coverage),
                "final_score": float(score),
            }
        )
        return float(score), debug

    def fast_detect(
        self,
        hsv_img: np.ndarray,
        lab_img: np.ndarray,
        color_range: ColorRange
    ) -> Tuple[bool, float]:
        # Black must compete with the other learned scores. A shortcut based on
        # hand-written mean/median gates bypassed its baseline and produced a
        # different contract from line inference.
        return False, 0.0

    def build_mask(
        self,
        hsv_img: np.ndarray,
        lab_img: np.ndarray,
        color_range: ColorRange,
        global_sat_mask: np.ndarray,
    ) -> np.ndarray:
        del global_sat_mask
        return (
            (hsv_img[:, :, 1] >= color_range.hsv_min[1])
            & (hsv_img[:, :, 1] <= color_range.hsv_max[1])
            & (hsv_img[:, :, 2] >= color_range.hsv_min[2])
            & (hsv_img[:, :, 2] <= color_range.hsv_max[2])
            & np.all(
                (lab_img >= color_range.lab_min)
                & (lab_img <= color_range.lab_max),
                axis=2,
            )
        )
