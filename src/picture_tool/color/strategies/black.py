import numpy as np
from typing import Any, Dict, Tuple

from picture_tool.color.strategies.base import (
    ColorRange,
    center_crop,
    safe_ratio,
    weighted_score,
)
from picture_tool.color.strategies.generic import GenericStrategy

BLACK_S_THRESHOLD = 50.0
BLACK_V_THRESHOLD = 80.0
BLACK_MIN_COVERAGE = 0.6


def _rule_margin(value: float, limit: float) -> float:
    """How far below ``limit`` a measurement sits, on a 0..1 scale."""
    if limit <= 0:
        return 0.0
    return float(min(1.0, max(0.0, 1.0 - value / limit)))

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
        hsv_ratio = safe_ratio(np.count_nonzero(h_mask), len(hsv_vals))
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
        lab_ratio = safe_ratio(np.count_nonzero(lab_mask), len(lab_vals))
        debug["lab_ratio"] = lab_ratio

        # Black's hue really is undefined, so the similarity term is dropped
        # and the remaining weights renormalized. Passing a hard-coded 1.0
        # instead did not ignore the term -- it awarded a perfect score for it,
        # handing Black a free 0.2 on every region, including regions with no
        # black pixels at all.
        weights = {"hsv": 0.5, "lab": 0.3, "hue_sim": 0.2, "lab_chroma": 0.0}
        final_score = weighted_score(
            {"hsv": hsv_ratio, "lab": lab_ratio, "hue_sim": None},
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
        """Detects if it's black early-on for short-circuiting."""
        center_region = center_crop(hsv_img)
        if center_region.size == 0:
            return False, 0.0

        mean_s = float(np.mean(center_region[:, :, 1]))
        mean_v = float(np.mean(center_region[:, :, 2]))
        median_s = float(np.median(center_region[:, :, 1]))
        median_v = float(np.median(center_region[:, :, 2]))

        black_mask = (center_region[:, :, 1] < BLACK_S_THRESHOLD) & (
            center_region[:, :, 2] < BLACK_V_THRESHOLD
        )
        black_coverage = safe_ratio(np.count_nonzero(black_mask), black_mask.size)

        # Confidence has to describe the rule that actually fired. Reporting
        # coverage unconditionally meant a decision reached by the mean or the
        # median rule was scored by an unrelated number, which could then fail
        # black's own threshold -- "it is black, and black is NG" at once.
        evidence = [0.0]
        if mean_s < BLACK_S_THRESHOLD and mean_v < BLACK_V_THRESHOLD:
            evidence.append(
                min(
                    _rule_margin(mean_s, BLACK_S_THRESHOLD),
                    _rule_margin(mean_v, BLACK_V_THRESHOLD),
                )
            )
        if median_s < BLACK_S_THRESHOLD * 0.8 and median_v < BLACK_V_THRESHOLD * 0.8:
            evidence.append(
                min(
                    _rule_margin(median_s, BLACK_S_THRESHOLD * 0.8),
                    _rule_margin(median_v, BLACK_V_THRESHOLD * 0.8),
                )
            )
        if black_coverage > BLACK_MIN_COVERAGE:
            evidence.append(black_coverage)

        is_black = len(evidence) > 1
        return is_black, (max(evidence) if is_black else 0.0)
