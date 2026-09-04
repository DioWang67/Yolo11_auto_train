import numpy as np
from typing import Any, Dict, Tuple

from picture_tool.color.strategies.base import (
    ColorRange,
    largest_matching_blob,
    DEFAULT_MIN_BLOB_PIXELS,
)
from picture_tool.color.strategies.generic import GenericStrategy

from picture_tool.color.strategies.registry import ColorStrategyRegistry  # noqa: E402

@ColorStrategyRegistry.register("Black")
class BlackStrategy(GenericStrategy):
    """Specific logic for verifying Black colors."""

    def match_ratio(
        self,
        hsv_img: np.ndarray,
        lab_img: np.ndarray,
        color_range: ColorRange,
    ) -> Tuple[float, Dict[str, Any]]:
        debug: Dict[str, float] = {}
        if hsv_img.size == 0 or lab_img.size == 0:
            return 0.0, debug

        # Hue is undefined for achromatic pixels, and unlike every other color
        # this is not saturation-gated first -- black *is* the desaturated
        # case. Otherwise this is the same measurement as every chromatic
        # color now gets: the whole detection box, restricted to its largest
        # connected match, because black's wire varies board to board the
        # same way a chromatic one does. Mirrors
        # ``core/stats_color_checker.py``'s ``_black_baseline_match`` in the
        # inference repository.
        sv_mask = (
            (hsv_img[:, :, 1] >= color_range.hsv_min[1])
            & (hsv_img[:, :, 1] <= color_range.hsv_max[1])
            & (hsv_img[:, :, 2] >= color_range.hsv_min[2])
            & (hsv_img[:, :, 2] <= color_range.hsv_max[2])
        )
        lab_mask = np.all(
            (lab_img >= color_range.lab_min) & (lab_img <= color_range.lab_max),
            axis=2,
        )
        blob_mask = largest_matching_blob(
            sv_mask & lab_mask, DEFAULT_MIN_BLOB_PIXELS
        )
        if blob_mask is None:
            debug.update({"blob_pixels": 0, "final_score": 0.0})
            return 0.0, debug

        blob_pixels = int(np.count_nonzero(blob_mask))
        total_pixels = hsv_img.shape[0] * hsv_img.shape[1]
        score = min(1.0, float(blob_pixels) / float(total_pixels)) if total_pixels else 0.0
        debug.update({"blob_pixels": blob_pixels, "final_score": float(score)})
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
