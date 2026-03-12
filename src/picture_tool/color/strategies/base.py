from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class ColorRange:
    """Data object definition, moved from color_verifier.py to be accessible by strategies."""
    def __init__(
        self,
        name: str,
        hsv_min: np.ndarray,
        hsv_max: np.ndarray,
        lab_min: np.ndarray,
        lab_max: np.ndarray,
        hsv_mean: Optional[np.ndarray] = None,
        lab_mean: Optional[np.ndarray] = None,
        coverage_mean: Optional[float] = None,
        hsv_p10: Optional[np.ndarray] = None,
        hsv_p90: Optional[np.ndarray] = None,
        lab_p10: Optional[np.ndarray] = None,
        lab_p90: Optional[np.ndarray] = None,
    ):
        self.name = name
        self.hsv_min = hsv_min
        self.hsv_max = hsv_max
        self.lab_min = lab_min
        self.lab_max = lab_max
        self.hsv_mean = hsv_mean
        self.lab_mean = lab_mean
        self.coverage_mean = coverage_mean
        self.hsv_p10 = hsv_p10
        self.hsv_p90 = hsv_p90
        self.lab_p10 = lab_p10
        self.lab_p90 = lab_p90


class ColorStrategy(ABC):
    """
    Abstract Base Class for Color Strategies.
    Every specific color (e.g. Red, Yellow) should implement this interface
    to provide its own matching logic and masking logic.
    """

    @abstractmethod
    def match_ratio(
        self,
        hsv_vals: np.ndarray,
        lab_vals: np.ndarray,
        color_range: ColorRange
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate how well the given pixels match this specific color.
        Returns:
            Tuple[float, Dict[str, Any]]: The confidence score (0-1) and debug details.
        """
        pass

    @abstractmethod
    def build_mask(
        self,
        hsv_img: np.ndarray,
        lab_img: np.ndarray,
        color_range: ColorRange,
        global_sat_mask: np.ndarray
    ) -> np.ndarray:
        """
        Build a boolean mask for pixels that match this color.
        Returns:
            np.ndarray: Boolean mask of the same shape as hsv_img.
        """
        pass

    def fast_detect(
        self,
        hsv_img: np.ndarray,
        lab_img: np.ndarray,
        color_range: ColorRange
    ) -> Tuple[bool, float]:
        """
        Optional fast-path detection for obvious color majorities (like Black/Yellow).
        Return (True, confidence) to short-circuit the evaluation pipeline.
        """
        return False, 0.0

    def post_correction(
        self,
        predicted_color: str,
        confidence: float,
        ratios: Dict[str, float],
        center_hsv: np.ndarray,
        center_lab: np.ndarray
    ) -> Optional[Tuple[str, float]]:
        """
        Optional late-stage correction logic (e.g. Green dominance override, Orange/Red tiebreak).
        Return a new (color, confidence) tuple to override the prediction.
        """
        return None
