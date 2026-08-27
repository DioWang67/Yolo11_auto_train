from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

#: Fraction trimmed from each side when isolating the center of a region.
CENTER_MARGIN_RATIO = 0.15


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


# ---------------------------------------------------------------------------
# Shared numeric helpers
#
# These mirror core/stats_color_checker.py in the inference repository. The two
# implementations are separate code bases but gate the same product, so a
# decision rule that differs between them lets a model clear the training gate
# and then behave differently on the line.
# ---------------------------------------------------------------------------


def circular_hue_mean(hue_values: np.ndarray) -> float:
    """Mean hue on OpenCV's 0..179 circle.

    A plain arithmetic mean breaks at the 0/179 seam: red pixels at 3 and 178
    average to ~90, which is green, so a seam-crossing red loses its similarity
    score to a color it does not resemble. Hue is expanded to a full turn,
    averaged as unit vectors, then mapped back.
    """
    values = np.asarray(hue_values, dtype=np.float64).ravel()
    if values.size == 0:
        return 0.0
    angles = values * (np.pi / 90.0)
    mean_angle = np.arctan2(
        float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles)))
    )
    return float(np.mod(mean_angle * (90.0 / np.pi), 180.0))


def hue_in_range(h_vals: np.ndarray, hue_min: float, hue_max: float) -> np.ndarray:
    """Hue membership test that survives the 0/179 seam.

    A margin can push a recorded range past either end of the circle, and a
    color calibrated around hue 0 records a range that wraps by construction.
    Both read as ``hue_min > hue_max`` once normalized, which the plain
    ``>= min and <= max`` test answered with "never matches".
    """
    lo = float(hue_min)
    hi = float(hue_max)
    if hi - lo >= 180.0:
        return np.ones(h_vals.shape, dtype=bool)
    lo %= 180.0
    hi %= 180.0
    if lo <= hi:
        return (h_vals >= lo) & (h_vals <= hi)
    return (h_vals >= lo) | (h_vals <= hi)


def center_crop(img: np.ndarray, margin_ratio: float = CENTER_MARGIN_RATIO) -> np.ndarray:
    """Crop a centered region, falling back to the full image when it cannot.

    Shared so every fast path and the main scoring path judge the *same*
    pixels. They previously mixed per-axis margins with ``min(h, w)`` margins,
    so the decisions could legitimately disagree about an elongated region.
    """
    if img.size == 0:
        return img
    h, w = img.shape[:2]
    margin = int(min(h, w) * margin_ratio)
    if margin <= 0 or margin * 2 >= h or margin * 2 >= w:
        return img
    cropped = img[margin : h - margin, margin : w - margin]
    return cropped if cropped.size else img


def safe_ratio(count: int, total: int) -> float:
    """Fraction of ``total``, answering 0.0 rather than dividing by zero."""
    return float(count) / total if total else 0.0


def weighted_score(
    terms: Dict[str, Optional[float]], weights: Dict[str, float]
) -> float:
    """Combine scored terms, dropping absent ones and renormalizing.

    A term with no baseline behind it used to default to a perfect 1.0 and
    still collect its full weight, so a color with incomplete statistics
    outscored one with complete statistics -- exactly the wrong way round.
    Renormalizing is a no-op when every term is present, because each weight
    set already sums to 1.
    """
    used = [
        (value, weights[key])
        for key, value in terms.items()
        if value is not None and weights.get(key)
    ]
    total_weight = sum(weight for _, weight in used)
    if total_weight <= 0.0:
        return 0.0
    return float(sum(value * weight for value, weight in used) / total_weight)
