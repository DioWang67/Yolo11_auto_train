from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

#: Fraction trimmed from each side when isolating the center of a region.
CENTER_MARGIN_RATIO = 0.15
#: Pixels a chromatic color's per-pixel envelope test must clear before the
#: pixel counts toward any color at all. Mirrors
#: ``core/stats_color_checker.py`` in the inference repository.
DEFAULT_SAT_THRESHOLD = 20.0
#: Floor on the largest connected matching region, in pixels. Below this, a
#: color is unmeasurable rather than scored from noise. A real deployment
#: value needs measuring against real crops before it can move, same as any
#: other value compared against a threshold.
DEFAULT_MIN_BLOB_PIXELS = 8.0


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
        hsv_img: np.ndarray,
        lab_img: np.ndarray,
        color_range: ColorRange
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate how well the whole detection box matches this specific color.

        ``hsv_img``/``lab_img`` are the whole box in its own 2D shape, not a
        pre-flattened, pre-cropped pixel list: connected-component selection
        (see ``measure_color_region``) needs the spatial layout a flat array
        throws away.

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
        Optional recognizer for an obvious color majority, kept for diagnostics.

        Returning ``(True, confidence)`` no longer short-circuits anything: the
        caller records it in debug info and scores every color anyway. It used
        to return early, which reported whichever color was recognized *first*
        rather than the one that won -- a 40% yellow band beat a 60% green one.
        The inference runtime removed the same shortcut, so do not reinstate the
        early return here without changing both sides and the shared fixture.
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
        Optional late-stage correction logic (e.g. the Orange/Red tiebreak).

        Return a new ``(color, confidence)`` tuple to override the prediction. A
        correction decides *which* color the region is; it measures nothing new
        about how strongly the region matches, so it must carry an existing
        score across rather than invent one from a raw pixel ratio.
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


def circular_hue_distance(h1: float, h2: float) -> float:
    """Shortest distance between two hues on OpenCV's 0..179 circle."""
    diff = abs(h1 - h2)
    return float(min(diff, 180 - diff))


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
    pixels. The ratio is evaluated independently on width and height, mirroring
    ``center_crop_by_ratio`` in the inference runtime
    (``yolo11_inference/core/color_sampling.py``) which the baseline's
    statistics are measured with. A single ``min(h, w)`` margin keeps a
    different region of an elongated ROI -- on a wire that is enough to change
    which color wins, so the two sides have to derive the crop the same way.
    """
    if not isinstance(img, np.ndarray) or img.ndim < 2 or img.size == 0:
        return img
    ratio = float(margin_ratio)
    if not np.isfinite(ratio) or ratio <= 0.0:
        return img
    h, w = img.shape[:2]
    margin_y = int(h * ratio)
    margin_x = int(w * ratio)
    if margin_y <= 0 and margin_x <= 0 or margin_y * 2 >= h or margin_x * 2 >= w:
        return img
    cropped = img[margin_y : h - margin_y, margin_x : w - margin_x]
    return cropped if cropped.size else img


def safe_ratio(count: int, total: int) -> float:
    """Fraction of ``total``, answering 0.0 rather than dividing by zero."""
    return float(count) / total if total else 0.0


def largest_matching_blob(
    match_mask: np.ndarray, min_pixels: float
) -> Optional[np.ndarray]:
    """The largest 4-connected region of ``match_mask``, or ``None`` below floor.

    Mirror of ``_largest_matching_blob`` in
    ``core/stats_color_checker.py`` (inference repository). A detection box is
    scored for one known expected color, not classified from scratch, so this
    never has to guess *which* color a region is -- only which pixels, among
    those already matching that color's envelope, belong to one coherent
    object rather than to scattered, unrelated pixels elsewhere in the box
    (board silkscreen, a reflection, a neighboring wire). A wire's position and
    curve vary board to board; a fixed geometric crop used to exclude that
    scattered matter by luck, when the wire happened to sit where the crop
    assumed it would. This excludes it by construction instead.

    ``min_pixels`` rejects a match too small to trust -- a handful of stray
    pixels sharing a hue is not a wire.
    """
    if match_mask.size == 0:
        return None
    count, labels = cv2.connectedComponents(
        match_mask.astype(np.uint8), connectivity=4
    )
    if count <= 1:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # label 0 is background, never a candidate
    largest_label = int(np.argmax(sizes))
    if sizes[largest_label] < min_pixels:
        return None
    return labels == largest_label


def measure_color_region(
    hsv_img: np.ndarray,
    lab_img: np.ndarray,
    h_mask: np.ndarray,
    sat_threshold: float = DEFAULT_SAT_THRESHOLD,
    min_blob_pixels: float = DEFAULT_MIN_BLOB_PIXELS,
) -> Tuple[float, np.ndarray, np.ndarray, int]:
    """Restrict a chromatic color's scoring to its largest connected match.

    ``h_mask`` is this color's own hue/sat/value test over the whole
    detection box, not a fixed geometric sub-crop -- a wire's position and
    curve vary board to board, so assuming it sits in a fixed fraction of the
    box discards real wire pixels on some boards and admits board background
    on others. Combined with the shared saturation gate, only the largest
    connected region passing both is scored; a stray same-hue pixel elsewhere
    in the box no longer inflates the ratio just because a rectangle used to
    exclude it by luck.

    Returns ``(hsv_ratio, measured_hsv, measured_lab, candidate_pixels)``.
    When nothing forms a connected region at all, ``measured_*`` fall back to
    the whole saturation-gated pool rather than an empty selection: this is
    not the contamination problem blob selection exists to fix (a wire sitting
    somewhere a fixed crop did not expect); it is a genuinely weak hue signal
    (real desaturation, for instance), and the LAB/hue-mean terms computed
    from the fallback still carry information a zero ``hsv_ratio`` does not
    erase.
    """
    s_vals = hsv_img[:, :, 1]
    sat_mask = s_vals >= sat_threshold
    candidate_pixels = int(np.count_nonzero(sat_mask))
    if candidate_pixels == 0:
        empty = np.zeros((0, 3), dtype=hsv_img.dtype)
        return 0.0, empty, np.zeros((0, 3), dtype=lab_img.dtype), 0

    blob_mask = largest_matching_blob(h_mask & sat_mask, min_blob_pixels)
    if blob_mask is not None:
        hsv_ratio = safe_ratio(int(np.count_nonzero(blob_mask)), candidate_pixels)
        measured_mask = blob_mask
    else:
        hsv_ratio = 0.0
        measured_mask = sat_mask

    return hsv_ratio, hsv_img[measured_mask], lab_img[measured_mask], candidate_pixels


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
