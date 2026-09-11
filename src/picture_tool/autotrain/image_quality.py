"""Offline image quality metrics for collected production samples.

These are computed from images the inference project already wrote to disk,
in this process, long after the inspection finished. Nothing here runs on the
inference thread, so a slow or failing metric costs production nothing.

The three metrics answer different questions about why a sample might be
worth retraining on: ``brightness`` and ``saturation`` describe the lighting
and colour the frame was captured under, ``blur_score`` describes whether the
frame is sharp enough to be worth labelling at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageQuality:
    """Scalar descriptors of one image.

    ``brightness`` and ``saturation`` are HSV channel means on a 0-255 scale.
    ``blur_score`` is the variance of the Laplacian: higher is sharper, and
    the useful threshold is station-specific, so nothing here judges it.
    """

    brightness: float
    saturation: float
    blur_score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "brightness": self.brightness,
            "saturation": self.saturation,
            "blur_score": self.blur_score,
        }


def measure_array(image: Any) -> ImageQuality:
    """Measure an in-memory BGR or grayscale image.

    Raises ValueError for anything that is not a usable 2-D or 3-channel
    image, so a caller cannot silently record zeros for a broken read.
    """
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("image is empty")
    if array.ndim == 2:
        bgr = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif array.ndim == 3 and array.shape[2] == 3:
        bgr = array.astype(np.uint8)
    elif array.ndim == 3 and array.shape[2] == 4:
        bgr = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError(f"unsupported image shape: {array.shape}")

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    value = np.asarray(hsv[:, :, 2], dtype=np.float64)
    saturation = np.asarray(hsv[:, :, 1], dtype=np.float64)
    return ImageQuality(
        brightness=float(np.mean(value)),
        saturation=float(np.mean(saturation)),
        blur_score=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    )


def measure_file(path: str | Path) -> ImageQuality | None:
    """Measure an image on disk, returning None when it cannot be read.

    A missing or unreadable image is an ordinary outcome here: production
    retention may already have deleted it. Callers record the absence rather
    than failing the collection pass.
    """
    resolved = Path(path)
    try:
        image = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
    except cv2.error as exc:  # pragma: no cover - defensive, cv2 usually returns None
        LOGGER.warning("Image quality read failed for %s: %s", resolved, exc)
        return None
    if image is None:
        LOGGER.debug("Image quality skipped, unreadable or missing: %s", resolved)
        return None
    try:
        return measure_array(image)
    except (ValueError, cv2.error) as exc:
        LOGGER.warning("Image quality measurement failed for %s: %s", resolved, exc)
        return None
