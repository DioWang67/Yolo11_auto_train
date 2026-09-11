"""Offline image quality metrics.

These run long after the inspection, on images production already saved, so
the behaviour that matters is that a bad image costs one measurement and
never raises into a collection pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from picture_tool.autotrain.image_quality import (
    ImageQuality,
    measure_array,
    measure_file,
)

cv2 = pytest.importorskip("cv2", reason="cv2 is required for image quality")


def _bgr(value=(10, 20, 200), size=16):
    array = np.zeros((size, size, 3), dtype=np.uint8)
    array[:, :] = value
    return array


# ---------------------------------------------------------------------------
# Measurement


def test_a_colour_image_yields_all_three_metrics():
    quality = measure_array(_bgr())

    assert isinstance(quality, ImageQuality)
    assert quality.brightness > 0
    assert quality.saturation > 0
    assert quality.blur_score >= 0


def test_a_brighter_image_measures_brighter():
    dark = measure_array(_bgr((10, 10, 10)))
    bright = measure_array(_bgr((240, 240, 240)))

    assert bright.brightness > dark.brightness


def test_a_saturated_image_measures_more_saturated():
    grey = measure_array(_bgr((128, 128, 128)))
    vivid = measure_array(_bgr((0, 0, 255)))

    assert vivid.saturation > grey.saturation


def test_a_sharp_image_scores_higher_than_a_flat_one():
    flat = measure_array(_bgr((128, 128, 128)))
    edges = _bgr((0, 0, 0))
    edges[::2, :] = (255, 255, 255)

    assert measure_array(edges).blur_score > flat.blur_score


def test_grayscale_and_rgba_images_are_accepted():
    grayscale = np.full((8, 8), 128, dtype=np.uint8)
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[:, :] = (10, 20, 200, 255)

    assert measure_array(grayscale).brightness > 0
    assert measure_array(rgba).saturation > 0


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((0, 0, 3), dtype=np.uint8),
        np.zeros((4, 4, 2), dtype=np.uint8),
        np.zeros((2, 2, 2, 2), dtype=np.uint8),
    ],
    ids=["empty", "two-channel", "four-dimensional"],
)
def test_an_unusable_array_is_refused_rather_than_scored_as_zero(bad):
    with pytest.raises(ValueError):
        measure_array(bad)


def test_metrics_serialise():
    payload = measure_array(_bgr()).to_dict()

    assert set(payload) == {"brightness", "saturation", "blur_score"}


# ---------------------------------------------------------------------------
# Files


def test_measuring_a_real_file_works(tmp_path):
    path = tmp_path / "image.jpg"
    cv2.imwrite(str(path), _bgr())

    assert measure_file(path) is not None


def test_a_missing_file_returns_none_rather_than_raising(tmp_path):
    """Retention may already have deleted the image; that is not an error."""
    assert measure_file(tmp_path / "gone.jpg") is None


def test_a_file_that_is_not_an_image_returns_none(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("this is not an image", encoding="utf-8")

    assert measure_file(path) is None


def test_a_directory_passed_by_mistake_returns_none(tmp_path):
    assert measure_file(tmp_path) is None
