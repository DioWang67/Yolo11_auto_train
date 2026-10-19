"""The probe image is a cross-repository contract.

yolo11_inference rebuilds this image from the stamp's parameters with its own
copy of the formula (the repositories never import each other) and pins the
same reference values in tests/test_anomalib_compatibility.py. Changing the
formula here without a new probe ``kind`` would make the station reject every
model packaged afterwards.
"""

import numpy as np
import pytest

from picture_tool.train import anomalib_compat


def test_probe_is_larger_than_the_model_input_so_the_resize_runs():
    assert anomalib_compat.probe_size(256) == 391
    assert anomalib_compat.probe_image(256).shape == (3, 391, 391)


def test_probe_matches_the_reference_values_the_station_pins():
    image = anomalib_compat.probe_image(256).astype(np.float64)

    assert anomalib_compat.PROBE_KIND == "sincos_v2"
    assert image.mean() == pytest.approx(0.449995586, abs=1e-8)
    assert image.std() == pytest.approx(0.031929384, abs=1e-8)
    for index, expected in {
        (0, 0, 0): 0.449999988,
        (1, 17, 200): 0.450729549,
        (2, 390, 390): 0.476070642,
        (0, 123, 45): 0.420663834,
        (2, 77, 301): 0.446561217,
    }.items():
        assert image[index] == pytest.approx(expected, abs=1e-6)


def test_probe_stays_inside_the_unit_range():
    image = anomalib_compat.probe_image(256)

    assert image.dtype == np.float32
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
