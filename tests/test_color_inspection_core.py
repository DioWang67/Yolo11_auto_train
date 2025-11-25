import numpy as np

from picture_tool.color import color_inspection as color


def test_safe_percentile_returns_default_for_empty_array():
    assert np.isnan(color.safe_percentile(np.array([]), 95))
    assert color.safe_percentile(np.array([]), 50, default=42.0) == 42.0


def test_safe_ratio_handles_zero_denominator():
    assert np.isnan(color.safe_ratio(np.ones(3), 0))
    assert color.safe_ratio(np.zeros(3), 5, default=0.0) == 0.0


def test_color_palette_normalizes_names_and_aliases():
    palette = color.ColorPalette(
        names=["Red", "Amber", "Red"],
        aliases={"amb": "Amber", "amber": "Amber"},
        hue_ranges={"Amber": (10.0, 20.0)},
    )

    assert palette.names == ("Red", "Amber")
    assert palette.aliases["amb"] == "Amber"
    assert palette.aliases["red"] == "Red"
    assert palette.aliases["amber"] == "Amber"
