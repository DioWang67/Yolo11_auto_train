from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from picture_tool.color.strategies.base import ColorRange
from picture_tool.color.strategies.green import GreenStrategy
from picture_tool.color.strategies.red_orange import RedOrangeStrategy
from picture_tool.color.strategies.registry import ColorStrategyRegistry
from picture_tool.color.strategies.yellow import YellowStrategy
from picture_tool.color.strategies import registry as registry_module


def _range(name: str, *, mean: bool = True) -> ColorRange:
    return ColorRange(
        name=name,
        hsv_min=np.array([0, 0, 0]),
        hsv_max=np.array([180, 255, 255]),
        lab_min=np.array([0, 0, 0]),
        lab_max=np.array([255, 255, 255]),
        hsv_mean=np.array([30, 100, 100]) if mean else None,
        lab_mean=np.array([100, 120, 110]) if mean else None,
    )


def test_green_strategy_match_and_post_correction_paths() -> None:
    strategy = GreenStrategy()
    assert strategy.match_ratio(np.array([]), np.array([]), _range("Green")) == (0.0, {})
    hsv = np.array([[80, 100, 60], [10, 10, 200]], dtype=float)
    lab = np.array([[100, 120, 110], [100, 120, 110]], dtype=float)

    score, debug = strategy.match_ratio(hsv, lab, _range("Green"))
    assert 0.0 < score <= 1.0
    assert debug["hsv_ratio"] == 0.5
    assert debug["final_score"] == score

    image = np.full((2, 2, 3), [80, 100, 100], dtype=float)
    assert strategy.post_correction("Orange", 0.5, {}, image, image) is None
    assert strategy.post_correction("Red", 0.5, {}, np.array([]), image) is None
    assert strategy.post_correction("Red", 0.5, {}, image, image) == ("Green", 1.0)
    image[:, :, 0] = 20
    assert strategy.post_correction("Red", 0.5, {}, image, image) is None


@pytest.mark.parametrize("color", ["Red", "Orange"])
@pytest.mark.parametrize("with_lab_mean", [False, True])
def test_red_orange_match_uses_color_specific_hue_and_optional_lab_chroma(
    color: str,
    with_lab_mean: bool,
) -> None:
    strategy = RedOrangeStrategy()
    hue = 2 if color == "Red" else 12
    hsv = np.array([[hue, 200, 180], [50, 200, 180]], dtype=float)
    lab = np.array([[100, 120, 110], [100, 120, 110]], dtype=float)
    color_range = _range(color, mean=with_lab_mean)

    score, debug = strategy.match_ratio(hsv, lab, color_range)

    assert score > 0.3
    assert debug["hsv_ratio"] == 0.5
    assert ("lab_chroma_similarity" in debug) is with_lab_mean
    assert strategy.match_ratio(np.array([]), lab, color_range) == (0.0, {})


@pytest.mark.parametrize(
    ("hues", "lab_ab", "expected"),
    [
        ([10, 10], (100, 120), "Orange"),
        ([2, 2], (100, 120), "Red"),
        ([2, 10], (120, 100), "Red"),
        ([30, 30], (100, 100), "Orange"),
        ([2, 2], (120, 100), "Red"),
    ],
)
def test_red_orange_post_correction_exercises_each_vote_path(
    hues: list[int],
    lab_ab: tuple[int, int],
    expected: str,
) -> None:
    hsv = np.array([[[hues[0], 100, 100], [hues[1], 100, 100]]], dtype=float)
    lab = np.array(
        [[[100, lab_ab[0], lab_ab[1]], [100, lab_ab[0], lab_ab[1]]]],
        dtype=float,
    )

    result = RedOrangeStrategy().post_correction(
        "Orange",
        0.5,
        {"Orange": 0.6, "Red": 0.5},
        hsv,
        lab,
    )

    assert result is not None
    assert result[0] == expected
    assert result[1] > 0


@pytest.mark.parametrize(
    ("predicted", "ratios"),
    [
        ("Red", {"Red": 0.5}),
        ("Green", {"Red": 0.5, "Orange": 0.5}),
        ("Red", {"Red": 0.1, "Orange": 0.9}),
    ],
)
def test_red_orange_post_correction_rejects_inapplicable_inputs(
    predicted: str,
    ratios: dict[str, float],
) -> None:
    image = np.ones((1, 1, 3), dtype=float)
    assert RedOrangeStrategy().post_correction(predicted, 0.5, ratios, image, image) is None


def test_red_orange_post_correction_rejects_empty_and_unsaturated_centers() -> None:
    strategy = RedOrangeStrategy()
    ratios = {"Red": 0.5, "Orange": 0.5}
    image = np.ones((1, 1, 3), dtype=float)
    assert strategy.post_correction("Red", 0.5, ratios, np.array([]), image) is None
    assert strategy.post_correction("Red", 0.5, ratios, image, np.array([])) is None
    image[:, :, 1] = 0
    assert strategy.post_correction("Red", 0.5, ratios, image, image) is None


def test_yellow_strategy_match_and_fast_detection() -> None:
    strategy = YellowStrategy()
    color_range = _range("Yellow")
    assert strategy.match_ratio(np.array([]), np.array([]), color_range) == (0.0, {})
    hsv = np.array([[25, 150, 200], [10, 30, 50]], dtype=float)
    lab = np.array([[100, 120, 110], [100, 120, 110]], dtype=float)
    score, debug = strategy.match_ratio(hsv, lab, color_range)
    assert score > 0
    assert debug["hsv_ratio"] == 0.5

    yellow = np.full((10, 10, 3), [25, 150, 220], dtype=float)
    assert strategy.fast_detect(yellow, yellow, color_range) == (True, 1.0)

    orange = np.full((10, 10, 3), [10, 150, 220], dtype=float)
    detected, confidence = strategy.fast_detect(orange, orange, color_range)
    assert not detected
    assert confidence == 0.0


def test_registry_registration_initialization_fuzzy_lookup_and_missing_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ColorStrategyRegistry, "_strategies", {})
    monkeypatch.setattr(ColorStrategyRegistry, "_fallback", None)
    monkeypatch.setattr(ColorStrategyRegistry, "_initialized", False)

    @ColorStrategyRegistry.register("Exact")
    class _Exact(GreenStrategy):
        pass

    @ColorStrategyRegistry.register_fallback()
    class _Fallback(YellowStrategy):
        pass

    assert ColorStrategyRegistry.get_strategy("EXACT").__class__ is _Exact
    assert ColorStrategyRegistry.get_strategy("prefix-exact-suffix").__class__ is _Exact
    assert ColorStrategyRegistry.get_strategy("unknown").__class__ is _Fallback
    assert "exact" in ColorStrategyRegistry.all_strategies()

    monkeypatch.setattr(ColorStrategyRegistry, "_fallback", None)
    with pytest.raises(RuntimeError, match="Fallback"):
        ColorStrategyRegistry.get_strategy("unknown")

    monkeypatch.setattr(ColorStrategyRegistry, "_initialized", False)
    imported: list[str] = []
    monkeypatch.setattr(
        registry_module.pkgutil,
        "iter_modules",
        lambda path: [(None, "base", False), (None, "custom", False), (None, "registry", False)],
    )
    monkeypatch.setattr(
        registry_module.importlib,
        "import_module",
        lambda name: imported.append(name) or SimpleNamespace(),
    )
    ColorStrategyRegistry.initialize()
    assert imported == ["picture_tool.color.strategies.custom"]
    ColorStrategyRegistry.initialize()
    assert imported == ["picture_tool.color.strategies.custom"]


# ---------------------------------------------------------------------------
# Regressions from the color-detection review
#
# These mirror the inference repository's
# tests/test_stats_color_checker_unit.py. The two code bases gate the same
# product, so a rule that holds there has to hold here too.
# ---------------------------------------------------------------------------


def test_orange_red_tiebreak_cannot_overtake_an_unrelated_color() -> None:
    """The tie-breaker disambiguates; it must not manufacture confidence.

    Reproduces the Black-reported-as-Orange failure: Black legitimately scored
    highest, then the Orange/Red tie-breaker multiplied Orange's score by 1.3
    and Orange won a comparison it had lost on the evidence.
    """
    from picture_tool.color.strategies.red_orange import RedOrangeStrategy

    ratios = {
        "Black": 0.385626,
        "Yellow": 0.362031,
        "Orange": 0.311123,
        "Red": 0.219895,
    }
    # Hue and Lab both vote Orange, so this takes the strongest correction path.
    hsv = np.array([[[10.0, 66.0, 89.0]] * 100], dtype=float)
    lab = np.array([[[87.0, 120.0, 136.0]] * 100], dtype=float)

    winner, confidence = RedOrangeStrategy().post_correction(
        "Orange", ratios["Orange"], ratios, hsv, lab
    )

    assert winner == "Orange"
    assert confidence == pytest.approx(max(ratios["Orange"], ratios["Red"]))
    ratios[winner] = confidence
    assert max(ratios, key=ratios.get) == "Black"


def test_black_scores_nothing_on_pixels_that_are_not_black() -> None:
    """Black's hue term was hard-coded to a perfect 1.0 and kept its weight.

    That handed Black a free 0.2 on every region, whatever was in it.
    """
    from picture_tool.color.strategies.black import BlackStrategy

    color_range = ColorRange(
        name="Black",
        hsv_min=np.array([0.0, 0.0, 0.0]),
        hsv_max=np.array([180.0, 50.0, 80.0]),
        lab_min=np.array([0.0, 0.0, 0.0]),
        lab_max=np.array([10.0, 10.0, 10.0]),
    )
    bright = np.array([[30.0, 240.0, 250.0]] * 50)
    lab = np.array([[240.0, 140.0, 200.0]] * 50)

    score, _debug = BlackStrategy().match_ratio(bright, lab, color_range)

    assert score == pytest.approx(0.0)


def test_circular_hue_mean_crosses_the_seam() -> None:
    from picture_tool.color.strategies.base import circular_hue_mean

    assert circular_hue_mean(np.array([3.0, 178.0])) == pytest.approx(0.5, abs=1e-6)
    assert circular_hue_mean(np.array([10.0, 20.0])) == pytest.approx(15.0, abs=1e-6)
    assert circular_hue_mean(np.array([])) == 0.0


def test_generic_strategy_uses_a_circular_hue_mean() -> None:
    """Red pixels at 3 and 178 averaged to ~90 -- which is green."""
    from picture_tool.color.strategies.generic import GenericStrategy

    seam = np.array([[3.0, 200.0, 150.0]] * 50 + [[178.0, 200.0, 150.0]] * 50)
    lab = np.array([[150.0, 150.0, 150.0]] * 100)

    _score, debug = GenericStrategy().match_ratio(seam, lab, _range("Red"))

    assert debug["mean_hue"] == pytest.approx(0.5, abs=0.1)


def test_hue_range_test_wraps_around_zero() -> None:
    from picture_tool.color.strategies.base import hue_in_range

    h_vals = np.array([0.0, 5.0, 90.0, 175.0])

    assert hue_in_range(h_vals, 80.0, 100.0).tolist() == [False, False, True, False]
    # A margin pushed the recorded range past the seam.
    assert hue_in_range(h_vals, 170.0, 190.0).tolist() == [True, True, False, True]
    assert hue_in_range(h_vals, -10.0, 190.0).all()


def test_weighted_score_drops_absent_terms_rather_than_scoring_them_perfect() -> None:
    """The contract is that an absent term does not participate.

    Note this is *not* "removing a term can never raise the score": with the
    remaining terms at 1.0 the renormalized score is 1.0. What it rules out is
    an absent term contributing a perfect 1.0 *and* its full weight, which is
    what made a color with incomplete statistics outscore a complete one.
    """
    from picture_tool.color.strategies.base import weighted_score

    weights = {"a": 0.5, "b": 0.3, "c": 0.2}

    # Every term present: a plain weighted sum, unchanged from before.
    assert weighted_score({"a": 1.0, "b": 0.0, "c": 0.0}, weights) == pytest.approx(0.5)
    # "c" absent: the score is the other two renormalized, not 0.5 + 1.0 * 0.2.
    assert weighted_score({"a": 1.0, "b": 0.0, "c": None}, weights) == pytest.approx(
        0.5 / 0.8
    )
    assert weighted_score({"a": None, "b": None, "c": None}, weights) == 0.0


@pytest.mark.parametrize("strategy_path", ["black.BlackStrategy", "yellow.YellowStrategy"])
def test_fast_detect_survives_an_empty_region(strategy_path: str) -> None:
    """An empty crop divided by a zero-sized mask and raised."""
    import importlib

    module_name, class_name = strategy_path.split(".")
    module = importlib.import_module(f"picture_tool.color.strategies.{module_name}")
    strategy = getattr(module, class_name)()

    empty = np.zeros((0, 5, 3))
    assert strategy.fast_detect(empty, empty, _range("Black")) == (False, 0.0)
