"""The training-pipeline color gate must match the shared conformance fixture.

The fixture is shared, byte for byte, with the inference repository's runtime
color checker. The two are separate code bases judging the same product, so a
decision rule that moves on one side and not the other lets a model clear this
gate and behave differently on the line -- and neither repository can notice
that on its own. Each side pins itself to the fixture here; the workspace CI
checks that the two copies of the fixture still match.

Regenerate with ``scripts/generate_color_conformance.py`` in the workspace
repository, which re-runs both implementations and refuses to hide a new
disagreement.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from picture_tool.color import color_verifier

FIXTURE = Path(__file__).parent / "fixtures" / "color_conformance.json"


def _payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


_PAYLOAD = _payload()


@pytest.fixture(scope="module")
def color_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The fixture embeds the model, so this suite needs nothing beside it.

    Referencing the model by path in the sibling inference checkout would make
    this whole suite skip in this repository's own CI, and a conformance guard
    that only runs somewhere else is half a guard.
    """
    path = tmp_path_factory.mktemp("conformance") / "color_stats.json"
    path.write_text(json.dumps(_PAYLOAD["color_model"]), encoding="utf-8")
    return path


def _render(spec: dict) -> np.ndarray:
    size = int(spec.get("size", 96))
    hsv = np.zeros((size, size, 3), np.uint8)
    bands = spec["bands"]
    edges = np.linspace(0, size, len(bands) + 1).astype(int)
    for index, band in enumerate(bands):
        low, high = edges[index], edges[index + 1]
        hsv[low:high, :, 0] = band[0]
        hsv[low:high, :, 1] = band[1]
        hsv[low:high, :, 2] = band[2]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _verdict(model: Path, spec: dict) -> tuple[str, bool]:
    image = _render(spec)
    ranges = color_verifier.load_color_ranges(model)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    ratios, _masks, debug = color_verifier._evaluate_image_improved(
        image, hsv, lab, ranges
    )
    color, confidence = color_verifier._initial_prediction(ratios)
    context = color_verifier.DecisionContext(
        ratios=ratios, debug_info=debug or {}, hsv_img=hsv, lab_img=lab
    )
    color, confidence = color_verifier._apply_color_rules(color, confidence, context)
    threshold = color_verifier._confidence_threshold_for(
        color, _PAYLOAD["default_threshold"]
    )
    return str(color).casefold(), bool(confidence >= threshold)


def _case_ids(cases: list[dict]) -> list[str]:
    return [case["name"] for case in cases]


def test_fixture_is_not_silently_empty() -> None:
    """A conformance suite that stopped covering anything protects nothing."""
    assert len(_PAYLOAD["cases"]) >= 10
    assert _PAYLOAD["color_model"]["summary"]


@pytest.mark.parametrize("case", _PAYLOAD["cases"], ids=_case_ids(_PAYLOAD["cases"]))
def test_gate_matches_the_shared_conformance_fixture(
    case: dict, color_model: Path
) -> None:
    color, confident = _verdict(color_model, case["spec"])

    assert confident is case["confident"], (
        f"{case['name']}: confidence disagrees with the shared fixture"
    )
    if case["confident"]:
        assert color == case["color"], (
            f"{case['name']}: color disagrees with the shared fixture"
        )


@pytest.mark.parametrize(
    "case",
    _PAYLOAD["known_divergences"],
    ids=_case_ids(_PAYLOAD["known_divergences"]),
)
def test_recorded_divergences_still_diverge_the_way_they_are_recorded(
    case: dict, color_model: Path
) -> None:
    """A divergence is pinned, not tolerated.

    If this side moves -- toward the runtime or away from it -- the recorded
    reason is stale and someone has to look at it rather than discover the
    change later on the line.
    """
    color, confident = _verdict(color_model, case["spec"])

    assert confident is case["training_confident"]
    if case["color"] is not None:
        assert color == case["color"]
