"""The vision model as evidence, and the calibration that makes it proposable.

Two load-bearing tests. One is that a proposer without a calibration refuses:
uncorrected boxes measured mean IoU 0.36, which is accurate enough to look
right in every count and class total and wrong enough to teach a detector the
wrong extents, and that failure would surface much later with nothing pointing
back here. The other is that a box with no reply near it gets no opinion --- an
evidence source that guesses stops being independent of what it is checking.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from picture_tool.autotrain.class_schema import ClassSchema
from picture_tool.bootstrap.evidence import Box, BoxOpinion, EvidenceError
from picture_tool.bootstrap.profile import ProductProfile
from picture_tool.bootstrap.vision_client import VisionClientError, VisionReply
from picture_tool.bootstrap.vision_evidence import (
    SOURCE_NAME,
    BoxCalibration,
    CalibrationError,
    VisionLLMEvidence,
    VisionLLMProposer,
    boxes_from_reply,
    calibrate,
    compare_opinions,
    match_by_centre,
)

NAMES = ("Black", "Green", "Orange", "Red", "Yellow")


def profile() -> ProductProfile:
    return ProductProfile(
        product="Cable1",
        area="A",
        class_schema=ClassSchema(names=NAMES, source="test"),
        expected_counts={"Black": 2, "Green": 1, "Orange": 1, "Red": 1, "Yellow": 1},
    )


def box(name: str, cx: float, cy: float = 0.59, w: float = 0.045, h: float = 0.07) -> Box:
    return Box(class_name=name, cx=cx, cy=cy, width=w, height=h, confidence=0.9)


class _FakeClient:
    """Answers with a canned reply. The network is never touched in tests."""

    def __init__(self, boxes: list[dict] | None = None, error: str = "") -> None:
        self.boxes = boxes or []
        self.error = error
        self.calls = 0

    def ask(self, request: Any, *, require_verdict: bool = True) -> VisionReply:
        self.calls += 1
        if self.error:
            raise VisionClientError(self.error)
        return VisionReply(fields={"boxes": self.boxes})


def reply(name: str, cx: float, cy: float = 0.59, w: float = 0.03, h: float = 0.045) -> dict:
    return {"class": name, "bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]}


# -- the calibration, and the refusal to work without one -------------------


def test_a_proposer_without_a_calibration_refuses() -> None:
    proposer = VisionLLMProposer(_FakeClient([reply("Red", 0.35)]))

    with pytest.raises(CalibrationError, match="0.36"):
        proposer.propose("unused.jpg", profile())


def test_a_calibration_from_another_station_is_refused() -> None:
    """The bias comes from this station's framing; it is not transferable."""
    other = BoxCalibration(product="Cable2", area="B", width_scale=1.38, height_scale=1.6)
    proposer = VisionLLMProposer(_FakeClient(), calibration=other)

    with pytest.raises(CalibrationError, match="Cable2"):
        proposer.propose("unused.jpg", profile())


def test_uncalibrated_is_possible_but_has_to_be_asked_for() -> None:
    proposer = VisionLLMProposer(_FakeClient(), allow_uncalibrated=True)

    assert proposer._checked_calibration(profile()) is None


def test_calibration_measures_the_bias_it_was_shown() -> None:
    """Boxes two-thirds the right size, centred correctly, as measured."""
    pairs = [
        (box("Red", 0.35, w=0.030, h=0.045), box("Red", 0.35, w=0.045, h=0.072)),
        (box("Green", 0.40, w=0.030, h=0.045), box("Green", 0.40, w=0.045, h=0.072)),
        (box("Orange", 0.45, w=0.030, h=0.045), box("Orange", 0.45, w=0.045, h=0.072)),
        (box("Yellow", 0.50, w=0.030, h=0.045), box("Yellow", 0.50, w=0.045, h=0.072)),
        (box("Black", 0.55, w=0.030, h=0.045), box("Black", 0.55, w=0.045, h=0.072)),
        (box("Black", 0.60, w=0.030, h=0.045), box("Black", 0.60, w=0.045, h=0.072)),
    ]
    cal = calibrate(pairs, product="Cable1", area="A", sample_images=1)

    assert cal.width_scale == pytest.approx(1.5)
    assert cal.height_scale == pytest.approx(1.6)
    assert cal.cx_shift == pytest.approx(0.0)
    assert cal.sample_boxes == 6


def test_calibration_uses_medians_so_one_stray_box_cannot_set_it() -> None:
    """A box on the wrong object would drag a mean into fitting nothing."""
    good = [
        (box("Red", 0.35, w=0.030), box("Red", 0.35, w=0.045)) for _ in range(6)
    ]
    stray = (box("Green", 0.40, w=0.001), box("Green", 0.40, w=0.045))
    cal = calibrate([*good, stray], product="Cable1", area="A")

    assert cal.width_scale == pytest.approx(1.5)


def test_too_few_boxes_is_refused_rather_than_fitted() -> None:
    pairs = [(box("Red", 0.35), box("Red", 0.35))] * 3

    with pytest.raises(CalibrationError, match="too few"):
        calibrate(pairs, product="Cable1", area="A")


def test_a_calibration_survives_a_round_trip(tmp_path) -> None:
    cal = BoxCalibration(
        product="Cable1", area="A", width_scale=1.38, height_scale=1.597,
        cy_shift=0.0188, sample_boxes=18, sample_images=3,
        derived_from="Result/20260915 PASS frames",
    )
    path = cal.write(tmp_path / "calibration.json")

    back = BoxCalibration.read(path)

    assert back == cal
    assert json.loads(path.read_text(encoding="utf-8"))["sample_images"] == 3


def test_a_zero_scale_is_refused_at_construction() -> None:
    with pytest.raises(CalibrationError, match="positive"):
        BoxCalibration(product="Cable1", area="A", width_scale=0.0, height_scale=1.0)


def test_applying_a_calibration_grows_the_box_around_its_centre() -> None:
    cal = BoxCalibration(product="Cable1", area="A", width_scale=1.5, height_scale=1.6)
    out = cal.apply(box("Red", 0.35, cy=0.59, w=0.030, h=0.045))

    assert out.cx == pytest.approx(0.35)
    assert out.width == pytest.approx(0.045)
    assert out.height == pytest.approx(0.072)
    assert out.class_name == "Red"


# -- evidence: an opinion per box, or none ----------------------------------


def test_one_request_answers_every_box() -> None:
    """Six calls would cost six times as much to answer the same question."""
    client = _FakeClient([reply(n, cx) for n, cx in
                          zip(NAMES, (0.35, 0.40, 0.45, 0.50, 0.55))])
    source = VisionLLMEvidence(client)
    boxes = [box(n, cx) for n, cx in zip(NAMES, (0.35, 0.40, 0.45, 0.50, 0.55))]

    opinions = source.read(np.zeros((640, 640, 3), dtype=np.uint8), boxes, profile())

    assert client.calls == 1
    assert len(opinions) == len(boxes)
    assert [o.class_name for o in opinions] == list(NAMES)
    assert all(o.source == SOURCE_NAME for o in opinions)


def test_a_box_with_nothing_reported_near_it_gets_no_opinion() -> None:
    """Silence keeps this source independent of the one it is checking."""
    client = _FakeClient([reply("Red", 0.35)])
    source = VisionLLMEvidence(client)

    opinions = source.read(
        np.zeros((640, 640, 3), dtype=np.uint8), [box("Red", 0.90)], profile()
    )

    assert opinions[0].class_name == ""
    assert opinions[0].confidence == 0.0
    assert "no object reported" in opinions[0].detail


def test_the_model_may_disagree_with_the_box_it_was_given() -> None:
    """The point of a third source: it is not told what the detector said."""
    client = _FakeClient([reply("Orange", 0.35)])
    source = VisionLLMEvidence(client)

    opinions = source.read(
        np.zeros((640, 640, 3), dtype=np.uint8), [box("Red", 0.35)], profile()
    )

    assert opinions[0].class_name == "Orange"


def test_no_boxes_means_no_request() -> None:
    client = _FakeClient()
    source = VisionLLMEvidence(client)

    assert source.read(np.zeros((640, 640, 3), dtype=np.uint8), [], profile()) == []
    assert client.calls == 0


def test_an_unreachable_model_is_an_evidence_error() -> None:
    source = VisionLLMEvidence(_FakeClient(error="endpoint refused"))

    with pytest.raises(EvidenceError, match="endpoint refused"):
        source.read(np.zeros((640, 640, 3), dtype=np.uint8), [box("Red", 0.35)], profile())


# -- reading the reply ------------------------------------------------------


def test_a_class_outside_the_contract_is_dropped_not_renumbered() -> None:
    boxes = boxes_from_reply({"boxes": [reply("Purple", 0.35), reply("Red", 0.40)]},
                             profile())

    assert [b.class_name for b in boxes] == ["Red"]


def test_a_reversed_or_empty_box_is_dropped() -> None:
    payload = {"boxes": [
        {"class": "Red", "bbox": [0.5, 0.5, 0.4, 0.4]},   # reversed, still valid once sorted
        {"class": "Green", "bbox": [0.3, 0.3, 0.3, 0.3]},  # zero area
        {"class": "Black", "bbox": [0.1, 0.1]},            # malformed
    ]}
    boxes = boxes_from_reply(payload, profile())

    assert [b.class_name for b in boxes] == ["Red"]
    assert boxes[0].width == pytest.approx(0.1)


def test_a_reply_without_boxes_yields_nothing() -> None:
    assert boxes_from_reply({}, profile()) == []
    assert boxes_from_reply({"boxes": "not a list"}, profile()) == []


# -- matching ---------------------------------------------------------------


def test_matching_prefers_the_nearest_centre_not_the_best_overlap() -> None:
    """Sizes are the one thing known to be wrong, so IoU must not rank them."""
    proposed = [box("Red", 0.350, w=0.005), box("Green", 0.365, w=0.060)]
    matched = match_by_centre(proposed, [box("Red", 0.351)])

    assert matched[0] is not None
    assert matched[0].class_name == "Red"


def test_a_distant_box_does_not_match_at_all() -> None:
    assert match_by_centre([box("Red", 0.35)], [box("Red", 0.80)]) == [None]


# -- comparing two sources --------------------------------------------------


def test_comparison_reports_where_two_sources_part_company() -> None:
    left = [BoxOpinion(source="detector", class_name="Red", confidence=0.6),
            BoxOpinion(source="detector", class_name="Black", confidence=0.9)]
    right = [BoxOpinion(source=SOURCE_NAME, class_name="Orange", confidence=0.9),
             BoxOpinion(source=SOURCE_NAME, class_name="Black", confidence=0.9)]

    report = compare_opinions(left, right)

    assert report.compared == 2
    assert report.agreed == 1
    assert report.disagreements == (("box_1", "Red", "Orange"),)
    assert report.rate == pytest.approx(0.5)


def test_an_undecided_opinion_is_not_counted_as_agreement() -> None:
    left = [BoxOpinion(source="detector", class_name="Red", confidence=0.6)]
    right = [BoxOpinion(source=SOURCE_NAME, class_name="", confidence=0.0)]

    report = compare_opinions(left, right)

    assert report.compared == 0
    assert report.undecided == 1
    assert report.rate == 0.0


def test_mismatched_opinion_lists_are_refused() -> None:
    with pytest.raises(EvidenceError, match="line up"):
        compare_opinions([BoxOpinion(source="a", class_name="Red", confidence=1.0)], [])
