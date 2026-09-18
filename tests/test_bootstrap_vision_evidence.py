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
    VerticalStructure,
    VisionLLMEvidence,
    VisionLLMProposer,
    boxes_from_reply,
    calibrate,
    compare_opinions,
    match_by_centre,
    measure_vertical_structure,
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


def test_a_flattened_cy_cannot_veto_a_correct_cx() -> None:
    """The bug that made a 6/6 frame measure 0/6.

    The model puts every box on one horizontal line while the board is
    tilted, so cy is out by up to 0.06 --- three times the match radius ---
    on boxes whose cx is right to 0.005. Under a combined distance none of
    them matched and the frame scored zero.
    """
    # A tilted board, as the hand labels measure it: the ends are collinear
    # with a slope of about 0.17, and where that line sits varies frame to
    # frame. The model draws its own flat line at its own height.
    xs = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
    targets = [box(n, cx, cy=0.620 + 0.17 * (cx - 0.5))
               for n, cx in zip((*NAMES, "Black"), xs)]
    proposed = [box(t.class_name, t.cx + 0.004, cy=0.680) for t in targets]

    matched = match_by_centre(proposed, targets)

    assert [m.class_name for m in matched if m] == [t.class_name for t in targets]
    assert all(np.hypot(m.cx - t.cx, m.cy - t.cy) > 0.02
               for m, t in zip(matched, targets) if m is not None), (
        "the 2-D distance these pass at must still exceed the radius, "
        "or the test is not reproducing the failure it is named for"
    )


def test_cy_may_gate_when_a_station_stacks_objects_but_never_ranks() -> None:
    """Two objects at one cx need cy; it excludes, it does not choose."""
    upper, lower = box("Red", 0.35, cy=0.30), box("Green", 0.35, cy=0.80)

    assert match_by_centre([upper], [lower], vertical_radius=0.10) == [None]
    # Without the gate the same pair matches: cx alone cannot separate them,
    # which is exactly the condition a caller passes vertical_radius for.
    assert match_by_centre([upper], [lower])[0] is upper


def test_cx_decides_the_match_even_when_another_box_is_nearer_overall() -> None:
    near_in_cx = box("Red", 0.352, cy=0.10)
    near_in_cy = box("Green", 0.362, cy=0.59)
    matched = match_by_centre([near_in_cx, near_in_cy], [box("Red", 0.350)])

    assert matched[0] is near_in_cx


def test_a_non_positive_radius_is_refused_rather_than_matching_nothing() -> None:
    with pytest.raises(EvidenceError, match="positive"):
        match_by_centre([box("Red", 0.35)], [box("Red", 0.35)], radius=0.0)
    with pytest.raises(EvidenceError, match="positive"):
        match_by_centre([box("Red", 0.35)], [box("Red", 0.35)],
                        vertical_radius=-1.0)


# -- where the vertical error lives -----------------------------------------


def tilted_frame(intercept: float, slope: float, flat_at: float
                 ) -> list[tuple[Box, Box]]:
    """One frame: collinear truth at some angle, a flat reply across it."""
    xs = (0.35, 0.42, 0.49, 0.56, 0.63)
    return [
        (box("Red", cx, cy=flat_at),
         box("Red", cx, cy=intercept + slope * (cx - 0.5)))
        for cx in xs
    ]


def test_a_shift_that_fits_every_frame_is_reported_as_one() -> None:
    """The control: when a constant is the truth, it must be found."""
    frames = [tilted_frame(0.60 + 0.02, slope=0.0, flat_at=0.60)
              for _ in range(8)]

    measured = measure_vertical_structure(frames)

    assert measured.between_frames.median == pytest.approx(0.02)
    assert measured.between_frames.mad == pytest.approx(0.0)
    assert "constant vertical shift fits" in measured.reading


def test_frames_placed_differently_rule_out_one_station_shift() -> None:
    heights = (0.42, 0.51, 0.58, 0.66, 0.73, 0.79)
    frames = [tilted_frame(h, slope=0.05, flat_at=0.62) for h in heights]

    measured = measure_vertical_structure(frames)

    assert measured.between_frames.mad > VerticalStructure.MEANINGFUL_OFFSET
    assert "no one station-wide cy_shift" in measured.reading


def test_a_tilt_that_reverses_between_frames_is_not_a_station_tilt() -> None:
    """The finding that kept a tilt term out of BoxCalibration."""
    frames = [tilted_frame(0.62, slope=s, flat_at=0.62)
              for s in (-0.18, -0.06, 0.02, 0.07, 0.13, 0.30)]

    measured = measure_vertical_structure(frames)

    assert not measured.frame_slopes.settled
    assert measured.frame_slopes.negative == 2
    assert "no station tilt to store" in measured.reading


def test_a_tilt_every_frame_agrees_on_would_be_worth_storing() -> None:
    """The opposite verdict has to be reachable, or the test above is vacuous."""
    frames = [tilted_frame(0.62, slope=s, flat_at=0.62)
              for s in (0.16, 0.17, 0.17, 0.18, 0.18, 0.19)]

    measured = measure_vertical_structure(frames)

    assert measured.frame_slopes.settled
    assert "would earn its place" in measured.reading


def test_flattening_the_frames_would_hide_the_distinction() -> None:
    """Why the pairs stay grouped: one bag of pairs cannot tell these apart."""
    spread_out = [tilted_frame(h, slope=0.0, flat_at=0.62)
                  for h in (0.50, 0.56, 0.62, 0.68, 0.74)]
    all_alike = [tilted_frame(0.62, slope=0.0, flat_at=0.62) for _ in range(5)]

    assert measure_vertical_structure(spread_out).between_frames.mad > 0.05
    assert measure_vertical_structure(all_alike).between_frames.mad == 0.0
    # Pooled, the two are the same multiset of dy values in a different order.
    assert (sorted(round(r.cy - p.cy, 6) for f in spread_out for p, r in f)
            != sorted(round(r.cy - p.cy, 6) for f in all_alike for p, r in f))


def test_an_empty_measurement_says_so_rather_than_dividing_by_zero() -> None:
    measured = measure_vertical_structure([])

    assert measured.between_frames.n == 0
    assert not measured.frame_slopes.settled
    assert "nothing measured" in measured.reading


def test_the_slope_ignores_box_pairs_too_close_in_cx_to_carry_one() -> None:
    """Dividing by a near-zero cx gap amplifies noise into a gradient."""
    stacked = [(box("Red", 0.500, cy=0.60), box("Red", 0.500, cy=0.70)),
               (box("Red", 0.5001, cy=0.60), box("Red", 0.5001, cy=0.30)),
               (box("Red", 0.5002, cy=0.60), box("Red", 0.5002, cy=0.90)),
               (box("Red", 0.5003, cy=0.60), box("Red", 0.5003, cy=0.20))]

    assert measure_vertical_structure([stacked]).frame_slopes.n == 0


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


# -- the vision model inside the decision ------------------------------------

from picture_tool.bootstrap.auto_label import (  # noqa: E402
    AUTO_ACCEPT,
    NEEDS_REVIEW,
    R_INDEPENDENT_CONSENSUS,
    R_VISION_DISAGREEMENT,
    decide,
)
from picture_tool.bootstrap.sample_quality import QUALITY_PASS, SampleQuality  # noqa: E402


def _decide(box_class: str, colour: str, vision: str | None):
    """One box, three sources, and whatever the bootstrapper makes of it."""
    boxes = [box(box_class, 0.35)]
    sources = {
        "detector": [BoxOpinion(source="detector", class_name=box_class, confidence=0.9)],
        "colour": [BoxOpinion(source="colour", class_name=colour, confidence=0.9,
                              scores={colour: 0.8, "Other": 0.1})],
    }
    if vision is not None:
        sources[SOURCE_NAME] = [
            BoxOpinion(source=SOURCE_NAME, class_name=vision, confidence=0.9)
        ]
    return decide(
        sample_id="s", image_path="s.jpg", boxes=boxes,
        opinions_by_source=sources,
        quality=SampleQuality(sample_id="s", image_path="s.jpg", status=QUALITY_PASS),
        profile=ProductProfile(
            product="Cable1", area="A",
            class_schema=ClassSchema(names=NAMES, source="t"),
            expected_counts={box_class: 1},
            confusion_pairs=(("Red", "Orange"),),
        ),
    )


def test_two_independent_sources_agreeing_is_reported_as_a_proposed_answer() -> None:
    """The red_orange_critical case: both non-detector sources say Orange."""
    result = _decide("Red", colour="Orange", vision="Orange")

    assert result.decision == NEEDS_REVIEW
    assert R_INDEPENDENT_CONSENSUS in result.reasons
    assert result.boxes[0].consensus_class == "Orange"
    assert "detector read Red" in result.detail


def test_one_source_objecting_is_not_reported_as_consensus() -> None:
    result = _decide("Red", colour="Red", vision="Orange")

    assert R_VISION_DISAGREEMENT in result.reasons
    assert R_INDEPENDENT_CONSENSUS not in result.reasons
    assert result.boxes[0].consensus_class == ""


def test_the_vision_model_cannot_clear_an_objection_the_colour_made() -> None:
    """It may only add doubt: AUTO_ACCEPT is reached by having no reasons."""
    without = _decide("Red", colour="Orange", vision=None)
    agreeing = _decide("Red", colour="Orange", vision="Red")

    assert without.decision == NEEDS_REVIEW
    assert agreeing.decision == NEEDS_REVIEW


def test_all_three_agreeing_still_reaches_auto_accept() -> None:
    result = _decide("Red", colour="Red", vision="Red")

    assert result.decision == AUTO_ACCEPT


def test_an_undecided_vision_opinion_changes_nothing() -> None:
    """Silence from a source is not a vote against."""
    result = _decide("Red", colour="Red", vision="")

    assert result.decision == AUTO_ACCEPT
