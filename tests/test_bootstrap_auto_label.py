"""The decision layer, and the rule it exists to enforce.

The load-bearing test is the one that refuses to accept a box the detector
vouched for alone. Everything else here is a way that rule, or the refusals
around it, could be quietly weakened.
"""

from __future__ import annotations

import json

import pytest
import yaml

from picture_tool.autotrain.class_schema import normalize_class_names
from picture_tool.bootstrap.auto_label import (
    AUTO_ACCEPT,
    NEEDS_REVIEW,
    R_ALL_CHECKS_AGREE,
    R_COLOUR_AMBIGUOUS,
    R_COLOUR_DISAGREEMENT,
    R_COUNT_MISMATCH,
    R_CRITICAL_DISAGREEMENT,
    R_EXACT_DUPLICATE,
    R_IMPOSSIBLE_GEOMETRY,
    R_NO_DETECTIONS,
    R_NO_INDEPENDENT_EVIDENCE,
    R_QUALITY_FAIL,
    R_TOTAL_MISMATCH,
    REJECT,
    decide,
    summarise,
)
from picture_tool.bootstrap.evidence import Box, BoxOpinion
from picture_tool.bootstrap.export import (
    ExportError,
    export_accepted,
    write_reports,
)
from picture_tool.bootstrap.profile import ProductProfile, expected_counts_from_items
from picture_tool.bootstrap.sample_quality import (
    QUALITY_FAIL,
    QUALITY_PASS,
    QUALITY_SUSPECT,
    REASON_EXACT_DUPLICATE,
    REASON_NEAR_DUPLICATE,
    REASON_SEVERE_BLUR,
    REASON_UNREADABLE,
    SampleQuality,
)

SCHEMA = normalize_class_names(
    ["Black", "Green", "Orange", "Red", "Yellow"], source="test"
)
#: The real Cable1/A multiset. Black twice because the fixture has two.
EXPECTED_ITEMS = ["Red", "Green", "Orange", "Yellow", "Black", "Black"]

PROFILE = ProductProfile(
    product="Cable1",
    area="A",
    class_schema=SCHEMA,
    expected_counts=expected_counts_from_items(EXPECTED_ITEMS),
    confusion_pairs=(("Red", "Orange"),),
)

LAYOUT = [
    ("Black", 0.20),
    ("Black", 0.30),
    ("Green", 0.40),
    ("Orange", 0.50),
    ("Red", 0.60),
    ("Yellow", 0.70),
]


def _boxes(overrides=None, confidence=0.9):
    overrides = overrides or {}
    boxes = []
    for index, (name, y) in enumerate(LAYOUT):
        boxes.append(
            Box(
                class_name=overrides.get(index, {}).get("class_name", name),
                cx=0.5,
                cy=y,
                width=0.08,
                height=0.08,
                confidence=overrides.get(index, {}).get("confidence", confidence),
                source="detector",
            )
        )
    return boxes


def _opinions(boxes, colour_overrides=None, colour_scores=None, with_colour=True):
    colour_overrides = colour_overrides or {}
    detector = [
        BoxOpinion(source="detector", class_name=b.class_name, confidence=b.confidence)
        for b in boxes
    ]
    if not with_colour:
        return {"detector": detector}
    colour = []
    for index, box in enumerate(boxes):
        name = colour_overrides.get(index, box.class_name)
        scores = colour_scores.get(index) if colour_scores else None
        if scores is None:
            scores = {n: 0.05 for n in SCHEMA.names}
            scores[name] = 0.9
        colour.append(
            BoxOpinion(
                source="colour",
                class_name=name,
                confidence=scores[name],
                scores=scores,
            )
        )
    return {"detector": detector, "colour": colour}


def _quality(status=QUALITY_PASS, reasons=(), duplicate_of=""):
    return SampleQuality(
        sample_id="s0",
        image_path="D:/prod/s0.jpg",
        status=status,
        reasons=tuple(reasons),
        sha256="a" * 64,
        perceptual_hash="ph",
        width=1920,
        height=1080,
        brightness=70.0,
        saturation=40.0,
        blur_score=80.0,
        duplicate_of=duplicate_of,
    )


def _decide(boxes=None, opinions=None, quality=None, sample_id="s0"):
    boxes = _boxes() if boxes is None else boxes
    return decide(
        sample_id=sample_id,
        image_path=f"D:/prod/{sample_id}.jpg",
        boxes=boxes,
        opinions_by_source=_opinions(boxes) if opinions is None else opinions,
        quality=quality or _quality(),
        profile=PROFILE,
    )


# ---------------------------------------------------------------------------
# Acceptance


def test_a_fully_corroborated_sample_is_accepted():
    result = _decide()

    assert result.decision == AUTO_ACCEPT
    assert result.reasons == (R_ALL_CHECKS_AGREE,)
    assert result.class_counts == {
        "Black": 2, "Green": 1, "Orange": 1, "Red": 1, "Yellow": 1
    }


def test_the_detector_may_not_accept_its_own_output():
    """The rule this whole module exists for.

    Perfect boxes, perfect confidence, counts matching the station -- and no
    second source. Accepting this would make the bootstrapper a machine for
    copying the champion's mistakes into its successor's training data.
    """
    boxes = _boxes(confidence=0.99)

    result = _decide(boxes=boxes, opinions=_opinions(boxes, with_colour=False))

    assert result.decision == NEEDS_REVIEW
    assert R_NO_INDEPENDENT_EVIDENCE in result.reasons


def test_high_confidence_does_not_override_a_disagreement():
    boxes = _boxes(confidence=0.99)
    opinions = _opinions(boxes, colour_overrides={2: "Yellow"})

    result = _decide(boxes=boxes, opinions=opinions)

    assert result.decision == NEEDS_REVIEW
    assert R_COLOUR_DISAGREEMENT in result.reasons


# ---------------------------------------------------------------------------
# Review


def test_a_red_orange_disagreement_is_flagged_as_critical():
    """The pair the line actually confuses gets its own reason."""
    boxes = _boxes()
    opinions = _opinions(boxes, colour_overrides={4: "Orange"})

    result = _decide(boxes=boxes, opinions=opinions)

    assert result.decision == NEEDS_REVIEW
    assert R_CRITICAL_DISAGREEMENT in result.reasons
    assert "detector Red vs colour Orange" in result.detail
    assert any(box.critical for box in result.boxes)


def test_a_non_critical_disagreement_is_not_called_critical():
    boxes = _boxes()
    opinions = _opinions(boxes, colour_overrides={2: "Yellow"})

    result = _decide(boxes=boxes, opinions=opinions)

    assert R_CRITICAL_DISAGREEMENT not in result.reasons
    assert R_COLOUR_DISAGREEMENT in result.reasons


def test_colour_that_cannot_decide_is_not_agreement():
    """Red 0.51 / Orange 0.49 corroborates nothing; its argmax is noise."""
    boxes = _boxes()
    scores = {n: 0.05 for n in SCHEMA.names}
    scores.update({"Red": 0.51, "Orange": 0.49})
    opinions = _opinions(boxes, colour_scores={4: scores})

    result = _decide(boxes=boxes, opinions=opinions)

    assert result.decision == NEEDS_REVIEW
    assert R_COLOUR_AMBIGUOUS in result.reasons


def test_a_missing_object_is_reported_per_class():
    boxes = _boxes()[:5]

    result = _decide(boxes=boxes, opinions=_opinions(boxes))

    assert result.decision == NEEDS_REVIEW
    assert R_TOTAL_MISMATCH in result.reasons
    assert R_COUNT_MISMATCH in result.reasons
    assert "Yellow 0/1" in result.detail


def test_an_impossible_class_count_goes_to_review_not_the_dataset():
    """Three Blacks on a two-Black fixture. Policy: review, never accept."""
    boxes = _boxes(overrides={2: {"class_name": "Black"}})
    opinions = _opinions(boxes)

    result = _decide(boxes=boxes, opinions=opinions)

    assert result.decision == NEEDS_REVIEW
    assert R_COUNT_MISMATCH in result.reasons
    assert "Black 3/2" in result.detail


def test_a_near_duplicate_is_reviewed_not_discarded():
    result = _decide(
        quality=_quality(QUALITY_SUSPECT, (REASON_NEAR_DUPLICATE,), duplicate_of="s1")
    )

    assert result.decision == NEEDS_REVIEW
    assert "s1" in result.detail


# ---------------------------------------------------------------------------
# Refusal


def test_an_unreadable_image_is_rejected():
    result = _decide(quality=_quality(QUALITY_FAIL, (REASON_UNREADABLE,)))

    assert result.decision == REJECT
    assert R_QUALITY_FAIL in result.reasons


def test_severe_blur_is_rejected():
    result = _decide(quality=_quality(QUALITY_FAIL, (REASON_SEVERE_BLUR,)))

    assert result.decision == REJECT
    assert REASON_SEVERE_BLUR in result.reasons


def test_an_exact_duplicate_is_rejected_rather_than_weighted_twice():
    result = _decide(
        quality=_quality(QUALITY_PASS, (REASON_EXACT_DUPLICATE,), duplicate_of="s1")
    )

    assert result.decision == REJECT
    assert R_EXACT_DUPLICATE in result.reasons


def test_an_image_with_no_detections_is_rejected():
    result = _decide(boxes=[], opinions={})

    assert result.decision == REJECT
    assert R_NO_DETECTIONS in result.reasons


def test_degenerate_geometry_is_rejected():
    boxes = _boxes()
    boxes[1] = Box("Black", 0.5, 0.3, 0.0, 0.08, 0.9, "detector")

    result = _decide(boxes=boxes, opinions=_opinions(boxes))

    assert result.decision == REJECT
    assert R_IMPOSSIBLE_GEOMETRY in result.reasons


# ---------------------------------------------------------------------------
# Export


def _accepted_and_others(tmp_path):
    image = tmp_path / "prod" / "s0.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"an-image")
    boxes = _boxes()
    good = decide(
        sample_id="s0",
        image_path=str(image),
        boxes=boxes,
        opinions_by_source=_opinions(boxes),
        quality=_quality(),
        profile=PROFILE,
    )
    review = _decide(sample_id="s1", boxes=boxes, opinions=_opinions(boxes, {4: "Orange"}))
    rejected = _decide(sample_id="s2", quality=_quality(QUALITY_FAIL, (REASON_UNREADABLE,)))
    return [good, review, rejected]


def test_only_accepted_samples_reach_the_dataset(tmp_path):
    decisions = _accepted_and_others(tmp_path)

    result = export_accepted(decisions, tmp_path / "ds", profile=PROFILE)

    stems = sorted(p.stem for p in (result.root / "images" / "train").iterdir())
    assert stems == ["s0"]
    assert sorted(p.stem for p in (result.root / "labels" / "train").iterdir()) == ["s0"]
    assert result.images == 1
    assert result.instances == 6


def test_class_ids_come_from_the_contract_not_the_detector(tmp_path):
    decisions = _accepted_and_others(tmp_path)

    result = export_accepted(decisions, tmp_path / "ds", profile=PROFILE)

    lines = (result.root / "labels" / "train" / "s0.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    ids = sorted(int(line.split()[0]) for line in lines)
    assert ids == [0, 0, 1, 2, 3, 4]


def test_the_exported_descriptor_is_one_ultralytics_accepts(tmp_path):
    decisions = _accepted_and_others(tmp_path)

    result = export_accepted(decisions, tmp_path / "ds", profile=PROFILE)

    payload = yaml.safe_load(result.data_yaml.read_text(encoding="utf-8"))
    assert payload["train"] and payload["val"]
    assert payload["names"] == dict(enumerate(SCHEMA.names))


def test_every_exported_label_carries_its_provenance(tmp_path):
    decisions = _accepted_and_others(tmp_path)

    result = export_accepted(decisions, tmp_path / "ds", profile=PROFILE)

    payload = json.loads(
        (result.root / "label_provenance.json").read_text(encoding="utf-8")
    )
    record = payload["s0"]
    assert record["decision"] == AUTO_ACCEPT
    assert record["reference_source"] == "existing_validated_color_stats"
    assert record["class_schema_hash"] == SCHEMA.schema_hash
    assert record["produced_at"]
    assert record["evidence"] and record["evidence"][0]["opinions"]


def test_exporting_nothing_is_refused_rather_than_writing_an_empty_set(tmp_path):
    rejected = [_decide(quality=_quality(QUALITY_FAIL, (REASON_UNREADABLE,)))]

    with pytest.raises(ExportError, match="Nothing was accepted"):
        export_accepted(rejected, tmp_path / "ds", profile=PROFILE)


def test_the_reports_refuse_to_claim_an_accuracy(tmp_path):
    decisions = _accepted_and_others(tmp_path)

    written = write_reports(decisions, tmp_path / "report", profile=PROFILE)

    payload = json.loads(written["manifest"].read_text(encoding="utf-8"))
    assert payload["accuracy"]["measured"] is False
    assert "memorisation" in payload["accuracy"]["reason"]


def test_the_summary_counts_the_critical_pair_by_direction():
    boxes = _boxes()
    decisions = [
        _decide(boxes=boxes, opinions=_opinions(boxes, {4: "Orange"})),
        _decide(boxes=boxes, opinions=_opinions(boxes, {3: "Red"})),
    ]

    stats = summarise(decisions)

    assert stats["critical_pair_disagreements"] == {
        "detector_Orange__colour_Red": 1,
        "detector_Red__colour_Orange": 1,
    }
