"""Metrics comparison and the deterministic promotion decision.

These cover the two outcomes the request called out by name: a regressed
model must be REJECTED, an improved one must be marked PROMOTION_CANDIDATE
and nothing must be deployed either way.
"""

from __future__ import annotations

import pytest

from picture_tool.autotrain import golden
from picture_tool.autotrain.config import PromotionConfig
from picture_tool.autotrain.metrics import (
    ClassMetrics,
    ModelMetrics,
    compare,
    extract_metrics,
)
from picture_tool.autotrain.promotion import (
    PROMOTION_CANDIDATE,
    REJECTED,
    decide,
    describe,
)


def _metrics(
    *,
    map50=0.80,
    precision=0.85,
    recall=0.90,
    map50_95=0.55,
    per_class=None,
    false_negatives=0,
    false_positives=0,
):
    classes = per_class or {"Red": 0.90, "Orange": 0.88, "Yellow": 0.86}
    return ModelMetrics(
        overall={
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
        },
        per_class={
            name: ClassMetrics(name=name, precision=0.9, recall=value, map50=value)
            for name, value in classes.items()
        },
        class_names=tuple(classes),
        false_negatives=false_negatives,
        false_positives=false_positives,
    )


def _golden(status=golden.OK):
    return golden.GoldenStatus(status=status, detail="" if status == golden.OK else "x")


def _policy(**overrides):
    defaults = {
        "min_map50_delta": 0.0,
        "max_overall_regression": 0.02,
        "critical_classes": (),
        "max_critical_class_recall_drop": 0.005,
        "max_false_negatives": 0,
        "require_golden_pass": True,
    }
    defaults.update(overrides)
    return PromotionConfig(**defaults)


# ---------------------------------------------------------------------------
# The two headline outcomes


def test_an_improved_model_is_marked_a_promotion_candidate():
    champion = _metrics(map50=0.80)
    challenger = _metrics(map50=0.812, per_class={"Red": 0.901, "Orange": 0.917, "Yellow": 0.858})

    decision = decide(compare(champion, challenger), _golden(), _policy())

    assert decision.decision == PROMOTION_CANDIDATE
    assert decision.is_candidate is True
    assert decision.reasons == ()


def test_a_regressed_model_is_rejected():
    champion = _metrics(map50=0.80)
    challenger = _metrics(map50=0.74)

    decision = decide(compare(champion, challenger), _golden(), _policy())

    assert decision.decision == REJECTED
    assert any("mAP50" in reason for reason in decision.reasons)


def test_a_promotion_candidate_is_a_recommendation_not_a_deployment():
    """The strongest verdict available is still just advice for a human."""
    decision = decide(
        compare(_metrics(map50=0.80), _metrics(map50=0.90)), _golden(), _policy()
    )

    assert decision.decision == PROMOTION_CANDIDATE
    assert not hasattr(decision, "deploy")
    assert set(decision.to_dict()) == {
        "decision",
        "reasons",
        "passed_checks",
        "golden_status",
    }


# ---------------------------------------------------------------------------
# Individual rules


def test_an_overall_metric_regression_blocks_promotion():
    champion = _metrics(map50=0.80, recall=0.90)
    challenger = _metrics(map50=0.81, recall=0.85)  # recall -0.05

    decision = decide(compare(champion, challenger), _golden(), _policy())

    assert decision.decision == REJECTED
    assert any("regressed beyond" in reason for reason in decision.reasons)


def test_a_regression_within_tolerance_is_allowed():
    champion = _metrics(map50=0.80, recall=0.90)
    challenger = _metrics(map50=0.81, recall=0.895)  # -0.005, inside 0.02

    decision = decide(compare(champion, challenger), _golden(), _policy())

    assert decision.decision == PROMOTION_CANDIDATE


def test_a_critical_class_recall_drop_blocks_promotion():
    champion = _metrics(map50=0.80, per_class={"Red": 0.90, "Orange": 0.88})
    challenger = _metrics(map50=0.85, per_class={"Red": 0.80, "Orange": 0.95})

    decision = decide(
        compare(champion, challenger),
        _golden(),
        _policy(critical_classes=("Red",)),
    )

    assert decision.decision == REJECTED
    assert any("Red" in reason for reason in decision.reasons)


def test_a_non_critical_class_may_drop():
    champion = _metrics(map50=0.80, per_class={"Red": 0.90, "Yellow": 0.88})
    challenger = _metrics(map50=0.85, per_class={"Red": 0.91, "Yellow": 0.70})

    decision = decide(
        compare(champion, challenger),
        _golden(),
        _policy(critical_classes=("Red",)),
    )

    assert decision.decision == PROMOTION_CANDIDATE


def test_a_critical_class_the_challenger_dropped_blocks_promotion():
    """A class that vanished cannot be shown to have held its recall."""
    champion = _metrics(map50=0.80, per_class={"Red": 0.90, "Orange": 0.88})
    challenger = _metrics(map50=0.85, per_class={"Orange": 0.95})

    decision = decide(
        compare(champion, challenger),
        _golden(),
        _policy(critical_classes=("Red",)),
    )

    assert decision.decision == REJECTED
    assert any("could not be compared" in reason for reason in decision.reasons)


def test_more_false_negatives_than_the_champion_blocks_promotion():
    champion = _metrics(map50=0.80, false_negatives=2)
    challenger = _metrics(map50=0.85, false_negatives=5)

    decision = decide(compare(champion, challenger), _golden(), _policy())

    assert decision.decision == REJECTED
    assert any("false negatives rose" in reason for reason in decision.reasons)


def test_holding_false_negatives_level_is_not_a_regression():
    champion = _metrics(map50=0.80, false_negatives=2)
    challenger = _metrics(map50=0.85, false_negatives=2)

    decision = decide(compare(champion, challenger), _golden(), _policy())

    assert decision.decision == PROMOTION_CANDIDATE


def test_fewer_false_negatives_is_an_improvement():
    champion = _metrics(map50=0.80, false_negatives=5)
    challenger = _metrics(map50=0.85, false_negatives=1)

    decision = decide(compare(champion, challenger), _golden(), _policy())

    assert decision.decision == PROMOTION_CANDIDATE


# ---------------------------------------------------------------------------
# Fail-closed


@pytest.mark.parametrize(
    "status",
    [golden.NOT_CONFIGURED, golden.MISSING, golden.MISMATCH, golden.CONTAMINATED, golden.INVALID],
)
def test_any_non_passing_golden_status_blocks_promotion(status):
    decision = decide(
        compare(_metrics(map50=0.80), _metrics(map50=0.95)), _golden(status), _policy()
    )

    assert decision.decision == REJECTED
    assert any("golden dataset status" in reason for reason in decision.reasons)


def test_the_golden_requirement_can_be_switched_off_explicitly():
    decision = decide(
        compare(_metrics(map50=0.80), _metrics(map50=0.95)),
        _golden(golden.NOT_CONFIGURED),
        _policy(require_golden_pass=False),
    )

    assert decision.decision == PROMOTION_CANDIDATE
    assert decision.golden_status == golden.NOT_CONFIGURED


def test_an_unmeasurable_metric_blocks_rather_than_being_skipped():
    champion = _metrics(map50=0.80)
    challenger = ModelMetrics(overall={"precision": 0.9}, per_class={})

    decision = decide(compare(champion, challenger), _golden(), _policy())

    assert decision.decision == REJECTED
    assert any("could not be compared" in reason for reason in decision.reasons)


def test_every_failing_rule_is_reported_not_just_the_first():
    champion = _metrics(map50=0.80, recall=0.90, false_negatives=0)
    challenger = _metrics(map50=0.60, recall=0.60, false_negatives=9)

    decision = decide(
        compare(champion, challenger), _golden(golden.MISSING), _policy()
    )

    assert len(decision.reasons) >= 4


def test_thresholds_are_all_configurable():
    champion = _metrics(map50=0.80)
    challenger = _metrics(map50=0.79)

    strict = decide(compare(champion, challenger), _golden(), _policy())
    lenient = decide(
        compare(champion, challenger), _golden(), _policy(min_map50_delta=-0.05)
    )

    assert strict.decision == REJECTED
    assert lenient.decision == PROMOTION_CANDIDATE


def test_describe_lists_reasons_and_passed_checks():
    decision = decide(
        compare(_metrics(map50=0.80), _metrics(map50=0.60)), _golden(), _policy()
    )

    lines = list(describe(decision))

    assert lines[0] == "Decision: REJECTED"
    assert any("Blocking reasons" in line for line in lines)


# ---------------------------------------------------------------------------
# Metric extraction


class _FakeBox:
    def __init__(self):
        self.mp = 0.85
        self.mr = 0.90
        self.map50 = 0.80
        self.map = 0.55
        self.ap_class_index = [0, 1]
        self.p = [0.9, 0.8]
        self.r = [0.95, 0.7]
        self.ap50 = [0.92, 0.75]
        self.ap = [[0.6, 0.5], [0.4, 0.3]]


class _FakeConfusion:
    # 2 classes + background band.
    matrix = [
        [10.0, 0.0, 3.0],
        [0.0, 8.0, 1.0],
        [2.0, 4.0, 0.0],
    ]


class _FakeResults:
    def __init__(self):
        self.box = _FakeBox()
        self.names = {0: "Red", 1: "Orange"}
        self.confusion_matrix = _FakeConfusion()
        self.results_dict = {"metrics/precision(B)": 0.85, "metrics/recall(B)": 0.90}


def test_extraction_produces_overall_and_per_class_metrics():
    metrics = extract_metrics(_FakeResults(), source="challenger")

    assert metrics.overall["map50"] == pytest.approx(0.80)
    assert metrics.per_class["Red"].recall == pytest.approx(0.95)
    assert metrics.per_class["Orange"].precision == pytest.approx(0.80)
    assert metrics.per_class["Red"].map50_95 == pytest.approx(0.55)
    assert metrics.source == "challenger"


def test_false_positive_and_negative_counts_come_from_the_background_band():
    metrics = extract_metrics(_FakeResults())

    assert metrics.false_positives == 4  # 3 + 1, predictions with no ground truth
    assert metrics.false_negatives == 6  # 2 + 4, ground truth with no prediction


def test_extraction_survives_a_result_without_per_class_arrays():
    class Sparse:
        box = None
        names = {}

    metrics = extract_metrics(Sparse())

    assert metrics.per_class == {}
    assert metrics.overall == {}
    assert metrics.false_negatives == 0


def test_metrics_round_trip_through_json_shape():
    original = extract_metrics(_FakeResults(), source="champion")

    restored = ModelMetrics.from_dict(original.to_dict())

    assert restored.overall == original.overall
    assert restored.per_class["Red"].recall == original.per_class["Red"].recall
    assert restored.false_negatives == original.false_negatives


def test_comparison_covers_classes_from_either_model():
    champion = _metrics(per_class={"Red": 0.9, "Gone": 0.8})
    challenger = _metrics(per_class={"Red": 0.91, "New": 0.7})

    comparison = compare(champion, challenger)
    names = [item.name for item in comparison.per_class_recall]

    assert names == ["Gone", "New", "Red"]
    assert comparison.recall_delta("Red") == pytest.approx(0.01)
    assert comparison.recall_delta("Gone") is None
