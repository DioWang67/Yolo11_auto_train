"""What the cycle report actually shows a person.

A number that is measured but never rendered is not a check on anything, so
these tests are about what reaches the page --- particularly the golden
split, whose whole purpose is to make a trade visible that the overall score
hides.
"""

from __future__ import annotations

from picture_tool.autotrain import golden
from picture_tool.autotrain.evaluator import (
    COMPLETED,
    GROUP_INSUFFICIENT,
    GROUP_MEASURED,
    GROUP_NO_LABELS,
    EvaluationReport,
    GroupEvaluation,
)
from picture_tool.autotrain.golden_candidates import HARD_CASE, REPRESENTATIVE
from picture_tool.autotrain.metrics import ClassMetrics, ModelMetrics, compare
from picture_tool.autotrain.promotion import (
    PROMOTION_CANDIDATE,
    PromotionDecision,
)
from picture_tool.autotrain.reports import build_payload, render_text


def _metrics(map50: float, recall: float, source: str) -> ModelMetrics:
    return ModelMetrics(
        overall={
            "precision": 0.9,
            "recall": recall,
            "map50": map50,
            "map50_95": 0.5,
        },
        per_class={
            "Black": ClassMetrics(name="Black", precision=0.9, recall=recall)
        },
        source=source,
    )


def _comparison(champion_map50: float, challenger_map50: float):
    return compare(
        _metrics(champion_map50, 0.80, "champion"),
        _metrics(challenger_map50, 0.70, "challenger"),
    )


def _report(groups: tuple[GroupEvaluation, ...]) -> EvaluationReport:
    return EvaluationReport(
        status=COMPLETED,
        comparison=_comparison(0.80, 0.85),
        golden=golden.GoldenStatus(status=golden.OK),
        champion_weights="champion.pt",
        challenger_weights="challenger.pt",
        champion_sha256="abc",
        data_yaml="data.yaml",
        split="test",
        confidence=0.4,
        imgsz=640,
        golden_comparison=_comparison(0.80, 0.84),
        golden_groups=groups,
    )


def _render(report: EvaluationReport) -> str:
    return render_text(
        cycle_id="cycle-1",
        champion="v1",
        challenger="candidate1",
        dataset_version="dataset_v001",
        evaluation=report,
        decision=PromotionDecision(
            decision=PROMOTION_CANDIDATE,
            reasons=(),
            passed_checks=("golden dataset passed",),
            golden_status=golden.OK,
        ),
    )


def test_the_golden_overall_numbers_reach_the_page():
    """They were being measured and then dropped on the floor."""
    text = _render(_report(()))

    assert "Golden overall:" in text
    assert "mAP50" in text


def test_both_groups_are_shown_with_their_sample_counts():
    text = _render(
        _report(
            (
                GroupEvaluation(
                    group=HARD_CASE,
                    status=GROUP_MEASURED,
                    sample_count=40,
                    comparison=_comparison(0.70, 0.60),
                ),
                GroupEvaluation(
                    group=REPRESENTATIVE,
                    status=GROUP_MEASURED,
                    sample_count=120,
                    comparison=_comparison(0.85, 0.92),
                ),
            )
        )
    )

    assert f"Golden [{HARD_CASE}] (40 samples):" in text
    assert f"Golden [{REPRESENTATIVE}] (120 samples):" in text


def test_a_regression_hidden_by_the_overall_score_is_visible_per_group():
    """The trade the split exists to expose: overall up, hard cases down."""
    text = _render(
        _report(
            (
                GroupEvaluation(
                    group=HARD_CASE,
                    status=GROUP_MEASURED,
                    sample_count=40,
                    comparison=_comparison(0.70, 0.60),
                ),
            )
        )
    )

    hard_case_block = text.split(f"Golden [{HARD_CASE}]")[1]
    assert "-10.0%" in hard_case_block


def test_an_unmeasured_group_shows_its_reason_not_a_blank():
    """An unmeasured group and a steady one must not read the same."""
    text = _render(
        _report(
            (
                GroupEvaluation(
                    group=HARD_CASE,
                    status=GROUP_INSUFFICIENT,
                    sample_count=3,
                    detail="3 sample(s) is below the 10 required",
                ),
                GroupEvaluation(
                    group=REPRESENTATIVE,
                    status=GROUP_NO_LABELS,
                    sample_count=50,
                    detail="No label file could be located",
                ),
            )
        )
    )

    assert GROUP_INSUFFICIENT in text
    assert "below the 10 required" in text
    assert GROUP_NO_LABELS in text
    assert "No label file could be located" in text


def test_the_report_still_says_nothing_was_deployed():
    text = _render(_report(()))

    assert "Nothing has been deployed" in text


def test_an_evaluation_that_never_ran_renders_without_golden_lines():
    text = render_text(
        cycle_id="cycle-1",
        champion="v1",
        challenger="",
        dataset_version="",
        evaluation=None,
        decision=None,
    )

    assert "Evaluation:  not run" in text
    assert "Golden overall:" not in text


def test_the_groups_are_carried_in_the_machine_readable_payload():
    payload = build_payload(
        cycle_id="cycle-1",
        product="Cable1",
        area="A",
        champion=None,
        challenger="candidate1",
        dataset_version="dataset_v001",
        evaluation=_report(
            (
                GroupEvaluation(
                    group=HARD_CASE,
                    status=GROUP_MEASURED,
                    sample_count=40,
                    comparison=_comparison(0.70, 0.60),
                ),
            )
        ),
        decision=None,
        steps={},
    )

    groups = payload["evaluation"]["golden_groups"]
    assert [item["group"] for item in groups] == [HARD_CASE]
    assert groups[0]["sample_count"] == 40
    assert payload["deployed"] is False
