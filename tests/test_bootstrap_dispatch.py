"""The fence around a non-deterministic decider.

The load-bearing tests are the refusals. Sampling is on, so the same metrics
can produce different answers on different runs; what makes that survivable
is not that the model is reliable but that everything it can return is
either an action the caller already implements or an error.
"""

from __future__ import annotations

from typing import Any

import pytest

from picture_tool.bootstrap.dispatch import (
    ACCEPT,
    DEFAULT_ACTIONS,
    MORE_DATA,
    STOP,
    TRAIN_LONGER,
    Decision,
    DispatchError,
    Flow,
    choose_flow,
    decide_next,
    decision_from_reply,
    metrics_from_results_csv,
    render_prompt,
)
from picture_tool.bootstrap.vision_client import VisionClientError, VisionReply

METRICS = {"mAP50": 0.719, "mAP50-95": 0.117, "images": 46}


class _FakeClient:
    def __init__(self, fields: dict | None = None, error: str = "") -> None:
        self.fields = fields or {}
        self.error = error
        self.prompts: list[str] = []

    def ask(self, request: Any, *, require_verdict: bool = True) -> VisionReply:
        self.prompts.append(request.prompt)
        if self.error:
            raise VisionClientError(self.error)
        return VisionReply(fields=self.fields)


def ask(**overrides) -> Decision:
    fields = {"action": ACCEPT, "reason": "metrics are good enough"}
    fields.update(overrides)
    return decision_from_reply(fields, allowed=DEFAULT_ACTIONS)


# -- the fence --------------------------------------------------------------


def test_an_action_nobody_implements_is_refused_not_rounded() -> None:
    """Mapping it to the nearest allowed action would be deciding, quietly."""
    with pytest.raises(DispatchError, match="not one of"):
        ask(action="RETRAIN_FROM_SCRATCH")


def test_deploy_is_refused_by_name_even_when_chosen() -> None:
    with pytest.raises(DispatchError, match="never on offer"):
        ask(action="DEPLOY")


def test_deploy_cannot_even_be_offered() -> None:
    """Refused where the fence is set, not only where it is tested."""
    with pytest.raises(DispatchError, match="cannot be offered"):
        decide_next(
            _FakeClient(), METRICS, product="Cable1", area="A",
            inventory="6 wire ends", images=46,
            actions=(ACCEPT, "DEPLOY"),
        )


def test_a_decision_without_a_reason_is_refused() -> None:
    """An unwatched chain of choices is auditable only if each carries why."""
    with pytest.raises(DispatchError, match="without a reason"):
        ask(reason="   ")


def test_an_empty_reply_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(DispatchError, match="named no action"):
        decision_from_reply({}, allowed=DEFAULT_ACTIONS)


def test_an_unreachable_model_does_not_become_a_decision() -> None:
    with pytest.raises(DispatchError, match="Could not reach"):
        decide_next(
            _FakeClient(error="connection refused"), METRICS,
            product="Cable1", area="A", inventory="6 wire ends", images=46,
        )


# -- what it does when the reply is usable ----------------------------------


def test_a_usable_reply_becomes_a_decision_carrying_what_it_saw() -> None:
    client = _FakeClient({"action": MORE_DATA, "reason": "Orange looks short",
                          "detail": {"classes": ["Orange"]}})

    decision = decide_next(
        client, METRICS, product="Cable1", area="A",
        inventory="6 wire ends", images=46,
    )

    assert decision.action == MORE_DATA
    assert decision.detail == {"classes": ["Orange"]}
    # The metrics travel with the decision so it can be re-read against what
    # was in front of it, not against where the run finished up.
    assert decision.saw == METRICS
    assert decision.to_dict()["reason"] == "Orange looks short"


def test_case_and_whitespace_in_the_action_are_tolerated() -> None:
    """Formatting noise is not a different decision."""
    assert ask(action=" accept ").action == ACCEPT


def test_a_detail_that_is_not_an_object_is_dropped_not_crashed_on() -> None:
    assert ask(action=STOP, reason="x", detail="nonsense").detail == {}


# -- the prompt -------------------------------------------------------------


def test_the_prompt_carries_the_numbers_and_the_choices() -> None:
    text = render_prompt(
        METRICS, product="Cable1", area="A", inventory="6 wire ends",
        images=46, actions=DEFAULT_ACTIONS,
    )

    assert "0.719" in text and "Cable1" in text
    for action in DEFAULT_ACTIONS:
        assert action in text


# -- the evidence the decision is made on -----------------------------------


HEADER = ("epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),"
          "metrics/mAP50-95(B)\n")


def test_accuracy_numbers_are_read_with_the_curve_that_produced_them() -> None:
    """The final row cannot say whether the curve had flattened."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/results.csv"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(HEADER)
            for i, m in enumerate([0.41, 0.63, 0.77, 0.80, 0.806], start=1):
                handle.write(f"{i},0.998,1.0,0.995,{m}\n")

        out = metrics_from_results_csv(path)

    assert out["metrics_available"] is True
    assert out["mAP50_95"] == 0.806 and out["precision"] == 0.998
    assert out["epochs_run"] == 5
    # The trend travels with it, unjudged: deciding whether 0.80 -> 0.806 is
    # "still improving" is the decider's call, not this function's.
    assert out["mAP50_95_last_epochs"] == [0.41, 0.63, 0.77, 0.8, 0.806]
    assert "still_improving" not in out


def test_a_missing_or_empty_results_file_says_so_rather_than_zeroing() -> None:
    """Zeros would read as a terrible model instead of an absent measurement."""
    import tempfile

    missing = metrics_from_results_csv("nowhere/results.csv")
    assert missing["metrics_available"] is False
    assert "mAP50_95" not in missing

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/results.csv"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(HEADER)
        assert metrics_from_results_csv(path)["metrics_available"] is False


def test_rows_without_readable_accuracy_are_not_metrics() -> None:
    """A row count is not a measurement.

    Reporting available here let the decider be asked to choose between
    ACCEPT and TRAIN_LONGER from epochs_run alone, which is the judgement on
    no evidence that the caller checks this flag to prevent.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/results.csv"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(HEADER)
            for i in range(1, 4):
                handle.write(f"{i},,,,\n")

        out = metrics_from_results_csv(path)

    assert out["metrics_available"] is False
    assert out["epochs_run"] == 3
    assert "none of" in out["detail"] and "read as a number" in out["detail"]
    assert "mAP50_95" not in out


def test_a_flow_is_chosen_by_the_name_it_was_registered_under() -> None:
    """Names are matched case-insensitively and come back as declared.

    The fence still runs through decision_from_reply, which upper-cases
    before comparing: handing it the declared spelling refused every flow
    whose name was not already all-caps.
    """
    flows = [Flow(name="seeded_retrain", summary="retrain from the seed set"),
             Flow(name="cold_start", summary="label from scratch")]
    client = _FakeClient({"action": "Seeded_Retrain", "reason": "他要重訓"})

    decision = choose_flow(client, "重新訓練一次", flows=flows)

    assert decision.action == "seeded_retrain"
    assert decision.saw == {"instruction": "重新訓練一次"}

    unknown = _FakeClient({"action": "deploy_it", "reason": "no"})
    with pytest.raises(DispatchError, match="not one of"):
        choose_flow(unknown, "上線", flows=flows)


def test_earlier_steps_are_shown_so_it_does_not_circle() -> None:
    history = [Decision(action=TRAIN_LONGER, reason="mAP still climbing")]

    text = render_prompt(
        METRICS, product="Cable1", area="A", inventory="6 wire ends",
        images=46, actions=DEFAULT_ACTIONS, history=history,
    )

    assert "mAP still climbing" in text
    assert "already failed to help" in text
