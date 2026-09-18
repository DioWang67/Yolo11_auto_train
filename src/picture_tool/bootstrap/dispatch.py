"""Letting the vision model choose the next step, within a fence.

The model is put here rather than at the labelling because that is where it
is good. Measured on this station's 46 hand-labelled frames: naming and
counting were right 276 times out of 276, while the boxes it drew scored
mean IoU 0.35. Judging is what it does well; drawing is not. A workflow that
asks it to read numbers and say what to do next is asking the question it
can answer.

**It chooses; it does not act.** The caller declares which actions exist and
carries them out, and a reply naming anything else is refused rather than
mapped onto the nearest thing that was allowed. This is the whole safety
argument for handing a non-deterministic decider the wheel: sampling is on
(no temperature is set anywhere in this project, so the same metrics can
produce different answers), and the way that stays survivable is that the
set of things it can produce is small, fixed, and each one already
understood by the program.

:data:`DEPLOY` is named here only so it can be refused by name. Nothing in
the autotrain path may deploy --- ``trainer.assert_no_forbidden_tasks``
enforces that a layer down --- and a station new enough to need this
workflow has no golden set, so there is nothing that could tell a good
model from a bad one yet. An action that cannot be judged is not an action
to hand to a judge.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from picture_tool.autotrain import AutoTrainError
from picture_tool.bootstrap.vision_client import VisionClientError, VisionRequest

#: What ultralytics calls the numbers, and what this module calls them.
#: Renamed because the decision is read back by people as often as by a
#: model, and "metrics/mAP50(B)" carries a framework's internal punctuation
#: into a record that outlives the framework.
RESULT_COLUMNS: Mapping[str, str] = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50_95",
}

#: How many epochs of history to show. Enough to see a curve flatten, few
#: enough that the numbers do not crowd out the question being asked.
TREND_EPOCHS = 5


def metrics_from_results_csv(path: str | Path) -> dict[str, Any]:
    """The accuracy numbers a training run produced, plus how they moved.

    The final row alone cannot answer the one question that decides between
    stopping and training longer --- whether the curve had flattened --- so
    the last few epochs come too. The trend is reported and not judged:
    reading it is the decider's job, and precomputing "still improving"
    here would be this module quietly making the call it is assembling
    evidence for.
    """
    source = Path(path)
    if not source.is_file():
        return {"metrics_available": False,
                "detail": f"no results.csv at {source}"}
    with source.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row]
    if not rows:
        return {"metrics_available": False,
                "detail": f"{source} has no rows"}

    def number(row: Mapping[str, Any], column: str) -> float | None:
        try:
            return round(float(str(row.get(column, "")).strip()), 5)
        except (TypeError, ValueError):
            return None

    accuracy = {
        name: value
        for column, name in RESULT_COLUMNS.items()
        if (value := number(rows[-1], column)) is not None
    }
    # Rows are not metrics. A results.csv whose accuracy columns are
    # absent, blank or unparseable still has a row count, and calling that
    # available would let the decider be asked to choose between ACCEPT and
    # TRAIN_LONGER from `epochs_run` alone -- a judgement on no evidence,
    # which is the one thing the caller checks this flag to prevent.
    if not accuracy:
        return {
            "metrics_available": False,
            "epochs_run": len(rows),
            "detail": (
                f"{source} has {len(rows)} row(s) but none of "
                f"{sorted(RESULT_COLUMNS)} could be read as a number"
            ),
        }

    out: dict[str, Any] = {"metrics_available": True, "epochs_run": len(rows)}
    out.update(accuracy)
    trend = [number(row, "metrics/mAP50-95(B)") for row in rows[-TREND_EPOCHS:]]
    out["mAP50_95_last_epochs"] = [v for v in trend if v is not None]
    return out

#: What the workflow knows how to do. Each name is a branch the caller has
#: already written; the model picks between them and never invents one.
ACCEPT = "ACCEPT"
TRAIN_LONGER = "TRAIN_LONGER"
MORE_DATA = "MORE_DATA"
STOP = "STOP"

DEFAULT_ACTIONS: tuple[str, ...] = (ACCEPT, TRAIN_LONGER, MORE_DATA, STOP)

#: Never offered, and checked for rather than assumed absent.
DEPLOY = "DEPLOY"


class DispatchError(AutoTrainError):
    """Raised when the model's answer cannot be taken as a decision."""


@dataclass(frozen=True)
class Decision:
    """One step's worth of judgement, with what it was based on.

    ``reason`` is not decoration. A run of this workflow is a chain of
    choices nobody watched, and the only way to audit it afterwards is for
    each link to carry why it was taken --- so an answer without one is
    refused rather than stored as an empty string.
    """

    action: str
    reason: str
    detail: Mapping[str, Any] = field(default_factory=dict)
    #: The metrics the model was shown, so a decision can be re-read later
    #: against what it actually saw rather than what the run ended up with.
    saw: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "detail": dict(self.detail),
            "saw": dict(self.saw),
        }


PROMPT = (
    "You are choosing the next step in a model-training workflow for a "
    "{product} {area} inspection station.\n\n"
    "The station expects these objects in every frame: {inventory}.\n\n"
    "A model has just been trained on {images} labelled image(s) and "
    "measured on a held-out split. Results:\n"
    "{metrics}\n\n"
    "{history}"
    "Choose exactly one of these actions:\n"
    "{actions}\n\n"
    'Reply with ONLY this JSON and nothing else:\n'
    '{{"action":"<one of the above>","reason":"<one sentence>",'
    '"detail":{{}}}}'
)

ACTION_HELP: Mapping[str, str] = {
    ACCEPT: "the model is good enough to hand back for a person to review",
    TRAIN_LONGER: "the metrics are still improving and more epochs may help",
    MORE_DATA: (
        "the labelled set is too small or misses cases; put which classes "
        "look short in detail as {\"classes\":[...]}"
    ),
    STOP: "something is wrong that more training or data will not fix",
}


def render_prompt(
    metrics: Mapping[str, Any],
    *,
    product: str,
    area: str,
    inventory: str,
    images: int,
    actions: Sequence[str],
    history: Sequence[Decision] = (),
) -> str:
    """What the model is shown. Numbers, not pictures.

    The decision is about measurements, and sending frames alongside them
    would invite the model to weigh how the boxes *look* --- the one thing
    this module's opening measurement says it reads badly.
    """
    past = ""
    if history:
        lines = "\n".join(
            f"  {i}. chose {d.action}: {d.reason}" for i, d in enumerate(history, 1)
        )
        past = (
            "Earlier steps in this same run, most recent last:\n"
            f"{lines}\n\n"
            "Do not repeat a step that has already failed to help.\n\n"
        )
    return PROMPT.format(
        product=product,
        area=area,
        inventory=inventory,
        images=images,
        metrics=json.dumps(dict(metrics), indent=2, sort_keys=True),
        history=past,
        actions="\n".join(f"- {a}: {ACTION_HELP.get(a, '')}" for a in actions),
    )


def decision_from_reply(
    fields: Mapping[str, Any],
    *,
    allowed: Sequence[str],
    saw: Mapping[str, Any] | None = None,
) -> Decision:
    """One reply, as a decision, or an error saying why it is not one.

    Nothing here falls back to a default action. A reply that cannot be read
    means the decider did not decide, and continuing on a guess would put
    the workflow somewhere no one chose --- which is exactly the failure the
    fenced action set exists to prevent.
    """
    action = str(fields.get("action") or "").strip().upper()
    if not action:
        raise DispatchError("The reply named no action.")
    if action == DEPLOY:
        raise DispatchError(
            "The reply chose DEPLOY, which is never on offer: this path "
            "cannot deploy, and a station without a golden set has nothing "
            "to judge a deployment against."
        )
    if action not in set(allowed):
        raise DispatchError(
            f"The reply chose {action!r}, which is not one of "
            f"{sorted(set(allowed))}. It is refused rather than mapped onto "
            "the nearest allowed action, which would be this module "
            "deciding while appearing to relay a decision."
        )
    reason = str(fields.get("reason") or "").strip()
    if not reason:
        raise DispatchError(
            f"The reply chose {action} without a reason. A chain of "
            "unwatched choices can only be audited if each carries why."
        )
    detail = fields.get("detail")
    return Decision(
        action=action,
        reason=reason,
        detail=dict(detail) if isinstance(detail, Mapping) else {},
        saw=dict(saw or {}),
    )


@dataclass(frozen=True)
class Flow:
    """One thing the operator can ask for, and what it actually does.

    ``steps`` is written for the person, not the model: a sentence gets
    turned into a job that runs for half an hour, and the cheap protection
    against a misread instruction is showing what was understood in the
    words of what will happen, before it happens.
    """

    name: str
    summary: str
    steps: tuple[str, ...] = ()

    def describe(self) -> str:
        lines = "\n".join(f"     {i}. {s}" for i, s in enumerate(self.steps, 1))
        return f"{self.name}: {self.summary}\n{lines}" if lines else self.name


INTAKE_PROMPT = (
    "An operator has asked for something. Decide which of the available "
    "workflows they mean.\n\n"
    "Their words (they may be in Chinese or English):\n"
    "{instruction}\n\n"
    "Available workflows:\n"
    "{flows}\n\n"
    "If none of them clearly matches, choose NONE rather than the closest "
    "one: running the wrong workflow wastes the operator's time and they "
    "would rather be asked again.\n\n"
    'Reply with ONLY this JSON and nothing else:\n'
    '{{"action":"<workflow name or NONE>","reason":"<one sentence, in the '
    'operator\'s language>","detail":{{}}}}'
)

#: What an intake reply says when the request matched nothing.
NONE = "NONE"


def choose_flow(
    client: Any,
    instruction: str,
    *,
    flows: Sequence[Flow],
) -> Decision:
    """Which workflow an operator's sentence asked for.

    The same fence as :func:`decide_next`, for the same reason: the model
    picks between things that already exist, and a reply naming anything
    else is refused rather than steered to the nearest match. ``NONE`` is
    offered deliberately --- a decider with no way to say "I am not sure"
    will always produce a confident answer, and here that answer would
    start a job nobody asked for.
    """
    names = [f.name for f in flows]
    if DEPLOY in {n.upper() for n in names}:
        raise DispatchError("A flow named DEPLOY cannot be offered here.")
    if not instruction.strip():
        raise DispatchError("No instruction was given to interpret.")
    request = VisionRequest(
        sample_id="intake",
        prompt=INTAKE_PROMPT.format(
            instruction=instruction.strip(),
            flows="\n".join(f"- {f.describe()}" for f in flows),
        ),
    )
    try:
        reply = client.ask(request, require_verdict=False)
    except VisionClientError as exc:
        raise DispatchError(f"Could not reach the model: {exc}") from None
    action = str(reply.fields.get("action") or "").strip()
    if action.upper() == NONE:
        raise DispatchError(
            "The request did not clearly match any workflow: "
            + str(reply.fields.get("reason") or "no reason given")
        )
    # Flow names are matched case-insensitively but returned as declared, so
    # the caller looks them up by the name it registered. The fence is still
    # walked through decision_from_reply --- one place refuses an actionless
    # or reasonless reply --- and that function upper-cases before comparing
    # against `allowed`, so it is handed the upper form and the declared
    # spelling is put back on the way out. Handing it the declared spelling
    # instead refused every flow whose name was not already all-caps.
    for flow in flows:
        if flow.name.lower() == action.lower():
            decision = decision_from_reply(
                {**reply.fields, "action": flow.name.upper()},
                allowed=[flow.name.upper()],
                saw={"instruction": instruction},
            )
            return replace(decision, action=flow.name)
    raise DispatchError(
        f"The reply chose {action!r}, which is not one of {names}."
    )


def decide_next(
    client: Any,
    metrics: Mapping[str, Any],
    *,
    product: str,
    area: str,
    inventory: str,
    images: int,
    actions: Sequence[str] = DEFAULT_ACTIONS,
    history: Sequence[Decision] = (),
) -> Decision:
    """Ask for the next step and come back with one, or raise."""
    if DEPLOY in {a.upper() for a in actions}:
        raise DispatchError(
            "DEPLOY cannot be offered as an action here. It is refused at "
            "the point it would be offered, not only at the point it would "
            "be chosen, so a caller cannot widen the fence by accident."
        )
    request = VisionRequest(
        sample_id=f"{product}_{area}_dispatch",
        prompt=render_prompt(
            metrics, product=product, area=area, inventory=inventory,
            images=images, actions=actions, history=history,
        ),
    )
    try:
        reply = client.ask(request, require_verdict=False)
    except VisionClientError as exc:
        raise DispatchError(f"Could not reach the model: {exc}") from None
    return decision_from_reply(reply.fields, allowed=actions, saw=metrics)
