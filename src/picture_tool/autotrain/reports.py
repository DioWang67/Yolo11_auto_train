"""Training cycle reports.

Two renderings of the same facts: a short text block a person can read at a
glance, and a JSON document a tool (or a later agent) can consume. Both are
written into the cycle directory.

Deltas are shown as percentage points, not ratios: "Red Recall +0.1%" means
recall moved from 0.900 to 0.901, which is how the request expressed it and
how the numbers are discussed on the line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from picture_tool.autotrain.evaluator import EvaluationReport
from picture_tool.autotrain.metrics import MetricDelta
from picture_tool.autotrain.promotion import PromotionDecision

#: Shown when a metric could not be measured on one side of the comparison.
UNMEASURED = "n/a"


def render_text(
    *,
    cycle_id: str,
    champion: str,
    challenger: str,
    dataset_version: str,
    evaluation: EvaluationReport | None,
    decision: PromotionDecision | None,
    notes: Sequence[str] = (),
) -> str:
    """Render the human-facing cycle summary."""
    lines: list[str] = [
        f"Training Cycle: {cycle_id}",
        "",
        f"Champion:    {champion or UNMEASURED}",
        f"Challenger:  {challenger or UNMEASURED}",
        f"Dataset:     {dataset_version or UNMEASURED}",
        "",
    ]

    if evaluation is None:
        lines.append("Evaluation:  not run")
    else:
        lines.append("Evaluation:")
        lines.append("  Overall:")
        for item in evaluation.comparison.overall:
            lines.append(f"    {_label(item.name):<18}{_delta(item)}")
        recalls = evaluation.comparison.per_class_recall
        if recalls:
            lines.append("  Per-class recall:")
            for item in recalls:
                lines.append(f"    {item.name + ' Recall':<18}{_delta(item)}")
        if evaluation.status != "COMPLETED":
            lines.append(f"  Status: {evaluation.status} - {evaluation.detail}")

    lines.append("")
    golden_status = evaluation.golden.status if evaluation else "NOT_EVALUATED"
    lines.append(f"Golden Dataset:  {golden_status}")
    if evaluation is not None and evaluation.golden.detail:
        lines.append(f"  {evaluation.golden.detail}")
    if evaluation is not None:
        lines.extend(_golden_lines(evaluation))

    lines.append("")
    if decision is None:
        lines.append("Decision:        not reached")
    else:
        lines.append(f"Decision:        {decision.decision}")
        for reason in decision.reasons:
            lines.append(f"  - {reason}")

    lines.append("")
    lines.append(
        "This is a recommendation only. Nothing has been deployed: adopting a "
        "challenger remains a named human action in the existing inspection "
        "release flow."
    )
    if notes:
        lines.append("")
        lines.append("Notes:")
        lines.extend(f"  - {note}" for note in notes)
    return "\n".join(lines) + "\n"


def build_payload(
    *,
    cycle_id: str,
    product: str,
    area: str,
    champion: Mapping[str, Any] | None,
    challenger: str,
    dataset_version: str,
    evaluation: EvaluationReport | None,
    decision: PromotionDecision | None,
    steps: Mapping[str, Any],
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the machine-readable cycle report."""
    return {
        "schema_version": 1,
        "cycle_id": cycle_id,
        "product": product,
        "area": area,
        "champion": dict(champion) if champion else None,
        "challenger": challenger,
        "dataset_version": dataset_version,
        "evaluation": evaluation.to_dict() if evaluation else None,
        "decision": decision.to_dict() if decision else None,
        "deployed": False,
        "steps": dict(steps),
        "notes": list(notes),
    }


def write_reports(
    directory: Path,
    *,
    text: str,
    payload: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Write both renderings into the cycle directory."""
    directory.mkdir(parents=True, exist_ok=True)
    text_path = directory / "report.md"
    json_path = directory / "report.json"
    text_path.write_text(text, encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return text_path, json_path


def _golden_lines(evaluation: EvaluationReport) -> list[str]:
    """Render the golden numbers, overall and per group.

    Shown together on purpose. The overall golden score is the headline, but
    a challenger that gains on ``representative`` while losing ``hard_case``
    can raise it --- and that is the trade the split exists to expose, so
    neither half is legible without the other.
    """
    lines: list[str] = []
    if evaluation.golden_comparison is not None:
        lines.append("  Golden overall:")
        for item in evaluation.golden_comparison.overall:
            lines.append(f"    {_label(item.name):<18}{_delta(item)}")

    for group in evaluation.golden_groups:
        header = f"  Golden [{group.group}] ({group.sample_count} samples):"
        if not group.is_measured or group.comparison is None:
            # The reason, never a blank or a zero: an unmeasured group and a
            # steady one must not read the same way.
            lines.append(f"{header} {group.status}")
            if group.detail:
                lines.append(f"    {group.detail}")
            continue
        lines.append(header)
        for item in group.comparison.overall:
            lines.append(f"    {_label(item.name):<18}{_delta(item)}")
        for item in group.comparison.per_class_recall:
            lines.append(f"    {item.name + ' Recall':<18}{_delta(item)}")
    return lines


def _label(name: str) -> str:
    return {
        "precision": "Precision",
        "recall": "Recall",
        "map50": "mAP50",
        "map50_95": "mAP50-95",
    }.get(name, name)


def _delta(item: MetricDelta) -> str:
    """Format one metric as a signed percentage-point change.

    Both absolute values are shown alongside: a delta on its own hides
    whether +1.2% moved a strong model or a weak one.
    """
    if item.delta is None:
        return UNMEASURED
    return (
        f"{item.delta * 100:+.1f}%"
        f"  ({_number(item.champion)} -> {_number(item.challenger)})"
    )


def _number(value: float | None) -> str:
    return UNMEASURED if value is None else f"{value:.3f}"
