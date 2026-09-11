"""The promotion decision.

Deterministic by construction: a pure function of measured metrics, the
golden-set status, and configured thresholds. No model, agent or heuristic
gets a vote on whether a challenger is better --- "it looks better to me" is
exactly the reasoning this gate exists to exclude.

The strongest outcome available here is ``PROMOTION_CANDIDATE``, which means
"a human should now consider this". It is a recommendation, never a
deployment: adopting a challenger stays a named action in the existing
inspection-release flow.

Every rule is fail-closed. A metric that could not be measured blocks
promotion rather than being skipped, because an unmeasured regression and no
regression look identical from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from picture_tool.autotrain import golden as golden_module
from picture_tool.autotrain.config import PromotionConfig
from picture_tool.autotrain.metrics import MetricComparison

#: The challenger is worth a human's attention.
PROMOTION_CANDIDATE = "PROMOTION_CANDIDATE"
#: The challenger failed at least one rule.
REJECTED = "REJECTED"


@dataclass(frozen=True)
class PromotionDecision:
    """The verdict, and every reason behind it."""

    decision: str
    reasons: tuple[str, ...]
    passed_checks: tuple[str, ...]
    golden_status: str

    @property
    def is_candidate(self) -> bool:
        return self.decision == PROMOTION_CANDIDATE

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reasons": list(self.reasons),
            "passed_checks": list(self.passed_checks),
            "golden_status": self.golden_status,
        }


def decide(
    comparison: MetricComparison,
    golden_status: golden_module.GoldenStatus,
    policy: PromotionConfig,
) -> PromotionDecision:
    """Apply every promotion rule and return the verdict.

    All rules are evaluated even after the first failure: a report that lists
    one problem invites a retry that hits the next one, and the operator
    deserves the whole picture at once.
    """
    failures: list[str] = []
    passed: list[str] = []

    _check_overall_improvement(comparison, policy, failures, passed)
    _check_overall_regression(comparison, policy, failures, passed)
    _check_critical_classes(comparison, policy, failures, passed)
    _check_false_negatives(comparison, policy, failures, passed)
    _check_golden(golden_status, policy, failures, passed)

    return PromotionDecision(
        decision=REJECTED if failures else PROMOTION_CANDIDATE,
        reasons=tuple(failures),
        passed_checks=tuple(passed),
        golden_status=golden_status.status,
    )


# ---------------------------------------------------------------------------
# Rules


def _check_overall_improvement(
    comparison: MetricComparison,
    policy: PromotionConfig,
    failures: list[str],
    passed: list[str],
) -> None:
    delta = comparison.overall_delta("map50")
    if delta is None:
        failures.append(
            "mAP50 could not be compared against the champion; an unmeasured "
            "model cannot be recommended."
        )
        return
    if delta < policy.min_map50_delta:
        failures.append(
            f"mAP50 changed by {delta:+.4f}, below the required "
            f"{policy.min_map50_delta:+.4f}."
        )
        return
    passed.append(f"mAP50 {delta:+.4f} meets the required {policy.min_map50_delta:+.4f}")


def _check_overall_regression(
    comparison: MetricComparison,
    policy: PromotionConfig,
    failures: list[str],
    passed: list[str],
) -> None:
    limit = policy.max_overall_regression
    regressed: list[str] = []
    unmeasured: list[str] = []
    for item in comparison.overall:
        if item.delta is None:
            unmeasured.append(item.name)
            continue
        if item.delta < -limit:
            regressed.append(f"{item.name} {item.delta:+.4f}")
    if unmeasured:
        failures.append(
            "these overall metrics could not be compared: " + ", ".join(unmeasured)
        )
    if regressed:
        failures.append(
            f"overall metrics regressed beyond {limit:.4f}: " + ", ".join(regressed)
        )
    if not unmeasured and not regressed:
        passed.append(f"no overall metric regressed beyond {limit:.4f}")


def _check_critical_classes(
    comparison: MetricComparison,
    policy: PromotionConfig,
    failures: list[str],
    passed: list[str],
) -> None:
    if not policy.critical_classes:
        passed.append("no critical classes configured")
        return

    limit = policy.max_critical_class_recall_drop
    problems: list[str] = []
    for class_name in policy.critical_classes:
        delta = comparison.recall_delta(class_name)
        if delta is None:
            problems.append(
                f"{class_name}: recall could not be compared (class missing from "
                "one of the models)"
            )
            continue
        if delta < -limit:
            problems.append(f"{class_name}: recall {delta:+.4f}")
    if problems:
        failures.append(
            f"critical class recall must not drop by more than {limit:.4f}: "
            + "; ".join(problems)
        )
        return
    passed.append(
        "critical classes held recall within "
        f"{limit:.4f}: {', '.join(policy.critical_classes)}"
    )


def _check_false_negatives(
    comparison: MetricComparison,
    policy: PromotionConfig,
    failures: list[str],
    passed: list[str],
) -> None:
    """False negatives are escapes: a missed defect reaches the customer.

    Judged against the champion, not against zero. A challenger that leaves
    the escape count unchanged has not made the line worse, and on a station
    whose champion already misses some defects an absolute threshold of zero
    would reject every possible improvement.
    """
    champion = comparison.champion.false_negatives
    challenger = comparison.challenger.false_negatives
    increase = challenger - champion
    if increase > policy.max_false_negatives:
        failures.append(
            f"false negatives rose by {increase} "
            f"({champion} -> {challenger}), above the permitted "
            f"{policy.max_false_negatives}."
        )
        return
    passed.append(
        f"false negatives {champion} -> {challenger} "
        f"(increase {increase} within {policy.max_false_negatives})"
    )


def _check_golden(
    golden_status: golden_module.GoldenStatus,
    policy: PromotionConfig,
    failures: list[str],
    passed: list[str],
) -> None:
    if not policy.require_golden_pass:
        passed.append(
            f"golden dataset check not required (status {golden_status.status})"
        )
        return
    if golden_status.is_passing:
        passed.append("golden dataset passed")
        return
    failures.append(
        f"golden dataset status is {golden_status.status}, not a pass"
        + (f": {golden_status.detail}" if golden_status.detail else "")
    )


def describe(decision: PromotionDecision) -> Sequence[str]:
    """Human-readable lines for the report."""
    lines = [f"Decision: {decision.decision}"]
    if decision.reasons:
        lines.append("Blocking reasons:")
        lines.extend(f"  - {reason}" for reason in decision.reasons)
    if decision.passed_checks:
        lines.append("Checks passed:")
        lines.extend(f"  - {check}" for check in decision.passed_checks)
    return lines
