"""Select inspections a human already judged the model got wrong.

These are the highest-value candidates in the pool: an operator has looked at
the board and disagreed with the machine. The disagreement is real evidence;
the *correct boxes* still are not, so these too enter as ``NEEDS_LABEL``.

The signals come from the production inspection database's own review
columns, which the operator review screen already writes.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from picture_tool.autotrain.candidate_pool import CandidateSample, build_candidate

DEFAULT_MAX_SAMPLES = 500

#: Review outcomes that mean "the machine was wrong", ranked by how much a
#: corrected label is worth. An escape (real defect shipped as PASS) outranks
#: an over-kill (good board failed), because the first one reaches a customer.
_OUTCOME_SCORES: dict[str, float] = {
    "false_accept": 1.0,
    "missed_defect": 1.0,
    "escape": 1.0,
    "false_reject": 0.8,
    "over_kill": 0.8,
    "overkill": 0.8,
    "wrong_class": 0.7,
    "label_fix_required": 0.6,
    "annotation_fix": 0.6,
}

#: Outcomes that explicitly mean "do not train on this".
_EXCLUDED_OUTCOMES = frozenset(
    {"undecidable", "not_usable", "skip", "confirmed_ok", "confirmed_ng"}
)


class ReviewCorrectionSelector:
    """Pick records an operator marked as a machine error."""

    name = "review_correction"

    def select(
        self,
        records: Sequence[object],
        *,
        options: Mapping[str, Any],
        selected_at: str | None = None,
    ) -> list[CandidateSample]:
        limit = int(options.get("max_samples", DEFAULT_MAX_SAMPLES))
        if limit <= 0:
            return []
        extra_scores = options.get("outcome_scores")
        scores = dict(_OUTCOME_SCORES)
        if isinstance(extra_scores, dict):
            scores.update({str(k).lower(): float(v) for k, v in extra_scores.items()})

        scored: list[tuple[float, str, object]] = []
        for record in records:
            outcome = str(getattr(record, "review_outcome", "") or "").strip().lower()
            if not outcome or outcome in _EXCLUDED_OUTCOMES:
                continue
            score = scores.get(outcome)
            if score is None:
                continue
            scored.append((score, outcome, record))

        scored.sort(key=lambda item: item[0], reverse=True)

        candidates: list[CandidateSample] = []
        for score, outcome, record in scored[:limit]:
            category = str(getattr(record, "failure_category", "") or "")
            candidate = build_candidate(
                record,
                selector=self.name,
                reason=(
                    f"operator review recorded {outcome!r}"
                    + (f" ({category})" if category else "")
                ),
                score=score,
                metrics={
                    "review_outcome": outcome,
                    "failure_category": category,
                    "action_route": str(getattr(record, "action_route", "") or ""),
                },
                selected_at=selected_at,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates
