"""Select inspections the deployed model was least sure about.

Low confidence is the classic active-learning signal: the model is telling
you where its decision boundary is thin. It says nothing about whether the
prediction was right, which is exactly why these samples go to a human
labelling queue rather than into training as-is.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from picture_tool.autotrain.candidate_pool import CandidateSample, build_candidate

DEFAULT_CONF_BELOW = 0.55
DEFAULT_MAX_SAMPLES = 200


class LowConfidenceSelector:
    """Pick records whose weakest prediction falls under a threshold."""

    name = "low_confidence"

    def select(
        self,
        records: Sequence[object],
        *,
        options: Mapping[str, Any],
        selected_at: str | None = None,
    ) -> list[CandidateSample]:
        threshold = float(options.get("conf_below", DEFAULT_CONF_BELOW))
        limit = int(options.get("max_samples", DEFAULT_MAX_SAMPLES))
        if limit <= 0:
            return []

        scored: list[tuple[float, object]] = []
        for record in records:
            confidence = getattr(record, "min_confidence", None)
            if confidence is None or confidence >= threshold:
                continue
            scored.append((confidence, record))

        # Least confident first: with a limit in play, those are the samples
        # whose labels buy the most.
        scored.sort(key=lambda item: item[0])

        candidates: list[CandidateSample] = []
        for confidence, record in scored[:limit]:
            candidate = build_candidate(
                record,
                selector=self.name,
                reason=(
                    f"lowest prediction confidence {confidence:.3f} is below "
                    f"the {threshold:.2f} review threshold"
                ),
                # Score rises as confidence falls, so the pool's merge rule
                # keeps the most uncertain reason as the headline.
                score=float(threshold - confidence),
                metrics={"min_confidence": confidence, "conf_below": threshold},
                selected_at=selected_at,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates
