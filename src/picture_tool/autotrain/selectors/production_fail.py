"""Select inspections the line itself failed that nobody has reviewed yet.

This closes a blind spot the other selectors share. A model that misses a
wire outright draws no box for it, so it is never *unsure* about it and
``low_confidence`` cannot pick the image; ``review_correction`` only sees
what an operator already looked at. The line still notices --- the station
counts fewer items than it expects and fails the board --- so an unreviewed
failure is where total misses surface.

Most failures are real defects rather than model errors. That is fine: either
way the image is a hard case worth a verified label, and it enters the pool as
``NEEDS_LABEL`` like every other pick. Reviewed failures are left to
``review_correction``, which ranks them by what the reviewer found. ``ERROR``
style statuses are system faults whose images may be unusable, and are not
picked.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from picture_tool.autotrain.candidate_pool import CandidateSample, build_candidate

DEFAULT_MAX_SAMPLES = 100
#: What inference records for a board it judged bad. ``detection_system``
#: turns a detector's FAIL into DETECTION_FAIL, so both occur.
DEFAULT_STATUSES = ("FAIL", "DETECTION_FAIL")
#: Below review_correction's lowest (0.6): a reviewed error is stronger
#: evidence than an unreviewed failure when both selectors pick one image.
_SCORE = 0.5


class ProductionFailSelector:
    """Pick failed inspections without an operator review."""

    name = "production_fail"

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
        statuses = {
            str(status).strip().upper()
            for status in options.get("statuses", DEFAULT_STATUSES)
        }

        candidates: list[CandidateSample] = []
        for record in records:
            if len(candidates) >= limit:
                break
            status = str(getattr(record, "status", "") or "").strip().upper()
            if status not in statuses:
                continue
            if str(getattr(record, "review_outcome", "") or "").strip():
                continue
            reasons = [str(r) for r in getattr(record, "fail_reasons", ()) or () if r]
            candidate = build_candidate(
                record,
                selector=self.name,
                reason=f"production judged {status}"
                + (f": {', '.join(reasons)}" if reasons else ""),
                score=_SCORE,
                metrics={"fail_reasons": reasons},
                selected_at=selected_at,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates
