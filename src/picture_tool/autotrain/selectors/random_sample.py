"""Select a small unbiased sample of ordinary production inspections.

Every other selector is biased by construction: it picks what the model or an
operator already flagged. That leaves a blind spot --- drift the model is
confidently wrong about produces no signal at all. A random sample is the
only candidate source that can see it.

Selection is seeded and deterministic, so a cycle can be reproduced and a
report can be checked.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from picture_tool.autotrain.candidate_pool import CandidateSample, build_candidate

DEFAULT_FRACTION = 0.01
DEFAULT_MAX_SAMPLES = 50
DEFAULT_SEED = 20260910


class RandomSampleSelector:
    """Pick a seeded random fraction of the records, capped by ``max_samples``."""

    name = "random_sample"

    def select(
        self,
        records: Sequence[object],
        *,
        options: Mapping[str, Any],
        selected_at: str | None = None,
    ) -> list[CandidateSample]:
        fraction = float(options.get("fraction", DEFAULT_FRACTION))
        limit = int(options.get("max_samples", DEFAULT_MAX_SAMPLES))
        seed = int(options.get("seed", DEFAULT_SEED))
        if not records or limit <= 0 or fraction <= 0:
            return []

        wanted = min(limit, max(1, round(len(records) * fraction)))
        # Sort before sampling: the collector's directory-walk order is not
        # guaranteed across platforms, and an unstable order would make the
        # seed meaningless.
        ordered = sorted(records, key=_identity)
        chosen = random.Random(seed).sample(ordered, k=min(wanted, len(ordered)))

        candidates: list[CandidateSample] = []
        for record in chosen:
            candidate = build_candidate(
                record,
                selector=self.name,
                reason=(
                    f"random audit sample (fraction={fraction:g}, seed={seed})"
                ),
                # Lowest score of any selector: a random pick should never
                # outrank a real signal when the pool merges duplicates.
                score=0.0,
                metrics={"fraction": fraction, "seed": seed},
                selected_at=selected_at,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates


def _identity(record: object) -> str:
    return str(getattr(record, "inspection_id", "") or id(record))
