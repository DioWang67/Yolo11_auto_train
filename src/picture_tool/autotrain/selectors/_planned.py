"""Reserved selectors, declared but deliberately not implemented.

The configuration accepts these four names so the file shape is stable and a
station can be prepared for them, but switching one on is refused in
``config.py`` rather than silently ignored. Each needs infrastructure this
phase does not build:

``model_disagreement``
    Needs a second model run over the same frames, and somewhere to keep the
    second model's predictions.
``class_imbalance``
    Needs the *labelled* class distribution of the current dataset version,
    which only exists after labelling, not at collection time.
``embedding_novelty``
    Needs an embedding model and a stored reference distribution.
``distribution_drift``
    Needs a baseline window to compare against, and a decision about what
    counts as drift on this station's own data.

They are listed here, rather than left out, so the next phase extends a
declared shape instead of inventing one --- and so nobody concludes from
their absence that these signals were forgotten.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from picture_tool.autotrain.candidate_pool import CandidateSample

PLANNED_SELECTOR_NOTES: dict[str, str] = {
    "model_disagreement": "requires a second model's predictions over the same frames",
    "class_imbalance": "requires the labelled class distribution of a dataset version",
    "embedding_novelty": "requires an embedding model and a reference distribution",
    "distribution_drift": "requires a baseline window and a station-specific threshold",
}


class NotImplementedSelector:
    """Placeholder that names what it would need, if it were ever created.

    Never registered by default; ``config.py`` refuses to enable these names,
    so this exists to document the contract a future selector must satisfy.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def select(
        self,
        records: Sequence[object],
        *,
        options: Mapping[str, Any],
        selected_at: str | None = None,
    ) -> list[CandidateSample]:
        raise NotImplementedError(
            f"Selector {self.name!r} is reserved for a later phase: "
            + PLANNED_SELECTOR_NOTES.get(self.name, "not implemented")
        )
