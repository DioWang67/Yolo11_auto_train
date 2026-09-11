"""Sample selectors: which production inspections are worth learning from.

A selector reads production records and returns candidates with a stated
reason and a score. It never decides that something is *correct* --- only
that it is *interesting*. Labels come from humans afterwards.

Registration follows the same shape as the inference project's pipeline step
registry: factories keyed by name, so a later phase can add a selector
without touching this module's callers.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from picture_tool.autotrain import AutoTrainError
from picture_tool.autotrain.candidate_pool import CandidateSample

LOGGER = logging.getLogger(__name__)


class SelectorError(AutoTrainError):
    """Raised when a selector is misconfigured."""


class SampleSelector(Protocol):
    """Callable that picks candidates out of production records."""

    name: str

    def select(
        self,
        records: Sequence[object],
        *,
        options: Mapping[str, Any],
        selected_at: str | None = None,
    ) -> list[CandidateSample]:
        """Return the candidates this selector wants, with reasons and scores."""
        ...


SelectorFactory = Callable[[], SampleSelector]

_REGISTRY: dict[str, SelectorFactory] = {}


def register_selector(name: str, factory: SelectorFactory) -> None:
    """Register a selector factory by name (case-insensitive)."""
    key = name.strip().lower()
    if not key:
        raise SelectorError("Selector name must not be empty")
    _REGISTRY[key] = factory


def unregister_selector(name: str) -> None:
    """Remove a registered selector, if present."""
    _REGISTRY.pop(name.strip().lower(), None)


def available_selectors() -> list[str]:
    """Names of every registered selector."""
    return sorted(_REGISTRY)


def create_selector(name: str) -> SampleSelector:
    """Instantiate one registered selector."""
    key = name.strip().lower()
    factory = _REGISTRY.get(key)
    if factory is None:
        raise SelectorError(
            f"Unknown selector: {name}. Available: {', '.join(available_selectors())}"
        )
    return factory()


def run_selectors(
    records: Sequence[object],
    settings: Iterable[object],
    *,
    selected_at: str | None = None,
) -> list[CandidateSample]:
    """Run every enabled selector and return the combined candidates.

    One selector raising does not lose the others' work: the failure is
    logged and the pass continues, because a partly-filled pool is more
    useful than none, and the next cycle will try again.
    """
    from picture_tool.autotrain.candidate_pool import dedupe_preserving_best

    collected: list[CandidateSample] = []
    for setting in settings:
        name = getattr(setting, "name", "")
        if not getattr(setting, "enabled", False):
            continue
        try:
            selector = create_selector(name)
            picked = selector.select(
                records,
                options=getattr(setting, "options", {}) or {},
                selected_at=selected_at,
            )
        except (SelectorError, ValueError, TypeError, AttributeError) as exc:
            LOGGER.warning("Selector %s failed and was skipped: %s", name, exc)
            continue
        LOGGER.info("Selector %s picked %s candidate(s)", name, len(picked))
        collected.extend(picked)
    return dedupe_preserving_best(collected)


def _register_defaults() -> None:
    """Register the selectors this phase implements."""
    from picture_tool.autotrain.selectors.low_confidence import LowConfidenceSelector
    from picture_tool.autotrain.selectors.random_sample import RandomSampleSelector
    from picture_tool.autotrain.selectors.review_correction import (
        ReviewCorrectionSelector,
    )

    register_selector("low_confidence", LowConfidenceSelector)
    register_selector("review_correction", ReviewCorrectionSelector)
    register_selector("random_sample", RandomSampleSelector)


_register_defaults()
