"""The training candidate pool.

A candidate is a production image a selector thought was worth learning from,
together with *why* it was picked. Two properties matter:

1. **Selected is not labelled.** Every candidate enters as ``NEEDS_LABEL``.
   Predictions are never promoted to ground truth --- doing that teaches the
   model its own mistakes, which is why the operator workflow already forbids
   it. Labels arrive through :mod:`picture_tool.autotrain.labeling`.
2. **The image is copied in immediately.** Production retention can delete a
   PASS image within 30 days, so a pool that only stored a path would rot.

Sample identity is the SHA-256 of the image bytes, matching the convention
the operator review flow already uses, so the same photo selected twice is
one candidate rather than two.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from picture_tool.autotrain import AutoTrainError
from picture_tool.dataset_manifest_lock import autotrain_store_lock

LOGGER = logging.getLogger(__name__)

POOL_MANIFEST_NAME = "pool.jsonl"
POOL_IMAGES_DIRNAME = "images"
POOL_LABELS_DIRNAME = "labels"

#: A candidate has been selected but carries no trustworthy annotation yet.
NEEDS_LABEL = "NEEDS_LABEL"
#: A human has annotated it and the label passed validation.
VERIFIED = "VERIFIED"
#: A human looked at it and decided it should not be trained on.
DISCARDED = "DISCARDED"

LABEL_STATES = (NEEDS_LABEL, VERIFIED, DISCARDED)


class CandidatePoolError(AutoTrainError):
    """Raised when the pool cannot be read or written safely."""


@dataclass(frozen=True)
class CandidateSample:
    """One pooled candidate and the provenance of its selection."""

    sample_id: str
    inspection_id: str
    product: str
    area: str
    selector: str
    selected_reason: str
    score: float
    source_model: str
    source_model_version: str
    selected_at: str
    production_timestamp: str
    image_path: str
    source_image_path: str
    status: str = ""
    label_state: str = NEEDS_LABEL
    label_path: str = ""
    also_selected_by: tuple[str, ...] = ()
    image_quality: Mapping[str, float] | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "inspection_id": self.inspection_id,
            "product": self.product,
            "area": self.area,
            "selector": self.selector,
            "selected_reason": self.selected_reason,
            "score": self.score,
            "source_model": self.source_model,
            "source_model_version": self.source_model_version,
            "selected_at": self.selected_at,
            "production_timestamp": self.production_timestamp,
            "image_path": self.image_path,
            "source_image_path": self.source_image_path,
            "status": self.status,
            "label_state": self.label_state,
            "label_path": self.label_path,
            "also_selected_by": list(self.also_selected_by),
            "image_quality": dict(self.image_quality) if self.image_quality else None,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateSample":
        quality = payload.get("image_quality")
        return cls(
            sample_id=str(payload["sample_id"]),
            inspection_id=str(payload.get("inspection_id", "")),
            product=str(payload.get("product", "")),
            area=str(payload.get("area", "")),
            selector=str(payload.get("selector", "")),
            selected_reason=str(payload.get("selected_reason", "")),
            score=float(payload.get("score", 0.0)),
            source_model=str(payload.get("source_model", "")),
            source_model_version=str(payload.get("source_model_version", "")),
            selected_at=str(payload.get("selected_at", "")),
            production_timestamp=str(payload.get("production_timestamp", "")),
            image_path=str(payload.get("image_path", "")),
            source_image_path=str(payload.get("source_image_path", "")),
            status=str(payload.get("status", "")),
            label_state=str(payload.get("label_state", NEEDS_LABEL)),
            label_path=str(payload.get("label_path", "")),
            also_selected_by=tuple(
                str(name) for name in payload.get("also_selected_by", []) or []
            ),
            image_quality=dict(quality) if isinstance(quality, dict) else None,
            metrics=dict(payload.get("metrics") or {}),
        )


@dataclass(frozen=True)
class PoolAddResult:
    """What one ``add`` transaction changed."""

    added: tuple[str, ...]
    merged: tuple[str, ...]
    skipped_missing_image: tuple[str, ...]

    def summary(self) -> dict[str, int]:
        return {
            "added": len(self.added),
            "merged": len(self.merged),
            "skipped_missing_image": len(self.skipped_missing_image),
        }


class CandidatePool:
    """A per-station, append-mostly store of training candidates.

    Reads are lock-free; every mutation takes a cross-process lock and
    rewrites the manifest atomically, so a crash mid-write cannot leave a
    half-written pool behind.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest_path = self.root / POOL_MANIFEST_NAME
        self.images_dir = self.root / POOL_IMAGES_DIRNAME
        self.labels_dir = self.root / POOL_LABELS_DIRNAME

    # -- reading ----------------------------------------------------------------

    def load(self) -> tuple[CandidateSample, ...]:
        """Return every candidate, skipping lines that cannot be parsed."""
        if not self.manifest_path.is_file():
            return ()
        samples: list[CandidateSample] = []
        try:
            lines = self.manifest_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise CandidatePoolError(
                f"Unable to read candidate pool {self.manifest_path}: {exc}"
            ) from exc
        for number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                samples.append(CandidateSample.from_dict(json.loads(stripped)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                LOGGER.warning(
                    "Skipping unreadable pool entry %s:%s: %s",
                    self.manifest_path,
                    number,
                    exc,
                )
        return tuple(samples)

    def by_label_state(self, state: str) -> tuple[CandidateSample, ...]:
        """Return candidates in one label state."""
        return tuple(s for s in self.load() if s.label_state == state)

    def statistics(self) -> dict[str, Any]:
        """Counts by label state and by selector, for reports and the facade."""
        samples = self.load()
        by_state = {state: 0 for state in LABEL_STATES}
        by_selector: dict[str, int] = {}
        for sample in samples:
            by_state[sample.label_state] = by_state.get(sample.label_state, 0) + 1
            for name in (sample.selector, *sample.also_selected_by):
                if name:
                    by_selector[name] = by_selector.get(name, 0) + 1
        return {
            "total": len(samples),
            "by_label_state": by_state,
            "by_selector": dict(sorted(by_selector.items())),
        }

    # -- writing ----------------------------------------------------------------

    def add(self, samples: Iterable[CandidateSample]) -> PoolAddResult:
        """Add candidates, copying each image into the pool.

        A candidate already present is merged rather than duplicated: its
        additional selector is recorded, and an existing label state is left
        alone so re-running selection cannot reset human work.
        """
        incoming = list(samples)
        if not incoming:
            return PoolAddResult((), (), ())

        self.root.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        added: list[str] = []
        merged: list[str] = []
        skipped: list[str] = []

        with autotrain_store_lock(self.root, name="pool"):
            existing = {sample.sample_id: sample for sample in self.load()}
            for sample in incoming:
                current = existing.get(sample.sample_id)
                if current is not None:
                    existing[sample.sample_id] = _merge(current, sample)
                    merged.append(sample.sample_id)
                    continue
                stored = self._store_image(sample)
                if stored is None:
                    skipped.append(sample.sample_id)
                    continue
                existing[sample.sample_id] = stored
                added.append(sample.sample_id)
            self._write(existing.values())

        return PoolAddResult(tuple(added), tuple(merged), tuple(skipped))

    def update_label_state(
        self,
        sample_id: str,
        state: str,
        *,
        label_path: str = "",
    ) -> CandidateSample:
        """Record the outcome of human labelling for one candidate."""
        if state not in LABEL_STATES:
            raise CandidatePoolError(
                f"Unknown label state {state!r}; expected one of {LABEL_STATES}."
            )
        with autotrain_store_lock(self.root, name="pool"):
            samples = {sample.sample_id: sample for sample in self.load()}
            current = samples.get(sample_id)
            if current is None:
                raise CandidatePoolError(f"Unknown candidate: {sample_id}")
            updated = replace(current, label_state=state, label_path=label_path)
            samples[sample_id] = updated
            self._write(samples.values())
        return updated

    # -- internals ---------------------------------------------------------------

    def _store_image(self, sample: CandidateSample) -> CandidateSample | None:
        """Copy the production image into the pool, or drop the candidate.

        The copy is what makes the pool survive production retention. A
        candidate whose image has already gone is not pooled at all: there is
        nothing left to label.
        """
        source = Path(sample.source_image_path)
        if not source.is_file():
            LOGGER.warning(
                "Candidate %s has no readable source image at %s; not pooled.",
                sample.sample_id,
                source,
            )
            return None
        destination = self.images_dir / f"{sample.sample_id}{source.suffix.lower()}"
        if not destination.exists():
            try:
                shutil.copy2(source, destination)
            except OSError as exc:
                LOGGER.warning(
                    "Could not copy candidate image %s -> %s: %s",
                    source,
                    destination,
                    exc,
                )
                return None
        return replace(sample, image_path=str(destination))

    def _write(self, samples: Iterable[CandidateSample]) -> None:
        """Rewrite the manifest atomically, newest selection last."""
        ordered = sorted(samples, key=lambda s: (s.selected_at, s.sample_id))
        payload = "".join(
            json.dumps(sample.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for sample in ordered
        )
        handle, temporary_name = tempfile.mkstemp(
            dir=str(self.root), prefix=".pool-", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.manifest_path)
        except OSError as exc:
            raise CandidatePoolError(
                f"Unable to write candidate pool {self.manifest_path}: {exc}"
            ) from exc
        finally:
            if os.path.exists(temporary_name):
                try:
                    os.remove(temporary_name)
                except OSError:  # pragma: no cover - best effort cleanup
                    LOGGER.debug("Could not remove temporary pool file %s", temporary_name)


def _merge(current: CandidateSample, incoming: CandidateSample) -> CandidateSample:
    """Fold a repeat selection into the candidate already in the pool.

    The stronger score wins the headline reason so the pool explains itself by
    its most compelling evidence, but every selector that fired is kept, and
    human label state is never overwritten.
    """
    selectors = set(current.also_selected_by) | {current.selector, incoming.selector}
    selectors.discard("")
    if incoming.score > current.score:
        headline = replace(
            current,
            selector=incoming.selector,
            selected_reason=incoming.selected_reason,
            score=incoming.score,
        )
    else:
        headline = current
    selectors.discard(headline.selector)
    return replace(headline, also_selected_by=tuple(sorted(selectors)))


def sample_id_for_image(path: str | Path) -> str:
    """Content hash of an image, used as the stable candidate identity."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CandidatePoolError(f"Unable to hash image {path}: {exc}") from exc
    return digest.hexdigest()


def utc_now_iso() -> str:
    """Timestamp used for ``selected_at``; UTC so cycles sort correctly."""
    return datetime.now(timezone.utc).isoformat()


def build_candidate(
    record: Any,
    *,
    selector: str,
    reason: str,
    score: float,
    metrics: Mapping[str, Any] | None = None,
    selected_at: str | None = None,
) -> CandidateSample | None:
    """Turn a :class:`ProductionRecord` into a candidate, or None.

    Returns None when the production image is gone, because a candidate
    nobody can label is not a candidate.
    """
    source_image = getattr(record, "original_path", "")
    if not source_image or not Path(source_image).is_file():
        return None
    try:
        sample_id = sample_id_for_image(source_image)
    except CandidatePoolError:
        return None

    quality = getattr(record, "image_quality", None)
    timestamp = getattr(record, "timestamp", None)
    return CandidateSample(
        sample_id=sample_id,
        inspection_id=getattr(record, "inspection_id", ""),
        product=getattr(record, "product", ""),
        area=getattr(record, "area", ""),
        selector=selector,
        selected_reason=reason,
        score=float(score),
        source_model=getattr(record, "model_weights", ""),
        source_model_version=getattr(record, "model_version", ""),
        selected_at=selected_at or utc_now_iso(),
        production_timestamp=timestamp.isoformat() if timestamp else "",
        image_path="",
        source_image_path=str(source_image),
        status=getattr(record, "status", ""),
        label_state=NEEDS_LABEL,
        image_quality=quality.to_dict() if quality is not None else None,
        metrics=dict(metrics or {}),
    )


def dedupe_preserving_best(samples: Sequence[CandidateSample]) -> list[CandidateSample]:
    """Collapse repeats within a single selection pass."""
    best: dict[str, CandidateSample] = {}
    for sample in samples:
        current = best.get(sample.sample_id)
        best[sample.sample_id] = sample if current is None else _merge(current, sample)
    return list(best.values())
