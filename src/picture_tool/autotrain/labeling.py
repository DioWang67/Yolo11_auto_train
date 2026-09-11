"""The human labelling queue.

This is the deliberate seam in the autonomous path. Selection is automatic;
annotation is not. Low-confidence and randomly-sampled images carry no ground
truth, and the model's own predictions are not evidence --- training on them
teaches the model its own mistakes, which is exactly why the operator
workflow already refuses to do it.

So a cycle exports a request, a person labels it with whatever tool they
already use, and the cycle imports the result. Labels are validated with the
same :func:`picture_tool.pending_annotations.validate_yolo_label_text` the
operator flow uses, so a label that would be rejected there is rejected here.

Import is all-or-nothing per request: a batch containing one invalid label
changes nothing, so the person fixes it and retries against an unchanged
pool rather than reconciling a half-applied import.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from picture_tool.autotrain import AutoTrainError
from picture_tool.autotrain.candidate_pool import (
    NEEDS_LABEL,
    VERIFIED,
    CandidatePool,
    CandidateSample,
)
from picture_tool.pending_annotations import validate_yolo_label_text

LOGGER = logging.getLogger(__name__)

REQUEST_MANIFEST_NAME = "request.json"
CLASSES_FILENAME = "classes.txt"
IMAGES_DIRNAME = "images"
LABELS_DIRNAME = "labels"
REQUEST_SCHEMA_VERSION = 1


class LabelingError(AutoTrainError):
    """Raised when a labelling request cannot be exported or imported."""


@dataclass(frozen=True)
class LabelingRequest:
    """One exported batch of images awaiting annotation."""

    request_id: str
    root: Path
    product: str
    area: str
    class_names: tuple[str, ...]
    sample_ids: tuple[str, ...]
    created_at: str
    schema_version: int = REQUEST_SCHEMA_VERSION

    @property
    def images_dir(self) -> Path:
        return self.root / IMAGES_DIRNAME

    @property
    def labels_dir(self) -> Path:
        return self.root / LABELS_DIRNAME

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "product": self.product,
            "area": self.area,
            "class_names": list(self.class_names),
            "sample_ids": list(self.sample_ids),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ImportResult:
    """What one import applied, and what it refused."""

    verified: tuple[str, ...] = ()
    skipped_unlabelled: tuple[str, ...] = ()
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def applied(self) -> bool:
        return bool(self.verified)

    def summary(self) -> dict[str, Any]:
        return {
            "verified": len(self.verified),
            "skipped_unlabelled": len(self.skipped_unlabelled),
            "errors": list(self.errors),
        }


def export_request(
    pool: CandidatePool,
    destination: Path,
    *,
    product: str,
    area: str,
    class_names: Sequence[str],
    request_id: str | None = None,
    limit: int | None = None,
    samples: Sequence[CandidateSample] | None = None,
) -> LabelingRequest:
    """Write the pending candidates out for a human to annotate.

    Filenames are prefixed with the selector that picked each image, so an
    annotator can work through one kind of case at a time --- the same reason
    the operator flow prefixes its filenames by repair type.
    """
    if not class_names:
        raise LabelingError(
            "class_names is required: an annotation tool with the wrong class "
            "list produces labels that cannot be trained on."
        )
    pending = list(samples if samples is not None else pool.by_label_state(NEEDS_LABEL))
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        raise LabelingError("There are no candidates waiting for labels.")

    identifier = request_id or datetime.now(timezone.utc).strftime("req_%Y%m%dT%H%M%SZ")
    root = Path(destination) / identifier
    images = root / IMAGES_DIRNAME
    labels = root / LABELS_DIRNAME
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    exported: list[str] = []
    for sample in pending:
        source = Path(sample.image_path)
        if not source.is_file():
            LOGGER.warning(
                "Candidate %s has no pooled image at %s; not exported.",
                sample.sample_id,
                source,
            )
            continue
        name = f"{sample.selector}-{sample.sample_id}{source.suffix.lower()}"
        shutil.copy2(source, images / name)
        exported.append(sample.sample_id)

    if not exported:
        raise LabelingError(
            "No pooled images could be exported; the pool may be damaged."
        )

    (root / CLASSES_FILENAME).write_text(
        "\n".join(class_names) + "\n", encoding="utf-8"
    )
    request = LabelingRequest(
        request_id=identifier,
        root=root,
        product=product,
        area=area,
        class_names=tuple(class_names),
        sample_ids=tuple(exported),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    (root / REQUEST_MANIFEST_NAME).write_text(
        json.dumps(request.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return request


def load_request(root: str | Path) -> LabelingRequest:
    """Read an exported request back from disk."""
    directory = Path(root).expanduser().resolve()
    manifest = directory / REQUEST_MANIFEST_NAME
    if not manifest.is_file():
        raise LabelingError(f"Not a labelling request: {manifest} is missing.")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LabelingError(f"Unable to read {manifest}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LabelingError(f"{manifest} is not a request manifest.")
    return LabelingRequest(
        request_id=str(payload.get("request_id", directory.name)),
        root=directory,
        product=str(payload.get("product", "")),
        area=str(payload.get("area", "")),
        class_names=tuple(str(n) for n in payload.get("class_names", [])),
        sample_ids=tuple(str(s) for s in payload.get("sample_ids", [])),
        created_at=str(payload.get("created_at", "")),
        schema_version=int(payload.get("schema_version", 0)),
    )


def import_request(pool: CandidatePool, root: str | Path) -> ImportResult:
    """Validate annotated labels and mark their candidates verified.

    All-or-nothing: every label is validated before any is stored, so a batch
    with one bad file leaves the pool exactly as it was.
    """
    request = load_request(root)
    if not request.class_names:
        raise LabelingError(
            f"Request {request.request_id} declares no classes; cannot validate "
            "labels against an unknown class list."
        )

    by_id = {sample.sample_id: sample for sample in pool.load()}
    accepted: list[tuple[str, str]] = []
    unlabelled: list[str] = []
    errors: list[str] = []

    for sample_id in request.sample_ids:
        if sample_id not in by_id:
            errors.append(f"{sample_id}: no longer in the candidate pool")
            continue
        label_path = _find_label(request, sample_id)
        if label_path is None:
            unlabelled.append(sample_id)
            continue
        try:
            text = label_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"{sample_id}: unreadable label ({exc})")
            continue
        problems = validate_yolo_label_text(text, len(request.class_names))
        if problems:
            errors.append(f"{sample_id}: " + "; ".join(problems))
            continue
        accepted.append((sample_id, text))

    if errors:
        return ImportResult(
            verified=(),
            skipped_unlabelled=tuple(unlabelled),
            errors=tuple(errors),
        )

    pool.labels_dir.mkdir(parents=True, exist_ok=True)
    verified: list[str] = []
    for sample_id, text in accepted:
        stored = pool.labels_dir / f"{sample_id}.txt"
        stored.write_text(text, encoding="utf-8")
        pool.update_label_state(sample_id, VERIFIED, label_path=str(stored))
        verified.append(sample_id)

    return ImportResult(
        verified=tuple(verified),
        skipped_unlabelled=tuple(unlabelled),
        errors=(),
    )


def verified_samples(pool: CandidatePool) -> tuple[CandidateSample, ...]:
    """Pool entries with a human-verified label that still exists on disk."""
    usable: list[CandidateSample] = []
    for sample in pool.by_label_state(VERIFIED):
        if not sample.label_path or not Path(sample.label_path).is_file():
            LOGGER.warning(
                "Candidate %s is marked verified but its label is missing at %s",
                sample.sample_id,
                sample.label_path,
            )
            continue
        if not sample.image_path or not Path(sample.image_path).is_file():
            LOGGER.warning(
                "Candidate %s is marked verified but its image is missing at %s",
                sample.sample_id,
                sample.image_path,
            )
            continue
        usable.append(sample)
    return tuple(usable)


def _find_label(request: LabelingRequest, sample_id: str) -> Path | None:
    """Locate the annotator's output for one sample.

    Accepts either the exported ``<selector>-<sample_id>.txt`` name or a bare
    ``<sample_id>.txt``, because annotation tools differ in whether they keep
    the image's stem.
    """
    exact = request.labels_dir / f"{sample_id}.txt"
    if exact.is_file():
        return exact
    if not request.labels_dir.is_dir():
        return None
    for path in sorted(request.labels_dir.glob(f"*{sample_id}.txt")):
        if path.is_file():
            return path
    return None


def request_statistics(request: LabelingRequest) -> Mapping[str, int]:
    """How much of a request has been annotated so far."""
    labelled = sum(
        1 for sample_id in request.sample_ids if _find_label(request, sample_id)
    )
    return {
        "requested": len(request.sample_ids),
        "labelled": labelled,
        "outstanding": len(request.sample_ids) - labelled,
    }
