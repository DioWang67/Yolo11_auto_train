"""Human labelling, checked --- and the states between "a file exists" and
"this is ground truth".

A review pack asks a person for boxes. This module is what happens when they
come back with them. It validates, it records who approved what, and it
refuses to let either of those stand in for the other.

The distinction it exists to enforce is that **a label file is not an
approval**. A directory full of ``.txt`` files proves that someone drew
boxes; it says nothing about whether the boxes are right, whether the image
belongs in a yardstick at all, or whether anybody looked. Treating the
presence of a file as consent is how a golden set quietly fills with
half-finished work. So validation produces a *derived* state from the
content, a person produces a *decision*, and only the two together --
agreeing, on the same bytes -- make a sample eligible.

Approval is bound to content. Every decision records the sha256 of the image
and of the label it was made against. Edit the label afterwards and the
approval does not follow: it goes stale and the sample drops back to
awaiting review. This is the same reasoning that locks a golden manifest,
applied one step earlier, and it closes the obvious hole in any
approve-then-edit workflow.

Identity is the image's content hash, joined against the review pack. Not
the filename: a reviewer will rename, re-case and re-extension files, and a
pack image renamed by hand must still be the same sample. Joining on bytes
also means an image that was never in the pack cannot be smuggled in, which
matters because the pack is where the training-contamination checks were
done.

Nothing here registers a golden set, and nothing here writes a label.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from picture_tool.autotrain import AutoTrainError
from picture_tool.autotrain.class_schema import ClassSchema
from picture_tool.autotrain.golden_candidates import (
    IMAGE_SUFFIXES,
    UNKNOWN_SOURCE,
)
from picture_tool.pending_annotations import validate_yolo_label_text

LOGGER = logging.getLogger(__name__)

REVIEW_STATE_FILENAME = "review_state.json"
REVIEW_SCHEMA_VERSION = 1
IMAGES_DIRNAME = "images"
LABELS_DIRNAME = "labels"

#: No label file yet. The reviewer has not started this one.
NEEDS_LABEL = "NEEDS_LABEL"
#: A valid, complete label, waiting for a person to say it is right.
LABELED = "LABELED"
#: Something a person has to judge before this can go further.
NEEDS_REVIEW = "NEEDS_REVIEW"
#: A person said this is ground truth, against these exact bytes.
APPROVED = "APPROVED"
#: A person said this one is not usable.
REJECTED = "REJECTED"

REVIEW_STATES = (NEEDS_LABEL, LABELED, NEEDS_REVIEW, APPROVED, REJECTED)
#: The only states a person may set. The others are derived from content.
DECISION_STATES = (APPROVED, REJECTED)

# -- problem codes ----------------------------------------------------------

PROBLEM_MISSING_LABEL = "missing_label"
PROBLEM_MISSING_IMAGE = "label_without_image"
PROBLEM_UNREADABLE_IMAGE = "unreadable_image"
PROBLEM_INVALID_SYNTAX = "invalid_label_syntax"
PROBLEM_LABEL_INCOMPLETE = "label_incomplete"
PROBLEM_EMPTY_LABEL = "empty_label"
PROBLEM_NOT_IN_PACK = "not_in_review_pack"
PROBLEM_SOURCE_UNKNOWN = "source_untraceable"
PROBLEM_TRAINING_CONTAMINATION = "training_contamination"
PROBLEM_SCHEMA_MISMATCH = "class_schema_mismatch"
PROBLEM_APPROVAL_STALE = "approval_stale"

#: Problems that stop a sample being eligible however tidy it looks.
BLOCKING_PROBLEMS = (
    PROBLEM_MISSING_IMAGE,
    PROBLEM_UNREADABLE_IMAGE,
    PROBLEM_INVALID_SYNTAX,
    PROBLEM_EMPTY_LABEL,
    PROBLEM_NOT_IN_PACK,
    PROBLEM_SOURCE_UNKNOWN,
    PROBLEM_TRAINING_CONTAMINATION,
    PROBLEM_SCHEMA_MISMATCH,
    PROBLEM_LABEL_INCOMPLETE,
)


class LabelReviewError(AutoTrainError):
    """Raised when a labelling directory cannot be read or judged."""


@dataclass(frozen=True)
class Decision:
    """One human judgement, bound to the bytes it was made against."""

    state: str
    reviewed_by: str
    reviewed_at: str
    image_sha256: str
    label_sha256: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "image_sha256": self.image_sha256,
            "label_sha256": self.label_sha256,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Decision":
        return cls(
            state=str(payload.get("state", "")),
            reviewed_by=str(payload.get("reviewed_by", "")),
            reviewed_at=str(payload.get("reviewed_at", "")),
            image_sha256=str(payload.get("image_sha256", "")),
            label_sha256=str(payload.get("label_sha256", "")),
            note=str(payload.get("note", "")),
        )


@dataclass(frozen=True)
class SampleReview:
    """One labelled sample, as validation found it."""

    source_image_id: str
    image_path: str
    label_path: str
    image_sha256: str
    label_sha256: str
    state: str
    group: str = ""
    problems: tuple[str, ...] = ()
    detail: str = ""
    class_counts: Mapping[str, int] = field(default_factory=dict)
    box_count: int = 0
    reviewed_by: str = ""
    reviewed_at: str = ""

    @property
    def is_approved(self) -> bool:
        return self.state == APPROVED

    @property
    def is_eligible(self) -> bool:
        """Approved *and* clean. Never one without the other."""
        return self.state == APPROVED and not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_image_id": self.source_image_id,
            "image_path": self.image_path,
            "label_path": self.label_path,
            "image_sha256": self.image_sha256,
            "label_sha256": self.label_sha256,
            "state": self.state,
            "group": self.group,
            "problems": list(self.problems),
            "detail": self.detail,
            "class_counts": dict(self.class_counts),
            "box_count": self.box_count,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "golden_eligible": self.is_eligible,
        }


@dataclass(frozen=True)
class PackSample:
    """What the review pack said about one image, keyed by content hash."""

    source_image_id: str
    group: str
    image_sha256: str


@dataclass(frozen=True)
class LabelReviewReport:
    """Every sample in a labelling directory, and what may happen next."""

    root: str
    samples: tuple[SampleReview, ...]
    class_schema: ClassSchema | None = None
    expected_counts: Mapping[str, int] = field(default_factory=dict)
    orphan_labels: tuple[str, ...] = ()

    def by_state(self) -> dict[str, int]:
        counts = {state: 0 for state in REVIEW_STATES}
        for sample in self.samples:
            counts[sample.state] = counts.get(sample.state, 0) + 1
        return counts

    def eligible(self) -> tuple[SampleReview, ...]:
        return tuple(sample for sample in self.samples if sample.is_eligible)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "root": self.root,
            "by_state": self.by_state(),
            "eligible": len(self.eligible()),
            "orphan_labels": list(self.orphan_labels),
            "class_schema": (
                self.class_schema.to_dict() if self.class_schema else None
            ),
            "expected_class_counts": dict(self.expected_counts),
            "samples": [sample.to_dict() for sample in self.samples],
        }


# ---------------------------------------------------------------------------
# The decision store


def decisions_path(root: str | Path) -> Path:
    return Path(root) / REVIEW_STATE_FILENAME


def load_decisions(root: str | Path) -> dict[str, Decision]:
    """Read the recorded human judgements, if any have been made."""
    path = decisions_path(root)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LabelReviewError(
            f"Could not read {path}: {exc}. Refusing to continue as though no "
            "decisions had been made: that would silently discard approvals."
        ) from exc
    raw = payload.get("decisions") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        raise LabelReviewError(f"{path} is not a review state file.")
    return {
        str(key): Decision.from_dict(value)
        for key, value in raw.items()
        if isinstance(value, dict)
    }


def save_decisions(root: str | Path, decisions: Mapping[str, Decision]) -> Path:
    path = decisions_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "decisions": {
            key: value.to_dict() for key, value in sorted(decisions.items())
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def record_decision(
    root: str | Path,
    report: LabelReviewReport,
    *,
    source_image_ids: Iterable[str],
    state: str,
    reviewed_by: str,
    note: str = "",
) -> tuple[list[str], dict[str, str]]:
    """Record a person's judgement on some samples.

    Returns the ids recorded and, for each id refused, why. Approving a
    sample that validation has not cleared is refused rather than recorded:
    an approval that sits on top of a known problem is exactly the thing
    this module exists to prevent. Rejection is always allowed --- a person
    may discard anything, including something that never got a label.
    """
    if state not in DECISION_STATES:
        raise LabelReviewError(
            f"{state} is not a decision a person makes; it is derived from "
            f"the content. Use one of {', '.join(DECISION_STATES)}."
        )
    if not reviewed_by.strip():
        raise LabelReviewError(
            "reviewed_by is required: an approval with nobody's name on it is "
            "not an approval."
        )

    known = {sample.source_image_id: sample for sample in report.samples}
    decisions = load_decisions(root)
    now = datetime.now(timezone.utc).isoformat()
    recorded: list[str] = []
    refused: dict[str, str] = {}

    for source_id in source_image_ids:
        sample = known.get(source_id)
        if sample is None:
            refused[source_id] = "not present in this labelling directory"
            continue
        if state == APPROVED:
            if sample.state == NEEDS_LABEL:
                refused[source_id] = "has no label yet"
                continue
            blocking = [p for p in sample.problems if p != PROBLEM_APPROVAL_STALE]
            if blocking:
                refused[source_id] = (
                    "validation problems must be resolved first: "
                    + ", ".join(blocking)
                )
                continue
        decisions[source_id] = Decision(
            state=state,
            reviewed_by=reviewed_by.strip(),
            reviewed_at=now,
            image_sha256=sample.image_sha256,
            label_sha256=sample.label_sha256,
            note=note,
        )
        recorded.append(source_id)

    save_decisions(root, decisions)
    return recorded, refused


# ---------------------------------------------------------------------------
# Validation


def read_pack_samples(pack: str | Path) -> dict[str, PackSample]:
    """Index a review pack by image content hash.

    The pack is the authority on which images may be labelled at all and
    which group each belongs to, and it is where the training-contamination
    checks were performed. Joining on the hash rather than the filename
    means a reviewer may rename freely, and means an image that was never in
    the pack cannot arrive as though it had been.
    """
    import csv

    directory = Path(pack)
    csv_path = directory / "review_pack.csv" if directory.is_dir() else directory
    if not csv_path.is_file():
        raise LabelReviewError(f"Review pack not found: {csv_path}")
    try:
        with open(csv_path, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError) as exc:
        raise LabelReviewError(f"Could not read {csv_path}: {exc}") from exc

    samples: dict[str, PackSample] = {}
    for row in rows:
        digest = str(row.get("image_sha256") or "").strip()
        if not digest:
            continue
        samples[digest] = PackSample(
            source_image_id=str(row.get("source_image_id") or "").strip(),
            group=str(row.get("group") or "").strip(),
            image_sha256=digest,
        )
    if not samples:
        raise LabelReviewError(
            f"{csv_path} lists no images with a content hash; it cannot "
            "identify what was labelled."
        )
    return samples


def validate_labels(
    root: str | Path,
    *,
    schema: ClassSchema,
    expected_counts: Mapping[str, int] | None = None,
    pack_samples: Mapping[str, PackSample] | None = None,
    trained_sha256: Iterable[str] = (),
    trained_source_ids: Iterable[str] = (),
    expected_schema_hash: str = "",
) -> LabelReviewReport:
    """Check a labelling directory and derive each sample's state.

    Validates only. It does not write labels, does not correct them, and
    does not register anything.

    A box count that disagrees with the station is reported as
    ``label_incomplete`` and sends the sample to ``NEEDS_REVIEW``. It is
    never quietly fixed: the count is the strongest available signal that a
    reviewer stopped halfway, and a tool that silently completes or trims
    boxes would erase the evidence that anything was wrong.
    """
    directory = Path(root)
    images_dir = directory / IMAGES_DIRNAME
    labels_dir = directory / LABELS_DIRNAME
    if not images_dir.is_dir():
        raise LabelReviewError(
            f"{images_dir} not found. A labelling directory holds "
            f"{IMAGES_DIRNAME}/ and {LABELS_DIRNAME}/ side by side, which is "
            "the layout the golden registration and the evaluator both expect."
        )

    counts = dict(expected_counts or {})
    expected_total = sum(counts.values())
    trained_hashes = {str(value) for value in trained_sha256}
    trained_sources = {str(value) for value in trained_source_ids}
    decisions = load_decisions(directory)

    samples: list[SampleReview] = []
    seen_labels: set[Path] = set()

    for image in sorted(images_dir.iterdir()):
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        review = _review_one(
            image,
            labels_dir=labels_dir,
            schema=schema,
            expected_counts=counts,
            expected_total=expected_total,
            pack_samples=pack_samples,
            trained_hashes=trained_hashes,
            trained_sources=trained_sources,
            expected_schema_hash=expected_schema_hash,
            decisions=decisions,
            seen_labels=seen_labels,
        )
        samples.append(review)

    orphans = tuple(
        sorted(
            path.name
            for path in (labels_dir.glob("*.txt") if labels_dir.is_dir() else [])
            if path not in seen_labels
        )
    )
    return LabelReviewReport(
        root=str(directory),
        samples=tuple(samples),
        class_schema=schema,
        expected_counts=counts,
        orphan_labels=orphans,
    )


def _review_one(
    image: Path,
    *,
    labels_dir: Path,
    schema: ClassSchema,
    expected_counts: Mapping[str, int],
    expected_total: int,
    pack_samples: Mapping[str, PackSample] | None,
    trained_hashes: set[str],
    trained_sources: set[str],
    expected_schema_hash: str,
    decisions: Mapping[str, Decision],
    seen_labels: set[Path],
) -> SampleReview:
    problems: list[str] = []
    details: list[str] = []

    image_sha = _sha256_file(image)
    if image_sha is None:
        return SampleReview(
            source_image_id=image.stem,
            image_path=str(image),
            label_path="",
            image_sha256="",
            label_sha256="",
            state=NEEDS_REVIEW,
            problems=(PROBLEM_UNREADABLE_IMAGE,),
            detail=f"{image.name} could not be read",
        )

    pack_entry = (pack_samples or {}).get(image_sha)
    if pack_samples is not None and pack_entry is None:
        problems.append(PROBLEM_NOT_IN_PACK)
        details.append(
            "this image is not in the review pack, so the contamination "
            "checks made there do not cover it"
        )
    source_id = pack_entry.source_image_id if pack_entry else image.stem
    group = pack_entry.group if pack_entry else ""
    if not source_id or source_id == UNKNOWN_SOURCE:
        problems.append(PROBLEM_SOURCE_UNKNOWN)
        details.append("no traceable source capture")

    if image_sha in trained_hashes or source_id in trained_sources:
        problems.append(PROBLEM_TRAINING_CONTAMINATION)
        details.append("this capture, or a derivative of it, was trained on")

    if expected_schema_hash and expected_schema_hash != schema.schema_hash:
        problems.append(PROBLEM_SCHEMA_MISMATCH)
        details.append(
            f"class schema hash {schema.schema_hash[:12]} does not match the "
            f"expected {expected_schema_hash[:12]}"
        )

    label = labels_dir / f"{image.stem}.txt"
    if not label.is_file():
        return SampleReview(
            source_image_id=source_id,
            image_path=str(image),
            label_path="",
            image_sha256=image_sha,
            label_sha256="",
            state=NEEDS_LABEL if not problems else NEEDS_REVIEW,
            group=group,
            problems=tuple(problems) if problems else (PROBLEM_MISSING_LABEL,),
            detail="; ".join(details) or "no label file yet",
        )
    seen_labels.add(label)

    label_sha = _sha256_file(label) or ""
    try:
        text = label.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        problems.append(PROBLEM_INVALID_SYNTAX)
        details.append(f"label unreadable: {exc}")
        text = ""

    # One validator, shared with the operator handoff import, so the two
    # paths cannot disagree about what a valid label is.
    syntax_errors = validate_yolo_label_text(text, len(schema.names))
    if syntax_errors:
        problems.append(PROBLEM_INVALID_SYNTAX)
        details.extend(syntax_errors[:5])

    class_counts = _count_classes(text, schema)
    box_count = sum(class_counts.values())
    if box_count == 0 and not syntax_errors:
        problems.append(PROBLEM_EMPTY_LABEL)
        details.append(
            "no boxes. A blank label is a valid negative sample elsewhere, "
            "but this station always has objects in frame."
        )
    elif expected_total and not syntax_errors:
        mismatches = [
            f"{name} {class_counts.get(name, 0)}/{expected}"
            for name, expected in sorted(expected_counts.items())
            if class_counts.get(name, 0) != expected
        ]
        if mismatches:
            problems.append(PROBLEM_LABEL_INCOMPLETE)
            details.append(
                f"{box_count} of {expected_total} expected boxes; "
                + ", ".join(mismatches)
                + ". Not corrected here: a person decides whether the label "
                "is unfinished or the image genuinely differs."
            )

    state = LABELED if not problems else NEEDS_REVIEW
    reviewed_by = ""
    reviewed_at = ""
    decision = decisions.get(source_id)
    if decision is not None and decision.state in DECISION_STATES:
        stale = (
            decision.image_sha256 != image_sha
            or decision.label_sha256 != label_sha
        )
        if stale:
            problems.append(PROBLEM_APPROVAL_STALE)
            details.append(
                f"a {decision.state.lower()} decision by "
                f"{decision.reviewed_by or 'someone'} was made against "
                "different bytes; the image or label changed afterwards"
            )
            state = NEEDS_REVIEW
        elif decision.state == REJECTED:
            state = REJECTED
            reviewed_by, reviewed_at = decision.reviewed_by, decision.reviewed_at
        elif not problems:
            state = APPROVED
            reviewed_by, reviewed_at = decision.reviewed_by, decision.reviewed_at

    return SampleReview(
        source_image_id=source_id,
        image_path=str(image),
        label_path=str(label),
        image_sha256=image_sha,
        label_sha256=label_sha,
        state=state,
        group=group,
        problems=tuple(problems),
        detail="; ".join(details),
        class_counts=class_counts,
        box_count=box_count,
        reviewed_by=reviewed_by,
        reviewed_at=reviewed_at,
    )


def _count_classes(text: str, schema: ClassSchema) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            continue
        if 0 <= class_id < len(schema.names):
            counts[schema.names[class_id]] += 1
    return dict(counts)


def _sha256_file(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Coverage


def stage_approved(
    report: LabelReviewReport,
    destination: str | Path,
    *,
    schema: ClassSchema,
    split: str = "val",
    overwrite: bool = False,
) -> tuple[Path, dict[str, str]]:
    """Copy the eligible samples into a directory ready to be registered.

    Only ``is_eligible`` samples are staged --- approved *and* free of
    problems. Anything else is left where it is.

    The layout is the one ultralytics needs, ``images/<split>/`` beside
    ``labels/<split>/``, because a golden set that cannot be validated is
    not a yardstick. The descriptor carries both ``train`` and ``val`` keys
    pointing at the same split: check_det_dataset raises without both, and
    pointing train at the evaluation images is safe here precisely because
    nothing ever trains on a golden set.

    Returns the staged directory and the ``sha256 -> group`` map to hand to
    :func:`picture_tool.autotrain.golden.register`, so the split a reviewer
    worked to travels with the set instead of being re-derived.
    """
    import shutil

    import yaml

    target = Path(destination)
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise LabelReviewError(
            f"{target} already exists and is not empty. Staging into it would "
            "mix two golden sets; pass overwrite=True only if you mean to "
            "replace it."
        )
    images_dir = target / IMAGES_DIRNAME / split
    labels_dir = target / LABELS_DIRNAME / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, str] = {}
    staged = 0
    for sample in report.eligible():
        image = Path(sample.image_path)
        label = Path(sample.label_path)
        name = sample.source_image_id or image.stem
        shutil.copy2(image, images_dir / f"{name}{image.suffix}")
        shutil.copy2(label, labels_dir / f"{name}.txt")
        if sample.group:
            groups[sample.image_sha256] = sample.group
        staged += 1

    if not staged:
        raise LabelReviewError(
            "No sample is both approved and free of validation problems, so "
            "there is nothing to register. A golden set of nothing would "
            "still be a golden set as far as the promotion gate is concerned."
        )

    (target / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(target.resolve()),
                "train": f"{IMAGES_DIRNAME}/{split}",
                "val": f"{IMAGES_DIRNAME}/{split}",
                "names": {i: name for i, name in enumerate(schema.names)},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return target, groups


def coverage(
    samples: Sequence[SampleReview],
    *,
    groups: Sequence[str],
    min_group_samples: int = 10,
    critical_group: str = "red_orange_critical",
) -> dict[str, Any]:
    """What an eligible set actually covers, group by group.

    Reports ``INSUFFICIENT`` per group on the evaluator's own floor rather
    than describing a thin set as complete. A partial golden set is a
    legitimate thing to build --- waiting for all 250 images before any
    evaluation is possible would stall the whole path --- but it has to say
    which of its groups cannot yet answer a question.
    """
    eligible = [sample for sample in samples if sample.is_eligible]
    per_group: dict[str, dict[str, Any]] = {}
    for group in groups:
        members = [sample for sample in eligible if sample.group == group]
        per_group[group] = {
            "images": len(members),
            "instances": sum(sample.box_count for sample in members),
            "status": (
                "OK" if len(members) >= min_group_samples else "INSUFFICIENT"
            ),
            "shortfall": max(0, min_group_samples - len(members)),
        }

    per_class: collections.Counter[str] = collections.Counter()
    for sample in eligible:
        per_class.update(sample.class_counts)

    ungrouped = [sample for sample in eligible if sample.group not in groups]
    return {
        "total_images": len(eligible),
        "total_instances": sum(sample.box_count for sample in eligible),
        "per_group": per_group,
        "per_class_instances": dict(sorted(per_class.items())),
        "critical_images": per_group.get(critical_group, {}).get("images", 0),
        "ungrouped_images": len(ungrouped),
        "min_group_samples": min_group_samples,
        "insufficient_groups": sorted(
            group
            for group, stats in per_group.items()
            if stats["status"] == "INSUFFICIENT"
        ),
    }
