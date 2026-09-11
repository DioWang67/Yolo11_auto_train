"""Assembling a *candidate* list for a golden evaluation set.

Nothing here produces a golden set. It produces a list for a person to read,
and :mod:`picture_tool.autotrain.golden` is what turns a directory a person
assembled into a locked one. That separation is the point: a golden set
chosen by a script out of whatever was lying around is worse than none,
because it looks like evidence.

Two kinds of candidate come out, and they are not interchangeable:

``ready_to_review``
    Images that already carry human-drawn ground-truth boxes, from operator
    handoff jobs. A reviewer confirms or drops them.

``needs_annotation``
    Production images with strong evidence that they are worth including ---
    a human correction, a low-confidence detection, a missed detection ---
    but **no ground truth**. What production records is the set of boxes the
    model *found*, with a human's verdict on each. A missed object leaves no
    trace there at all, so those records cannot be promoted to ground truth:
    doing so would build a yardstick that scores every false negative as
    correct. They must be annotated before they can be golden.

The same split runs across a second axis, ``representative`` versus
``hard_case``, because a golden set that is all hard cases stops describing
the line and a golden set that is all routine stops catching regressions.
Neither axis is balanced to equal counts here --- see
:func:`summarise` for the statistics, and leave the sizing to a person.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import logging
from dataclasses import dataclass, field
from dataclasses import replace as _replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from picture_tool.autotrain import AutoTrainError
from picture_tool.autotrain.class_schema import ClassSchema
from picture_tool.autotrain.image_quality import measure_file

LOGGER = logging.getLogger(__name__)

REPORT_SCHEMA_VERSION = 1

#: Has human-drawn boxes already; a reviewer only has to agree.
READY_TO_REVIEW = "ready_to_review"
#: Worth including on the evidence, but carries no ground truth yet.
NEEDS_ANNOTATION = "needs_annotation"

#: Describes the line as it normally runs.
REPRESENTATIVE = "representative"
#: Selected because something went wrong, or nearly did.
HARD_CASE = "hard_case"

#: Reasons a sample lands in the hard-case set. Recorded per candidate so a
#: reviewer can see *why* without re-deriving it, and so coverage gaps are
#: visible as reasons that never fired.
REASON_CORRECTION = "human_correction"
REASON_LOW_CONFIDENCE = "low_confidence"
REASON_MISSED_DETECTION = "missed_detection"
REASON_EXTRA_DETECTION = "extra_detection"
REASON_QUALITY_OUTLIER = "quality_outlier"

HARD_CASE_REASONS = (
    REASON_CORRECTION,
    REASON_LOW_CONFIDENCE,
    REASON_MISSED_DETECTION,
    REASON_EXTRA_DETECTION,
    REASON_QUALITY_OUTLIER,
)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


class GoldenCandidateError(AutoTrainError):
    """Raised when a candidate pass cannot be built."""


@dataclass(frozen=True)
class Candidate:
    """One image put forward for a person to consider."""

    sample_id: str
    image_path: str
    status: str
    group: str
    source: str
    timestamp: str = ""
    model_version: str = ""
    camera_id: str = ""
    reasons: tuple[str, ...] = ()
    detail: str = ""
    class_counts: Mapping[str, int] = field(default_factory=dict)
    detection_count: int = 0
    min_confidence: float | None = None
    corrections: tuple[str, ...] = ()
    brightness: float | None = None
    saturation: float | None = None
    blur_score: float | None = None
    duplicate_group: str = ""
    label_path: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "label_path": self.label_path,
            "status": self.status,
            "group": self.group,
            "source": self.source,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "camera_id": self.camera_id,
            "reasons": ";".join(self.reasons),
            "detail": self.detail,
            "class_counts": json.dumps(dict(self.class_counts), ensure_ascii=False),
            "detection_count": self.detection_count,
            "min_confidence": (
                "" if self.min_confidence is None else f"{self.min_confidence:.4f}"
            ),
            "corrections": ";".join(self.corrections),
            "brightness": "" if self.brightness is None else f"{self.brightness:.2f}",
            "saturation": "" if self.saturation is None else f"{self.saturation:.2f}",
            "blur_score": "" if self.blur_score is None else f"{self.blur_score:.2f}",
            "duplicate_group": self.duplicate_group,
        }


# ---------------------------------------------------------------------------
# Duplicate detection


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: Path, size: int = 8) -> str | None:
    """A perceptual hash, for images that are near-identical but not equal.

    The station photographs the same fixture repeatedly, so byte equality
    catches almost none of the redundancy: re-encoding, a one-pixel shift or a
    lighting flicker all defeat it while leaving two pictures a person would
    call the same. dHash compares each pixel with its right-hand neighbour on
    a downscaled greyscale image, which survives all three.

    Deliberately not a similarity *score*: equal hashes group, and that is all
    this is used for. A threshold on Hamming distance would be one more number
    to tune with no evidence to tune it against.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:  # pragma: no cover - cv2 is a hard dependency in practice
        LOGGER.warning("OpenCV unavailable; skipping perceptual hashing")
        return None
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    except (OSError, ValueError) as exc:
        LOGGER.warning("Could not read %s: %s", path, exc)
        return None
    if image is None:
        return None
    small = np.asarray(
        cv2.resize(image, (size + 1, size), interpolation=cv2.INTER_AREA),
        dtype=np.int32,
    )
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flatten().tolist():
        value = (value << 1) | int(bit)
    return f"{value:0{(size * size) // 4}x}"


def group_duplicates(paths: Sequence[Path]) -> dict[Path, str]:
    """Map each image to a duplicate-group id, exact matches first.

    Exact duplicates get the sha group; everything else falls back to its
    perceptual hash. An image in no group at all gets its own id, so a caller
    can always ask "how many groups" without special cases.
    """
    by_sha: dict[str, list[Path]] = collections.defaultdict(list)
    for path in paths:
        try:
            by_sha[sha256_file(path)].append(path)
        except OSError as exc:
            LOGGER.warning("Could not hash %s: %s", path, exc)

    groups: dict[Path, str] = {}
    for digest, members in by_sha.items():
        if len(members) > 1:
            for path in members:
                groups[path] = f"sha:{digest[:12]}"

    remaining = [path for members in by_sha.values() for path in members
                 if path not in groups]
    by_phash: dict[str, list[Path]] = collections.defaultdict(list)
    for path in remaining:
        value = difference_hash(path)
        if value:
            by_phash[value].append(path)
    for value, members in by_phash.items():
        for path in members:
            groups[path] = f"phash:{value}"

    for path in paths:
        groups.setdefault(path, f"single:{path.name}")
    return groups


def merge_group_evidence(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Give every byte-identical copy the union of what is known about them.

    A handoff job copies the production image it was built from, so the same
    picture exists twice: once with human-drawn boxes and no production
    history, once with the confidence, corrections and failure reasons but no
    ground truth. Thinning then has to discard one of them, and either choice
    loses something real.

    So the evidence is merged before anything is discarded. Only exact
    duplicates are merged --- two *similar* photographs have genuinely
    different histories, and pooling those would attribute one image's
    correction to another.
    """
    by_group: dict[str, list[Candidate]] = collections.defaultdict(list)
    for candidate in candidates:
        by_group[candidate.duplicate_group].append(candidate)

    merged: list[Candidate] = []
    for group, members in by_group.items():
        if not group.startswith("sha:") or len(members) < 2:
            merged.extend(members)
            continue
        reasons = tuple(
            sorted({reason for member in members for reason in member.reasons})
        )
        corrections = tuple(
            sorted({pair for member in members for pair in member.corrections})
        )
        confidences = [
            member.min_confidence
            for member in members
            if member.min_confidence is not None
        ]
        details = sorted({member.detail for member in members if member.detail})
        for member in members:
            merged.append(
                _replace(
                    member,
                    reasons=reasons,
                    corrections=corrections,
                    detail="; ".join(details),
                    min_confidence=min(confidences) if confidences else None,
                    group=HARD_CASE if reasons else member.group,
                )
            )
    return merged


def thin_by_group(
    candidates: Sequence[Candidate], *, per_group: int = 2
) -> tuple[list[Candidate], int]:
    """Keep at most ``per_group`` candidates from each duplicate group.

    Byte-identical images are the exception: those keep exactly one, whatever
    ``per_group`` says. Two copies of the same file are not two samples, and
    the station's handoff jobs copy the same pictures between them --- 510
    files for 71 distinct images in this project's own data. Only the
    perceptually-grouped sets keep several, because those really are different
    photographs that happen to look alike, and a reviewer may want to see more
    than one.

    Hard cases are kept ahead of representative ones within a group, and the
    lowest-confidence sample ahead of the rest: if the same picture recurs, the
    copy worth reviewing is the one the model struggled with most.

    Returns the kept candidates and how many were dropped. Nothing on disk is
    touched --- thinning is a decision about the *list*.
    """
    ordered = sorted(
        candidates,
        key=lambda c: (
            # Ground truth outranks everything. A handoff job copies the
            # production image it was built from, so the same picture arrives
            # twice: once labelled, once as production evidence. Dropping the
            # labelled copy to keep the "more interesting" one would throw
            # away the only thing that makes it usable as a yardstick.
            0 if c.status == READY_TO_REVIEW else 1,
            0 if c.group == HARD_CASE else 1,
            c.min_confidence if c.min_confidence is not None else 1.0,
            c.sample_id,
        ),
    )
    seen: collections.Counter[str] = collections.Counter()
    kept: list[Candidate] = []
    dropped = 0
    for candidate in ordered:
        key = candidate.duplicate_group or candidate.sample_id
        allowed = 1 if key.startswith("sha:") else per_group
        if seen[key] >= allowed:
            dropped += 1
            continue
        seen[key] += 1
        kept.append(candidate)
    return kept, dropped


# ---------------------------------------------------------------------------
# Reading the evidence


@dataclass(frozen=True)
class DetectionEvidence:
    """What one production row says about one image."""

    sample_id: str
    image_path: Path
    timestamp: str
    model_version: str
    camera_id: str
    status: str
    detections: tuple[Mapping[str, Any], ...]

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    @property
    def min_confidence(self) -> float | None:
        values = [
            float(d["confidence"])
            for d in self.detections
            if d.get("confidence") is not None
        ]
        return min(values) if values else None

    @property
    def corrections(self) -> tuple[str, ...]:
        """``predicted->verified`` for every box a person changed."""
        pairs = []
        for detection in self.detections:
            predicted = str(detection.get("class") or "")
            verified = str(detection.get("verified_class") or "")
            if predicted and verified and predicted != verified:
                pairs.append(f"{predicted}->{verified}")
        return tuple(pairs)

    @property
    def verified_class_counts(self) -> dict[str, int]:
        counts: collections.Counter[str] = collections.Counter()
        for detection in self.detections:
            name = str(
                detection.get("verified_class") or detection.get("class") or ""
            )
            if name:
                counts[name] += 1
        return dict(counts)


def read_review_manifests(
    manifest_paths: Iterable[str | Path], *, product: str, area: str
) -> list[DetectionEvidence]:
    """Read production review rows, keeping only those with a live image.

    Read-only, and tolerant: these files come in several generations with
    different columns, and a row this cannot parse is skipped rather than
    failing the pass.
    """
    evidence: list[DetectionEvidence] = []
    for manifest in manifest_paths:
        path = Path(manifest)
        try:
            with open(path, encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            LOGGER.warning("Skipping unreadable manifest %s: %s", path, exc)
            continue
        for row in rows:
            if (row.get("product") or "").strip() != product:
                continue
            if (row.get("area") or "").strip() not in ("", area):
                continue
            image = _first_existing(
                row.get("original_path"), row.get("preprocessed_path")
            )
            if image is None:
                continue
            detections = _parse_detections(row.get("detections_json"))
            evidence.append(
                DetectionEvidence(
                    sample_id=image.stem,
                    image_path=image,
                    timestamp=(row.get("timestamp") or "").strip(),
                    model_version=(row.get("model_version") or "").strip(),
                    camera_id=(row.get("camera_id") or "").strip(),
                    status=(row.get("status") or "").strip(),
                    detections=detections,
                )
            )
    return evidence


def _first_existing(*values: str | None) -> Path | None:
    for value in values:
        if not value:
            continue
        path = Path(str(value).strip())
        if path.is_file():
            return path
    return None


def _parse_detections(raw: str | None) -> tuple[Mapping[str, Any], ...]:
    if not raw:
        return ()
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return ()
    if not isinstance(payload, list):
        return ()
    return tuple(item for item in payload if isinstance(item, dict))


# ---------------------------------------------------------------------------
# Selection


def classify(
    evidence: DetectionEvidence,
    *,
    expected_boxes: int,
    low_confidence_below: float,
) -> tuple[str, tuple[str, ...], str]:
    """Decide whether one production row is a hard case, and say why.

    ``expected_boxes`` is how many objects this station should see in a good
    image. It is a property of the station, not of the class schema: Cable1/A
    has five classes but six objects, because two of them are black.
    """
    reasons: list[str] = []
    details: list[str] = []

    corrections = evidence.corrections
    if corrections:
        reasons.append(REASON_CORRECTION)
        details.append(f"corrected {len(corrections)}: {', '.join(sorted(set(corrections)))}")

    lowest = evidence.min_confidence
    if lowest is not None and lowest < low_confidence_below:
        reasons.append(REASON_LOW_CONFIDENCE)
        details.append(f"min confidence {lowest:.3f}")

    count = evidence.detection_count
    if count < expected_boxes:
        reasons.append(REASON_MISSED_DETECTION)
        details.append(f"{count} of {expected_boxes} objects detected")
    elif count > expected_boxes:
        reasons.append(REASON_EXTRA_DETECTION)
        details.append(f"{count} boxes for {expected_boxes} objects")

    group = HARD_CASE if reasons else REPRESENTATIVE
    return group, tuple(reasons), "; ".join(details)


def quality_outliers(
    candidates: Sequence[Candidate], *, percentile: float = 0.05
) -> set[str]:
    """Sample ids in the extreme tails of brightness, saturation or blur.

    Tails rather than thresholds, because what counts as "too dark" for this
    station is not known here and inventing a number would be a threshold
    nobody chose. A reviewer sees the actual values in the report.
    """
    flagged: set[str] = set()
    for attribute in ("brightness", "saturation", "blur_score"):
        measured = [
            (getattr(c, attribute), c.sample_id)
            for c in candidates
            if getattr(c, attribute) is not None
        ]
        if len(measured) < 20:
            continue
        measured.sort()
        cut = max(1, int(len(measured) * percentile))
        flagged.update(sample_id for _, sample_id in measured[:cut])
        flagged.update(sample_id for _, sample_id in measured[-cut:])
    return flagged


def read_labelled_samples(
    roots: Iterable[str | Path], schema: ClassSchema
) -> list[Candidate]:
    """Candidates that already carry human-drawn boxes.

    Their class ids are read against ``schema`` so the counts in the report
    are names rather than numbers, and so an id the schema cannot explain is
    visible here rather than at evaluation time.
    """
    candidates: list[Candidate] = []
    for root in roots:
        raw = Path(root)
        images_dir, labels_dir = raw / "images", raw / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue
        for image in sorted(images_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = labels_dir / f"{image.stem}.txt"
            if not label.is_file():
                continue
            counts: collections.Counter[str] = collections.Counter()
            unknown = 0
            for line in label.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    class_id = int(float(line.split()[0]))
                except (ValueError, IndexError):
                    unknown += 1
                    continue
                if 0 <= class_id < len(schema.names):
                    counts[schema.names[class_id]] += 1
                else:
                    unknown += 1
            candidates.append(
                Candidate(
                    sample_id=image.stem,
                    image_path=str(image),
                    label_path=str(label),
                    status=READY_TO_REVIEW,
                    group=REPRESENTATIVE,
                    source=f"handoff:{_job_id(raw)}",
                    class_counts=dict(counts),
                    detection_count=sum(counts.values()) + unknown,
                    detail=f"{unknown} unreadable class id(s)" if unknown else "",
                )
            )
    return candidates


def _job_id(raw: Path) -> str:
    for parent in raw.parents:
        if parent.parent.name == "jobs":
            return parent.name
    return raw.name


def build_candidates(
    *,
    schema: ClassSchema,
    review_manifests: Iterable[str | Path] = (),
    labelled_roots: Iterable[str | Path] = (),
    product: str = "Cable1",
    area: str = "A",
    expected_boxes: int = 6,
    low_confidence_below: float = 0.55,
    per_duplicate_group: int = 2,
    measure_quality: bool = True,
) -> tuple[list[Candidate], dict[str, Any]]:
    """One candidate pass. Read-only over every input.

    Returns the candidates and the statistics, and writes nothing --- pass
    the pair to :func:`write_report` to put them somewhere. Keeping the two
    apart is what lets a caller inspect a pass before committing it to disk.
    """
    candidates: list[Candidate] = []

    for evidence in read_review_manifests(
        review_manifests, product=product, area=area
    ):
        group, reasons, detail = classify(
            evidence,
            expected_boxes=expected_boxes,
            low_confidence_below=low_confidence_below,
        )
        candidates.append(
            Candidate(
                sample_id=evidence.sample_id,
                image_path=str(evidence.image_path),
                status=NEEDS_ANNOTATION,
                group=group,
                source=f"production:{evidence.status or 'unknown'}",
                timestamp=evidence.timestamp,
                model_version=evidence.model_version,
                camera_id=evidence.camera_id,
                reasons=reasons,
                detail=detail,
                class_counts=evidence.verified_class_counts,
                detection_count=evidence.detection_count,
                min_confidence=evidence.min_confidence,
                corrections=evidence.corrections,
            )
        )

    candidates.extend(read_labelled_samples(labelled_roots, schema))

    # One image can appear in several manifests. Keep the richest row --- the
    # one that found the most reasons --- rather than whichever came first.
    best: dict[str, Candidate] = {}
    for candidate in candidates:
        existing = best.get(candidate.image_path)
        if existing is None or len(candidate.reasons) > len(existing.reasons):
            best[candidate.image_path] = candidate
    candidates = list(best.values())

    if measure_quality:
        measured = []
        for candidate in candidates:
            quality = measure_file(candidate.image_path)
            measured.append(
                candidate
                if quality is None
                else _with_quality(candidate, quality)
            )
        candidates = measured

        outliers = quality_outliers(candidates)
        candidates = [
            _add_reason(c, REASON_QUALITY_OUTLIER)
            if c.sample_id in outliers and c.status == NEEDS_ANNOTATION
            else c
            for c in candidates
        ]

    groups = group_duplicates([Path(c.image_path) for c in candidates])
    candidates = [
        _with_group(c, groups.get(Path(c.image_path), "")) for c in candidates
    ]

    candidates = merge_group_evidence(candidates)
    kept, dropped = thin_by_group(candidates, per_group=per_duplicate_group)
    summary = summarise(kept, schema)
    summary["thinned_out_as_near_duplicates"] = dropped
    summary["examined_before_thinning"] = len(candidates)
    return kept, summary


def _with_quality(candidate: Candidate, quality: Any) -> Candidate:
    return _replace(
        candidate,
        brightness=quality.brightness,
        saturation=quality.saturation,
        blur_score=quality.blur_score,
    )


def _with_group(candidate: Candidate, group: str) -> Candidate:
    from dataclasses import replace

    return replace(candidate, duplicate_group=group)


def _add_reason(candidate: Candidate, reason: str) -> Candidate:
    if reason in candidate.reasons:
        return candidate
    return _replace(
        candidate, reasons=candidate.reasons + (reason,), group=HARD_CASE
    )


# ---------------------------------------------------------------------------
# Reporting


def summarise(
    candidates: Sequence[Candidate], schema: ClassSchema
) -> dict[str, Any]:
    """Counts a person needs to decide how big a golden set should be.

    Reports what is there; chooses nothing. Reasons that never fired are
    reported as coverage gaps rather than omitted, because an empty count is
    the finding.
    """
    by_status = collections.Counter(c.status for c in candidates)
    by_group = collections.Counter(c.group for c in candidates)
    by_source = collections.Counter(c.source.split(":")[0] for c in candidates)
    by_reason = collections.Counter(r for c in candidates for r in c.reasons)
    per_class: collections.Counter[str] = collections.Counter()
    for candidate in candidates:
        per_class.update(candidate.class_counts)
    confusions: collections.Counter[str] = collections.Counter()
    for candidate in candidates:
        confusions.update(candidate.corrections)
    by_day = collections.Counter(
        c.timestamp[:10] for c in candidates if c.timestamp
    )
    by_model = collections.Counter(c.model_version for c in candidates if c.model_version)

    quality: dict[str, Any] = {}
    for attribute in ("brightness", "saturation", "blur_score"):
        values = sorted(
            getattr(c, attribute) for c in candidates if getattr(c, attribute) is not None
        )
        if values:
            quality[attribute] = {
                "min": round(values[0], 2),
                "p50": round(values[len(values) // 2], 2),
                "max": round(values[-1], 2),
                "n": len(values),
            }

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "class_schema": schema.to_dict(),
        "image_count": len(candidates),
        "instance_count": sum(per_class.values()),
        "by_status": dict(by_status),
        "by_group": dict(by_group),
        "by_source": dict(by_source),
        "per_class": dict(per_class),
        "hard_case_reasons": {
            reason: by_reason.get(reason, 0) for reason in HARD_CASE_REASONS
        },
        "coverage_gaps": [
            reason for reason in HARD_CASE_REASONS if not by_reason.get(reason)
        ],
        "confusions": dict(confusions.most_common()),
        "duplicate_groups": len({c.duplicate_group for c in candidates}),
        "by_day": dict(sorted(by_day.items())),
        "by_model_version": dict(by_model),
        "quality": quality,
    }


CSV_COLUMNS = (
    "sample_id",
    "image_path",
    "label_path",
    "status",
    "group",
    "source",
    "timestamp",
    "model_version",
    "camera_id",
    "reasons",
    "detail",
    "class_counts",
    "detection_count",
    "min_confidence",
    "corrections",
    "brightness",
    "saturation",
    "blur_score",
    "duplicate_group",
)


def write_report(
    candidates: Sequence[Candidate],
    summary: Mapping[str, Any],
    destination: str | Path,
) -> dict[str, Path]:
    """Write the candidate list for a person to work through.

    Writes only into ``destination``. Nothing is copied out of production and
    no source image is touched --- the report references images where they
    already live, so reviewing it cannot disturb the data it describes.
    """
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)

    csv_path = directory / "candidates.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for candidate in sorted(
            candidates, key=lambda c: (c.group != HARD_CASE, c.sample_id)
        ):
            writer.writerow(candidate.to_row())

    json_path = directory / "summary.json"
    json_path.write_text(
        json.dumps(dict(summary), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    readme_path = directory / "REVIEW.md"
    readme_path.write_text(_review_text(summary), encoding="utf-8")
    return {"csv": csv_path, "summary": json_path, "readme": readme_path}


def _review_text(summary: Mapping[str, Any]) -> str:
    schema = summary.get("class_schema") or {}
    names = schema.get("names") or []
    mapping = ", ".join(f"{i}={n}" for i, n in enumerate(names))
    gaps = summary.get("coverage_gaps") or []
    if gaps:
        listed = "\n".join(f"  - {gap}" for gap in gaps)
        gap_text = (
            "These hard-case kinds are **not** present in the available "
            "data:\n\n" + listed + "\n\nDo not fabricate them. Record the gap "
            "and revisit when the line produces such a case."
        )
    else:
        gap_text = "None --- every hard-case reason is represented."
    return f"""# Golden candidates --- for review

**These are candidates. None of this is a golden set yet.** Nothing becomes
golden until a person assembles a directory and registers it; see
"Approving" below.

## Class contract

    {mapping}
    hash {schema.get('schema_hash', '')}

Note that this station expects *six* objects per image across *five* classes:
Black appears twice physically. The duplicate is a property of the station,
not of the class list.

## What is in candidates.csv

Two columns decide how to read a row.

`status`
  - `{READY_TO_REVIEW}` --- already has human-drawn boxes. Confirm or drop.
  - `{NEEDS_ANNOTATION}` --- no ground truth. Production records only the
    boxes the model *found*; an object it missed leaves no trace, so these
    cannot be used as a yardstick until someone annotates them.

`group`
  - `{REPRESENTATIVE}` --- the line running normally.
  - `{HARD_CASE}` --- something went wrong or nearly did. `reasons` says what.

## Reviewing

1. Sort by `group`, then by `reasons`. Hard cases first; they are the ones
   worth arguing about.
2. `duplicate_group` marks near-identical images. At most two candidates per
   group are listed, but check that the survivors are the ones you want.
3. For `{NEEDS_ANNOTATION}` rows, annotate before including. `corrections`
   and `min_confidence` say what the model got wrong, which is a hint about
   where to look --- not an answer to copy.
4. Copy the images you accept into one directory, with their labels.

## Approving

    picture-tool-autotrain golden register <your directory> --registered-by "<name>"

Then put the printed path and manifest hash into
`configs/autonomous_training.yaml` under `golden:`. Until that is done,
evaluation reports NOT_CONFIGURED and promotion is refused --- which is
deliberate, not a bug.

## Coverage gaps

{gap_text}

## Sizing

Deliberately not decided here. `summary.json` has the counts; the split
between representative and hard cases, and how many of each, is a judgement
about what this station's evaluation should mean.
"""
