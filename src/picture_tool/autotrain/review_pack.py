"""Assembling a *review pack*: a day's work for a person, not a golden set.

A candidate pass produces well over a thousand rows. Nobody labels a
thousand images, so the list is not actually actionable --- and an
unactionable list is how a golden set ends up never being built at all.
This module cuts that down to a few hundred images a reviewer can work
through, and says for each one why it is in the pile.

Three properties separate this from "take the first two hundred rows":

**Selection is at source level, not file level.** The same capture exists
many times over --- handoff jobs copy their datasets forward, augmentation
writes ``_aug_<n>`` derivatives, production stores the same picture under
different paths. Counting files would mean a pack of 200 rows that is
really 40 photographs. Everything here is keyed by
:func:`~picture_tool.autotrain.golden_candidates.source_image_id`, and the
totals reported are distinct sources.

**Anything the model has trained on is removed, not flagged.** A candidate
pass keeps contaminated rows with a reason attached, which is right for a
report a person reads. It is wrong for a pack a person is about to spend
days annotating: labelling an image that cannot ever be a yardstick is
wasted work. Derivatives go with their source, because training on a
flipped, brightened copy is training on the image.

**Near-duplicates are thinned by spreading, not by deleting.** The station
photographs the same fixture all day, so perceptual clusters are large and
mostly redundant --- but not entirely: two pictures that look alike can
differ in the way that matters. Dropping a cluster to one sample throws
that away; keeping all of them fills the pack with the same picture. So a
cluster contributes a few members chosen to be as unlike each other as the
available measurements allow.

Nothing here labels anything, registers anything, or writes into
production. Every entry leaves with status ``NEEDS_LABEL``, which is the
only honest status for an image whose boxes a person has not yet drawn.
"""

from __future__ import annotations

import collections
import csv
import json
import logging
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from picture_tool.autotrain import AutoTrainError
from picture_tool.autotrain.class_schema import ClassSchema
from picture_tool.autotrain.golden_candidates import (
    HARD_CASE,
    NEEDS_ANNOTATION,
    NEEDS_LABEL,
    REPRESENTATIVE,
    UNKNOWN_SOURCE,
    Candidate,
    source_image_id,
)

LOGGER = logging.getLogger(__name__)

PACK_SCHEMA_VERSION = 1

#: Red and Orange are the pair this station actually confuses, so images
#: where that call was wrong or is missing get their own group rather than
#: being averaged into the hard cases.
RED_ORANGE_CRITICAL = "red_orange_critical"

#: Every group a pack can contain, in priority order. Priority decides who
#: survives trimming to the target size.
PACK_GROUPS = (RED_ORANGE_CRITICAL, HARD_CASE, REPRESENTATIVE)

#: Why a source was selected. These are the reviewer's answer to "why am I
#: looking at this picture", and they are kept per entry.
SELECTED_CRITICAL_CORRECTION = "critical_class_corrected"
SELECTED_CRITICAL_COUNT = "critical_class_count_off"
SELECTED_HARD_EVIDENCE = "hard_case_evidence"
SELECTED_CLUSTER_DIVERSITY = "cluster_diversity_pick"
SELECTED_CLUSTER_UNIQUE = "no_near_duplicates"
SELECTED_ROUTINE = "routine_line_sample"

#: Why a source was excluded before selection ever ran.
EXCLUDED_TRAINED_SHA = "trained_exact_image"
EXCLUDED_TRAINED_SOURCE = "trained_source_or_derivative"
#: Perceptually indistinguishable from a training image, though its bytes
#: and its filename lineage both say otherwise.
EXCLUDED_TRAINED_PERCEPTUAL = "trained_perceptual_match"
EXCLUDED_UNTRACEABLE = "source_untraceable"
EXCLUDED_ALREADY_LABELLED = "not_a_needs_label_candidate"

#: How the target is shared out when there are more sources than places.
#: Deliberately not equal: a pack that is all hard cases stops describing
#: the line, and a pack with no critical cases misses the failure the
#: station actually has.
DEFAULT_SHARES: Mapping[str, float] = {
    RED_ORANGE_CRITICAL: 0.30,
    HARD_CASE: 0.45,
    REPRESENTATIVE: 0.25,
}


class ReviewPackError(AutoTrainError):
    """Raised when a review pack cannot be assembled."""


@dataclass(frozen=True)
class PackEntry:
    """One distinct source image, put in front of a person to annotate."""

    source_image_id: str
    sample_id: str
    image_path: str
    image_sha256: str
    group: str
    selected_reasons: tuple[str, ...]
    #: The candidate pass's own evidence reasons, carried through unchanged.
    evidence_reasons: tuple[str, ...] = ()
    detail: str = ""
    timestamp: str = ""
    camera_id: str = ""
    model_version: str = ""
    detection_count: int = 0
    min_confidence: float | None = None
    class_counts: Mapping[str, int] = field(default_factory=dict)
    corrections: tuple[str, ...] = ()
    brightness: float | None = None
    saturation: float | None = None
    blur_score: float | None = None
    perceptual_group: str = ""
    cluster_size: int = 1
    #: Always NEEDS_LABEL. A review pack contains no ground truth by
    #: construction: that is the work it is asking for.
    status: str = NEEDS_LABEL
    #: Where the reviewer opens the image, once the pack has copied it.
    pack_image: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "source_image_id": self.source_image_id,
            "sample_id": self.sample_id,
            "group": self.group,
            "selected_reasons": ";".join(self.selected_reasons),
            "evidence_reasons": ";".join(self.evidence_reasons),
            "status": self.status,
            "pack_image": self.pack_image,
            "image_path": self.image_path,
            "image_sha256": self.image_sha256,
            "timestamp": self.timestamp,
            "camera_id": self.camera_id,
            "model_version": self.model_version,
            "detection_count": self.detection_count,
            "min_confidence": (
                "" if self.min_confidence is None else f"{self.min_confidence:.4f}"
            ),
            "class_counts": json.dumps(
                dict(self.class_counts), ensure_ascii=False
            ),
            "corrections": ";".join(self.corrections),
            "brightness": (
                "" if self.brightness is None else f"{self.brightness:.2f}"
            ),
            "saturation": (
                "" if self.saturation is None else f"{self.saturation:.2f}"
            ),
            "blur_score": (
                "" if self.blur_score is None else f"{self.blur_score:.2f}"
            ),
            "perceptual_group": self.perceptual_group,
            "cluster_size": self.cluster_size,
            "detail": self.detail,
        }


PACK_COLUMNS = tuple(
    PackEntry(
        source_image_id="", sample_id="", image_path="", image_sha256="",
        group="", selected_reasons=(),
    ).to_row()
)


# ---------------------------------------------------------------------------
# Exclusion


def expected_class_counts(expected_items: Sequence[str]) -> dict[str, int]:
    """Turn a station's ``expected_items`` into per-class counts.

    ``expected_items`` is the multiset of objects the station expects to
    see, which is why Cable1/A lists Black twice across five classes. It is
    not a class schema and must never be used as one --- see
    :mod:`picture_tool.autotrain.class_schema` --- but counting it is
    exactly what it is for.
    """
    counts: collections.Counter[str] = collections.Counter()
    for item in expected_items:
        name = str(item).strip()
        if name:
            counts[name] += 1
    return dict(counts)


def exclude_trained(
    candidates: Sequence[Candidate],
    *,
    trained_source_ids: Iterable[str] = (),
    trained_sha256: Iterable[str] = (),
    perceptually_trained: Iterable[str] = (),
    require_traceable_source: bool = True,
) -> tuple[list[Candidate], dict[str, list[str]]]:
    """Drop everything the model may already have seen.

    Removed rather than flagged: a reviewer asked to annotate an image that
    can never serve as a yardstick is being asked to waste a day.

    Three checks, because the first two have a hole between them that this
    station's own data falls through. Bytes catch a copied file. Filename
    lineage catches an augmented derivative. Neither catches a handoff copy
    that was re-encoded *and* renamed --- and the handoff names here
    (``<uuid>-yolo_Cable1_A_142254.jpg``) share no structure at all with the
    production ones (``yolo_Cable1_A_142252_433429_<hex>.jpg``), so lineage
    matching between the two is not merely failing, it is incapable of
    succeeding. Five images in the first real pack were perceptually
    identical to training images while passing both other checks.

    ``perceptually_trained`` is therefore a set of sample ids the caller has
    already matched by perceptual hash. It is computed outside this function
    because that needs the image bytes, and everything else here is pure.
    It is a coarser signal --- two genuinely different photographs of the
    same fixture can hash alike, and at this station they sometimes do --- so
    it is applied fail-closed and reported under its own reason. Losing a
    handful of usable candidates costs the pack nothing; keeping one image
    the model has memorised costs the yardstick its meaning.

    ``require_traceable_source`` also drops images whose lineage cannot be
    read. That is deliberately stricter than "known to be contaminated":
    untraceable is not the same as clean, and this project already refuses
    such images at registration time for the same reason.
    """
    sources = {str(value) for value in trained_source_ids}
    hashes = {str(value) for value in trained_sha256}
    perceptual = {str(value) for value in perceptually_trained}
    kept: list[Candidate] = []
    excluded: dict[str, list[str]] = collections.defaultdict(list)

    for candidate in candidates:
        if candidate.status != NEEDS_ANNOTATION or candidate.label_path:
            excluded[EXCLUDED_ALREADY_LABELLED].append(candidate.sample_id)
            continue
        if candidate.image_sha256 and candidate.image_sha256 in hashes:
            excluded[EXCLUDED_TRAINED_SHA].append(candidate.sample_id)
            continue
        source = candidate.source_image_id or source_image_id(
            candidate.image_path
        )
        if source in sources:
            excluded[EXCLUDED_TRAINED_SOURCE].append(candidate.sample_id)
            continue
        if candidate.sample_id in perceptual:
            excluded[EXCLUDED_TRAINED_PERCEPTUAL].append(candidate.sample_id)
            continue
        if require_traceable_source and source == UNKNOWN_SOURCE:
            excluded[EXCLUDED_UNTRACEABLE].append(candidate.sample_id)
            continue
        kept.append(candidate)
    return kept, dict(excluded)


def perceptual_matches(
    candidates: Sequence[Candidate],
    trained_hashes: Iterable[str],
    *,
    hasher: Any,
) -> set[str]:
    """Sample ids that look like a training image to a perceptual hash.

    ``hasher`` takes an image path and returns a hash or ``None``; injected
    so this stays testable without image files, and so the caller decides
    which hash --- the project already has one in
    :func:`~picture_tool.autotrain.golden_candidates.difference_hash`, and a
    second scheme would disagree with the deduplication that uses it.

    An image the hasher cannot read is *not* matched. Refusing it here would
    conflate "unreadable" with "trained on"; the pack drops unreadable
    images later anyway, when the copy fails.
    """
    trained = {str(value) for value in trained_hashes if value}
    if not trained:
        return set()
    matched: set[str] = set()
    for candidate in candidates:
        digest = hasher(candidate.image_path)
        if digest and str(digest) in trained:
            matched.add(candidate.sample_id)
    return matched


def collapse_to_sources(
    candidates: Sequence[Candidate],
) -> tuple[list[Candidate], int]:
    """One row per distinct source capture, keeping the richest evidence.

    Exact duplicates and augmented derivatives of the same capture are not
    separate samples however many files they occupy. Within a source the
    survivor is the row that knows the most: the most evidence reasons
    first, then the lowest confidence, because if the same capture recurs
    the copy worth a reviewer's time is the one the model struggled with.
    """
    by_source: dict[str, list[Candidate]] = collections.defaultdict(list)
    for candidate in candidates:
        key = candidate.source_image_id or source_image_id(candidate.image_path)
        by_source[key].append(candidate)

    kept: list[Candidate] = []
    for _, members in sorted(by_source.items()):
        members.sort(
            key=lambda c: (
                -len(c.reasons),
                c.min_confidence if c.min_confidence is not None else 1.0,
                c.sample_id,
            )
        )
        kept.append(members[0])
    return kept, len(candidates) - len(kept)


# ---------------------------------------------------------------------------
# Grouping


def critical_reasons(
    candidate: Candidate,
    *,
    critical_classes: Sequence[str],
    expected_counts: Mapping[str, int],
) -> tuple[str, ...]:
    """Why this image is a Red/Orange problem, if it is.

    Two signals, both drawn from what production already recorded:

    * a person changed a box to or from one of the critical classes --- the
      confusion was observed, not inferred;
    * the image otherwise looks complete, but the critical classes are
      distributed wrongly within it --- two Oranges and no Red, say. That is
      a substitution, which is the failure this group is about.

    The second signal is deliberately conditional on the total being right.
    Without that condition it fires on every image the model failed on
    entirely: 935 of this station's production rows carry no detections at
    all, and each of those has zero Reds, which is a missed detection and
    not a colour confusion. Letting them in drowned the group at 63% of all
    candidates on the first real run --- a critical group that contains
    most things is not a critical group.

    Not inferred from confidence: the per-detection class of a
    low-confidence box is not carried this far, and guessing which class a
    weak detection belonged to would put images in the critical group on
    the strength of an assumption.
    """
    critical = {name.casefold() for name in critical_classes}
    reasons: list[str] = []

    for pair in candidate.corrections:
        sides = [side.strip().casefold() for side in pair.split("->")]
        if any(side in critical for side in sides):
            reasons.append(SELECTED_CRITICAL_CORRECTION)
            break

    expected_total = sum(expected_counts.values())
    observed_total = sum(candidate.class_counts.values())
    if expected_total and observed_total == expected_total:
        for name in critical_classes:
            expected = expected_counts.get(name)
            if expected is None:
                continue
            if candidate.class_counts.get(name, 0) != expected:
                reasons.append(SELECTED_CRITICAL_COUNT)
                break

    return tuple(reasons)


def assign_group(
    candidate: Candidate,
    *,
    critical_classes: Sequence[str],
    expected_counts: Mapping[str, int],
) -> tuple[str, tuple[str, ...]]:
    """Put one candidate in a pack group and record why."""
    critical = critical_reasons(
        candidate,
        critical_classes=critical_classes,
        expected_counts=expected_counts,
    )
    if critical:
        return RED_ORANGE_CRITICAL, critical
    if candidate.group == HARD_CASE or candidate.reasons:
        return HARD_CASE, (SELECTED_HARD_EVIDENCE,)
    return REPRESENTATIVE, (SELECTED_ROUTINE,)


# ---------------------------------------------------------------------------
# Diversity


def _feature_vector(candidate: Candidate) -> list[float | None]:
    return [
        _timestamp_seconds(candidate.timestamp),
        candidate.brightness,
        candidate.saturation,
        candidate.blur_score,
        candidate.min_confidence,
        float(candidate.detection_count),
    ]


def _timestamp_seconds(raw: str) -> float | None:
    text = (raw or "").strip()
    if not text:
        return None
    for pattern in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d_%H%M%S"):
        try:
            return (
                datetime.strptime(text[:19], pattern)
                .replace(tzinfo=timezone.utc)
                .timestamp()
            )
        except ValueError:
            continue
    return None


def diversity_select(
    members: Sequence[Candidate], limit: int
) -> list[Candidate]:
    """Choose up to ``limit`` members that are as unlike each other as possible.

    Greedy farthest-point over the measurements already on hand --- capture
    time, brightness, saturation, blur, confidence, box count --- each
    normalised across the cluster so that no single unit dominates. A
    missing measurement contributes nothing rather than being imputed,
    which keeps a cluster with no quality data falling back to spreading
    over time alone.

    The first pick is the most interesting member, not an arbitrary one, so
    a cluster trimmed to a single place still surrenders its best sample.
    """
    if limit <= 0:
        return []
    if len(members) <= limit:
        return list(members)

    ordered = sorted(
        members,
        key=lambda c: (
            -len(c.reasons),
            c.min_confidence if c.min_confidence is not None else 1.0,
            c.sample_id,
        ),
    )
    vectors = [_feature_vector(c) for c in ordered]
    spans: list[tuple[float, float] | None] = []
    for index in range(len(vectors[0])):
        values = [
            value
            for value in (vector[index] for vector in vectors)
            if value is not None
        ]
        if len(values) < 2:
            spans.append(None)
            continue
        low, high = min(values), max(values)
        spans.append(None if math.isclose(low, high) else (low, high))

    def distance(left: int, right: int) -> float:
        total = 0.0
        for index, span in enumerate(spans):
            if span is None:
                continue
            a, b = vectors[left][index], vectors[right][index]
            if a is None or b is None:
                continue
            low, high = span
            total += ((a - b) / (high - low)) ** 2
        return math.sqrt(total)

    chosen = [0]
    while len(chosen) < limit:
        best_index, best_distance = None, -1.0
        for index in range(len(ordered)):
            if index in chosen:
                continue
            nearest = min(distance(index, taken) for taken in chosen)
            if nearest > best_distance:
                best_index, best_distance = index, nearest
        if best_index is None:
            break
        chosen.append(best_index)
    return [ordered[index] for index in chosen]


# ---------------------------------------------------------------------------
# Sizing


def allocate(
    available: Mapping[str, int],
    target: int,
    shares: Mapping[str, float] = DEFAULT_SHARES,
) -> dict[str, int]:
    """Share ``target`` places out between the groups.

    A group that cannot fill its share gives the remainder to the others in
    priority order, so an under-supplied critical group costs the pack
    nothing. No group is padded past what exists: the pack reports being
    short rather than inventing places.
    """
    quota = {
        group: min(available.get(group, 0), int(target * shares.get(group, 0.0)))
        for group in PACK_GROUPS
    }
    spare = target - sum(quota.values())
    for group in PACK_GROUPS:
        if spare <= 0:
            break
        room = available.get(group, 0) - quota[group]
        if room > 0:
            take = min(room, spare)
            quota[group] += take
            spare -= take
    return quota


def _round_robin_by_day(candidates: Sequence[Candidate], limit: int) -> list[Candidate]:
    """Take ``limit`` candidates spread across capture days.

    Taking the head of a priority order would pull a whole pack out of the
    worst afternoon the line ever had, which describes that afternoon
    rather than the station.
    """
    buckets: dict[str, list[Candidate]] = collections.defaultdict(list)
    for candidate in candidates:
        buckets[(candidate.timestamp or "")[:10]].append(candidate)
    for members in buckets.values():
        members.sort(
            key=lambda c: (
                -len(c.reasons),
                c.min_confidence if c.min_confidence is not None else 1.0,
                c.sample_id,
            )
        )

    taken: list[Candidate] = []
    days = sorted(buckets)
    while len(taken) < limit and any(buckets[day] for day in days):
        for day in days:
            if len(taken) >= limit:
                break
            if buckets[day]:
                taken.append(buckets[day].pop(0))
    return taken


# ---------------------------------------------------------------------------
# The pass


def build_review_pack(
    candidates: Sequence[Candidate],
    *,
    trained_source_ids: Iterable[str] = (),
    trained_sha256: Iterable[str] = (),
    perceptually_trained: Iterable[str] = (),
    critical_classes: Sequence[str] = ("Red", "Orange"),
    expected_counts: Mapping[str, int] | None = None,
    target_min: int = 150,
    target_max: int = 250,
    per_cluster: int = 3,
    shares: Mapping[str, float] = DEFAULT_SHARES,
) -> tuple[list[PackEntry], dict[str, Any]]:
    """Assemble the pack. Reads nothing from disk and writes nothing.

    The order is forced by what each step needs from the one before:
    exclusion first, so no effort is spent on images that can never be a
    yardstick; source collapse next, so every later count is in distinct
    captures; grouping before diversity, so a cluster's spread is chosen
    knowing which of its members are critical; sizing last.
    """
    if target_min > target_max:
        raise ReviewPackError(
            f"target_min {target_min} exceeds target_max {target_max}."
        )
    counts = dict(expected_counts or {})

    kept, excluded = exclude_trained(
        candidates,
        trained_source_ids=trained_source_ids,
        trained_sha256=trained_sha256,
        perceptually_trained=perceptually_trained,
    )
    after_exclusion = len(kept)

    sources, collapsed_away = collapse_to_sources(kept)

    grouped: dict[str, tuple[str, tuple[str, ...]]] = {}
    for candidate in sources:
        grouped[candidate.sample_id] = assign_group(
            candidate,
            critical_classes=critical_classes,
            expected_counts=counts,
        )

    # Diversity runs per perceptual cluster, across groups: two look-alike
    # pictures are redundant whether or not they landed in the same group.
    clusters: dict[str, list[Candidate]] = collections.defaultdict(list)
    for candidate in sources:
        clusters[candidate.duplicate_group or candidate.sample_id].append(
            candidate
        )

    diverse: list[Candidate] = []
    cluster_size: dict[str, int] = {}
    diversity_reason: dict[str, str] = {}
    thinned_by_diversity = 0
    for key, members in sorted(clusters.items()):
        picked = diversity_select(members, per_cluster)
        thinned_by_diversity += len(members) - len(picked)
        for candidate in picked:
            cluster_size[candidate.sample_id] = len(members)
            diversity_reason[candidate.sample_id] = (
                SELECTED_CLUSTER_UNIQUE
                if len(members) == 1
                else SELECTED_CLUSTER_DIVERSITY
            )
        diverse.extend(picked)

    by_group: dict[str, list[Candidate]] = collections.defaultdict(list)
    for candidate in diverse:
        by_group[grouped[candidate.sample_id][0]].append(candidate)

    available = {group: len(by_group.get(group, [])) for group in PACK_GROUPS}
    target = min(target_max, sum(available.values()))
    quota = allocate(available, target, shares)

    selected: list[Candidate] = []
    for group in PACK_GROUPS:
        selected.extend(_round_robin_by_day(by_group.get(group, []), quota[group]))

    entries = [
        PackEntry(
            source_image_id=candidate.source_image_id
            or source_image_id(candidate.image_path),
            sample_id=candidate.sample_id,
            image_path=candidate.image_path,
            image_sha256=candidate.image_sha256,
            group=grouped[candidate.sample_id][0],
            selected_reasons=grouped[candidate.sample_id][1]
            + (diversity_reason[candidate.sample_id],),
            evidence_reasons=candidate.reasons,
            detail=candidate.detail,
            timestamp=candidate.timestamp,
            camera_id=candidate.camera_id,
            model_version=candidate.model_version,
            detection_count=candidate.detection_count,
            min_confidence=candidate.min_confidence,
            class_counts=dict(candidate.class_counts),
            corrections=candidate.corrections,
            brightness=candidate.brightness,
            saturation=candidate.saturation,
            blur_score=candidate.blur_score,
            perceptual_group=candidate.duplicate_group,
            cluster_size=cluster_size[candidate.sample_id],
        )
        for candidate in selected
    ]
    entries.sort(key=lambda e: (PACK_GROUPS.index(e.group), e.source_image_id))

    distinct_sources = {entry.source_image_id for entry in entries}
    summary: dict[str, Any] = {
        "schema_version": PACK_SCHEMA_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "candidates_in": len(candidates),
        "after_training_exclusion": after_exclusion,
        "excluded": {key: len(value) for key, value in excluded.items()},
        "distinct_sources_after_collapse": len(sources),
        "collapsed_duplicate_rows": collapsed_away,
        "thinned_by_cluster_diversity": thinned_by_diversity,
        "available_per_group": available,
        "quota_per_group": quota,
        "selected": len(entries),
        "distinct_sources_selected": len(distinct_sources),
        "per_group": {
            group: sum(1 for e in entries if e.group == group)
            for group in PACK_GROUPS
        },
        "target_min": target_min,
        "target_max": target_max,
        "per_cluster": per_cluster,
        "critical_classes": list(critical_classes),
        "expected_class_counts": counts,
        "meets_target_min": len(distinct_sources) >= target_min,
        "all_need_label": all(e.status == NEEDS_LABEL for e in entries),
    }
    if not summary["meets_target_min"]:
        summary["shortfall"] = target_min - len(distinct_sources)
    return entries, summary


def write_pack(
    entries: Sequence[PackEntry],
    summary: Mapping[str, Any],
    destination: str | Path,
    *,
    schema: ClassSchema | None = None,
    copy_images: bool = True,
) -> dict[str, Path]:
    """Write the pack for a person to work through.

    Images are copied by default, and that is not a convenience: production
    applies a retention cleanup that deletes passing images after thirty
    days, and a pack referencing paths that expire mid-review would lose
    the work. Copying reads production and writes only here.
    """
    directory = Path(destination)
    images_dir = directory / "images"
    directory.mkdir(parents=True, exist_ok=True)

    written: list[PackEntry] = []
    missing: list[str] = []
    if copy_images:
        images_dir.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            source = Path(entry.image_path)
            target = images_dir / f"{entry.group}__{entry.source_image_id}{source.suffix}"
            try:
                shutil.copy2(source, target)
            except OSError as exc:
                LOGGER.warning("Could not copy %s: %s", source, exc)
                missing.append(entry.source_image_id)
                written.append(entry)
                continue
            written.append(
                PackEntry(
                    **{
                        **entry.__dict__,
                        "pack_image": target.relative_to(directory).as_posix(),
                    }
                )
            )
    else:
        written = list(entries)

    csv_path = directory / "review_pack.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PACK_COLUMNS))
        writer.writeheader()
        for entry in written:
            writer.writerow(entry.to_row())

    payload = dict(summary)
    payload["images_copied"] = copy_images
    payload["images_not_copied"] = missing
    summary_path = directory / "summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    readme_path = directory / "REVIEW.md"
    readme_path.write_text(_review_text(payload, schema), encoding="utf-8")
    return {"csv": csv_path, "summary": summary_path, "readme": readme_path}


def _review_text(
    summary: Mapping[str, Any], schema: ClassSchema | None
) -> str:
    per_group = summary.get("per_group") or {}
    names = list(schema.names) if schema else []
    mapping = ", ".join(f"{i}={n}" for i, n in enumerate(names))
    return f"""# Review pack --- images to annotate

**This is not a golden set and nothing here is registered.** It is a list of
{summary.get('distinct_sources_selected', 0)} distinct source images chosen
for a person to draw boxes on. Every row's status is `NEEDS_LABEL`, and it
stays `NEEDS_LABEL` until a human has drawn a complete set of boxes: a
production record says what the model *found*, never what was there, so
nothing in here may be promoted to ground truth by copying predictions.

## Class contract

    {mapping}
    hash {schema.schema_hash if schema else ''}

This station expects **six** objects across **five** classes --- Black
appears twice, physically. A complete annotation therefore has six boxes.

## What is in the pile, and why

| Group | Count | What it means |
| --- | --- | --- |
| `{RED_ORANGE_CRITICAL}` | {per_group.get(RED_ORANGE_CRITICAL, 0)} | A person corrected a Red/Orange call, or the verified count for one of them does not match the station. This is the confusion the line actually has. |
| `{HARD_CASE}` | {per_group.get(HARD_CASE, 0)} | Missed or extra detections, low confidence, a human correction, or a quality outlier. |
| `{REPRESENTATIVE}` | {per_group.get(REPRESENTATIVE, 0)} | The line running normally. A yardstick made only of hard cases stops describing the station. |

The `selected_reasons` column says why each individual image is here. Read
it before deciding an image looks boring: `cluster_diversity_pick` means it
was chosen to be *unlike* its look-alikes, and dropping it collapses that
spread.

## How to work through it

1. Open `review_pack.csv`. The `pack_image` column is the copy in
   `images/`; `image_path` is where it came from in production.
2. Annotate all six objects in every image. Partial annotation is worse
   than none here --- a box count that disagrees with the station is what
   marks a label incomplete, and an incomplete label that looks finished
   will be caught only much later.
3. Work `{RED_ORANGE_CRITICAL}` first. It is the smallest group and the
   one carrying the failure mode.
4. When a batch is annotated, keep the images and labels together in the
   YOLO layout (`images/` and `labels/` beside each other) --- that is
   what the golden registration and the evaluator both expect.

## What happens next, and what does not

Registering a golden set is a separate, deliberate human act:
`picture-tool-autotrain golden register <dir> --registered-by <name>`,
optionally with `--groups` pointing at this pack so the
representative/hard_case split travels with it. Nothing registers itself.
No labels were generated automatically, and none will be.

## Exclusions applied before you were shown this

{json.dumps(summary.get('excluded', {}), ensure_ascii=False, indent=2)}

Images the model has already trained on --- including augmented
derivatives of them, matched by source lineage rather than by bytes --- are
removed outright rather than flagged, because annotating one would be
wasted work: it can never be a yardstick for a model that has seen it.
"""

