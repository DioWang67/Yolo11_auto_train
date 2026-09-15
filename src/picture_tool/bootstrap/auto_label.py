"""Turning evidence into a decision, and saying why.

The output is never a single number. A confidence score compresses away the
one thing a reviewer needs --- *which* check was unhappy --- and a pipeline
that only sees the number cannot tell "the detector was unsure" from "the
detector and the pixels named different colours", which call for opposite
responses.

The rule that shapes everything here: **a detector may not accept its own
output.** Every AUTO_ACCEPT needs an independent source to have looked at
the same box and agreed. Without that, the bootstrapper is a machine for
copying the champion's mistakes into the data its successor learns from, and
the mistakes it copies are exactly the ones it is most confident about.

Acceptance is therefore conjunctive and deliberately mean. Anything unclear
is NEEDS_REVIEW, which costs a person some minutes; a wrong pseudo-label
costs a model, silently, later.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from picture_tool.bootstrap.evidence import Box, BoxOpinion
from picture_tool.bootstrap.profile import ProductProfile
from picture_tool.bootstrap.vision_evidence import SOURCE_NAME as VISION_SOURCE
from picture_tool.bootstrap.sample_quality import (
    QUALITY_FAIL,
    QUALITY_SUSPECT,
    REASON_EXACT_DUPLICATE,
    REASON_NEAR_DUPLICATE,
    SampleQuality,
)

AUTO_ACCEPT = "AUTO_ACCEPT"
NEEDS_REVIEW = "NEEDS_REVIEW"
REJECT = "REJECT"

# -- reasons ---------------------------------------------------------------

R_QUALITY_FAIL = "quality_failed"
R_NO_DETECTIONS = "no_detections"
R_IMPOSSIBLE_GEOMETRY = "impossible_geometry"
R_UNKNOWN_CLASS = "class_outside_contract"
R_COUNT_MISMATCH = "class_count_mismatch"
R_TOTAL_MISMATCH = "object_count_mismatch"
R_LOW_DETECTOR_CONFIDENCE = "low_detector_confidence"
R_MODERATE_DETECTOR_CONFIDENCE = "moderate_detector_confidence"
R_NO_INDEPENDENT_EVIDENCE = "no_independent_evidence"
R_COLOUR_DISAGREEMENT = "detector_colour_disagreement"
R_CRITICAL_DISAGREEMENT = "critical_pair_disagreement"
R_COLOUR_AMBIGUOUS = "colour_evidence_ambiguous"
R_VISION_DISAGREEMENT = "detector_vision_disagreement"
R_INDEPENDENT_CONSENSUS = "independent_sources_agree_against_detector"
R_NEAR_DUPLICATE = "near_duplicate_unresolved"
R_EXACT_DUPLICATE = "exact_duplicate"
R_ALL_CHECKS_AGREE = "all_evidence_agrees"


@dataclass(frozen=True)
class DecisionThresholds:
    """Where acceptance stops.

    ``min_accept_confidence`` sits well above the proposal threshold: a box
    is worth *looking* at from 0.25, and worth trusting unreviewed from a
    good deal higher.

    ``min_colour_margin`` is the gap between the best and second-best colour
    score. Equal-ish scores mean the pixels did not decide, and a colour
    source that says "Red 0.51, Orange 0.49" has not corroborated anything ---
    treating its argmax as agreement would manufacture the independence this
    whole design depends on.
    """

    min_accept_confidence: float = 0.60
    review_confidence: float = 0.35
    min_colour_confidence: float = 0.25
    min_colour_margin: float = 0.10
    min_box_side: float = 0.005
    max_box_side: float = 0.95


@dataclass(frozen=True)
class BoxDecision:
    """One box, every source's reading of it, and whether they agreed."""

    box: Box
    opinions: tuple[BoxOpinion, ...]
    agreed: bool
    disagreement: str = ""
    critical: bool = False
    #: Set when every independent source that had an opinion disagreed with
    #: the detector *and* named the same class. That is a different thing
    #: from one source objecting, and the difference is what a reviewer
    #: needs: it is a proposed answer, not merely a doubt.
    consensus_class: str = ""

    def opinion(self, source: str) -> BoxOpinion | None:
        for item in self.opinions:
            if item.source == source:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": self.box.to_dict(),
            "opinions": [item.to_dict() for item in self.opinions],
            "agreed": self.agreed,
            "disagreement": self.disagreement,
            "critical": self.critical,
            "consensus_class": self.consensus_class,
        }


@dataclass(frozen=True)
class SampleDecision:
    """What the bootstrapper concluded about one image, and on what grounds."""

    sample_id: str
    image_path: str
    decision: str
    reasons: tuple[str, ...]
    detail: str
    boxes: tuple[BoxDecision, ...] = ()
    quality: SampleQuality | None = None
    class_counts: Mapping[str, int] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_accepted(self) -> bool:
        return self.decision == AUTO_ACCEPT

    @property
    def disagreement_count(self) -> int:
        return sum(1 for box in self.boxes if not box.agreed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "decision": self.decision,
            "reasons": list(self.reasons),
            "detail": self.detail,
            "class_counts": dict(self.class_counts),
            "disagreements": self.disagreement_count,
            "boxes": [box.to_dict() for box in self.boxes],
            "quality": self.quality.to_dict() if self.quality else None,
            "provenance": dict(self.provenance),
        }


def decide(
    *,
    sample_id: str,
    image_path: str,
    boxes: Sequence[Box],
    opinions_by_source: Mapping[str, Sequence[BoxOpinion]],
    quality: SampleQuality,
    profile: ProductProfile,
    thresholds: DecisionThresholds | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> SampleDecision:
    """Weigh everything known about one image and say what to do with it."""
    limits = thresholds or DecisionThresholds()
    reasons: list[str] = []
    details: list[str] = []

    # -- refusals, in order of how completely they end the question --------
    if quality.status == QUALITY_FAIL:
        return _reject(
            sample_id,
            image_path,
            [R_QUALITY_FAIL, *quality.reasons],
            quality.detail or "the frame is not readable",
            quality,
            provenance,
        )
    if REASON_EXACT_DUPLICATE in quality.reasons:
        return _reject(
            sample_id,
            image_path,
            [R_EXACT_DUPLICATE],
            f"byte-identical to {quality.duplicate_of}; one copy is one sample",
            quality,
            provenance,
        )
    if not boxes:
        return _reject(
            sample_id,
            image_path,
            [R_NO_DETECTIONS],
            "nothing was detected, so there is no label to draw",
            quality,
            provenance,
        )

    decisions = _weigh_boxes(boxes, opinions_by_source, profile, limits)

    impossible = [
        box
        for box in decisions
        if not _geometry_is_sane(box.box, limits)
    ]
    if impossible:
        return _reject(
            sample_id,
            image_path,
            [R_IMPOSSIBLE_GEOMETRY],
            f"{len(impossible)} box(es) have degenerate or oversized geometry",
            quality,
            provenance,
            decisions,
        )

    unknown = sorted(
        {
            box.box.class_name
            for box in decisions
            if box.box.class_name not in profile.class_schema.names
        }
    )
    if unknown:
        return _reject(
            sample_id,
            image_path,
            [R_UNKNOWN_CLASS],
            f"detector produced {unknown}, which the class contract excludes",
            quality,
            provenance,
            decisions,
        )

    counts = collections.Counter(box.box.class_name for box in decisions)

    # -- everything below is a reason to hesitate, not to refuse ----------
    if profile.expected_total and sum(counts.values()) != profile.expected_total:
        reasons.append(R_TOTAL_MISMATCH)
        details.append(
            f"{sum(counts.values())} objects where the station expects "
            f"{profile.expected_total}"
        )
    mismatched = [
        f"{name} {counts.get(name, 0)}/{expected}"
        for name, expected in sorted(profile.expected_counts.items())
        if counts.get(name, 0) != expected
    ]
    if mismatched:
        reasons.append(R_COUNT_MISMATCH)
        details.append("; ".join(mismatched))

    confidences = [box.box.confidence for box in decisions]
    lowest = min(confidences) if confidences else 0.0
    if lowest < limits.review_confidence:
        reasons.append(R_LOW_DETECTOR_CONFIDENCE)
        details.append(f"weakest detection {lowest:.3f}")
    elif lowest < limits.min_accept_confidence:
        reasons.append(R_MODERATE_DETECTOR_CONFIDENCE)
        details.append(f"weakest detection {lowest:.3f}")

    # The rule this module exists for. A box nobody but the detector looked
    # at has no corroboration, whatever its score.
    uncorroborated = [box for box in decisions if len(box.opinions) < 2]
    if uncorroborated:
        reasons.append(R_NO_INDEPENDENT_EVIDENCE)
        details.append(
            f"{len(uncorroborated)} box(es) carry only the detector's own word"
        )

    ambiguous = [box for box in decisions if box.disagreement == R_COLOUR_AMBIGUOUS]
    if ambiguous:
        reasons.append(R_COLOUR_AMBIGUOUS)
        details.append(f"{len(ambiguous)} box(es) the pixels could not decide")

    critical = [box for box in decisions if box.critical]
    if critical:
        reasons.append(R_CRITICAL_DISAGREEMENT)
        details.append(
            ", ".join(
                f"detector {box.box.class_name} vs colour "
                f"{(box.opinion('colour') or BoxOpinion('', '', 0)).class_name}"
                for box in critical
            )
        )
    # Reported before the plainer disagreements, because it is the strongest
    # thing the evidence can say: two sources that never saw each other's
    # answer produced the same one, and it is not the detector's.
    consensus = [box for box in decisions if box.consensus_class]
    if consensus:
        reasons.append(R_INDEPENDENT_CONSENSUS)
        details.append(
            ", ".join(
                f"colour and vision both read {box.consensus_class} where the "
                f"detector read {box.box.class_name}"
                for box in consensus
            )
        )

    vision_only = [
        box
        for box in decisions
        if box.disagreement == R_VISION_DISAGREEMENT and not box.consensus_class
    ]
    if vision_only:
        reasons.append(R_VISION_DISAGREEMENT)
        details.append(f"{len(vision_only)} box(es) the vision model read differently")

    plain_disagreements = [
        box
        for box in decisions
        if not box.agreed
        and not box.critical
        and box.disagreement not in (R_COLOUR_AMBIGUOUS, R_VISION_DISAGREEMENT)
    ]
    if plain_disagreements:
        reasons.append(R_COLOUR_DISAGREEMENT)
        details.append(f"{len(plain_disagreements)} box(es) disagreed on class")

    if REASON_NEAR_DUPLICATE in quality.reasons:
        reasons.append(R_NEAR_DUPLICATE)
        details.append(f"looks like {quality.duplicate_of}")
    if quality.status == QUALITY_SUSPECT and quality.detail:
        details.append(quality.detail)

    decision = NEEDS_REVIEW if reasons else AUTO_ACCEPT
    if not reasons:
        reasons.append(R_ALL_CHECKS_AGREE)
        details.append(
            f"{len(decisions)} boxes, every one corroborated, counts match the "
            "station"
        )
    return SampleDecision(
        sample_id=sample_id,
        image_path=image_path,
        decision=decision,
        reasons=tuple(reasons),
        detail="; ".join(details),
        boxes=tuple(decisions),
        quality=quality,
        class_counts=dict(counts),
        provenance=_provenance(profile, provenance),
    )


def _weigh_boxes(
    boxes: Sequence[Box],
    opinions_by_source: Mapping[str, Sequence[BoxOpinion]],
    profile: ProductProfile,
    limits: DecisionThresholds,
) -> tuple[BoxDecision, ...]:
    results: list[BoxDecision] = []
    for index, box in enumerate(boxes):
        opinions = tuple(
            readings[index]
            for readings in opinions_by_source.values()
            if index < len(readings)
        )
        colour = next((o for o in opinions if o.source == "colour"), None)
        vision = next((o for o in opinions if o.source == VISION_SOURCE), None)
        agreed = True
        disagreement = ""
        critical = False
        consensus = ""
        if colour is not None:
            margin = _margin(colour.scores)
            if (
                not colour.class_name
                or colour.confidence < limits.min_colour_confidence
                or margin < limits.min_colour_margin
            ):
                agreed = False
                disagreement = R_COLOUR_AMBIGUOUS
            elif colour.class_name != box.class_name:
                agreed = False
                disagreement = R_COLOUR_DISAGREEMENT
                critical = profile.is_confusable(box.class_name, colour.class_name)
        # The vision model can only add doubt, never remove it. Its accuracy
        # against ground truth is unmeasured, and AUTO_ACCEPT is reached by
        # having no reasons at all, so letting it clear an objection would
        # loosen a gate on the strength of evidence nobody has checked.
        if vision is not None and vision.class_name:
            if vision.class_name != box.class_name:
                agreed = False
                disagreement = disagreement or R_VISION_DISAGREEMENT
                critical = critical or profile.is_confusable(
                    box.class_name, vision.class_name
                )
                if colour is not None and colour.class_name == vision.class_name:
                    consensus = vision.class_name
        results.append(
            BoxDecision(
                box=box,
                opinions=opinions,
                agreed=agreed,
                disagreement=disagreement,
                critical=critical,
                consensus_class=consensus,
            )
        )
    return tuple(results)


def _margin(scores: Mapping[str, float]) -> float:
    if len(scores) < 2:
        return 1.0 if scores else 0.0
    ordered = sorted(scores.values(), reverse=True)
    return ordered[0] - ordered[1]


def _geometry_is_sane(box: Box, limits: DecisionThresholds) -> bool:
    if box.width <= limits.min_box_side or box.height <= limits.min_box_side:
        return False
    if box.width > limits.max_box_side or box.height > limits.max_box_side:
        return False
    return all(0.0 <= value <= 1.0 for value in (box.cx, box.cy))


def _reject(
    sample_id: str,
    image_path: str,
    reasons: Sequence[str],
    detail: str,
    quality: SampleQuality,
    provenance: Mapping[str, Any] | None,
    boxes: Sequence[BoxDecision] = (),
) -> SampleDecision:
    return SampleDecision(
        sample_id=sample_id,
        image_path=image_path,
        decision=REJECT,
        reasons=tuple(dict.fromkeys(reasons)),
        detail=detail,
        boxes=tuple(boxes),
        quality=quality,
        provenance=dict(provenance or {}),
    )


def _provenance(
    profile: ProductProfile, extra: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Enough to answer "how was this label produced" a year from now."""
    record = {
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "product": profile.product,
        "area": profile.area,
        "class_schema_hash": profile.class_schema.schema_hash,
        "expected_counts": dict(profile.expected_counts),
        "reference_source": "existing_validated_color_stats",
        "colour_stats_path": (
            str(profile.colour_stats_path) if profile.colour_stats_path else ""
        ),
    }
    record.update(dict(extra or {}))
    return record


def summarise(decisions: Sequence[SampleDecision]) -> dict[str, Any]:
    """Counts and reasons, which is what a person reads first."""
    by_decision = collections.Counter(item.decision for item in decisions)
    reasons: dict[str, collections.Counter[str]] = {
        AUTO_ACCEPT: collections.Counter(),
        NEEDS_REVIEW: collections.Counter(),
        REJECT: collections.Counter(),
    }
    per_class: collections.Counter[str] = collections.Counter()
    disagreements: collections.Counter[str] = collections.Counter()
    critical_pairs: collections.Counter[str] = collections.Counter()
    ambiguous = 0

    for item in decisions:
        for reason in item.reasons:
            reasons.setdefault(item.decision, collections.Counter())[reason] += 1
        per_class.update(item.class_counts)
        for box in item.boxes:
            if box.disagreement == R_COLOUR_AMBIGUOUS:
                ambiguous += 1
            elif not box.agreed:
                colour = box.opinion("colour")
                disagreements[box.box.class_name] += 1
                if box.critical and colour is not None:
                    critical_pairs[
                        f"detector_{box.box.class_name}__colour_{colour.class_name}"
                    ] += 1

    return {
        "samples": len(decisions),
        "by_decision": {key: by_decision.get(key, 0) for key in
                        (AUTO_ACCEPT, NEEDS_REVIEW, REJECT)},
        "reasons": {key: dict(value.most_common()) for key, value in reasons.items()},
        "proposed_instances_per_class": dict(sorted(per_class.items())),
        "disagreements_per_detector_class": dict(sorted(disagreements.items())),
        "critical_pair_disagreements": dict(sorted(critical_pairs.items())),
        "colour_ambiguous_boxes": ambiguous,
        "samples_with_any_disagreement": sum(
            1 for item in decisions if item.disagreement_count
        ),
    }
