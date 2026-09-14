"""Writing out only what was accepted, with a record of how it was made.

Two rules the rest of the pipeline depends on.

**Only AUTO_ACCEPT reaches the dataset.** NEEDS_REVIEW and REJECT are
written to reports so a person can work through them, and are not written
into ``images/``. A pseudo-label that nothing corroborated is worse than a
missing one: it trains a mistake in while looking like data.

**Every label says where it came from.** A year from now the only useful
question about a pseudo-label is "how was this produced", and the answer has
to survive the run that produced it. So each accepted sample carries the
detector and its weight hash, the colour reference it was checked against,
the profile's class-schema hash, the decision and its reasons, and a
timestamp --- beside the label, not only in a log.

The exported directory is laid out the way the existing dataset version
store and evaluator expect, because the point of the exercise is to hand
this to the trainer that already works, not to invent a second one.
"""

from __future__ import annotations

import collections
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from picture_tool.autotrain import AutoTrainError
from picture_tool.bootstrap.auto_label import (
    AUTO_ACCEPT,
    NEEDS_REVIEW,
    REJECT,
    SampleDecision,
    summarise,
)
from picture_tool.bootstrap.profile import ProductProfile

EXPORT_SCHEMA_VERSION = 1
PROVENANCE_FILENAME = "label_provenance.json"
MANIFEST_FILENAME = "bootstrap_manifest.json"


class ExportError(AutoTrainError):
    """Raised when an accepted set cannot be written out."""


@dataclass(frozen=True)
class ExportResult:
    root: Path
    data_yaml: Path
    images: int
    instances: int
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "data_yaml": str(self.data_yaml),
            "images": self.images,
            "instances": self.instances,
            "content_hash": self.content_hash,
        }


def to_yolo_lines(decision: SampleDecision, profile: ProductProfile) -> list[str]:
    """One YOLO line per box, with class ids from the profile's contract.

    Ids come from the schema, never from whatever the detector numbered
    them: two models can agree on names while disagreeing on order, and a
    dataset written against the wrong order is confidently mislabelled in a
    way no later check would catch.
    """
    names = list(profile.class_schema.names)
    lines: list[str] = []
    for item in decision.boxes:
        box = item.box
        if box.class_name not in names:
            raise ExportError(
                f"{decision.sample_id}: {box.class_name!r} is not in the class "
                "contract, so it has no id to write."
            )
        lines.append(
            f"{names.index(box.class_name)} "
            f"{box.cx:.6f} {box.cy:.6f} {box.width:.6f} {box.height:.6f}"
        )
    return lines


def export_accepted(
    decisions: Sequence[SampleDecision],
    destination: str | Path,
    *,
    profile: ProductProfile,
    split: str = "train",
    overwrite: bool = False,
) -> ExportResult:
    """Write the accepted samples as a YOLO dataset the trainer can read."""
    accepted = [item for item in decisions if item.decision == AUTO_ACCEPT]
    if not accepted:
        raise ExportError(
            "Nothing was accepted, so there is no dataset to write. That is a "
            "result, not a failure: the evidence did not corroborate anything."
        )

    root = Path(destination)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise ExportError(
            f"{root} already exists and is not empty; refusing to mix two "
            "bootstrap runs into one dataset."
        )
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    provenance: dict[str, Any] = {}
    instances = 0
    digest = hashlib.sha256()
    for item in sorted(accepted, key=lambda d: d.sample_id):
        source = Path(item.image_path)
        target = images_dir / f"{item.sample_id}{source.suffix}"
        # Copied, never moved: production keeps its originals untouched.
        shutil.copy2(source, target)
        lines = to_yolo_lines(item, profile)
        (labels_dir / f"{item.sample_id}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        instances += len(lines)
        digest.update(item.sample_id.encode("utf-8"))
        for line in lines:
            digest.update(line.encode("utf-8"))
        provenance[item.sample_id] = _label_provenance(item, profile)

    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root.resolve()),
                # Both keys: ultralytics' check_det_dataset raises without
                # them, and pointing train at the same split is safe because
                # the caller decides the real split downstream.
                "train": f"images/{split}",
                "val": f"images/{split}",
                "names": {i: n for i, n in enumerate(profile.class_schema.names)},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / PROVENANCE_FILENAME).write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return ExportResult(
        root=root,
        data_yaml=root / "data.yaml",
        images=len(accepted),
        instances=instances,
        content_hash=digest.hexdigest(),
    )


def _label_provenance(
    decision: SampleDecision, profile: ProductProfile
) -> dict[str, Any]:
    colour = [
        {
            "box": item.box.to_dict(),
            "opinions": [op.to_dict() for op in item.opinions],
            "agreed": item.agreed,
        }
        for item in decision.boxes
    ]
    return {
        "source_image_id": decision.sample_id,
        "source_image_path": decision.image_path,
        "decision": decision.decision,
        "reasons": list(decision.reasons),
        "detail": decision.detail,
        "class_counts": dict(decision.class_counts),
        "evidence": colour,
        "quality": decision.quality.to_dict() if decision.quality else None,
        "profile": profile.to_dict(),
        **dict(decision.provenance),
    }


def write_reports(
    decisions: Sequence[SampleDecision],
    destination: str | Path,
    *,
    profile: ProductProfile,
    batch_manifest: Mapping[str, Any] | None = None,
    export: ExportResult | None = None,
) -> dict[str, Path]:
    """Write the counts, the reasons, and the two piles a person must work."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)

    stats = summarise(decisions)
    payload: dict[str, Any] = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "reference_source": "existing_validated_color_stats",
        "accuracy": {
            "measured": False,
            "reason": (
                "No independent truth set exists. The only human labels for "
                "this station are images the champion trained on, and the "
                "champion is an evidence source here, so measuring against "
                "them would report memorisation as accuracy."
            ),
        },
        "profile": profile.to_dict(),
        "statistics": stats,
        "batch": dict(batch_manifest or {}),
        "export": export.to_dict() if export else None,
    }
    manifest = root / MANIFEST_FILENAME
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    written = {"manifest": manifest}
    for name, wanted in (
        ("needs_review", NEEDS_REVIEW),
        ("rejected", REJECT),
        ("accepted", AUTO_ACCEPT),
    ):
        path = root / f"{name}.json"
        path.write_text(
            json.dumps(
                [item.to_dict() for item in decisions if item.decision == wanted],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        written[name] = path
    return written


def reason_histogram(decisions: Sequence[SampleDecision]) -> dict[str, dict[str, int]]:
    """Which reasons actually drive each outcome, most common first."""
    buckets: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for item in decisions:
        for reason in item.reasons:
            buckets[item.decision][reason] += 1
    return {key: dict(value.most_common()) for key, value in buckets.items()}
