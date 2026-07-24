"""Dry-run analysis for byte-identical images and conflicting YOLO labels."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from picture_tool.utils.io_utils import DEFAULT_IMAGE_EXTS


HUMAN_SOURCE_TYPES = frozenset({"human_annotation", "human_empty"})
SOURCE_RANK = {
    "human_annotation": 3,
    "human_empty": 3,
    "ai_snapshot": 2,
    "legacy_unknown": 1,
}


class OperatorDatasetConflictError(ValueError):
    """Raised when equally authoritative labels disagree for one image."""

    def __init__(self, analysis: "DatasetConflictAnalysis") -> None:
        self.analysis = analysis
        preview = "; ".join(
            f"sha={item.image_sha256} samples={','.join(item.sample_ids)} "
            f"label_sha={','.join(item.label_sha256s)}"
            for item in analysis.conflicts[:5]
        )
        super().__init__(
            "Identical image content has conflicting authoritative labels. "
            "No canonical sample was selected; inspect the dry-run report. "
            + preview
        )


@dataclass(frozen=True)
class DatasetSample:
    """One image/label pair considered by the snapshot preflight."""

    sample_id: str
    image_path: Path
    label_path: Path
    image_sha256: str
    label_sha256: str
    source_type: str
    annotation_status: str


@dataclass(frozen=True)
class CanonicalSelection:
    """Auditable exclusion of one duplicate sample."""

    image_sha256: str
    kept_sample: str
    excluded_sample: str
    reason: str
    kept_label_sha256: str
    excluded_label_sha256: str
    kept_source_type: str
    excluded_source_type: str


@dataclass(frozen=True)
class LabelConflict:
    """One duplicate group that cannot be resolved without operator input."""

    image_sha256: str
    sample_ids: tuple[str, ...]
    label_sha256s: tuple[str, ...]
    source_types: tuple[str, ...]
    image_paths: tuple[str, ...]
    label_paths: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class DatasetConflictAnalysis:
    """Complete dry-run result for a raw YOLO dataset."""

    images_dir: Path
    labels_dir: Path
    manifest_path: Path | None
    sample_count: int
    selections: tuple[CanonicalSelection, ...]
    conflicts: tuple[LabelConflict, ...]
    excluded_image_paths: tuple[Path, ...]
    excluded_label_paths: tuple[Path, ...]

    @property
    def is_safe(self) -> bool:
        return not self.conflicts


def analyze_operator_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    manifest_path: str | Path | None = None,
    *,
    product: str = "",
    area: str = "",
) -> DatasetConflictAnalysis:
    """Analyze duplicate image contents without modifying any source file."""
    images_root = Path(images_dir).expanduser().resolve()
    labels_root = Path(labels_dir).expanduser().resolve()
    manifest = (
        Path(manifest_path).expanduser().resolve() if manifest_path is not None else None
    )
    manifest_rows = _read_manifest_rows(manifest)
    rows_by_name = _index_manifest_rows(manifest_rows)
    samples: list[DatasetSample] = []
    for image_path in sorted(images_root.iterdir() if images_root.is_dir() else ()):
        if (
            not image_path.is_file()
            or image_path.name.startswith(".")
            or image_path.suffix.lower() not in DEFAULT_IMAGE_EXTS
        ):
            continue
        label_path = labels_root / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        row = rows_by_name.get(image_path.name) or rows_by_name.get(image_path.stem) or {}
        image_sha256 = _sha256_file(image_path)
        sample_id = str(row.get("sample_id") or "").strip() or _sample_id_from_stem(
            image_path.stem
        )
        annotation_status = str(row.get("annotation_status") or "").strip().lower()
        samples.append(
            DatasetSample(
                sample_id=sample_id,
                image_path=image_path.resolve(),
                label_path=label_path.resolve(),
                image_sha256=image_sha256,
                label_sha256=_label_sha256(label_path),
                source_type=_source_type(annotation_status),
                annotation_status=annotation_status,
            )
        )

    selections: list[CanonicalSelection] = []
    conflicts: list[LabelConflict] = []
    excluded_images: list[Path] = []
    excluded_labels: list[Path] = []
    grouped: dict[str, list[DatasetSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.image_sha256, []).append(sample)
    for image_sha256, group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        kept, reason = _select_canonical(group, product=product, area=area)
        if kept is None:
            conflicts.append(_build_conflict(image_sha256, group, reason))
            continue
        for excluded in group:
            if excluded is kept:
                continue
            selections.append(
                CanonicalSelection(
                    image_sha256=image_sha256,
                    kept_sample=kept.sample_id,
                    excluded_sample=excluded.sample_id,
                    reason=reason,
                    kept_label_sha256=kept.label_sha256,
                    excluded_label_sha256=excluded.label_sha256,
                    kept_source_type=kept.source_type,
                    excluded_source_type=excluded.source_type,
                )
            )
            excluded_images.append(excluded.image_path)
            excluded_labels.append(excluded.label_path)
    return DatasetConflictAnalysis(
        images_dir=images_root,
        labels_dir=labels_root,
        manifest_path=manifest,
        sample_count=len(samples),
        selections=tuple(selections),
        conflicts=tuple(conflicts),
        excluded_image_paths=tuple(excluded_images),
        excluded_label_paths=tuple(excluded_labels),
    )


def analysis_payload(
    analysis: DatasetConflictAnalysis,
    *,
    scope: str,
    repair_mode: str = "dry_run_only",
) -> dict[str, Any]:
    """Return a stable JSON-serializable report with repair recommendations."""
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "repair_mode": repair_mode,
        "safe": analysis.is_safe,
        "sample_count": analysis.sample_count,
        "images_dir": str(analysis.images_dir),
        "labels_dir": str(analysis.labels_dir),
        "manifest_path": str(analysis.manifest_path or ""),
        "canonical_selections": [asdict(item) for item in analysis.selections],
        "conflicts": [asdict(item) for item in analysis.conflicts],
        "recommended_action": (
            "No mutation required. Canonical exclusions may be applied only to a new "
            "job snapshot and must retain this audit."
            if analysis.is_safe
            else "Do not retry training from this dataset. Review the conflicting "
            "authoritative labels and create an explicit corrected review revision."
        ),
    }


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write one diagnostic report with flush/fsync and atomic replacement."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def filter_manifest_rows(
    source_manifest: str | Path,
    destination_manifest: str | Path,
    excluded_sample_ids: set[str],
    *,
    excluded_output_image_names: set[str] | None = None,
) -> int:
    """Copy a manifest while omitting only audited canonical exclusions."""
    source = Path(source_manifest)
    destination = Path(destination_manifest)
    rows = _read_manifest_rows(source)
    if not rows:
        raise ValueError(f"Review dataset manifest has no rows: {source}")
    fieldnames = list(rows[0])
    excluded_names = excluded_output_image_names or set()

    def is_excluded(row: dict[str, str]) -> bool:
        output_name = Path(str(row.get("output_image") or "")).name
        if output_name and excluded_names:
            return output_name in excluded_names
        return str(row.get("sample_id") or "").strip() in excluded_sample_ids

    retained = [row for row in rows if not is_excluded(row)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retained)
    return len(retained)


def _select_canonical(
    group: list[DatasetSample], *, product: str, area: str
) -> tuple[DatasetSample | None, str]:
    highest_rank = max(SOURCE_RANK.get(item.source_type, 0) for item in group)
    authoritative = [
        item for item in group if SOURCE_RANK.get(item.source_type, 0) == highest_rank
    ]
    authoritative_labels = {item.label_sha256 for item in authoritative}
    if len(authoritative_labels) > 1:
        authority = "human" if highest_rank == 3 else "equally_authoritative"
        return None, f"conflicting_{authority}_labels"

    expected_id = _stable_sample_id(product, area, group[0].image_sha256)
    kept = min(
        authoritative,
        key=lambda item: (
            item.sample_id != expected_id,
            item.sample_id,
            item.image_path.name,
        ),
    )
    all_labels = {item.label_sha256 for item in group}
    non_human_sources = {
        item.source_type for item in group if item.source_type not in HUMAN_SOURCE_TYPES
    }
    if highest_rank == 3 and non_human_sources == {"ai_snapshot"}:
        return kept, (
            "identical_label_prefer_human_over_ai"
            if len(all_labels) == 1
            else "human_annotation_over_ai_snapshot"
        )
    if len(all_labels) == 1:
        return kept, (
            "identical_label_prefer_authoritative_source"
            if len({item.source_type for item in group}) > 1
            else "identical_label_prefer_canonical_sample"
        )
    return None, "conflicting_labels_without_canonical_authority"


def _build_conflict(
    image_sha256: str, group: list[DatasetSample], reason: str
) -> LabelConflict:
    ordered = sorted(group, key=lambda item: (item.sample_id, item.image_path.name))
    return LabelConflict(
        image_sha256=image_sha256,
        sample_ids=tuple(item.sample_id for item in ordered),
        label_sha256s=tuple(item.label_sha256 for item in ordered),
        source_types=tuple(item.source_type for item in ordered),
        image_paths=tuple(str(item.image_path) for item in ordered),
        label_paths=tuple(str(item.label_path) for item in ordered),
        reason=reason,
    )


def _read_manifest_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if not reader.fieldnames:
                raise ValueError(f"Manifest header is missing: {path}")
            return [dict(row) for row in reader]
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"Unable to read dataset manifest {path}: {exc}") from exc


def _index_manifest_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        output_image = Path(str(row.get("output_image") or ""))
        sample_id = str(row.get("sample_id") or "").strip()
        if output_image.name:
            indexed[output_image.name] = row
            indexed[output_image.stem] = row
        if sample_id:
            indexed[sample_id] = row
            indexed[f"review_{sample_id}"] = row
    return indexed


def _source_type(annotation_status: str) -> str:
    return {
        "verified_annotation": "human_annotation",
        "verified_empty": "human_empty",
        "verified_snapshot": "ai_snapshot",
    }.get(annotation_status, "legacy_unknown")


def _sample_id_from_stem(stem: str) -> str:
    return stem[len("review_") :] if stem.startswith("review_") else stem


def _stable_sample_id(product: str, area: str, image_sha256: str) -> str:
    identity = f"{product}\0{area}\0{image_sha256}".encode()
    return hashlib.sha256(identity).hexdigest()[:24]


def _label_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    normalized = "\n".join(
        line.strip() for line in text.splitlines() if line.strip()
    ).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
