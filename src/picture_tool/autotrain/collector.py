"""Read-only ingest of production inspection records.

The production inference project already writes one complete traceability
record per inspection --- ``*_config_snapshot.json`` under
``Result/<date>/<product>/<area>/<status>/metadata/<detector>/`` --- and
indexes it into ``Result/inspection_records.sqlite3``. Everything this
subsystem needs about a production inspection is already in those two places,
so the collector reads and never writes, never registers a pipeline step, and
never runs on the inference thread.

That is the whole safety argument: a collector failure cannot affect
production inference, because production does not call the collector.

The one thing not in the record is per-image brightness/saturation/blur;
those are computed here, offline, from images production already saved.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from picture_tool.autotrain.config import AutoTrainConfig
from picture_tool.autotrain.image_quality import ImageQuality, measure_file
from picture_tool.autotrain.paths import AutoTrainPaths

LOGGER = logging.getLogger(__name__)

#: Filename suffix production uses for the per-inspection record.
SNAPSHOT_SUFFIX = "_config_snapshot.json"

#: Highest snapshot schema this reader understands. Older records are read on
#: a best-effort basis; newer ones are read too, since unknown keys are
#: ignored, but the version is carried through so reports can say so.
KNOWN_SCHEMA_VERSION = 2

#: Camera settings live in the embedded merged config rather than at the top
#: level of the record.
_CAMERA_KEYS = ("exposure_time", "gain", "light_brightness")


@dataclass(frozen=True)
class DetectionRecord:
    """One predicted box from a production inspection."""

    class_name: str
    confidence: float | None
    bbox: tuple[float, float, float, float] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": list(self.bbox) if self.bbox is not None else None,
        }


@dataclass(frozen=True)
class ProductionRecord:
    """One production inspection, as far as autonomous training cares.

    Fields map onto the metadata the request asked to preserve: image id,
    timestamp, model version, prediction, class, confidence, bbox, image path,
    camera parameters, and the three offline image-quality scores.
    """

    inspection_id: str
    timestamp: datetime | None
    status: str
    detector: str
    product: str
    area: str
    model_version: str
    model_weights: str
    conf_threshold: float | None
    class_names: tuple[str, ...]
    detections: tuple[DetectionRecord, ...]
    original_path: str
    preprocessed_path: str
    annotated_path: str
    camera_parameters: Mapping[str, Any]
    equipment: Mapping[str, Any]
    fail_reasons: tuple[str, ...]
    config_hash: str
    snapshot_path: str
    schema_version: int
    image_quality: ImageQuality | None = None
    review_outcome: str = ""
    review_label: str = ""
    failure_category: str = ""
    action_route: str = ""

    @property
    def min_confidence(self) -> float | None:
        """Lowest confidence among predictions, or None when there are none."""
        scores = [d.confidence for d in self.detections if d.confidence is not None]
        return min(scores) if scores else None

    @property
    def has_image(self) -> bool:
        """Whether the source image production saved still exists."""
        return bool(self.original_path) and Path(self.original_path).is_file()

    def to_dict(self) -> dict[str, Any]:
        return {
            "inspection_id": self.inspection_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else "",
            "status": self.status,
            "detector": self.detector,
            "product": self.product,
            "area": self.area,
            "model_version": self.model_version,
            "model_weights": self.model_weights,
            "conf_threshold": self.conf_threshold,
            "class_names": list(self.class_names),
            "detections": [d.to_dict() for d in self.detections],
            "original_path": self.original_path,
            "preprocessed_path": self.preprocessed_path,
            "annotated_path": self.annotated_path,
            "camera_parameters": dict(self.camera_parameters),
            "equipment": dict(self.equipment),
            "fail_reasons": list(self.fail_reasons),
            "config_hash": self.config_hash,
            "snapshot_path": self.snapshot_path,
            "schema_version": self.schema_version,
            "image_quality": (
                self.image_quality.to_dict() if self.image_quality else None
            ),
            "review_outcome": self.review_outcome,
            "review_label": self.review_label,
            "failure_category": self.failure_category,
            "action_route": self.action_route,
        }


@dataclass(frozen=True)
class CollectionResult:
    """Records read, plus an honest account of what could not be read."""

    records: tuple[ProductionRecord, ...]
    scanned_files: int
    unreadable: tuple[str, ...] = ()
    missing_images: tuple[str, ...] = ()
    review_lookup_error: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "records": len(self.records),
            "scanned_files": self.scanned_files,
            "unreadable": len(self.unreadable),
            "missing_images": len(self.missing_images),
            "review_lookup_error": self.review_lookup_error,
        }


@dataclass
class _Counters:
    scanned: int = 0
    unreadable: list[str] = field(default_factory=list)
    missing_images: list[str] = field(default_factory=list)


def collect_production_records(
    paths: AutoTrainPaths,
    config: AutoTrainConfig,
    *,
    now: datetime | None = None,
) -> CollectionResult:
    """Read recent production inspections for the configured station.

    Refuses when the feature flag is off, before touching the filesystem.
    """
    config.require_enabled()
    return collect_from_results_root(
        paths.production_results_root(),
        product=config.station.product,
        area=config.station.area,
        lookback_days=config.collector.lookback_days,
        max_records=config.collector.max_records,
        compute_image_quality=config.collector.compute_image_quality,
        database_path=paths.production_database(),
        now=now,
    )


def collect_from_results_root(
    results_root: str | Path,
    *,
    product: str,
    area: str,
    lookback_days: int,
    max_records: int,
    compute_image_quality: bool = True,
    database_path: str | Path | None = None,
    now: datetime | None = None,
) -> CollectionResult:
    """Scan a results tree without needing a full configuration.

    Split out from :func:`collect_production_records` so the scanning logic is
    testable against a fixture tree, and so tools can point it at an archived
    results directory.
    """
    root = Path(results_root)
    counters = _Counters()
    records: list[ProductionRecord] = []

    for snapshot_path in _iter_snapshot_files(root, product, area, lookback_days, now):
        counters.scanned += 1
        record = _read_snapshot(snapshot_path, product=product, area=area, counters=counters)
        if record is None:
            continue
        if not record.has_image:
            counters.missing_images.append(record.inspection_id)
        elif compute_image_quality:
            quality = measure_file(record.original_path)
            if quality is not None:
                record = _with_quality(record, quality)
        records.append(record)
        if len(records) >= max_records:
            LOGGER.info(
                "Collector stopped at max_records=%s; older inspections were not read.",
                max_records,
            )
            break

    review_error = ""
    if database_path is not None:
        records, review_error = _attach_review_outcomes(records, Path(database_path))

    return CollectionResult(
        records=tuple(records),
        scanned_files=counters.scanned,
        unreadable=tuple(counters.unreadable),
        missing_images=tuple(counters.missing_images),
        review_lookup_error=review_error,
    )


# ---------------------------------------------------------------------------
# Scanning


def _iter_snapshot_files(
    results_root: Path,
    product: str,
    area: str,
    lookback_days: int,
    now: datetime | None,
) -> Iterator[Path]:
    """Yield snapshot files newest-date-first, within the lookback window.

    The tree is ``Result/<YYYYMMDD>/<product>/<area>/<status>/metadata/...``,
    so the date filter is applied on directory names before any file is
    opened --- on a station with a year of history that is the difference
    between reading a week and reading everything.
    """
    if not results_root.is_dir():
        LOGGER.warning("Production results root does not exist: %s", results_root)
        return
    reference = (now or datetime.now()).date()
    earliest = reference - timedelta(days=max(0, lookback_days - 1))

    for date_dir in sorted(_iter_dirs(results_root), key=lambda p: p.name, reverse=True):
        day = _parse_date_dir(date_dir.name)
        if day is None or day < earliest or day > reference:
            continue
        station_dir = date_dir / product / area
        if not station_dir.is_dir():
            continue
        for status_dir in _iter_dirs(station_dir):
            metadata_dir = status_dir / "metadata"
            if not metadata_dir.is_dir():
                continue
            for detector_dir in _iter_dirs(metadata_dir):
                yield from sorted(detector_dir.glob(f"*{SNAPSHOT_SUFFIX}"))


def _iter_dirs(root: Path) -> Iterator[Path]:
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        LOGGER.warning("Cannot list %s: %s", root, exc)
        return
    for entry in entries:
        if entry.is_dir():
            yield entry


def _parse_date_dir(name: str):
    try:
        return datetime.strptime(name, "%Y%m%d").date()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Parsing


def _read_snapshot(
    path: Path, *, product: str, area: str, counters: _Counters
) -> ProductionRecord | None:
    """Parse one record, or log and skip it.

    A single corrupt record must not end a collection pass: production keeps
    running while this reads, and a partially-written file is a normal thing
    to encounter.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        LOGGER.warning("Unreadable inspection record %s: %s", path, exc)
        counters.unreadable.append(str(path))
        return None
    if not isinstance(payload, dict):
        LOGGER.warning("Inspection record is not a mapping: %s", path)
        counters.unreadable.append(str(path))
        return None

    try:
        return _build_record(payload, path, product=product, area=area)
    except (TypeError, ValueError, AttributeError) as exc:
        LOGGER.warning("Malformed inspection record %s: %s", path, exc)
        counters.unreadable.append(str(path))
        return None


def _build_record(
    payload: Mapping[str, Any], path: Path, *, product: str, area: str
) -> ProductionRecord:
    model_info = _mapping(payload.get("model_info"))
    artifacts = _mapping(payload.get("artifacts"))
    embedded_config = _mapping(payload.get("config"))

    return ProductionRecord(
        inspection_id=str(payload.get("inspection_id") or path.stem),
        timestamp=_parse_timestamp(payload.get("timestamp")),
        status=str(payload.get("status") or ""),
        detector=str(payload.get("detector") or ""),
        product=str(payload.get("product") or product),
        area=str(payload.get("area") or area),
        model_version=str(model_info.get("model_version") or ""),
        model_weights=str(model_info.get("weights") or ""),
        conf_threshold=_optional_float(model_info.get("conf_thres")),
        class_names=tuple(str(name) for name in _sequence(model_info.get("class_names"))),
        detections=_parse_detections(payload.get("detections")),
        original_path=str(artifacts.get("original_path") or ""),
        preprocessed_path=str(artifacts.get("preprocessed_path") or ""),
        annotated_path=str(artifacts.get("annotated_path") or ""),
        camera_parameters={
            key: embedded_config.get(key)
            for key in _CAMERA_KEYS
            if key in embedded_config
        },
        equipment=dict(_mapping(payload.get("equipment"))),
        fail_reasons=tuple(str(r) for r in _sequence(payload.get("fail_reasons"))),
        config_hash=str(payload.get("config_hash") or ""),
        snapshot_path=str(path),
        schema_version=_optional_int(payload.get("schema_version")) or 0,
    )


def _parse_detections(value: Any) -> tuple[DetectionRecord, ...]:
    """Parse the prediction list, refusing a corrupt one.

    Strict on purpose. Everywhere else this reader is lenient, but "no
    predictions" and "the prediction field is damaged" must not collapse into
    the same record: a low-confidence selector would ignore the damaged one
    and a random selector might pick it, in both cases on the strength of a
    prediction list nobody actually read.
    """
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"detections must be a list, got {type(value).__name__}")

    detections: list[DetectionRecord] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"detection entries must be mappings, got {item!r}")
        detections.append(
            DetectionRecord(
                class_name=str(item.get("class") or item.get("class_name") or ""),
                confidence=_optional_float(item.get("confidence")),
                bbox=_parse_bbox(item.get("bbox") or item.get("box")),
            )
        )
    return tuple(detections)


def _parse_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return (x1, y1, x2, y2)


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, (list, tuple)) else ()


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _with_quality(record: ProductionRecord, quality: ImageQuality) -> ProductionRecord:
    """Return a copy carrying the measured quality (records are frozen)."""
    from dataclasses import replace

    return replace(record, image_quality=quality)


# ---------------------------------------------------------------------------
# Review outcomes, read-only from the production inspection database


def _attach_review_outcomes(
    records: list[ProductionRecord], database_path: Path
) -> tuple[list[ProductionRecord], str]:
    """Enrich records with operator review outcomes, if the database is there.

    Opened read-only through a URI so this cannot write to, lock, or migrate
    the production database. Any failure degrades to "no review information"
    rather than failing the pass: review data improves selection, it is not
    required for it.
    """
    if not records or not database_path.is_file():
        return records, ""

    from dataclasses import replace

    ids = [record.inspection_id for record in records]
    try:
        rows = _query_review_rows(database_path, ids)
    except sqlite3.Error as exc:
        LOGGER.warning("Review outcome lookup failed for %s: %s", database_path, exc)
        return records, str(exc)

    if not rows:
        return records, ""

    enriched: list[ProductionRecord] = []
    for record in records:
        row = rows.get(record.inspection_id)
        if row is None:
            enriched.append(record)
            continue
        # Named explicitly rather than splatted: the review columns are the
        # only fields this may touch, and spelling them out keeps a schema
        # change from silently overwriting something else on the record.
        enriched.append(
            replace(
                record,
                review_outcome=row["review_outcome"],
                review_label=row["review_label"],
                failure_category=row["failure_category"],
                action_route=row["action_route"],
            )
        )
    return enriched, ""


def _query_review_rows(
    database_path: Path, inspection_ids: Iterable[str]
) -> dict[str, dict[str, str]]:
    uri = f"file:{database_path.as_posix()}?mode=ro"
    rows: dict[str, dict[str, str]] = {}
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for chunk in _chunked(list(inspection_ids), 400):
            placeholders = ",".join("?" for _ in chunk)
            cursor = connection.execute(
                "SELECT inspection_id, review_outcome, review_label, "
                "failure_category, action_route FROM inspections "
                f"WHERE inspection_id IN ({placeholders})",
                tuple(chunk),
            )
            for row in cursor.fetchall():
                rows[str(row["inspection_id"])] = {
                    "review_outcome": str(row["review_outcome"] or ""),
                    "review_label": str(row["review_label"] or ""),
                    "failure_category": str(row["failure_category"] or ""),
                    "action_route": str(row["action_route"] or ""),
                }
    return rows


def _chunked(values: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]
