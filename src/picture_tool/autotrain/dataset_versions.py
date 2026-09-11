"""Immutable, lineage-tracked dataset versions.

A dataset version is a write-once snapshot of the labelled images one
training cycle used, plus a record of how it differs from its parent. New
training never edits the dataset a previous cycle trained on, and never
touches the operator workflow's ``raw/`` tree --- it copies out of it.

What a version deliberately does *not* contain is the train/val/test split.
Splitting is the existing :mod:`picture_tool.split.dataset_splitter`'s job and
runs per training run into the cycle's own working directory; writing its
output back here would make an immutable directory mutable. ``lineage.json``
records the split *policy* that was intended, and the cycle records the split
that actually happened, keyed by ``content_id``.

``content_id`` is a sha256 over the sorted sample ids (themselves image
content hashes), so two versions holding the same photos have the same id on
any machine. It is intentionally not the splitter's own ``dataset_id``, which
also folds in split assignment.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from picture_tool.autotrain import AutoTrainError
from picture_tool.dataset_manifest_lock import autotrain_store_lock

LOGGER = logging.getLogger(__name__)

LINEAGE_FILENAME = "lineage.json"
IMAGES_DIRNAME = "images"
LABELS_DIRNAME = "labels"
VERSION_PREFIX = "dataset_v"
VERSION_DIGITS = 3

LINEAGE_SCHEMA_VERSION = 1


class DatasetVersionError(AutoTrainError):
    """Raised when a dataset version is missing, malformed or would mutate."""


@dataclass(frozen=True)
class DatasetVersion:
    """One immutable dataset version and its provenance."""

    version: str
    parent_version: str
    created_at: str
    product: str
    area: str
    sample_ids: tuple[str, ...]
    added_samples: tuple[str, ...]
    removed_samples: tuple[str, ...]
    source: str
    label_source: str
    split_policy: Mapping[str, Any]
    description: str
    content_id: str
    root: Path | None = None
    schema_version: int = LINEAGE_SCHEMA_VERSION
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def images_dir(self) -> Path:
        if self.root is None:
            raise DatasetVersionError(f"{self.version} has no resolved location")
        return self.root / IMAGES_DIRNAME

    @property
    def labels_dir(self) -> Path:
        if self.root is None:
            raise DatasetVersionError(f"{self.version} has no resolved location")
        return self.root / LABELS_DIRNAME

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "parent_version": self.parent_version,
            "created_at": self.created_at,
            "product": self.product,
            "area": self.area,
            "sample_ids": list(self.sample_ids),
            "added_samples": list(self.added_samples),
            "removed_samples": list(self.removed_samples),
            "source": self.source,
            "label_source": self.label_source,
            "split_policy": dict(self.split_policy),
            "description": self.description,
            "content_id": self.content_id,
            "sample_count": len(self.sample_ids),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, root: Path | None = None
    ) -> "DatasetVersion":
        try:
            return cls(
                version=str(payload["version"]),
                parent_version=str(payload.get("parent_version", "")),
                created_at=str(payload.get("created_at", "")),
                product=str(payload.get("product", "")),
                area=str(payload.get("area", "")),
                sample_ids=tuple(str(s) for s in payload.get("sample_ids", [])),
                added_samples=tuple(str(s) for s in payload.get("added_samples", [])),
                removed_samples=tuple(
                    str(s) for s in payload.get("removed_samples", [])
                ),
                source=str(payload.get("source", "")),
                label_source=str(payload.get("label_source", "")),
                split_policy=dict(payload.get("split_policy") or {}),
                description=str(payload.get("description", "")),
                content_id=str(payload.get("content_id", "")),
                root=root,
                schema_version=int(payload.get("schema_version", 0)),
                extra=dict(payload.get("extra") or {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetVersionError(f"Malformed lineage record: {exc}") from exc


@dataclass(frozen=True)
class LabelledSample:
    """One labelled image entering a dataset version."""

    sample_id: str
    image_path: Path
    label_path: Path | None
    origin: str = ""


class DatasetVersionStore:
    """Creates and reads immutable dataset versions for one station."""

    def __init__(self, root: str | Path, *, product: str = "", area: str = "") -> None:
        self.root = Path(root).expanduser().resolve()
        self.product = product
        self.area = area

    # -- reading ----------------------------------------------------------------

    def list_versions(self) -> tuple[str, ...]:
        """Version names in creation order."""
        if not self.root.is_dir():
            return ()
        names = [
            entry.name
            for entry in self.root.iterdir()
            if entry.is_dir() and _version_number(entry.name) is not None
        ]
        return tuple(sorted(names, key=lambda name: _version_number(name) or 0))

    def latest(self) -> DatasetVersion | None:
        """Most recent complete version, or None when the store is empty."""
        for name in reversed(self.list_versions()):
            if (self.root / name / LINEAGE_FILENAME).is_file():
                return self.load(name)
        return None

    def incomplete_versions(self) -> tuple[str, ...]:
        """Version-named directories with no lineage record.

        ``create`` builds in staging and moves the finished version into place
        atomically, so one of these should never exist. When one does,
        something went wrong that a person needs to look at --- numbering past
        it would leave the damage in place and hide it.
        """
        return tuple(
            name
            for name in self.list_versions()
            if not (self.root / name / LINEAGE_FILENAME).is_file()
        )

    def load(self, version: str) -> DatasetVersion:
        """Read one version's lineage record."""
        directory = self.root / version
        lineage_path = directory / LINEAGE_FILENAME
        if not lineage_path.is_file():
            raise DatasetVersionError(f"No such dataset version: {version}")
        try:
            payload = json.loads(lineage_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatasetVersionError(
                f"Unable to read lineage for {version}: {exc}"
            ) from exc
        return DatasetVersion.from_dict(payload, root=directory)

    def next_version_name(self) -> str:
        """Name the version that would come next."""
        versions = self.list_versions()
        highest = _version_number(versions[-1]) if versions else 0
        return f"{VERSION_PREFIX}{(highest or 0) + 1:0{VERSION_DIGITS}d}"

    def history(self) -> tuple[DatasetVersion, ...]:
        """Every version, oldest first, skipping unreadable ones."""
        records: list[DatasetVersion] = []
        for name in self.list_versions():
            try:
                records.append(self.load(name))
            except DatasetVersionError as exc:
                LOGGER.warning("Skipping unreadable dataset version %s: %s", name, exc)
        return tuple(records)

    # -- writing ----------------------------------------------------------------

    def create(
        self,
        samples: Sequence[LabelledSample],
        *,
        source: str,
        label_source: str,
        description: str = "",
        split_policy: Mapping[str, Any] | None = None,
        parent_version: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> DatasetVersion:
        """Materialise a new immutable version from labelled samples.

        Built in a staging directory and moved into place only once complete,
        so an interrupted run cannot leave a half-populated version that a
        later cycle would treat as real.
        """
        if not samples:
            raise DatasetVersionError(
                "Refusing to create an empty dataset version; there is nothing to "
                "train on."
            )
        _assert_unique(samples)

        self.root.mkdir(parents=True, exist_ok=True)
        with autotrain_store_lock(self.root, name="datasets"):
            incomplete = self.incomplete_versions()
            if incomplete:
                raise DatasetVersionError(
                    "Refusing to create a dataset version while these have no "
                    "lineage record: " + ", ".join(incomplete) + ". Dataset "
                    "versions are immutable, so numbering past a damaged one "
                    "would leave it in place; inspect and remove it first."
                )
            version = self.next_version_name()
            destination = self.root / version
            if destination.exists():
                raise DatasetVersionError(
                    f"{version} already exists; dataset versions are immutable."
                )

            parent = (
                self.load(parent_version)
                if parent_version
                else self.latest()
            )
            parent_ids = set(parent.sample_ids) if parent else set()
            sample_ids = tuple(sorted(sample.sample_id for sample in samples))
            added = tuple(sorted(set(sample_ids) - parent_ids))
            removed = tuple(sorted(parent_ids - set(sample_ids)))

            record = DatasetVersion(
                version=version,
                parent_version=parent.version if parent else "",
                created_at=datetime.now(timezone.utc).isoformat(),
                product=self.product,
                area=self.area,
                sample_ids=sample_ids,
                added_samples=added,
                removed_samples=removed,
                source=source,
                label_source=label_source,
                split_policy=dict(split_policy or {}),
                description=description,
                content_id=content_id_for(sample_ids),
                root=destination,
                extra=dict(extra or {}),
            )
            self._materialise(record, samples, destination)
        return record

    def verify(self, version: str) -> tuple[str, ...]:
        """Re-derive a version's content from disk and report any drift.

        Returns the problems found, empty when the version still matches the
        lineage record written when it was created. This is what makes
        "immutable" checkable rather than merely intended.
        """
        record = self.load(version)
        problems: list[str] = []
        directory = self.root / version
        images = directory / IMAGES_DIRNAME

        if not images.is_dir():
            return (f"{version}: images directory is missing",)

        on_disk = {path.stem for path in images.iterdir() if path.is_file()}
        declared = set(record.sample_ids)
        for missing in sorted(declared - on_disk):
            problems.append(f"{version}: declared sample {missing} is missing on disk")
        for unexpected in sorted(on_disk - declared):
            problems.append(f"{version}: undeclared file {unexpected} appeared on disk")

        recomputed = content_id_for(tuple(sorted(declared)))
        if record.content_id and recomputed != record.content_id:
            problems.append(
                f"{version}: content_id no longer matches its lineage record"
            )
        return tuple(problems)

    # -- internals ---------------------------------------------------------------

    def _materialise(
        self,
        record: DatasetVersion,
        samples: Sequence[LabelledSample],
        destination: Path,
    ) -> None:
        staging = Path(
            tempfile.mkdtemp(dir=str(self.root), prefix=f".{record.version}-staging-")
        )
        try:
            images = staging / IMAGES_DIRNAME
            labels = staging / LABELS_DIRNAME
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)

            for sample in samples:
                if not sample.image_path.is_file():
                    raise DatasetVersionError(
                        f"Sample {sample.sample_id} has no image at "
                        f"{sample.image_path}"
                    )
                suffix = sample.image_path.suffix.lower() or ".jpg"
                shutil.copy2(sample.image_path, images / f"{sample.sample_id}{suffix}")
                if sample.label_path is not None:
                    if not sample.label_path.is_file():
                        raise DatasetVersionError(
                            f"Sample {sample.sample_id} has no label at "
                            f"{sample.label_path}"
                        )
                    shutil.copy2(
                        sample.label_path, labels / f"{sample.sample_id}.txt"
                    )
                else:
                    # A verified negative: no objects, which YOLO expresses as
                    # an empty label file. It must be explicit, never implied
                    # by an absent file.
                    (labels / f"{sample.sample_id}.txt").write_text("", encoding="utf-8")

            _write_json_atomic(staging / LINEAGE_FILENAME, record.to_dict())
            _make_read_only(staging)
            os.replace(staging, destination)
        except (OSError, DatasetVersionError):
            shutil.rmtree(staging, ignore_errors=True)
            raise


# ---------------------------------------------------------------------------


def content_id_for(sample_ids: Iterable[str]) -> str:
    """Stable id for a set of samples, independent of paths and machines."""
    digest = hashlib.sha256()
    for sample_id in sorted(set(sample_ids)):
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _assert_unique(samples: Sequence[LabelledSample]) -> None:
    seen: set[str] = set()
    for sample in samples:
        if sample.sample_id in seen:
            raise DatasetVersionError(
                f"Duplicate sample id in dataset version input: {sample.sample_id}"
            )
        seen.add(sample.sample_id)


def _version_number(name: str) -> int | None:
    if not name.startswith(VERSION_PREFIX):
        return None
    try:
        return int(name[len(VERSION_PREFIX) :])
    except ValueError:
        return None


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".lineage-")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            try:
                os.remove(temporary)
            except OSError:  # pragma: no cover - best effort cleanup
                LOGGER.debug("Could not remove temporary lineage file %s", temporary)


def _make_read_only(root: Path) -> None:
    """Clear the write bit on a finished version.

    Advisory rather than airtight --- an administrator can always chmod it
    back --- but it turns an accidental overwrite into an error instead of a
    silent mutation, which is the failure mode worth catching.
    """
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            try:
                path.chmod(path.stat().st_mode & 0o555)
            except OSError:  # pragma: no cover - platform dependent
                LOGGER.debug("Could not mark %s read-only", path)
