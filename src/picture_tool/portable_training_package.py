"""Verify and import a portable operator retraining package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping

import yaml

from picture_tool.dataset_manifest_lock import (
    DatasetManifestLockTimeoutError,
    dataset_manifest_lock,
    portable_import_lock,
)
from picture_tool.gui.operator_handoff import (
    OperatorHandoffError,
    _class_schema_hash,
    _load_training_options,
    load_operator_handoff,
)

PACKAGE_SCHEMA_VERSION = 1
PACKAGE_METADATA_NAME = "package.json"
MAX_PACKAGE_FILES = 100_000
MAX_PACKAGE_BYTES = 100 * 1024 * 1024 * 1024
SUPPORTED_SOURCE_HANDOFF_SCHEMAS = frozenset({3, 4, 5, 6})
PORTABLE_IMPORT_RECEIPT_SCHEMA_VERSION = 3
SUPPORTED_PORTABLE_IMPORT_RECEIPT_SCHEMAS = frozenset({1, 2, 3})
_RECEIPT_PREPARED = "prepared"
_RECEIPT_COMMITTED = "committed"
_LOGGER = logging.getLogger(__name__)


class PortableTrainingImportError(ValueError):
    """Raised when a portable package is invalid, unsafe, or conflicting."""


@dataclass(frozen=True)
class PortableTrainingImportReport:
    """Result of importing one portable operator job."""

    package_id: str
    handoff_path: Path
    product: str
    area: str
    ready_count: int
    pending_count: int
    reused_existing: bool = False


@dataclass(frozen=True)
class _OpenPackageIdentity:
    """Identity and bytes of the single package handle used for this import."""

    device: int
    inode: int
    size: int
    modified_ns: int
    sha256: str


@dataclass(frozen=True)
class _FileCopyPlan:
    source: Path
    destination: Path
    source_sha256: str
    write_required: bool


@dataclass(frozen=True)
class _CsvWritePlan:
    destination: Path
    rows: tuple[dict[str, str], ...]
    expected_destination_sha256: str | None


@dataclass(frozen=True)
class _SharedImportPlan:
    copies: tuple[_FileCopyPlan, ...]
    csv_writes: tuple[_CsvWritePlan, ...]


@dataclass(frozen=True)
class _MutationBackup:
    destination: Path
    backup: Path | None


class _ImportMutationJournal:
    """Restore every file changed before the final latest-pointer commit."""

    def __init__(self, data_root: Path) -> None:
        rollback_parent = data_root / ".portable_imports"
        rollback_parent.mkdir(parents=True, exist_ok=True)
        self._data_root = data_root.resolve()
        self._backup_root = Path(
            tempfile.mkdtemp(prefix=".rollback-", dir=rollback_parent)
        ).resolve()
        self._backups: list[_MutationBackup] = []
        self._recorded: set[Path] = set()
        self._closed = False

    def record_before_write(self, destination: Path) -> None:
        resolved = destination.resolve(strict=False)
        if not resolved.is_relative_to(self._data_root):
            raise PortableTrainingImportError(
                f"Portable import destination escapes training data: {destination}"
            )
        if resolved in self._recorded:
            return
        self._recorded.add(resolved)
        backup: Path | None = None
        if destination.is_symlink():
            raise PortableTrainingImportError(
                f"Portable import destination cannot be a symbolic link: {destination}"
            )
        if destination.exists():
            if not destination.is_file():
                raise PortableTrainingImportError(
                    f"Portable import destination is not a file: {destination}"
                )
            backup = self._backup_root / f"{len(self._backups):08d}.bak"
            shutil.copy2(destination, backup)
        self._backups.append(_MutationBackup(destination, backup))

    def rollback(self) -> None:
        if self._closed:
            return
        failures: list[str] = []
        for mutation in reversed(self._backups):
            try:
                if mutation.backup is None:
                    mutation.destination.unlink(missing_ok=True)
                else:
                    mutation.destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = mutation.destination.with_name(
                        f".{mutation.destination.name}.{uuid.uuid4().hex}.rollback"
                    )
                    try:
                        shutil.copy2(mutation.backup, temporary)
                        temporary.replace(mutation.destination)
                    finally:
                        temporary.unlink(missing_ok=True)
                _remove_empty_parents(
                    mutation.destination.parent,
                    stop=self._data_root,
                )
            except OSError as exc:
                failures.append(f"{mutation.destination}: {exc}")
        if failures:
            raise PortableTrainingImportError(
                "Portable import rollback was incomplete; recovery evidence was "
                f"retained at {self._backup_root}. Failures: {'; '.join(failures)}"
            )
        self._closed = True
        shutil.rmtree(self._backup_root, ignore_errors=True)

    def commit(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            shutil.rmtree(self._backup_root)
        except OSError as exc:
            _LOGGER.warning(
                "Portable import rollback evidence cleanup was deferred: path=%s "
                "error=%s",
                self._backup_root,
                exc,
            )


def import_portable_training_package(
    package_path: str | Path,
    training_root: str | Path,
) -> PortableTrainingImportReport:
    """Verify a package, merge its data, and create a local immutable handoff."""
    package = Path(package_path).expanduser().resolve()
    root = Path(training_root).expanduser().resolve()
    if not package.is_file():
        raise PortableTrainingImportError(f"Training package not found: {package}")
    data_root = (root / "data").resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    with _portable_import_lock(data_root), ExitStack() as manifest_locks:
        try:
            package_handle = package.open("rb")
        except OSError as exc:
            raise PortableTrainingImportError(
                f"Unable to open training package: {package}"
            ) from exc
        with package_handle:
            package_identity = _capture_open_package_identity(package_handle)
            package_sha256 = package_identity.sha256
            with zipfile.ZipFile(package_handle) as archive:
                metadata, inventory = _validate_archive(archive)
                package_id = _safe_segment(metadata.get("package_id"), "package_id")
                product = _safe_segment(metadata.get("product"), "product")
                area = _safe_segment(metadata.get("area"), "area")
                job_id = (
                    "portable-"
                    f"{hashlib.sha256(package_id.encode()).hexdigest()[:24]}"
                )
                job_dir = data_root / ".operator_handoff" / "jobs" / job_id
                handoff_path = job_dir / "handoff.json"
                status_path = job_dir / "status.json"
                latest_path = data_root / ".operator_handoff" / "latest.json"
                receipt_dir = data_root / ".portable_imports" / package_id
                receipt_path = receipt_dir / "import.json"
                if receipt_path.exists() or receipt_path.is_symlink():
                    _assert_open_package_unchanged(
                        package_handle,
                        package,
                        package_identity,
                    )
                    return _validate_existing_portable_import(
                        receipt_path=receipt_path,
                        expected_handoff_path=handoff_path,
                        expected_status_path=status_path,
                        latest_path=latest_path,
                        training_root=root,
                        package_id=package_id,
                        package_sha256=package_sha256,
                        product=product,
                        area=area,
                    )

                imports_root = data_root / ".portable_imports"
                imports_root.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    prefix=f".{package_id}-", dir=imports_root
                ) as temporary_directory:
                    staging = Path(temporary_directory)
                    _extract_verified_files(archive, inventory, staging)
                    dataset_source = _resolved_staging_path(
                        staging, metadata.get("dataset_path"), "dataset_path"
                    )
                    models_source = _resolved_staging_path(
                        staging, metadata.get("models_path"), "models_path"
                    )
                    handoff_source = _resolved_staging_path(
                        staging, metadata.get("handoff_path"), "handoff_path"
                    )
                    if not dataset_source.is_dir() or not models_source.is_dir():
                        raise PortableTrainingImportError(
                            "The portable dataset or model directory is missing."
                        )
                    original_handoff = _read_json_mapping(handoff_source)
                    (
                        normalized_training_options,
                        source_sample_ids,
                        source_pending_sample_ids,
                    ) = _validate_source_contract(
                        original_handoff,
                        metadata,
                        product=product,
                        area=area,
                    )
                    (
                        source_runtime_weight,
                        portable_runtime_weight,
                    ) = _rewrite_portable_runtime_config(
                        models_source,
                        product=product,
                        area=area,
                    )
                    _validate_staged_sample_contract(
                        dataset_source,
                        sample_ids=source_sample_ids,
                        pending_sample_ids=source_pending_sample_ids,
                    )
                    _assert_open_package_unchanged(
                        package_handle,
                        package,
                        package_identity,
                    )

                    dataset_destination = data_root / product / area
                    try:
                        manifest_locks.enter_context(
                            dataset_manifest_lock(dataset_destination)
                        )
                    except DatasetManifestLockTimeoutError as exc:
                        raise PortableTrainingImportError(
                            "This product/station dataset is already being updated."
                        ) from exc
                    portable_models = (
                        data_root / ".portable_models" / package_id / "models"
                    )
                    if job_dir.exists() or job_dir.is_symlink():
                        raise PortableTrainingImportError(
                            "A portable job exists without an import receipt; "
                            "operator reconciliation is required."
                        )
                    shared_plan = _preflight_shared_import(
                        dataset_source=dataset_source,
                        dataset_destination=dataset_destination,
                        models_source=models_source,
                        models_destination=portable_models,
                        package_id=package_id,
                        data_root=data_root,
                    )
                    journal = _ImportMutationJournal(data_root)
                    prepared_receipt_persisted = False
                    try:
                        _apply_shared_import_plan(shared_plan, journal)

                        sample_ids = list(source_sample_ids)
                        expected_pending = set(source_pending_sample_ids)
                        ready_ids = _manifest_ids(
                            dataset_destination
                            / "metadata"
                            / "review_dataset_manifest.csv"
                        )
                        pending_ids = _manifest_ids(
                            dataset_destination / "review_pending" / "manifest.csv"
                        )
                        job_pending_ids = sorted(expected_pending & pending_ids)
                        job_ready_count = len(set(sample_ids) & ready_ids)
                        total_ready_count = len(ready_ids)
                        local_handoff = dict(original_handoff)
                        # Source-machine roots are not part of the portable payload.
                        local_handoff.pop("inference_station_data_dir", None)
                        local_handoff.pop("inference_project_root", None)
                        local_handoff.update(
                            {
                                "schema_version": 4,
                                "job_id": job_id,
                                "source_package_id": package_id,
                                "source_package_sha256": package_sha256,
                                "portable_source_runtime_weight": (
                                    source_runtime_weight
                                ),
                                "portable_runtime_weight": portable_runtime_weight,
                                "training_options": normalized_training_options,
                                "data_root": str(data_root),
                                "status_path": str(status_path.resolve()),
                                "inference_models_dir": str(
                                    portable_models.resolve()
                                ),
                                "source_manifest": str(receipt_path.resolve()),
                                "ready_count": job_ready_count,
                                "total_ready_count": total_ready_count,
                                "pending_count": len(job_pending_ids),
                            }
                        )
                        original_targets = local_handoff.get("targets")
                        if (
                            not isinstance(original_targets, list)
                            or len(original_targets) != 1
                        ):
                            raise PortableTrainingImportError(
                                "The source handoff must contain exactly one target."
                            )
                        local_target = dict(original_targets[0])
                        local_target.update(
                            {
                                "product": product,
                                "area": area,
                                "dataset_root": str(dataset_destination.resolve()),
                                "ready_count": job_ready_count,
                                "total_ready_count": total_ready_count,
                                "pending_count": len(job_pending_ids),
                                "sample_ids": sample_ids,
                                "pending_sample_ids": list(
                                    source_pending_sample_ids
                                ),
                            }
                        )
                        local_handoff["targets"] = [local_target]
                        _write_json_atomic(handoff_path, local_handoff)
                        try:
                            load_operator_handoff(handoff_path, training_root=root)
                        except OperatorHandoffError as exc:
                            raise PortableTrainingImportError(
                                f"Imported operator handoff is invalid: {exc}"
                            ) from exc

                        initial_state = (
                            "waiting_annotation" if job_pending_ids else "queued"
                        )
                        status_payload = {
                            "schema_version": 1,
                            "job_id": job_id,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "product": product,
                            "area": area,
                            "ready_count": total_ready_count,
                            "pending_count": len(job_pending_ids),
                            "progress": 0,
                            "state": initial_state,
                            "message": (
                                "Portable package imported; annotation is required."
                                if job_pending_ids
                                else "Portable package imported; ready to train."
                            ),
                        }
                        _write_json_atomic(status_path, status_payload)
                        prepared_receipt = {
                            "schema_version": (
                                PORTABLE_IMPORT_RECEIPT_SCHEMA_VERSION
                            ),
                            "state": _RECEIPT_PREPARED,
                            "package_id": package_id,
                            "package_sha256": package_sha256,
                            "imported_at": datetime.now(timezone.utc).isoformat(),
                            "source_package": str(package),
                            "source_runtime_weight": source_runtime_weight,
                            "portable_runtime_weight": portable_runtime_weight,
                            "handoff_path": str(handoff_path.resolve()),
                            "handoff_sha256": _sha256_file(handoff_path),
                            "status_identity_sha256": _status_identity_sha256(
                                status_payload
                            ),
                            "product": product,
                            "area": area,
                            "ready_count": total_ready_count,
                            "pending_count": len(job_pending_ids),
                        }
                        _write_json_atomic_verified(
                            receipt_path,
                            prepared_receipt,
                            description="prepared portable import receipt",
                        )
                        prepared_receipt_persisted = True
                        _publish_latest_handoff(latest_path, local_handoff)
                    except BaseException as import_error:
                        try:
                            journal.rollback()
                        except PortableTrainingImportError as rollback_error:
                            add_note = getattr(import_error, "add_note", None)
                            if callable(add_note):
                                add_note(str(rollback_error))
                        if not prepared_receipt_persisted:
                            try:
                                shutil.rmtree(job_dir)
                            except FileNotFoundError:
                                pass
                            except OSError as cleanup_error:
                                add_note = getattr(import_error, "add_note", None)
                                if callable(add_note):
                                    add_note(
                                        "Portable job cleanup failed: "
                                        f"{cleanup_error}"
                                    )
                        raise
                    else:
                        journal.commit()
                        _finalize_portable_receipt(
                            receipt_path,
                            prepared_receipt,
                        )

                    return PortableTrainingImportReport(
                        package_id=package_id,
                        handoff_path=handoff_path,
                        product=product,
                        area=area,
                        ready_count=total_ready_count,
                        pending_count=len(job_pending_ids),
                    )


def _validate_existing_portable_import(
    *,
    receipt_path: Path,
    expected_handoff_path: Path,
    expected_status_path: Path,
    latest_path: Path,
    training_root: Path,
    package_id: str,
    package_sha256: str,
    product: str,
    area: str,
) -> PortableTrainingImportReport:
    """Fail closed on persisted import state before returning idempotent reuse."""
    if receipt_path.is_symlink():
        raise PortableTrainingImportError(
            "The existing portable import receipt cannot be a symbolic link."
        )
    receipt = _read_json_mapping(receipt_path)
    receipt_schema = receipt.get("schema_version")
    if (
        type(receipt_schema) is not int
        or receipt_schema not in SUPPORTED_PORTABLE_IMPORT_RECEIPT_SCHEMAS
    ):
        raise PortableTrainingImportError(
            "The existing portable import receipt schema is unsupported."
        )
    receipt_state = (
        _RECEIPT_COMMITTED
        if receipt_schema == 1
        else str(receipt.get("state") or "").strip()
    )
    if receipt_state not in {_RECEIPT_PREPARED, _RECEIPT_COMMITTED}:
        raise PortableTrainingImportError(
            "The existing portable import receipt state is invalid."
        )
    expected_receipt_fields = {
        "package_id": package_id,
        "package_sha256": package_sha256,
        "product": product,
        "area": area,
    }
    if any(receipt.get(key) != value for key, value in expected_receipt_fields.items()):
        raise PortableTrainingImportError(
            "The existing portable import receipt does not match this package."
        )
    try:
        receipt_handoff_path = Path(str(receipt.get("handoff_path") or "")).resolve()
    except (OSError, RuntimeError) as exc:
        raise PortableTrainingImportError(
            "The existing portable import receipt has an invalid handoff path."
        ) from exc
    if receipt_handoff_path != expected_handoff_path.resolve():
        raise PortableTrainingImportError(
            "The existing portable import receipt points outside its immutable job."
        )
    local_handoff = _read_json_mapping(expected_handoff_path)
    if (
        local_handoff.get("job_id") != expected_handoff_path.parent.name
        or local_handoff.get("source_package_id") != package_id
        or local_handoff.get("source_package_sha256") != package_sha256
    ):
        raise PortableTrainingImportError(
            "The existing portable handoff does not match its import receipt."
        )
    latest: dict[str, Any] | None = None
    if latest_path.exists() or latest_path.is_symlink():
        if latest_path.is_symlink():
            raise PortableTrainingImportError(
                "The latest operator handoff pointer cannot be a symbolic link."
            )
        latest = _read_json_mapping(latest_path)
        if not str(latest.get("job_id") or "").strip():
            raise PortableTrainingImportError(
                "The latest operator handoff pointer is invalid."
            )
    if receipt_state == _RECEIPT_PREPARED and latest != local_handoff:
        raise PortableTrainingImportError(
            "The prepared portable import receipt has no exact latest-pointer "
            "commit; operator reconciliation is required."
        )
    try:
        loaded = load_operator_handoff(
            expected_handoff_path,
            training_root=training_root,
        )
    except OperatorHandoffError as exc:
        raise PortableTrainingImportError(
            f"The existing portable handoff is invalid: {exc}"
        ) from exc
    target = loaded.selected_target
    if (target.product, target.area) != (product, area):
        raise PortableTrainingImportError(
            "The existing portable handoff target does not match its receipt."
        )
    status = _read_json_mapping(expected_status_path)
    if (
        type(status.get("schema_version")) is not int
        or status.get("schema_version") != 1
        or status.get("job_id") != loaded.job_id
        or status.get("product") != product
        or status.get("area") != area
    ):
        raise PortableTrainingImportError(
            "The existing portable job status does not match its handoff."
        )
    status_identity_sha256 = _status_identity_sha256(status)
    ready_count = target.total_ready_count
    pending_count = target.pending_count
    if (
        type(receipt.get("ready_count")) is not int
        or receipt["ready_count"] != ready_count
        or type(receipt.get("pending_count")) is not int
        or receipt["pending_count"] != pending_count
    ):
        raise PortableTrainingImportError(
            "The existing portable import counts do not match its handoff."
        )
    if receipt_schema in {2, PORTABLE_IMPORT_RECEIPT_SCHEMA_VERSION}:
        if receipt.get("handoff_sha256") != _sha256_file(expected_handoff_path):
            raise PortableTrainingImportError(
                "The existing portable handoff checksum does not match its receipt."
            )
    if (
        receipt_schema == PORTABLE_IMPORT_RECEIPT_SCHEMA_VERSION
        and receipt.get("status_identity_sha256") != status_identity_sha256
    ):
        raise PortableTrainingImportError(
            "The existing portable status identity does not match its receipt."
        )
    if receipt_schema == 2:
        receipt = dict(receipt)
        receipt["schema_version"] = PORTABLE_IMPORT_RECEIPT_SCHEMA_VERSION
        receipt["status_identity_sha256"] = status_identity_sha256
        receipt.pop("status_sha256", None)
        try:
            _write_json_atomic_verified(
                receipt_path,
                receipt,
                description="migrated portable import receipt",
            )
        except (OSError, TypeError, ValueError) as exc:
            raise PortableTrainingImportError(
                "The existing portable import receipt could not be migrated."
            ) from exc
    if (
        latest is not None
        and str(latest.get("job_id") or "").strip() == loaded.job_id
        and latest != local_handoff
    ):
        raise PortableTrainingImportError(
            "The latest operator handoff differs from its immutable job."
        )
    if receipt_state == _RECEIPT_PREPARED:
        _finalize_portable_receipt(receipt_path, receipt)
    return PortableTrainingImportReport(
        package_id=package_id,
        handoff_path=expected_handoff_path.resolve(),
        product=product,
        area=area,
        ready_count=ready_count,
        pending_count=pending_count,
        reused_existing=True,
    )


def _publish_latest_handoff(path: Path, payload: dict[str, Any]) -> None:
    """Publish the sole commit pointer and tolerate a verified post-replace error."""
    try:
        _write_json_atomic(path, payload)
    except OSError:
        try:
            committed = _read_json_mapping(path) == payload
        except PortableTrainingImportError:
            committed = False
        if not committed:
            raise


def _status_identity_sha256(status: Mapping[str, Any]) -> str:
    """Hash only immutable status identity fields, never lifecycle progress."""
    identity = {
        "schema_version": status.get("schema_version"),
        "job_id": status.get("job_id"),
        "created_at": status.get("created_at"),
        "product": status.get("product"),
        "area": status.get("area"),
    }
    if (
        type(identity["schema_version"]) is not int
        or identity["schema_version"] != 1
        or not all(
            isinstance(identity[field], str) and str(identity[field]).strip()
            for field in ("job_id", "created_at", "product", "area")
        )
    ):
        raise PortableTrainingImportError(
            "The portable job status has an invalid immutable identity."
        )
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _write_json_atomic_verified(
    path: Path,
    payload: dict[str, Any],
    *,
    description: str,
) -> None:
    """Treat a reported write error as success only when exact bytes committed."""
    try:
        _write_json_atomic(path, payload)
    except (OSError, TypeError, ValueError) as exc:
        try:
            committed = _read_json_mapping(path) == payload
        except PortableTrainingImportError:
            committed = False
        if not committed:
            raise
        _LOGGER.warning(
            "%s was committed although durability reporting failed: path=%s "
            "error=%s",
            description,
            path,
            exc,
        )


def _finalize_portable_receipt(
    receipt_path: Path,
    prepared_receipt: dict[str, Any],
) -> None:
    """Finalize evidence after latest committed; diagnostics cannot undo commit."""
    committed_receipt = dict(prepared_receipt)
    committed_receipt["state"] = _RECEIPT_COMMITTED
    committed_receipt["committed_at"] = datetime.now(timezone.utc).isoformat()
    try:
        _write_json_atomic_verified(
            receipt_path,
            committed_receipt,
            description="committed portable import receipt",
        )
    except (OSError, TypeError, ValueError) as exc:
        _LOGGER.warning(
            "Portable handoff is committed but receipt finalization was deferred; "
            "the prepared receipt is retained for reconciliation: path=%s error=%s",
            receipt_path,
            exc,
        )


def _validate_archive(
    archive: zipfile.ZipFile,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    members: dict[str, zipfile.ZipInfo] = {}
    total_bytes = 0
    for member in archive.infolist():
        if member.is_dir():
            continue
        relative = _safe_archive_path(member.filename)
        if relative in members:
            raise PortableTrainingImportError(
                f"Duplicate path in training package: {relative}"
            )
        file_type = (member.external_attr >> 16) & 0o170000
        if file_type == 0o120000:
            raise PortableTrainingImportError("Symbolic links are not allowed in ZIP files.")
        members[relative] = member
        total_bytes += member.file_size
    if len(members) > MAX_PACKAGE_FILES or total_bytes > MAX_PACKAGE_BYTES:
        raise PortableTrainingImportError("Training package exceeds the safe size limit.")
    metadata_member = members.get(PACKAGE_METADATA_NAME)
    if metadata_member is None or metadata_member.file_size > 2 * 1024 * 1024:
        raise PortableTrainingImportError("Training package metadata is missing or too large.")
    try:
        metadata = json.loads(archive.read(metadata_member).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PortableTrainingImportError(
            f"Invalid training package metadata: {exc}"
        ) from exc
    if not isinstance(metadata, dict) or metadata.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise PortableTrainingImportError("Unsupported training package schema.")
    raw_inventory = metadata.get("files")
    if not isinstance(raw_inventory, dict):
        raise PortableTrainingImportError("Training package file inventory is missing.")
    inventory: dict[str, dict[str, Any]] = {}
    for raw_path, raw_contract in raw_inventory.items():
        relative = _safe_archive_path(str(raw_path))
        if not isinstance(raw_contract, dict):
            raise PortableTrainingImportError(f"Invalid file contract: {relative}")
        digest = str(raw_contract.get("sha256") or "").lower()
        size = raw_contract.get("size")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise PortableTrainingImportError(f"Invalid SHA-256 contract: {relative}")
        if type(size) is not int or size < 0:
            raise PortableTrainingImportError(f"Invalid file size contract: {relative}")
        inventory_member = members.get(relative)
        if inventory_member is None or inventory_member.file_size != size:
            raise PortableTrainingImportError(f"File size mismatch: {relative}")
        inventory[relative] = {"sha256": digest, "size": size}
    allowed_untracked = {PACKAGE_METADATA_NAME}
    if set(members) - allowed_untracked != set(inventory):
        raise PortableTrainingImportError(
            "Training package contains files outside its verified inventory."
        )
    return metadata, inventory


def _extract_verified_files(
    archive: zipfile.ZipFile,
    inventory: dict[str, dict[str, Any]],
    destination_root: Path,
) -> None:
    for relative, contract in inventory.items():
        destination = (destination_root / Path(*PurePosixPath(relative).parts)).resolve()
        if not destination.is_relative_to(destination_root.resolve()):
            raise PortableTrainingImportError(f"Unsafe extraction path: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        written = 0
        with archive.open(relative) as source, destination.open("wb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                target.write(chunk)
                digest.update(chunk)
                written += len(chunk)
        if written != contract["size"] or digest.hexdigest() != contract["sha256"]:
            raise PortableTrainingImportError(f"Checksum mismatch: {relative}")


def _preflight_shared_import(
    *,
    dataset_source: Path,
    dataset_destination: Path,
    models_source: Path,
    models_destination: Path,
    package_id: str,
    data_root: Path,
) -> _SharedImportPlan:
    """Resolve every conflict and merged manifest before shared state changes."""
    copies: list[_FileCopyPlan] = []
    for relative_directory in (
        Path("raw/images"),
        Path("raw/labels"),
        Path("review_pending/images"),
        Path("review_pending/labels"),
        Path("color_review"),
    ):
        candidate = dataset_source / relative_directory
        if candidate.is_dir():
            copies.extend(
                _preflight_directory_files(
                    candidate,
                    dataset_destination / relative_directory,
                    data_root=data_root,
                )
            )
    metadata_source = dataset_source / "metadata"
    if metadata_source.is_dir():
        for source in sorted(metadata_source.iterdir()):
            if source.is_symlink():
                raise PortableTrainingImportError(
                    f"Symbolic link is not allowed: {source}"
                )
            if source.is_file() and source.name != "review_dataset_manifest.csv":
                copies.append(
                    _preflight_file_copy(
                        source,
                        dataset_destination / "metadata" / source.name,
                        data_root=data_root,
                    )
                )
    copies.extend(
        _preflight_directory_files(
            models_source,
            models_destination,
            data_root=data_root,
        )
    )

    csv_writes = [
        _preflight_csv_manifest(
            dataset_source / "metadata" / "review_dataset_manifest.csv",
            dataset_destination / "metadata" / "review_dataset_manifest.csv",
            dataset_root=dataset_destination,
            pending=False,
            package_id=package_id,
            data_root=data_root,
        )
    ]
    pending_manifest = dataset_source / "review_pending" / "manifest.csv"
    if pending_manifest.is_file():
        csv_writes.append(
            _preflight_csv_manifest(
                pending_manifest,
                dataset_destination / "review_pending" / "manifest.csv",
                dataset_root=dataset_destination,
                pending=True,
                package_id=package_id,
                data_root=data_root,
            )
        )
    return _SharedImportPlan(tuple(copies), tuple(csv_writes))


def _preflight_directory_files(
    source_root: Path,
    destination_root: Path,
    *,
    data_root: Path,
) -> list[_FileCopyPlan]:
    plans: list[_FileCopyPlan] = []
    for source in sorted(source_root.rglob("*")):
        if source.is_symlink():
            raise PortableTrainingImportError(
                f"Symbolic link is not allowed: {source}"
            )
        if source.is_file():
            plans.append(
                _preflight_file_copy(
                    source,
                    destination_root / source.relative_to(source_root),
                    data_root=data_root,
                )
            )
    return plans


def _preflight_file_copy(
    source: Path,
    destination: Path,
    *,
    data_root: Path,
) -> _FileCopyPlan:
    _validate_shared_destination(destination, data_root=data_root)
    source_sha256 = _sha256_file(source)
    if destination.exists():
        if not destination.is_file() or _sha256_file(destination) != source_sha256:
            raise PortableTrainingImportError(
                f"Imported file conflicts with existing data: {destination}"
            )
        return _FileCopyPlan(source, destination, source_sha256, False)
    return _FileCopyPlan(source, destination, source_sha256, True)


def _preflight_csv_manifest(
    source: Path,
    destination: Path,
    *,
    dataset_root: Path,
    pending: bool,
    package_id: str,
    data_root: Path,
) -> _CsvWritePlan:
    if not source.is_file():
        raise PortableTrainingImportError(f"Dataset manifest is missing: {source}")
    _validate_shared_destination(destination, data_root=data_root)
    imported_rows = _read_csv(source)
    existing_rows = _read_csv(destination)
    merged = {
        _manifest_row_key(row, index): row
        for index, row in enumerate(existing_rows)
    }
    for index, row in enumerate(imported_rows):
        normalized = _normalize_portable_manifest_row(
            row,
            dataset_root=dataset_root,
            pending=pending,
            package_id=package_id,
        )
        key = _manifest_row_key(normalized, index)
        previous = merged.get(key)
        if previous and previous.get("image_sha256") and normalized.get(
            "image_sha256"
        ):
            if previous["image_sha256"] != normalized["image_sha256"]:
                raise PortableTrainingImportError(
                    f"Conflicting manifest sample identity: {key}"
                )
        merged[key] = normalized
    rows = tuple(merged.values())
    if not rows:
        raise PortableTrainingImportError(
            f"Cannot write an empty dataset manifest: {destination}"
        )
    expected_destination_sha256 = (
        _sha256_file(destination) if destination.is_file() else None
    )
    return _CsvWritePlan(destination, rows, expected_destination_sha256)


def _normalize_portable_manifest_row(
    row: dict[str, str],
    *,
    dataset_root: Path,
    pending: bool,
    package_id: str,
) -> dict[str, str]:
    normalized = dict(row)
    image_root = dataset_root / (
        "review_pending/images" if pending else "raw/images"
    )
    label_root = dataset_root / (
        "review_pending/labels" if pending else "raw/labels"
    )
    image_name = Path(str(row.get("output_image") or "")).name
    label_name = Path(str(row.get("output_label") or "")).name
    if image_name:
        normalized["portable_original_source_image"] = str(
            row.get("source_image") or ""
        )
        normalized["output_image"] = str((image_root / image_name).resolve())
        normalized["source_image"] = normalized["output_image"]
    if label_name:
        normalized["output_label"] = str((label_root / label_name).resolve())
    if pending and image_name:
        normalized["detection_source_image"] = normalized["output_image"]
    normalized["portable_package_id"] = package_id
    return normalized


def _validate_shared_destination(destination: Path, *, data_root: Path) -> None:
    if destination.is_symlink():
        raise PortableTrainingImportError(
            f"Portable import destination cannot be a symbolic link: {destination}"
        )
    resolved = destination.resolve(strict=False)
    if not resolved.is_relative_to(data_root.resolve()):
        raise PortableTrainingImportError(
            f"Portable import destination escapes training data: {destination}"
        )


def _apply_shared_import_plan(
    plan: _SharedImportPlan,
    journal: _ImportMutationJournal,
) -> None:
    for copy_plan in plan.copies:
        if _sha256_file(copy_plan.source) != copy_plan.source_sha256:
            raise PortableTrainingImportError(
                f"Portable staged file changed after preflight: {copy_plan.source}"
            )
        if not copy_plan.write_required:
            if (
                not copy_plan.destination.is_file()
                or _sha256_file(copy_plan.destination) != copy_plan.source_sha256
            ):
                raise PortableTrainingImportError(
                    "Portable import destination changed after preflight: "
                    f"{copy_plan.destination}"
                )
            continue
        if copy_plan.destination.exists() or copy_plan.destination.is_symlink():
            raise PortableTrainingImportError(
                "Portable import destination appeared after preflight: "
                f"{copy_plan.destination}"
            )
        journal.record_before_write(copy_plan.destination)
        _copy_verified_with_sha(
            copy_plan.source,
            copy_plan.destination,
            copy_plan.source_sha256,
        )
    for csv_plan in plan.csv_writes:
        current_sha256 = (
            _sha256_file(csv_plan.destination)
            if csv_plan.destination.is_file()
            else None
        )
        if current_sha256 != csv_plan.expected_destination_sha256:
            raise PortableTrainingImportError(
                "Portable dataset manifest changed after preflight: "
                f"{csv_plan.destination}"
            )
        journal.record_before_write(csv_plan.destination)
        _write_csv_atomic(csv_plan.destination, list(csv_plan.rows))


def _copy_verified_with_sha(
    source: Path,
    destination: Path,
    expected_sha256: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        shutil.copy2(source, temporary)
        if _sha256_file(temporary) != expected_sha256:
            raise PortableTrainingImportError(
                f"Copied portable file failed checksum verification: {destination}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _merge_dataset(source: Path, destination: Path, *, package_id: str) -> None:
    for relative_directory in (
        Path("raw/images"),
        Path("raw/labels"),
        Path("review_pending/images"),
        Path("review_pending/labels"),
        Path("color_review"),
    ):
        candidate = source / relative_directory
        if candidate.is_dir():
            _merge_directory_files(candidate, destination / relative_directory)
    metadata_source = source / "metadata"
    if metadata_source.is_dir():
        for file_path in metadata_source.iterdir():
            if file_path.is_file() and file_path.name != "review_dataset_manifest.csv":
                _copy_verified(file_path, destination / "metadata" / file_path.name)
    _merge_csv_manifest(
        source / "metadata" / "review_dataset_manifest.csv",
        destination / "metadata" / "review_dataset_manifest.csv",
        dataset_root=destination,
        pending=False,
        package_id=package_id,
    )
    pending_manifest = source / "review_pending" / "manifest.csv"
    if pending_manifest.is_file():
        _merge_csv_manifest(
            pending_manifest,
            destination / "review_pending" / "manifest.csv",
            dataset_root=destination,
            pending=True,
            package_id=package_id,
        )


def _merge_csv_manifest(
    source: Path,
    destination: Path,
    *,
    dataset_root: Path,
    pending: bool,
    package_id: str,
) -> None:
    if not source.is_file():
        if pending:
            return
        raise PortableTrainingImportError(f"Dataset manifest is missing: {source}")
    imported_rows = _read_csv(source)
    existing_rows = _read_csv(destination)
    merged = {_manifest_row_key(row, index): row for index, row in enumerate(existing_rows)}
    for index, row in enumerate(imported_rows):
        normalized = dict(row)
        image_root = dataset_root / ("review_pending/images" if pending else "raw/images")
        label_root = dataset_root / ("review_pending/labels" if pending else "raw/labels")
        image_name = Path(str(row.get("output_image") or "")).name
        label_name = Path(str(row.get("output_label") or "")).name
        if image_name:
            normalized["portable_original_source_image"] = str(row.get("source_image") or "")
            normalized["output_image"] = str((image_root / image_name).resolve())
            normalized["source_image"] = normalized["output_image"]
        if label_name:
            normalized["output_label"] = str((label_root / label_name).resolve())
        if pending and image_name:
            normalized["detection_source_image"] = normalized["output_image"]
        normalized["portable_package_id"] = package_id
        key = _manifest_row_key(normalized, index)
        previous = merged.get(key)
        if previous and previous.get("image_sha256") and normalized.get("image_sha256"):
            if previous["image_sha256"] != normalized["image_sha256"]:
                raise PortableTrainingImportError(
                    f"Conflicting manifest sample identity: {key}"
                )
        merged[key] = normalized
    _write_csv_atomic(destination, list(merged.values()))


def _merge_directory_files(source_root: Path, destination_root: Path) -> None:
    for source in sorted(source_root.rglob("*")):
        if source.is_symlink():
            raise PortableTrainingImportError(f"Symbolic link is not allowed: {source}")
        if source.is_file():
            _copy_verified(source, destination_root / source.relative_to(source_root))


def _rewrite_portable_runtime_config(
    models_root: Path,
    *,
    product: str,
    area: str,
) -> tuple[str, str]:
    """Bind an imported config to the verified station weight in its payload."""
    resolved_models_root = models_root.resolve()
    station_dir = (resolved_models_root / product / area / "yolo").resolve()
    if not station_dir.is_relative_to(resolved_models_root):
        raise PortableTrainingImportError("Portable model station path is unsafe.")
    config_path = station_dir / "config.yaml"
    if config_path.is_symlink() or not config_path.is_file():
        raise PortableTrainingImportError(
            f"Portable runtime config is missing: {config_path}"
        )
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PortableTrainingImportError(
            f"Unable to read portable runtime config: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PortableTrainingImportError("Portable runtime config must be a mapping.")
    source_runtime_weight = str(payload.get("weights") or "").strip()
    if not source_runtime_weight:
        raise PortableTrainingImportError(
            "Portable runtime config does not declare a weight artifact."
        )
    normalized_source = source_runtime_weight.replace("\\", "/")
    weight_name = _safe_segment(
        normalized_source.rsplit("/", 1)[-1],
        "runtime weight filename",
    )
    weights_root = (station_dir / "weights").resolve()
    packaged_weight = (weights_root / weight_name).resolve()
    if (
        not packaged_weight.is_relative_to(weights_root)
        or packaged_weight.is_symlink()
        or not packaged_weight.is_file()
    ):
        raise PortableTrainingImportError(
            "Portable runtime config references a weight missing from its station "
            f"payload: {weight_name}"
        )
    portable_runtime_weight = PurePosixPath(
        "models",
        product,
        area,
        "yolo",
        "weights",
        weight_name,
    ).as_posix()
    payload["weights"] = portable_runtime_weight
    _write_yaml_atomic(config_path, payload)
    return source_runtime_weight, portable_runtime_weight


def _copy_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256_file(source) != _sha256_file(destination):
            raise PortableTrainingImportError(
                f"Imported file conflicts with existing data: {destination}"
            )
        return
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_source_contract(
    handoff: dict[str, Any],
    metadata: dict[str, Any],
    *,
    product: str,
    area: str,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    schema_version = handoff.get("schema_version")
    if (
        type(schema_version) is not int
        or schema_version not in SUPPORTED_SOURCE_HANDOFF_SCHEMAS
    ):
        raise PortableTrainingImportError(
            "Package handoff schema is unsupported; expected one of "
            f"{sorted(SUPPORTED_SOURCE_HANDOFF_SCHEMAS)}."
        )
    if str(handoff.get("job_id") or "") != str(metadata.get("source_job_id") or ""):
        raise PortableTrainingImportError("Package job_id does not match its handoff.")
    targets = handoff.get("targets")
    if not isinstance(targets, list) or len(targets) != 1 or not isinstance(targets[0], dict):
        raise PortableTrainingImportError("Package handoff target is invalid.")
    if str(targets[0].get("product") or "") != product or str(
        targets[0].get("area") or ""
    ) != area:
        raise PortableTrainingImportError("Package target does not match its metadata.")
    try:
        training_options = _load_training_options(
            handoff.get("training_options"),
            schema_version=schema_version,
        )
    except OperatorHandoffError as exc:
        raise PortableTrainingImportError(
            f"Package handoff training options are invalid: {exc}"
        ) from exc
    target = targets[0]
    class_names_value = target.get("class_names")
    class_names = (
        tuple(str(name) for name in class_names_value)
        if isinstance(class_names_value, list)
        else ()
    )
    contract_required = bool(target.get("class_contract_required", False))
    observed_value = target.get("observed_class_map")
    has_observed_contract = isinstance(observed_value, dict) and bool(observed_value)
    if contract_required and not (class_names or has_observed_contract):
        raise PortableTrainingImportError("Required class contract is missing.")
    if class_names:
        if any(not name.strip() for name in class_names):
            raise PortableTrainingImportError("Class names must not be empty.")
        if len(set(class_names)) != len(class_names):
            raise PortableTrainingImportError(
                "Class names must be unique and ordered."
            )
        if str(target.get("class_schema_hash") or "").strip() != _class_schema_hash(
            class_names
        ):
            raise PortableTrainingImportError(
                "Class contract checksum does not match the ordered class names."
            )
    metadata_sample_ids = _validated_sample_id_list(
        metadata.get("sample_ids"),
        "package sample_ids",
    )
    metadata_pending_ids = _validated_sample_id_list(
        metadata.get("pending_sample_ids"),
        "package pending_sample_ids",
    )
    target_sample_ids = _validated_sample_id_list(
        target.get("sample_ids"),
        "handoff sample_ids",
    )
    target_pending_ids = _validated_sample_id_list(
        target.get("pending_sample_ids"),
        "handoff pending_sample_ids",
    )
    if metadata_sample_ids != target_sample_ids:
        raise PortableTrainingImportError(
            "Package sample_ids do not match the verified handoff."
        )
    if metadata_pending_ids != target_pending_ids:
        raise PortableTrainingImportError(
            "Package pending_sample_ids do not match the verified handoff."
        )
    if not set(metadata_pending_ids).issubset(metadata_sample_ids):
        raise PortableTrainingImportError(
            "Package pending_sample_ids must be a subset of sample_ids."
        )
    return asdict(training_options), metadata_sample_ids, metadata_pending_ids


def _validated_sample_id_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise PortableTrainingImportError(
            f"{field_name} must be a list of non-empty strings."
        )
    normalized = tuple(item.strip() for item in value)
    if len(set(normalized)) != len(normalized):
        raise PortableTrainingImportError(f"{field_name} contains duplicates.")
    return normalized


def _validate_staged_sample_contract(
    dataset_root: Path,
    *,
    sample_ids: tuple[str, ...],
    pending_sample_ids: tuple[str, ...],
) -> None:
    ready_ids = _validated_manifest_sample_ids(
        dataset_root / "metadata" / "review_dataset_manifest.csv",
        "ready",
    )
    pending_manifest = dataset_root / "review_pending" / "manifest.csv"
    pending_ids = (
        _validated_manifest_sample_ids(pending_manifest, "pending")
        if pending_manifest.is_file()
        else ()
    )
    ready_set = set(ready_ids)
    pending_set = set(pending_ids)
    expected_pending = set(pending_sample_ids)
    expected_ready = set(sample_ids) - expected_pending
    if ready_set & pending_set:
        raise PortableTrainingImportError(
            "Portable samples cannot exist in both ready and pending manifests."
        )
    if ready_set != expected_ready or pending_set != expected_pending:
        raise PortableTrainingImportError(
            "Portable sample lists do not match the staged dataset manifests."
        )


def _validated_manifest_sample_ids(path: Path, kind: str) -> tuple[str, ...]:
    rows = _read_csv(path)
    sample_ids = tuple(str(row.get("sample_id") or "").strip() for row in rows)
    if any(not sample_id for sample_id in sample_ids):
        raise PortableTrainingImportError(
            f"Portable {kind} manifest contains an empty sample_id."
        )
    if len(set(sample_ids)) != len(sample_ids):
        raise PortableTrainingImportError(
            f"Portable {kind} manifest contains duplicate sample_ids."
        )
    return sample_ids


def _resolved_staging_path(root: Path, value: Any, field_name: str) -> Path:
    relative = _safe_archive_path(str(value or ""))
    resolved = (root / Path(*PurePosixPath(relative).parts)).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise PortableTrainingImportError(f"Unsafe {field_name}.")
    return resolved


def _safe_archive_path(value: str) -> str:
    if not value or "\\" in value:
        raise PortableTrainingImportError(f"Unsafe ZIP path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PortableTrainingImportError(f"Unsafe ZIP path: {value!r}")
    return path.as_posix()


def _safe_segment(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text or text in {".", ".."} or any(
        not (character.isalnum() or character in "._-") for character in text
    ):
        raise PortableTrainingImportError(f"Invalid {field_name}: {text!r}")
    return text


def _manifest_ids(path: Path) -> set[str]:
    return {
        str(row.get("sample_id") or "")
        for row in _read_csv(path)
        if str(row.get("sample_id") or "")
    }


def _manifest_row_key(row: dict[str, str], index: int) -> str:
    return str(
        row.get("sample_id")
        or row.get("image_sha256")
        or row.get("output_image")
        or f"row:{index}"
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise PortableTrainingImportError(f"Unable to read CSV {path}: {exc}") from exc


def _write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise PortableTrainingImportError(f"Cannot write an empty dataset manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(field for field in row if field not in fields)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_mapping(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise PortableTrainingImportError(f"JSON file not found: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PortableTrainingImportError(f"Unable to read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PortableTrainingImportError(f"Invalid JSON mapping: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_yaml_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _capture_open_package_identity(handle: BinaryIO) -> _OpenPackageIdentity:
    before = _package_stat_signature(os.fstat(handle.fileno()))
    sha256 = _sha256_open_handle(handle)
    after = _package_stat_signature(os.fstat(handle.fileno()))
    if after != before:
        raise PortableTrainingImportError(
            "Training package changed while its identity was being calculated."
        )
    return _OpenPackageIdentity(*after, sha256)


def _assert_open_package_unchanged(
    handle: BinaryIO,
    package_path: Path,
    expected: _OpenPackageIdentity,
) -> None:
    current_signature = _package_stat_signature(os.fstat(handle.fileno()))
    current_sha256 = _sha256_open_handle(handle)
    try:
        pathname_signature = _package_stat_signature(package_path.stat())
    except OSError as exc:
        raise PortableTrainingImportError(
            "Training package pathname changed during import."
        ) from exc
    if (
        current_signature
        != (expected.device, expected.inode, expected.size, expected.modified_ns)
        or pathname_signature != current_signature
        or current_sha256 != expected.sha256
    ):
        raise PortableTrainingImportError(
            "Training package changed while its ZIP payload was being verified."
        )


def _package_stat_signature(stat_result: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
    )


def _sha256_open_handle(handle: BinaryIO) -> str:
    position = handle.tell()
    digest = hashlib.sha256()
    try:
        handle.seek(0)
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    finally:
        handle.seek(position)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_empty_parents(path: Path, *, stop: Path) -> None:
    current = path.resolve(strict=False)
    resolved_stop = stop.resolve()
    while current != resolved_stop and current.is_relative_to(resolved_stop):
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


@contextmanager
def _portable_import_lock(data_root: Path, timeout_seconds: float = 15.0):
    try:
        with portable_import_lock(
            data_root,
            timeout_seconds=timeout_seconds,
        ):
            yield
    except DatasetManifestLockTimeoutError as exc:
        raise PortableTrainingImportError(
            "Another portable training package is being imported."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package")
    parser.add_argument("--training-root", default=".")
    args = parser.parse_args()
    try:
        report = import_portable_training_package(args.package, args.training_root)
    except (OSError, zipfile.BadZipFile, PortableTrainingImportError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(report.handoff_path)


if __name__ == "__main__":
    main()
