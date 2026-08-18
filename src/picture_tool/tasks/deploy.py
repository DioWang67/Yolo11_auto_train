"""Deploy trained artifacts directly into a yolo11_inference models directory.

After ``yolo_train`` completes, this task copies ``config.yaml`` and weights
into ``{inference_models_dir}/{product}/{area}/yolo/`` so that inference can
discover the model without manual path edits.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, List, Tuple

import yaml

from picture_tool.pipeline.core import Task
from picture_tool.pipeline.utils import detect_existing_weights
from picture_tool.position.position_gate import (
    POSITION_GATE_REPORT_SCHEMA_VERSION,
    PositionGateError,
    canonical_detection_config_sha256,
    canonical_position_config_sha256,
)
from picture_tool.runtime_pair_deployment import PairVerification, verify_runtime_pair
from picture_tool.tasks.bundle import (
    find_color_model_source,
    rewrite_detection_config,
    select_runtime_weight,
    validate_deployment_target,
)
from picture_tool.tasks.deployment_target import resolve_yolo_deployment_target
from picture_tool.utils.onnx_exporter import OnnxExporter


_VERSION_RE = re.compile(r"_v(\d+)\.(\d+)\.(\d+)_")

STATION_LOCAL_FIELDS = {
    "exposure_time",
    "gain",
    "light_brightness",
    "calibration",
    "output_dir",
    "save_original",
    "save_processed",
    "save_annotated",
    "save_crops",
    "save_fail_only",
    "jpeg_quality",
    "png_compression",
    "max_crops_per_frame",
    "buffer_limit",
    "flush_interval",
}
DEPLOYMENT_OWNED_FIELDS = frozenset(
    {"weights", "position_config", "color_model_path"}
)
_DEPLOY_LOGGER = logging.getLogger(__name__)
_COLOR_REVISION_PUBLICATION_THREAD_LOCK = threading.Lock()
_COLOR_REVISION_PUBLICATION_LOCK_NAME = "deployment-publication.lock"
_VALIDATED_ACCEPTANCE_REPORT_SHA_FIELD = "_validated_report_sha256"


class ConcurrentStationConfigChangeError(RuntimeError):
    """Raised when station settings change during a deployment gate."""


class ColorRevisionPublicationLockTimeoutError(TimeoutError):
    """Raised when color activation blocks final model publication."""


class _ColorRevisionPublicationLock:
    """Owned cross-process byte lock with non-raising cleanup."""

    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self._released = False

    def release(self, logger: logging.Logger) -> None:
        if self._released:
            return
        self._released = True
        try:
            try:
                self._handle.seek(0)
                _unlock_color_revision_publication_byte(self._handle)
            except OSError as exc:
                logger.warning(
                    "Color revision publication lock unlock was deferred: %s",
                    exc,
                )
        finally:
            try:
                self._handle.close()
            except OSError as exc:
                logger.warning(
                    "Color revision publication lock handle cleanup failed: %s",
                    exc,
                )
            finally:
                _COLOR_REVISION_PUBLICATION_THREAD_LOCK.release()


class DeploymentRollbackError(RuntimeError):
    """Raised when deployment rollback cannot restore every destination."""

    def __init__(
        self,
        backup_root: Path,
        failures: list[tuple[Path, OSError]],
    ) -> None:
        self.backup_root = backup_root
        self.failures = tuple(failures)
        details = "; ".join(
            f"{destination}: {error}" for destination, error in failures
        )
        super().__init__(
            "Deployment rollback was incomplete; recovery evidence was retained at "
            f"{backup_root}. Restore failures: {details}"
        )


@dataclass(frozen=True)
class _StationConfigSnapshot:
    """One parsed station config and the identity of the exact bytes read."""

    values: dict[str, Any] | None
    identity: str


def _parse_version(filename: str) -> Tuple[int, int, int] | None:
    """Parse a semantic version tuple from a versioned weights filename."""
    match = _VERSION_RE.search(Path(filename).name)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _version_str(version: Tuple[int, int, int]) -> str:
    """Format a semantic version tuple."""
    return f"{version[0]}.{version[1]}.{version[2]}"


def _resolve_version(
    version_cfg: str,
    weights_dest: Path,
    product: str,
    area: str,
    extension: str = ".pt",
) -> Tuple[int, int, int]:
    """Return the deployment version tuple.

    Args:
        version_cfg: Explicit ``major.minor.patch`` value or ``auto``.
        weights_dest: Destination weights directory.
        product: Deployment product name.
        area: Deployment area name.
        extension: Runtime weight extension to scan.

    Returns:
        Version tuple for the new deployed weight.

    Raises:
        ValueError: If an explicit version string is invalid.
    """
    prefix = f"{product}_{area}_v"

    if version_cfg and version_cfg.lower() != "auto":
        parts = version_cfg.split(".")
        if len(parts) == 3:
            try:
                return (int(parts[0]), int(parts[1]), int(parts[2]))
            except ValueError:
                pass
        raise ValueError(
            f"deploy.version '{version_cfg}' is not a valid semantic version "
            "(expected 'major.minor.patch' or 'auto')."
        )

    existing: list[Tuple[int, int, int]] = []
    if weights_dest.exists():
        # Version identity belongs to the station model, not to a particular
        # runtime extension. Switching a station from PT to ONNX must advance
        # from v1.0.4 to v1.0.5 instead of restarting at v1.0.0.
        for path in weights_dest.glob(f"{prefix}*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {
                ".pt",
                ".onnx",
                ".engine",
                ".torchscript",
                extension.lower(),
            }:
                continue
            version = _parse_version(path.name)
            if version:
                existing.append(version)

    if not existing:
        return (1, 0, 0)

    latest = max(existing)
    return (latest[0], latest[1], latest[2] + 1)


def _acquire_deploy_lock(dest_dir: Path, timeout_seconds: float) -> Path:
    """Create a per-target lock directory around versioning and file writes.

    Args:
        dest_dir: Target ``models/<product>/<area>/yolo`` directory.
        timeout_seconds: Maximum seconds to wait for a previous deploy.

    Returns:
        Created lock directory path.

    Raises:
        TimeoutError: If another deploy keeps the lock too long.
    """
    lock_dir = dest_dir / ".deploy.lock"
    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    while True:
        try:
            lock_dir.mkdir()
            return lock_dir
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for deploy lock at {lock_dir}.")
            time.sleep(0.2)


def _release_deploy_lock(lock_dir: Path) -> None:
    """Release a deploy lock created by this process."""
    try:
        lock_dir.rmdir()
    except FileNotFoundError:
        return
    except OSError as exc:
        _DEPLOY_LOGGER.warning(
            "Deployment lock cleanup was deferred; remove the retained lock "
            "after verifying no deployment is active: lock_dir=%s error=%s",
            lock_dir,
            exc,
        )


def _acquire_color_revision_publication_lock(
    revisions_root: Path,
    timeout_seconds: float,
) -> _ColorRevisionPublicationLock:
    """Acquire the lock shared with inference color activation."""
    timeout = max(float(timeout_seconds), 0.0)
    started_at = time.monotonic()
    if not _COLOR_REVISION_PUBLICATION_THREAD_LOCK.acquire(timeout=timeout):
        raise ColorRevisionPublicationLockTimeoutError(
            "Another deployment thread is publishing a color-linked runtime."
        )
    handle: BinaryIO | None = None
    try:
        resolved_root = revisions_root.expanduser().resolve()
        locks_root = resolved_root / "locks"
        if locks_root.is_symlink():
            raise ValueError(
                f"Color revision lock root cannot be a symlink: {locks_root}"
            )
        locks_root.mkdir(parents=True, exist_ok=True)
        if locks_root.is_symlink():
            raise ValueError(
                f"Color revision lock root cannot be a symlink: {locks_root}"
            )
        resolved_locks_root = locks_root.resolve()
        if not resolved_locks_root.is_relative_to(resolved_root):
            raise ValueError(
                f"Color revision lock root escapes its store: {locks_root}"
            )
        lock_path = (
            resolved_locks_root / _COLOR_REVISION_PUBLICATION_LOCK_NAME
        )
        if lock_path.is_symlink():
            raise ValueError(
                f"Color revision publication lock cannot be a symlink: {lock_path}"
            )
        handle = lock_path.open("a+b")
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        while True:
            handle.seek(0)
            try:
                _lock_color_revision_publication_byte(handle)
                return _ColorRevisionPublicationLock(handle)
            except OSError as exc:
                if time.monotonic() - started_at >= timeout:
                    raise ColorRevisionPublicationLockTimeoutError(
                        "Timed out waiting for active color revision publication "
                        f"lock: {lock_path}"
                    ) from exc
                time.sleep(0.05)
    except BaseException:
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        _COLOR_REVISION_PUBLICATION_THREAD_LOCK.release()
        raise


def _lock_color_revision_publication_byte(handle: BinaryIO) -> None:
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:  # pragma: no cover - production station is Windows
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_color_revision_publication_byte(handle: BinaryIO) -> None:
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:  # pragma: no cover - production station is Windows
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 identity of a deployment artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_runtime_training_pair(run_dir: Path, runtime_path: Path) -> Path:
    """Validate and return the exact PT checkpoint used for a runtime export."""
    resolved_run_dir = run_dir.resolve()
    resolved_runtime = runtime_path.resolve()
    if resolved_runtime.suffix.lower() == ".pt":
        return resolved_runtime

    contract_path = resolved_run_dir / "runtime_export_manifest.json"
    if not contract_path.is_file():
        raise FileNotFoundError(
            "Runtime export lineage contract is missing; re-export the runtime "
            f"artifact before deployment: {contract_path}"
        )
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Runtime export lineage contract is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Runtime export lineage contract must contain an object.")

    runtime_value = str(payload.get("runtime_file") or "").strip()
    training_value = str(payload.get("training_weight_file") or "").strip()
    if not runtime_value or not training_value:
        raise ValueError("Runtime export lineage contract is incomplete.")
    contracted_runtime = (resolved_run_dir / runtime_value).resolve()
    training_weight = (resolved_run_dir / training_value).resolve()
    if not contracted_runtime.is_relative_to(resolved_run_dir):
        raise ValueError("Runtime export lineage contains an unsafe runtime path.")
    if not training_weight.is_relative_to(resolved_run_dir):
        raise ValueError("Runtime export lineage contains an unsafe training path.")
    if contracted_runtime != resolved_runtime:
        raise ValueError(
            "Selected runtime artifact does not match runtime_export_manifest.json."
        )
    if training_weight.suffix.lower() != ".pt" or not training_weight.is_file():
        raise FileNotFoundError(
            f"Paired runtime training checkpoint is unavailable: {training_weight}"
        )
    runtime_sha256 = str(payload.get("runtime_sha256") or "").strip().lower()
    training_sha256 = str(payload.get("training_weight_sha256") or "").strip().lower()
    if not runtime_sha256 or _sha256_file(resolved_runtime) != runtime_sha256:
        raise ValueError("Runtime export checksum does not match its lineage contract.")
    if not training_sha256 or _sha256_file(training_weight) != training_sha256:
        raise ValueError("Training checkpoint checksum does not match runtime lineage.")
    return training_weight


def _atomic_copy_verified(source: Path, destination: Path) -> str:
    """Copy a file through a temporary path, verify it, then publish atomically."""
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copy2(source, temporary)
        source_hash = _sha256_file(source)
        copied_hash = _sha256_file(temporary)
        if source_hash != copied_hash:
            raise OSError(
                f"Artifact checksum mismatch while copying {source} to {destination}"
            )
        temporary.replace(destination)
        return source_hash
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_yaml_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Publish YAML with an atomic same-directory replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _runtime_config_path(artifact_path: Path, inference_project_root: Path) -> str:
    """Return a relocatable path that inference resolves from its project root."""
    artifact = artifact_path.expanduser().resolve()
    project_root = inference_project_root.expanduser().resolve()
    try:
        relative = Path(os.path.relpath(artifact, start=project_root))
    except ValueError:
        # Windows paths on different drives cannot be made relative. Inference
        # accepts absolute runtime paths, so retain correctness in that layout.
        return str(artifact)
    return relative.as_posix()


class _DeploymentTransaction:
    """Rollback journal for files and directories changed by one deployment."""

    def __init__(self, station_dir: Path) -> None:
        self._station_dir = station_dir.resolve()
        self._backup_root = Path(
            tempfile.mkdtemp(prefix=".deploy-rollback-", dir=self._station_dir)
        ).resolve()
        self._entries: dict[Path, tuple[str, Path | None]] = {}
        self._closed = False

    def copy_verified(self, source: Path, destination: Path) -> str:
        """Journal and atomically copy one verified artifact."""
        target = self._track(destination)
        return _atomic_copy_verified(source, target)

    def replace_tree(self, source: Path, destination: Path) -> None:
        """Journal and replace one exported runtime directory."""
        target = self._track(destination)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        shutil.copytree(source, target)

    def write_yaml(self, path: Path, payload: dict[str, Any]) -> None:
        """Journal and atomically publish one YAML document."""
        target = self._track(path)
        _write_yaml_atomic(target, payload)

    def commit(self) -> None:
        """Discard rollback evidence after every publication succeeds."""
        if self._closed:
            return
        self._closed = True
        try:
            shutil.rmtree(self._backup_root)
        except OSError as exc:
            _DEPLOY_LOGGER.warning(
                "Deployment committed; rollback evidence cleanup was deferred: "
                "backup_root=%s error=%s",
                self._backup_root,
                exc,
            )

    def rollback(self) -> None:
        """Restore every changed destination to its pre-deployment state."""
        if self._closed:
            return
        failures: list[tuple[Path, OSError]] = []
        for destination, (kind, backup) in reversed(self._entries.items()):
            try:
                if destination.is_dir():
                    shutil.rmtree(destination)
                elif destination.exists() or destination.is_symlink():
                    destination.unlink()

                if kind == "file" and backup is not None:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    _atomic_copy_verified(backup, destination)
                elif kind == "directory" and backup is not None:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(backup, destination)
            except OSError as exc:
                failures.append((destination, exc))

        if failures:
            self._closed = True
            raise DeploymentRollbackError(self._backup_root, failures)

        self._closed = True
        try:
            shutil.rmtree(self._backup_root)
        except OSError as exc:
            _DEPLOY_LOGGER.warning(
                "Deployment rollback completed; evidence cleanup was deferred: "
                "backup_root=%s error=%s",
                self._backup_root,
                exc,
            )

    def _track(self, destination: Path) -> Path:
        if self._closed:
            raise RuntimeError("Deployment transaction is already closed.")
        unresolved = destination.expanduser().absolute()
        if unresolved.is_symlink():
            raise ValueError(
                f"Deployment transaction refuses symbolic-link target: {unresolved}"
            )
        target = unresolved.resolve()
        if not target.is_relative_to(self._station_dir):
            raise ValueError(
                f"Deployment transaction target escapes station directory: {target}"
            )
        if target in self._entries:
            return target

        backup = self._backup_root / f"{len(self._entries):04d}"
        if target.is_dir():
            shutil.copytree(target, backup)
            entry: tuple[str, Path | None] = ("directory", backup)
        elif target.exists():
            shutil.copy2(target, backup)
            entry = ("file", backup)
        else:
            entry = ("missing", None)
        self._entries[target] = entry
        return target


def _station_config_identity(content: bytes | None) -> str:
    """Return a stable identity that distinguishes a missing config from bytes."""
    if content is None:
        return "missing"
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _read_station_config_identity(path: Path) -> str:
    """Read the current station config identity for optimistic concurrency checks."""
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        content = None
    except OSError as exc:
        raise ValueError(f"Unable to verify existing station config: {exc}") from exc
    return _station_config_identity(content)


def _load_station_config_snapshot(path: Path) -> _StationConfigSnapshot:
    """Read and identify the live station config in one filesystem operation."""
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return _StationConfigSnapshot(values=None, identity="missing")
    except OSError as exc:
        raise ValueError(f"Unable to read existing station config: {exc}") from exc
    try:
        existing = yaml.safe_load(content.decode("utf-8")) or {}
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Unable to read existing station config: {exc}") from exc
    if not isinstance(existing, dict):
        raise ValueError("Existing station config must be a mapping.")
    return _StationConfigSnapshot(
        values=existing,
        identity=_station_config_identity(content),
    )


def _assert_station_config_unchanged(
    path: Path,
    expected_identity: str,
) -> None:
    """Fail closed instead of overwriting station edits made during deployment."""
    if _read_station_config_identity(path) != expected_identity:
        raise ConcurrentStationConfigChangeError(
            "Station config changed during deployment; deployment was rolled back "
            f"without overwriting the concurrent edit: {path}"
        )


def _preserve_station_fields(
    generated: dict[str, Any],
    existing_config: dict[str, Any] | None,
    deploy_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Keep machine-local camera/output settings across model deployments."""
    if not deploy_cfg.get("preserve_station_settings", True):
        return generated
    if existing_config is None:
        return generated
    configured_fields = deploy_cfg.get("station_fields")
    fields = (
        {str(field) for field in configured_fields}
        if isinstance(configured_fields, list)
        else STATION_LOCAL_FIELDS
    )
    forbidden_fields = fields & DEPLOYMENT_OWNED_FIELDS
    if forbidden_fields:
        raise ValueError(
            "deploy.station_fields cannot preserve deployment-owned fields: "
            + ", ".join(sorted(forbidden_fields))
        )
    merged = dict(generated)
    for field in fields:
        if field in existing_config:
            merged[field] = existing_config[field]
    return merged


def _position_area_mapping(
    config: dict[str, Any],
    product: str,
    area: str,
) -> dict[str, Any] | None:
    position_config = config.get("position_config")
    if not isinstance(position_config, dict):
        return None
    product_config = position_config.get(product)
    if not isinstance(product_config, dict):
        return None
    area_config = product_config.get(area)
    return area_config if isinstance(area_config, dict) else None


def _apply_position_activation_policy(
    generated: dict[str, Any],
    existing_config: dict[str, Any] | None,
    deploy_cfg: dict[str, Any],
    *,
    product: str,
    area: str,
) -> dict[str, Any]:
    """Keep runtime activation explicit while still deploying calibrated rules."""

    candidate_area = _position_area_mapping(generated, product, area)
    if candidate_area is None:
        return generated
    policy = str(
        deploy_cfg.get("position_activation") or "preserve"
    ).strip().lower()
    if policy not in {"preserve", "enable", "disable"}:
        raise ValueError(
            "deploy.position_activation must be preserve, enable, or disable."
        )

    enabled = False
    if policy == "enable":
        enabled = True
    elif policy == "disable":
        enabled = False
    elif existing_config is not None:
        existing_area = _position_area_mapping(existing_config, product, area)
        enabled = bool(
            existing_area.get("enabled", False)
            if existing_area is not None
            else False
        )

    merged = dict(generated)
    position_config = dict(merged.get("position_config") or {})
    product_config = dict(position_config.get(product) or {})
    updated_area = dict(candidate_area)
    updated_area["enabled"] = enabled
    product_config[area] = updated_area
    position_config[product] = product_config
    merged["position_config"] = position_config
    return merged


def _preserve_disabled_station_position_contract(
    generated: dict[str, Any],
    existing_config: dict[str, Any] | None,
    deploy_cfg: dict[str, Any],
    *,
    product: str,
    area: str,
) -> dict[str, Any]:
    """Preserve an inactive station contract when this job has no golden set."""
    policy = str(
        deploy_cfg.get("position_contract_policy") or "validate_candidate"
    ).strip().lower()
    if policy == "validate_candidate":
        return generated
    if policy != "preserve_disabled_station":
        raise ValueError(
            "deploy.position_contract_policy must be validate_candidate or "
            "preserve_disabled_station."
        )
    if existing_config is None:
        raise ValueError(
            "An active or missing station position contract cannot bypass "
            "position validation."
        )
    existing_area = _position_area_mapping(existing_config, product, area)
    if existing_area is None or bool(existing_area.get("enabled", False)):
        raise ValueError(
            "An active or missing station position contract cannot bypass "
            "position validation."
        )
    existing_position = existing_config.get("position_config")
    if not isinstance(existing_position, dict):
        raise ValueError("Existing station position contract is invalid.")
    merged = dict(generated)
    merged["position_config"] = existing_position
    return merged


def _load_position_gate(
    run_dir: Path,
    *,
    required: bool,
) -> tuple[dict[str, Any], Path | None]:
    path = run_dir / "position_gate.json"
    if not path.is_file():
        if required:
            raise ValueError(f"Position gate report not found: {path}")
        return {}, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read position gate report: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Position gate report must contain an object.")
    if required and payload.get("passed") is not True:
        failures = payload.get("failures") or []
        raise ValueError(
            "Deployment blocked by position gate: "
            + "; ".join(str(item) for item in failures)
        )
    return payload, path


def _verify_position_gate_candidate_config(
    gate: dict[str, Any],
    candidate_detection_config: dict[str, Any],
    candidate_position_config: dict[str, Any] | None,
    *,
    required: bool,
) -> None:
    """Bind a passed gate to exact detection and target-position contracts."""

    if not required:
        return
    if gate.get("schema_version") != POSITION_GATE_REPORT_SCHEMA_VERSION:
        raise ValueError(
            "Position gate schema is unsupported; rerun position validation."
        )
    if candidate_position_config is None:
        raise ValueError("Candidate position config is missing during deployment.")
    try:
        actual_detection_hash = canonical_detection_config_sha256(
            candidate_detection_config
        )
        actual_position_hash = canonical_position_config_sha256(
            candidate_position_config
        )
    except PositionGateError as exc:
        raise ValueError(str(exc)) from exc
    for field_name, label, actual_hash in (
        (
            "candidate_position_config_sha256",
            "position config",
            actual_position_hash,
        ),
        (
            "candidate_detection_config_sha256",
            "detection config",
            actual_detection_hash,
        ),
    ):
        expected_hash = str(gate.get(field_name) or "").strip().lower()
        if len(expected_hash) != 64 or any(
            character not in "0123456789abcdef"
            for character in expected_hash
        ):
            raise ValueError(
                f"Position gate has no valid candidate {label} checksum."
            )
        if actual_hash != expected_hash:
            raise ValueError(
                f"Candidate {label} checksum changed after gate evaluation."
            )


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verified_position_evidence(
    gate: dict[str, Any],
    *,
    path_field: str,
    hash_field: str,
    label: str,
    required: bool = False,
) -> tuple[Path | None, str | None]:
    raw_path = str(gate.get(path_field) or "").strip()
    if not raw_path:
        if required:
            raise ValueError(f"Position gate has no {label} path.")
        return None, None
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise ValueError(f"Position {label} was not found: {path}")
    expected_hash = str(gate.get(hash_field) or "").strip().lower()
    if (
        len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
    ):
        raise ValueError(f"Position gate has no valid {label} checksum.")
    actual_hash = _sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Position {label} checksum changed after gate evaluation."
        )
    return path, actual_hash


def _load_training_metadata(run_dir: Path) -> dict[str, Any]:
    """Load optional dataset/config hashes recorded by the trainer."""
    path = run_dir / "last_run_metadata.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_evaluation_gate(run_dir: Path, *, required: bool) -> dict[str, Any]:
    """Load the evaluation decision required before an operator deployment."""
    path = run_dir / "evaluation_gate.json"
    if not path.exists():
        if required:
            raise ValueError(f"Evaluation gate report not found: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read evaluation gate report: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Evaluation gate report must be an object.")
    if required and not payload.get("passed", False):
        failures = payload.get("failures") or []
        raise ValueError(
            "Deployment blocked by evaluation gate: "
            + "; ".join(str(item) for item in failures)
        )
    return payload


def _load_model_acceptance_report(
    report_path: Path,
    *,
    required: bool,
) -> dict[str, Any]:
    """Load and validate the deployment-blocking acceptance decision."""
    if report_path.is_symlink():
        raise ValueError(
            f"Model acceptance gate report cannot be a symbolic link: {report_path}"
        )
    if not report_path.is_file():
        if required:
            raise ValueError(
                f"Model acceptance gate report not found: {report_path}"
            )
        return {}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to read model acceptance gate report: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Model acceptance gate report must be an object.")
    passed = payload.get("passed")
    if not isinstance(passed, bool):
        raise ValueError("Model acceptance gate report has no boolean decision.")
    if required and not passed:
        failures = payload.get("failures") or []
        raise ValueError(
            "Deployment blocked by model acceptance gate: "
            + "; ".join(str(item) for item in failures)
        )
    return payload


def _normalize_snapshot_sha256(value: Any, *, source: str) -> str:
    """Validate one configured or reported acceptance snapshot checksum."""
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{source} has no valid snapshot manifest checksum.")
    return digest


def _expected_snapshot_manifest_sha256(
    gate_config: dict[str, Any],
    snapshot_manifest: Path,
) -> str:
    """Resolve the immutable snapshot identity carried by the operator contract."""
    configured = gate_config.get("snapshot_manifest_sha256")
    if configured is not None:
        return _normalize_snapshot_sha256(
            configured,
            source="Model acceptance gate",
        )

    # Legacy handoffs predate the explicit gate field. Prefer the producer's
    # immutable summary when available; only old ad-hoc configs fall back to
    # binding the bytes observed immediately before the subprocess starts.
    summary_path = snapshot_manifest.parent / "snapshot.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Unable to read acceptance snapshot summary: {summary_path}: {exc}"
            ) from exc
        if not isinstance(summary, dict):
            raise ValueError(
                f"Acceptance snapshot summary must contain an object: {summary_path}"
            )
        return _normalize_snapshot_sha256(
            summary.get("manifest_sha256"),
            source="Acceptance snapshot summary",
        )
    return _sha256_file(snapshot_manifest)


def _create_acceptance_invocation_dir(run_dir: Path) -> Path:
    """Atomically allocate one acceptance workspace below the training run."""
    resolved_run_dir = run_dir.expanduser().resolve()
    invocation_root = resolved_run_dir / ".acceptance_gate_runs"
    if invocation_root.is_symlink():
        raise ValueError(
            f"Acceptance invocation root cannot be a symbolic link: {invocation_root}"
        )
    invocation_root.mkdir(parents=True, exist_ok=True)
    if invocation_root.is_symlink():
        raise ValueError(
            f"Acceptance invocation root cannot be a symbolic link: {invocation_root}"
        )
    resolved_invocation_root = invocation_root.resolve()
    if not resolved_invocation_root.is_relative_to(resolved_run_dir):
        raise ValueError(
            "Acceptance invocation root escapes the training run: "
            f"{resolved_invocation_root}"
        )
    invocation_dir = Path(
        tempfile.mkdtemp(prefix="gate-", dir=resolved_invocation_root)
    ).resolve()
    if not invocation_dir.is_relative_to(resolved_invocation_root):
        raise ValueError(
            f"Acceptance invocation directory is unsafe: {invocation_dir}"
        )
    return invocation_dir


def _validate_acceptance_target_segment(value: str, *, label: str) -> None:
    """Reject target identifiers that could address another filesystem path."""
    path = Path(value)
    if (
        not value
        or value != value.strip()
        or path.is_absolute()
        or bool(path.drive)
        or len(path.parts) != 1
        or value in {".", ".."}
    ):
        raise ValueError(f"Acceptance {label} is not a safe path segment: {value!r}")


def _acceptance_policy_contract(gate_config: dict[str, Any]) -> dict[str, Any]:
    """Return the exact policy sent to and expected back from the runner."""
    return {
        "min_confirmed": int(gate_config.get("min_confirmed", 1)),
        "max_false_positives": int(
            gate_config.get("max_false_positives", 0)
        ),
        "max_false_negatives": int(
            gate_config.get("max_false_negatives", 0)
        ),
        "max_regressions": int(gate_config.get("max_regressions", 0)),
        "require_all_confirmed": bool(
            gate_config.get("require_all_confirmed", True)
        ),
        "require_no_errors": bool(gate_config.get("require_no_errors", True)),
        "require_baseline_predictions": True,
    }


def _write_acceptance_runner_log(
    path: Path,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
) -> None:
    """Persist subprocess output without assuming TimeoutExpired text types."""

    def normalized(value: str | bytes | None) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value or ""

    standard_output = normalized(stdout)
    standard_error = normalized(stderr)
    path.write_text(
        standard_output
        + (f"\n[stderr]\n{standard_error}" if standard_error else ""),
        encoding="utf-8",
    )


def _verify_acceptance_color_revision_contract(
    *,
    inference_project_root: Path,
    color_revisions_root: Path,
    report_path: Path,
    logger: logging.Logger,
    stage: str,
    timeout_seconds: float,
) -> None:
    """Use the inference-owned verifier to detect stale active revisions."""
    command = [
        sys.executable,
        "-m",
        "app.acceptance.color_revision_contract",
        "--revisions-root",
        str(color_revisions_root),
        "--report",
        str(report_path),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=inference_project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(float(timeout_seconds), 1.0),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"Color revision verification timed out during {stage}."
        ) from exc
    runner_log_path = report_path.parent / "runner.log"
    if runner_log_path.is_symlink():
        raise ValueError(
            f"Color revision runner log cannot be a symbolic link: {runner_log_path}"
        )
    try:
        with runner_log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[color-revision-verification:{stage}]\n")
            handle.write(completed.stdout)
            if completed.stderr:
                handle.write("\n[stderr]\n")
                handle.write(completed.stderr)
    except OSError as exc:
        raise OSError(
            f"Unable to retain color revision verification log: {runner_log_path}"
        ) from exc
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout).strip()
        raise ValueError(
            f"Active color revisions changed during {stage}: "
            f"{diagnostic or 'verification failed'}"
        )
    logger.info("Active color revision contract verified during %s.", stage)


def _validate_acceptance_color_revision_report(
    report: dict[str, Any],
    *,
    expected_target: dict[str, str],
) -> None:
    """Validate the report-side contract before invoking its owner verifier."""
    contract = report.get("color_revisions")
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise ValueError(
            "Model acceptance report has no supported color revision contract."
        )
    if contract.get("target") != expected_target:
        raise ValueError(
            "Model acceptance color revision target does not match the gate target."
        )
    enabled = contract.get("enabled")
    checker_type = contract.get("checker_type")
    entries = contract.get("entries")
    if (
        not isinstance(enabled, bool)
        or not isinstance(checker_type, str)
        or not isinstance(entries, list)
        or (enabled and not checker_type)
        or (not enabled and checker_type != "")
    ):
        raise ValueError(
            "Model acceptance color revision contract has invalid runtime fields."
        )
    identity = str(contract.get("identity_sha256") or "").strip().lower()
    if len(identity) != 64 or any(
        character not in "0123456789abcdef" for character in identity
    ):
        raise ValueError(
            "Model acceptance color revision contract has no valid identity."
        )


def _bind_validated_acceptance_report(
    report_path: Path,
    report: dict[str, Any],
) -> None:
    """Bind the parsed decision to the exact report bytes retained on disk."""
    if report_path.is_symlink():
        raise ValueError(
            f"Model acceptance report cannot be a symbolic link: {report_path}"
        )
    try:
        content = report_path.read_bytes()
        current = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to bind model acceptance report bytes: {report_path}"
        ) from exc
    if not isinstance(current, dict) or current != report:
        raise ValueError(
            "Model acceptance report changed after its decision was parsed."
        )
    report[_VALIDATED_ACCEPTANCE_REPORT_SHA_FIELD] = hashlib.sha256(
        content
    ).hexdigest()


def _assert_validated_acceptance_report_unchanged(
    report_path: Path,
    accepted_report: dict[str, Any],
) -> None:
    """Recheck exact report bytes while the publication lock is held."""
    if report_path.is_symlink():
        raise ValueError(
            f"Model acceptance report cannot be a symbolic link: {report_path}"
        )
    expected_sha256 = str(
        accepted_report.get(_VALIDATED_ACCEPTANCE_REPORT_SHA_FIELD) or ""
    )
    expected_payload = {
        key: value
        for key, value in accepted_report.items()
        if key != _VALIDATED_ACCEPTANCE_REPORT_SHA_FIELD
    }
    try:
        content = report_path.read_bytes()
        current_payload = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to recheck model acceptance report: {report_path}"
        ) from exc
    current_sha256 = hashlib.sha256(content).hexdigest()
    if (
        not expected_sha256
        or current_sha256 != expected_sha256
        or current_payload != expected_payload
    ):
        raise ValueError(
            "Model acceptance report changed before deployment publication."
        )


def _run_model_acceptance_gate(
    *,
    config: dict[str, Any],
    run_dir: Path,
    inference_project_root: Path,
    color_revisions_root: Path,
    product: str,
    area: str,
    candidate_config: dict[str, Any],
    candidate_weight: Path,
    color_model: Path | None,
    logger: logging.Logger,
) -> tuple[dict[str, Any], Path | None]:
    """Run the inference-owned headless gate against a job-scoped bundle."""
    ycfg = config.get("yolo_training", {}) or {}
    deploy_cfg = ycfg.get("deploy", {}) or {}
    gate_cfg = deploy_cfg.get("acceptance_gate", {}) or {}
    if not isinstance(gate_cfg, dict) or not gate_cfg.get("enabled", False):
        logger.info("Model acceptance gate disabled.")
        return {}, None

    dataset_root = Path(str(gate_cfg.get("dataset_root") or "")).expanduser()
    snapshot_manifest = Path(
        str(gate_cfg.get("snapshot_manifest") or "")
    ).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (inference_project_root / dataset_root).resolve()
    else:
        dataset_root = dataset_root.resolve()
    if not snapshot_manifest.is_absolute():
        snapshot_manifest = (dataset_root / snapshot_manifest).resolve()
    else:
        snapshot_manifest = snapshot_manifest.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Acceptance dataset root not found: {dataset_root}"
        )
    if not snapshot_manifest.is_file():
        raise FileNotFoundError(
            f"Acceptance snapshot manifest not found: {snapshot_manifest}"
        )
    expected_snapshot_sha256 = _expected_snapshot_manifest_sha256(
        gate_cfg,
        snapshot_manifest,
    )
    if _sha256_file(snapshot_manifest) != expected_snapshot_sha256:
        raise ValueError(
            "Acceptance snapshot manifest changed before model acceptance started."
        )

    runner_path = inference_project_root / "app" / "acceptance" / "headless.py"
    global_config = inference_project_root / "config.yaml"
    if not runner_path.is_file() or not global_config.is_file():
        raise FileNotFoundError(
            "Inference acceptance runtime is incomplete: "
            f"runner={runner_path}, config={global_config}"
        )

    _validate_acceptance_target_segment(product, label="product")
    _validate_acceptance_target_segment(area, label="area")
    invocation_dir = _create_acceptance_invocation_dir(run_dir)
    invocation_id = invocation_dir.name
    candidate_models_root = (invocation_dir / "candidate_models").resolve()
    candidate_station_dir = (
        candidate_models_root / product / area / "yolo"
    ).resolve()
    if not candidate_station_dir.is_relative_to(candidate_models_root):
        raise ValueError(
            "Acceptance candidate target escapes its invocation directory: "
            f"product={product!r} area={area!r}"
        )
    candidate_config_path = candidate_station_dir / "config.yaml"
    report_path = invocation_dir / "report.json"
    runner_log_path = invocation_dir / "runner.log"
    policy_contract = _acceptance_policy_contract(gate_cfg)
    expected_target = {
        "product": product,
        "area": area,
        "inference_type": "yolo",
    }

    try:
        _write_yaml_atomic(candidate_config_path, candidate_config)
        expected_config_sha256 = _sha256_file(candidate_config_path)
        expected_weight_sha256 = _sha256_file(candidate_weight)
        expected_color_sha256 = (
            _sha256_file(color_model) if color_model is not None else ""
        )

        command = [
            sys.executable,
            "-m",
            "app.acceptance.headless",
            "--project-root",
            str(inference_project_root),
            "--models-root",
            str(candidate_models_root),
            "--global-config",
            str(global_config),
            "--color-revisions-root",
            str(color_revisions_root),
            "--dataset-root",
            str(dataset_root),
            "--snapshot-manifest",
            str(snapshot_manifest),
            "--report",
            str(report_path),
            "--product",
            product,
            "--area",
            area,
            "--inference-type",
            "yolo",
            "--candidate-version",
            invocation_id,
            "--candidate-weight",
            str(candidate_weight),
            "--candidate-config",
            str(candidate_config_path),
            "--min-confirmed",
            str(policy_contract["min_confirmed"]),
            "--max-false-positives",
            str(policy_contract["max_false_positives"]),
            "--max-false-negatives",
            str(policy_contract["max_false_negatives"]),
            "--max-regressions",
            str(policy_contract["max_regressions"]),
        ]
        if color_model is not None:
            command.extend(["--color-model", str(color_model)])
        if not policy_contract["require_all_confirmed"]:
            command.append("--allow-pending")
        if not policy_contract["require_no_errors"]:
            command.append("--allow-errors")

        timeout_seconds = max(
            float(gate_cfg.get("timeout_seconds", 1800.0)), 1.0
        )
        logger.info(
            "Running frozen model acceptance set for %s/%s (%s, invocation=%s).",
            product,
            area,
            snapshot_manifest.parent.name,
            invocation_id,
        )
        try:
            completed = subprocess.run(
                command,
                cwd=inference_project_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            _write_acceptance_runner_log(
                runner_log_path,
                exc.stdout,
                exc.stderr,
            )
            raise TimeoutError(
                f"Model acceptance exceeded {timeout_seconds:.0f} seconds."
            ) from exc
        _write_acceptance_runner_log(
            runner_log_path,
            completed.stdout,
            completed.stderr,
        )
        for line in completed.stdout.splitlines():
            if line.startswith("[acceptance]"):
                logger.info("%s", line)
        if completed.returncode != 0:
            diagnostic_lines = (
                completed.stderr.splitlines() or completed.stdout.splitlines()
            )[-20:]
            for line in diagnostic_lines:
                logger.warning("%s", line)

        report = _load_model_acceptance_report(report_path, required=True)
        if completed.returncode != 0:
            raise ValueError(
                "Model acceptance runner failed with exit code "
                f"{completed.returncode}."
            )
        if report.get("target") != expected_target:
            raise ValueError(
                "Model acceptance report does not match the deployment target."
            )
        if report.get("policy") != policy_contract:
            raise ValueError(
                "Model acceptance report does not match the deployment policy."
            )
        _validate_acceptance_color_revision_report(
            report,
            expected_target=expected_target,
        )
        dataset = report.get("dataset") or {}
        if not isinstance(dataset, dict):
            raise ValueError(
                "Model acceptance report dataset must contain an object."
            )
        reported_snapshot_sha256 = _normalize_snapshot_sha256(
            dataset.get("snapshot_manifest_sha256"),
            source="Model acceptance report",
        )
        if reported_snapshot_sha256 != expected_snapshot_sha256:
            raise ValueError(
                "Model acceptance report does not match the expected snapshot "
                "manifest."
            )
        if _sha256_file(snapshot_manifest) != expected_snapshot_sha256:
            raise ValueError(
                "Acceptance snapshot manifest changed while model acceptance was "
                "running."
            )
        candidate = report.get("candidate") or {}
        if not isinstance(candidate, dict):
            raise ValueError(
                "Model acceptance report candidate must contain an object."
            )
        if candidate.get("version") != invocation_id:
            raise ValueError(
                "Model acceptance report does not match this deployment invocation."
            )
        if candidate.get("sha256") != expected_weight_sha256:
            raise ValueError(
                "Model acceptance report does not match the candidate weight."
            )
        if candidate.get("runtime_config_sha256") != expected_config_sha256:
            raise ValueError(
                "Model acceptance report does not match the candidate config."
            )
        if candidate.get("color_model_sha256") != expected_color_sha256:
            raise ValueError(
                "Model acceptance report does not match the candidate color model."
            )
        if _sha256_file(candidate_weight) != expected_weight_sha256:
            raise ValueError(
                "Candidate weight changed while model acceptance was running."
            )
        if _sha256_file(candidate_config_path) != expected_config_sha256:
            raise ValueError(
                "Candidate config changed while model acceptance was running."
            )
        if color_model is not None and (
            _sha256_file(color_model) != expected_color_sha256
        ):
            raise ValueError(
                "Candidate color model changed while model acceptance was running."
            )
        _verify_acceptance_color_revision_contract(
            inference_project_root=inference_project_root,
            color_revisions_root=color_revisions_root,
            report_path=report_path,
            logger=logger,
            stage="post-run",
            timeout_seconds=120.0,
        )
        _bind_validated_acceptance_report(report_path, report)
    except Exception as exc:
        evidence_note = (
            "Model acceptance diagnostics retained at " f"{invocation_dir}"
        )
        add_note = getattr(exc, "add_note", None)
        if callable(add_note):
            add_note(evidence_note)
        logger.warning("%s", evidence_note)
        raise

    candidate_bundle_removed = False
    try:
        shutil.rmtree(candidate_models_root)
        candidate_bundle_removed = True
    except FileNotFoundError:
        candidate_bundle_removed = True
    except OSError as exc:
        logger.warning(
            "Model acceptance passed, but candidate bundle cleanup was deferred: "
            "path=%s error=%s",
            candidate_models_root,
            exc,
        )
    logger.info(
        "Model acceptance evidence recorded: report=%s runner_log=%s "
        "candidate_bundle_removed=%s",
        report_path,
        runner_log_path,
        candidate_bundle_removed,
    )
    return report, report_path


def _verify_deployment_runtime_pair(
    config: dict[str, Any],
    runtime_path: Path,
    training_weight_path: Path,
) -> PairVerification | None:
    """Run the deployment-only ONNX/PT equivalence gate when configured."""
    if runtime_path.suffix.lower() != ".onnx":
        return None
    ycfg = config.get("yolo_training", {}) or {}
    deploy_cfg = ycfg.get("deploy", {}) or {}
    verification_cfg = deploy_cfg.get("runtime_pair_verification", {}) or {}
    if not isinstance(verification_cfg, dict) or not verification_cfg.get(
        "enabled", False
    ):
        return None

    configured_size = verification_cfg.get("input_size", ycfg.get("imgsz", 640))
    if isinstance(configured_size, (list, tuple)):
        input_size = max(int(value) for value in configured_size)
    else:
        input_size = int(configured_size or 640)
    return verify_runtime_pair(
        runtime_path,
        training_weight_path,
        input_size=input_size,
        rtol=float(verification_cfg.get("rtol", 1e-3)),
        atol=float(verification_cfg.get("atol", 1e-3)),
    )


def _validated_runtime_pair_proof(
    verification: PairVerification,
    *,
    deployed_runtime_path: Path,
    deployed_training_path: Path,
) -> dict[str, Any]:
    """Bind numerical equivalence proof to the exact staged deployment bytes."""
    runtime_sha256 = _sha256_file(deployed_runtime_path)
    training_sha256 = _sha256_file(deployed_training_path)
    if (
        runtime_sha256 != verification.runtime_sha256
        or training_sha256 != verification.training_weight_sha256
    ):
        raise ValueError(
            "ONNX/PT artifacts changed after runtime pair verification."
        )
    if verification.comparison.passed is not True:
        raise ValueError("ONNX/PT pair proof does not contain a passing decision.")
    proof: dict[str, Any] = {
        "schema_version": 1,
        "method": "numerical_equivalence",
        "runtime_sha256": runtime_sha256,
        "training_weight_sha256": training_sha256,
        "input_size": verification.input_size,
        "runtime_shape": list(verification.comparison.runtime_shape),
        "training_shape": list(verification.comparison.training_shape),
        "max_abs_error": verification.comparison.max_abs_error,
        "mean_abs_error": verification.comparison.mean_abs_error,
        "p99_abs_error": verification.comparison.p99_abs_error,
        "class_names": list(verification.comparison.class_names),
    }
    encoded = json.dumps(
        proof,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    proof["identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    return proof


def run_deploy(config: dict, args: Any) -> None:
    """Copy training artifacts to the inference models directory.

    Args:
        config: Pipeline configuration containing ``yolo_training.deploy``.
        args: Runtime args. Currently unused by this task.

    Raises:
        FileNotFoundError: If required config, weights, or color model files are missing.
        TimeoutError: If another deploy holds the target lock too long.
        ValueError: If product, area, or version config is invalid.
    """
    logger = logging.getLogger(__name__)
    ycfg = config.get("yolo_training", {})
    dcfg = ycfg.get("deploy", {})
    if not dcfg.get("enabled", False):
        logger.info("Deploy task disabled (yolo_training.deploy.enabled is false).")
        return

    target = resolve_yolo_deployment_target(config, args)
    product = target.product
    area = target.area
    validate_deployment_target(product, area)

    raw_dir = dcfg.get("inference_models_dir")
    if not raw_dir:
        raise ValueError(
            "deploy.inference_models_dir is not set. Configure the yolo11_inference "
            "models directory before running deploy."
        )

    inference_models_dir = Path(raw_dir).expanduser()
    if not inference_models_dir.is_absolute():
        inference_models_dir = Path.cwd() / inference_models_dir
    inference_models_dir = inference_models_dir.resolve()
    raw_project_root = dcfg.get("inference_project_root")
    inference_project_root = Path(
        raw_project_root if raw_project_root else inference_models_dir.parent
    ).expanduser()
    if not inference_project_root.is_absolute():
        inference_project_root = Path.cwd() / inference_project_root
    inference_project_root = inference_project_root.resolve()
    raw_color_revisions_root = dcfg.get("color_revisions_root")
    color_revisions_root = Path(
        raw_color_revisions_root
        if raw_color_revisions_root
        else inference_models_dir.parent / ".color_revisions"
    ).expanduser()
    if not color_revisions_root.is_absolute():
        color_revisions_root = Path.cwd() / color_revisions_root
    color_revisions_root = color_revisions_root.resolve()

    _, run_dir = detect_existing_weights(config)
    if not run_dir or not run_dir.exists():
        raise FileNotFoundError(
            "No YOLO training run found for deploy. Run yolo_train first "
            "or configure yolo_evaluation.weights / yolo_training.position_validation.weights."
        )

    exported_runtime = OnnxExporter.ensure(config, run_dir, logger)
    if OnnxExporter.is_enabled(config) and exported_runtime is None:
        raise RuntimeError(
            "Deployment requires a validated runtime export, but export failed."
        )

    gate_required = bool(
        ((config.get("yolo_evaluation", {}) or {}).get("gate", {}) or {}).get(
            "enabled", False
        )
    )
    evaluation_gate = _load_evaluation_gate(run_dir, required=gate_required)

    det_cfg_path = run_dir / "detection_config.yaml"
    if not det_cfg_path.exists():
        logger.error(
            "detection_config.yaml not found at %s; deploy aborted.", det_cfg_path
        )
        raise FileNotFoundError(det_cfg_path)

    try:
        det_cfg_data = yaml.safe_load(det_cfg_path.read_text(encoding="utf-8")) or {}
        det_cfg_data = rewrite_detection_config(det_cfg_data, product, area)
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValueError, TypeError, KeyError) as exc:
        logger.error("Failed to read detection config: %s", exc)
        raise

    candidate_position_area = _position_area_mapping(
        det_cfg_data,
        product,
        area,
    )
    # A disabled candidate can become enabled through the preserve/enable
    # activation policy. Therefore every deployed position contract requires
    # gate evidence, independent of its pre-policy enabled flag.
    position_contract_policy = str(
        dcfg.get("position_contract_policy") or "validate_candidate"
    ).strip().lower()
    if position_contract_policy not in {
        "validate_candidate",
        "preserve_disabled_station",
    }:
        raise ValueError(
            "deploy.position_contract_policy must be validate_candidate or "
            "preserve_disabled_station."
        )
    position_gate_required = (
        candidate_position_area is not None
        and position_contract_policy == "validate_candidate"
    )
    position_gate, position_gate_path = _load_position_gate(
        run_dir,
        required=position_gate_required,
    )
    if position_gate:
        if str(position_gate.get("product") or "") != product:
            raise ValueError("Position gate product does not match deploy target.")
        if str(position_gate.get("area") or "") != area:
            raise ValueError("Position gate area does not match deploy target.")
    _verify_position_gate_candidate_config(
        position_gate,
        det_cfg_data,
        candidate_position_area,
        required=position_gate_required,
    )
    position_report_path, position_report_sha256 = _verified_position_evidence(
        position_gate,
        path_field="candidate_report",
        hash_field="candidate_report_sha256",
        label="validation evidence",
        required=position_gate_required,
    )
    position_baseline_path, position_baseline_sha256 = (
        _verified_position_evidence(
            position_gate,
            path_field="baseline_report",
            hash_field="baseline_report_sha256",
            label="baseline evidence",
        )
    )
    position_calibration_path, position_calibration_sha256 = (
        _verified_position_evidence(
            position_gate,
            path_field="calibration_manifest",
            hash_field="calibration_manifest_sha256",
            label="calibration evidence",
        )
    )

    selected_weights_path = select_runtime_weight(
        run_dir, str(det_cfg_data.get("weights") or "")
    )
    selected_weights_name = selected_weights_path.name
    selected_extension = selected_weights_path.suffix or ".pt"
    training_weight_source = _resolve_runtime_training_pair(
        run_dir,
        selected_weights_path,
    )
    if not training_weight_source.is_file():
        raise FileNotFoundError(
            "Deployment requires the PT checkpoint paired with the runtime "
            f"artifact: {training_weight_source}"
        )
    pair_verification = _verify_deployment_runtime_pair(
        config,
        selected_weights_path,
        training_weight_source,
    )
    if pair_verification is not None:
        logger.info(
            "ONNX/PT pair verified (max_abs_error=%.6g, mean_abs_error=%.6g).",
            pair_verification.comparison.max_abs_error,
            pair_verification.comparison.mean_abs_error,
        )
    dest_dir = inference_models_dir / product / area / "yolo"
    weights_dest = dest_dir / "weights"
    dest_dir.mkdir(parents=True, exist_ok=True)
    weights_dest.mkdir(exist_ok=True)

    color_cfg_name = Path(det_cfg_data.get("color_model_path", "")).name
    if det_cfg_data.get("enable_color_check") and not color_cfg_name:
        raise FileNotFoundError(
            "enable_color_check is true but color_model_path is not configured."
        )
    station_config_path = dest_dir / "config.yaml"

    lock_dir = _acquire_deploy_lock(dest_dir, float(dcfg.get("lock_timeout", 30.0)))
    transaction: _DeploymentTransaction | None = None
    publication_lock: _ColorRevisionPublicationLock | None = None
    try:
        version = _resolve_version(
            dcfg.get("version", "auto"),
            weights_dest,
            product,
            area,
            selected_extension,
        )
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        versioned_name = (
            f"{product}_{area}_v{_version_str(version)}_{date_str}{selected_extension}"
        )
        versioned_path = weights_dest / versioned_name
        paired_training_name = (
            versioned_name
            if selected_extension.lower() == ".pt"
            else f"{Path(versioned_name).stem}.training.pt"
        )
        paired_training_path = weights_dest / paired_training_name

        if versioned_path.exists() and not dcfg.get("force", False):
            raise FileExistsError(
                f"Deployment version already exists: {versioned_path}. "
                "Use version=auto or deploy.force=true intentionally."
            )

        color_source = find_color_model_source(run_dir, color_cfg_name)
        color_source_kind = "training_run"
        color_destination = (
            (dest_dir / color_cfg_name).resolve() if color_cfg_name else None
        )
        if (
            color_destination is not None
            and color_destination.parent != dest_dir.resolve()
        ):
            raise ValueError("Resolved station color model path is unsafe.")
        if (
            color_source is None
            and color_destination is not None
            and dcfg.get("preserve_station_settings", True)
            and color_destination.is_file()
        ):
            color_source = color_destination
            color_source_kind = "existing_station"
        if det_cfg_data.get("enable_color_check") and color_source is None:
            raise FileNotFoundError(
                "enable_color_check is true but no color model/stat file was found "
                f"for {color_cfg_name}."
            )

        station_config_snapshot = _load_station_config_snapshot(station_config_path)
        deploy_config = dict(det_cfg_data)
        deploy_config = _preserve_station_fields(
            deploy_config,
            station_config_snapshot.values,
            dcfg,
        )
        deploy_config = _preserve_disabled_station_position_contract(
            deploy_config,
            station_config_snapshot.values,
            dcfg,
            product=product,
            area=area,
        )
        deploy_config = _apply_position_activation_policy(
            deploy_config,
            station_config_snapshot.values,
            dcfg,
            product=product,
            area=area,
        )
        deploy_config["weights"] = _runtime_config_path(
            versioned_path,
            inference_project_root,
        )
        if color_destination is not None:
            deploy_config["color_model_path"] = _runtime_config_path(
                color_destination,
                inference_project_root,
            )

        # The gate and final publication share this single prepared station
        # config. Only artifact addresses differ while the candidate still
        # lives in the training run; station fields are never re-merged.
        candidate_deploy_config = dict(deploy_config)
        candidate_deploy_config["weights"] = str(selected_weights_path.resolve())
        if color_source is not None and color_destination is not None:
            candidate_deploy_config["color_model_path"] = str(color_source.resolve())
        acceptance_report, acceptance_report_path = _run_model_acceptance_gate(
            config=config,
            run_dir=run_dir,
            inference_project_root=inference_project_root,
            color_revisions_root=color_revisions_root,
            product=product,
            area=area,
            candidate_config=candidate_deploy_config,
            candidate_weight=selected_weights_path,
            color_model=color_source,
            logger=logger,
        )
        _assert_station_config_unchanged(
            station_config_path,
            station_config_snapshot.identity,
        )

        transaction = _DeploymentTransaction(dest_dir)

        logger.info(
            "Deploying %s/%s/yolo to %s (version %s)",
            product,
            area,
            dest_dir,
            _version_str(version),
        )

        weight_sha256 = transaction.copy_verified(
            selected_weights_path,
            versioned_path,
        )
        accepted_weight_sha256 = (
            (acceptance_report.get("candidate") or {}).get("sha256")
            if acceptance_report
            else None
        )
        accepted_color_model_sha256 = (
            (acceptance_report.get("candidate") or {}).get(
                "color_model_sha256"
            )
            if acceptance_report
            else None
        )
        if (
            accepted_weight_sha256 is not None
            and accepted_weight_sha256 != weight_sha256
        ):
            raise ValueError(
                "Candidate weight changed after model acceptance completed."
            )
        if paired_training_path == versioned_path:
            paired_training_sha256 = weight_sha256
        else:
            paired_training_sha256 = transaction.copy_verified(
                training_weight_source,
                paired_training_path,
            )
        transaction.copy_verified(
            selected_weights_path, weights_dest / selected_weights_name
        )
        logger.info(
            "Copied runtime weight %s as %s.", selected_weights_name, versioned_name
        )

        for filename in [
            "best.pt",
            "last.pt",
            "best.onnx",
            "best.engine",
            "best.torchscript",
        ]:
            src = run_dir / "weights" / filename
            if src.exists() and src.name != selected_weights_name:
                transaction.copy_verified(src, weights_dest / filename)
                logger.info("Copied %s.", filename)

        for src_dir in _iter_export_dirs(run_dir / "weights"):
            dest_export_dir = weights_dest / src_dir.name
            transaction.replace_tree(src_dir, dest_export_dir)
            logger.info("Copied exported runtime directory %s.", src_dir.name)

        color_model_sha256: str | None = None
        if color_source is not None and color_destination is not None:
            if color_source.resolve() == color_destination:
                color_model_sha256 = _sha256_file(color_destination)
                logger.info("Preserved station colour model %s.", color_cfg_name)
            else:
                color_model_sha256 = transaction.copy_verified(
                    color_source, color_destination
                )
                logger.info(
                    "Copied colour model from %s to %s.",
                    color_source,
                    color_cfg_name,
                )
        if (
            accepted_color_model_sha256 is not None
            and accepted_color_model_sha256 != (color_model_sha256 or "")
        ):
            raise ValueError(
                "Candidate color model changed after model acceptance completed."
            )
        runtime_pair_proof = (
            _validated_runtime_pair_proof(
                pair_verification,
                deployed_runtime_path=versioned_path,
                deployed_training_path=paired_training_path,
            )
            if pair_verification is not None
            else None
        )
        if acceptance_report_path is not None:
            publication_lock = _acquire_color_revision_publication_lock(
                color_revisions_root,
                float(
                    dcfg.get(
                        "color_revision_publication_lock_timeout",
                        30.0,
                    )
                ),
            )
            _assert_validated_acceptance_report_unchanged(
                acceptance_report_path,
                acceptance_report,
            )
            _verify_acceptance_color_revision_contract(
                inference_project_root=inference_project_root,
                color_revisions_root=color_revisions_root,
                report_path=acceptance_report_path,
                logger=logger,
                stage="pre-publish",
                timeout_seconds=float(
                    dcfg.get("color_revision_verification_timeout", 15.0)
                ),
            )
            _assert_validated_acceptance_report_unchanged(
                acceptance_report_path,
                acceptance_report,
            )

        position_gate_relative: str | None = None
        position_report_relative: str | None = None
        position_baseline_relative: str | None = None
        position_calibration_relative: str | None = None
        acceptance_report_relative: str | None = None
        acceptance_report_sha256: str | None = None
        evidence_dir = dest_dir / "versions"
        if acceptance_report_path is not None:
            evidence_dir.mkdir(parents=True, exist_ok=True)
            acceptance_destination = (
                evidence_dir / f"{versioned_name}.model_acceptance.json"
            )
            transaction.copy_verified(
                acceptance_report_path,
                acceptance_destination,
            )
            acceptance_report_relative = (
                f"versions/{acceptance_destination.name}"
            )
            acceptance_report_sha256 = _sha256_file(
                acceptance_destination
            )
            if acceptance_report_sha256 != acceptance_report.get(
                _VALIDATED_ACCEPTANCE_REPORT_SHA_FIELD
            ):
                raise ValueError(
                    "Published model acceptance report does not match the "
                    "validated decision."
                )
        if position_gate_path is not None:
            evidence_dir.mkdir(parents=True, exist_ok=True)
            gate_destination = evidence_dir / f"{versioned_name}.position_gate.json"
            transaction.copy_verified(position_gate_path, gate_destination)
            position_gate_relative = f"versions/{gate_destination.name}"
        if position_report_path is not None:
            evidence_dir.mkdir(parents=True, exist_ok=True)
            report_destination = (
                evidence_dir / f"{versioned_name}.position_validation.json"
            )
            transaction.copy_verified(position_report_path, report_destination)
            position_report_relative = f"versions/{report_destination.name}"
            position_report_sha256 = _sha256_file(report_destination)
        if position_baseline_path is not None:
            evidence_dir.mkdir(parents=True, exist_ok=True)
            baseline_destination = (
                evidence_dir / f"{versioned_name}.position_baseline.json"
            )
            transaction.copy_verified(position_baseline_path, baseline_destination)
            position_baseline_relative = f"versions/{baseline_destination.name}"
            position_baseline_sha256 = _sha256_file(baseline_destination)
        if position_calibration_path is not None:
            evidence_dir.mkdir(parents=True, exist_ok=True)
            calibration_destination = (
                evidence_dir / f"{versioned_name}.position_calibration.json"
            )
            transaction.copy_verified(
                position_calibration_path,
                calibration_destination,
            )
            position_calibration_relative = (
                f"versions/{calibration_destination.name}"
            )
            position_calibration_sha256 = _sha256_file(calibration_destination)

        config_path = dest_dir / "config.yaml"
        deployed_position_area = _position_area_mapping(
            deploy_config,
            product,
            area,
        )
        position_config_sha256 = (
            _sha256_json(deploy_config.get("position_config"))
            if deployed_position_area is not None
            else None
        )
        training_metadata = _load_training_metadata(run_dir)
        trained_at = training_metadata.get(
            "trained_at"
        ) or datetime.datetime.fromtimestamp(
            selected_weights_path.stat().st_mtime
        ).astimezone().isoformat(timespec="seconds")
        deployed_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        config_snapshot_relative = f"versions/{versioned_name}.config.yaml"
        manifest = {
            "schema_version": 2,
            "deployed_version": _version_str(version),
            "deployed_date": date_str,
            "deployed_at": deployed_at,
            "trained_at": trained_at,
            "deployed_file": versioned_name,
            "model_type": "yolo",
            "runtime_format": selected_extension.lstrip("."),
            "file_size": versioned_path.stat().st_size,
            "weight_sha256": weight_sha256,
            "training_weight_file": paired_training_name,
            "training_weight_sha256": paired_training_sha256,
            "runtime_source_training_weight": (
                selected_weights_name
                if selected_extension.lower() == ".pt"
                else training_weight_source.name
            ),
            "runtime_pair_verified": pair_verification is not None,
            "runtime_pair_verification": runtime_pair_proof,
            "deployed_to": str(config_path),
            "product": product,
            "area": area,
            "dataset_hash": training_metadata.get("dataset_hash"),
            "training_config_hash": training_metadata.get("config_hash"),
            "evaluation_metrics": evaluation_gate.get("metrics", {}),
            "evaluation_gate_passed": evaluation_gate.get("passed"),
            "model_acceptance_gate_passed": (
                acceptance_report.get("passed")
                if acceptance_report
                else None
            ),
            "model_acceptance_metrics": acceptance_report.get("metrics", {}),
            "model_acceptance_baseline_metrics": acceptance_report.get(
                "baseline_metrics", {}
            ),
            "model_acceptance_report": acceptance_report_relative,
            "model_acceptance_report_sha256": acceptance_report_sha256,
            "model_acceptance_snapshot_sha256": (
                (acceptance_report.get("dataset") or {}).get(
                    "snapshot_manifest_sha256"
                )
                if acceptance_report
                else None
            ),
            "model_acceptance_color_revision_contract_sha256": (
                (acceptance_report.get("color_revisions") or {}).get(
                    "identity_sha256"
                )
                if acceptance_report
                else None
            ),
            "config_snapshot": config_snapshot_relative,
            "color_model_file": color_cfg_name or None,
            "color_model_sha256": color_model_sha256,
            "color_model_source": color_source_kind if color_model_sha256 else None,
            "position_runtime_enabled": bool(
                deployed_position_area.get("enabled", False)
                if deployed_position_area is not None
                else False
            ),
            "position_config_sha256": position_config_sha256,
            "position_gate_required": position_gate_required,
            "position_gate_passed": (
                position_gate.get("passed")
                if position_gate
                else None
            ),
            "position_gate_sha256": (
                _sha256_file(position_gate_path)
                if position_gate_path is not None
                else None
            ),
            "position_gate_report": position_gate_relative,
            "position_validation_sha256": position_report_sha256,
            "position_validation_report": position_report_relative,
            "position_baseline_sha256": position_baseline_sha256,
            "position_baseline_report": position_baseline_relative,
            "position_calibration_sha256": position_calibration_sha256,
            "position_calibration_manifest": position_calibration_relative,
            "position_metrics": position_gate.get("metrics", {}),
            "position_baseline_metrics": position_gate.get(
                "baseline_metrics"
            ),
        }
        # Each deployed weight keeps immutable metadata and the matching runtime
        # config. This makes later rollback deterministic instead of guessing
        # which thresholds/classes belonged to an old artifact.
        transaction.write_yaml(dest_dir / config_snapshot_relative, deploy_config)
        transaction.write_yaml(
            versioned_path.with_name(f"{versioned_path.name}.manifest.yaml"),
            manifest,
        )
        transaction.write_yaml(dest_dir / "deployment_manifest.yaml", manifest)
        # Publish config last: inference never observes a pointer to an
        # unverified or partially copied runtime artifact.
        _assert_station_config_unchanged(
            station_config_path,
            station_config_snapshot.identity,
        )
        transaction.write_yaml(config_path, deploy_config)
        logger.info("config.yaml written with weights %s.", versioned_name)
        try:
            (run_dir / "version_manifest.yaml").write_text(
                yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            logger.info("version_manifest.yaml written to %s.", run_dir)
        except OSError as exc:
            logger.warning("Could not write version_manifest.yaml: %s", exc)

        logger.info(
            "Deploy complete: %s/%s/yolo, version=%s, file=%s, trained_at=%s",
            product,
            area,
            _version_str(version),
            versioned_name,
            trained_at,
        )
    except BaseException as deployment_error:
        if transaction is not None:
            try:
                transaction.rollback()
            except DeploymentRollbackError as rollback_error:
                deployment_error.add_note(str(rollback_error))
                logger.critical("%s", rollback_error, exc_info=True)
        raise
    else:
        if transaction is not None:
            transaction.commit()
    finally:
        if publication_lock is not None:
            publication_lock.release(logger)
        _release_deploy_lock(lock_dir)


TASKS: List[Any] = [
    Task(
        name="deploy",
        run=run_deploy,
        description="Deploy artifacts directly to yolo11_inference models directory.",
        dependencies=["yolo_evaluation"],
    ),
]


def _iter_export_dirs(weights_dir: Path) -> list[Path]:
    """Return directory-based YOLO runtime artifacts created by Ultralytics."""
    if not weights_dir.exists():
        return []
    return sorted(
        path
        for path in weights_dir.iterdir()
        if path.is_dir() and path.name.endswith("_openvino_model")
    )
