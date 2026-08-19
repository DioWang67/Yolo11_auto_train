"""Shared operator-training job status and target lifecycle locking."""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_HEARTBEAT_TIMEOUT_SECONDS = 45.0
REMOTE_LOCK_RECOVERY_SECONDS = 10 * 60.0


class OperatorJobError(RuntimeError):
    """Raised when an operator job cannot safely advance."""


@dataclass(frozen=True)
class TargetTrainingLock:
    """Exclusive product/station lock held for one training lifecycle."""

    path: Path
    job_id: str
    lease_id: str


@dataclass(frozen=True)
class OperatorControlRequest:
    """One validated cross-process operator request."""

    request_id: str
    action: str
    requested_at: str


def update_job_status(
    status_path: Path | None,
    *,
    state: str,
    message: str,
    **values: Any,
) -> None:
    """Atomically publish a lifecycle state for the inference GUI.

    Args:
        status_path: Job-specific status file, or ``None`` for legacy handoffs.
        state: Stable lifecycle state.
        message: Concise operator-facing message.
        **values: Additional JSON-serializable status fields.

    Raises:
        OperatorJobError: If the status path is unsafe or cannot be written.
    """
    if status_path is None:
        return
    path = status_path.expanduser().resolve()
    if path.name != "status.json" or path.parent.name == "":
        raise OperatorJobError(f"Invalid operator job status path: {path}")
    payload = _read_json_mapping(path)
    payload.update({key: value for key, value in values.items() if value is not None})
    payload.update(
        {
            "schema_version": 1,
            "state": str(state),
            "message": str(message),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "training_process_id": os.getpid(),
            "training_process_host": socket.gethostname(),
            "heartbeat_at": datetime.now(timezone.utc).isoformat(),
            "heartbeat_timeout_seconds": DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
        }
    )
    try:
        _write_json_atomic(path, payload)
    except OSError as exc:
        raise OperatorJobError(f"Unable to update operator job status: {exc}") from exc


def acquire_target_training_lock(
    data_root: Path,
    *,
    product: str,
    area: str,
    job_id: str,
    timeout_seconds: float = 2.0,
) -> TargetTrainingLock:
    """Acquire one lock across snapshot, training, evaluation, and deployment.

    Args:
        data_root: Validated training data root.
        product: Product identifier.
        area: Station identifier.
        job_id: Immutable operator job identifier.
        timeout_seconds: Maximum wait for another active job.

    Returns:
        Lock handle that must be passed to :func:`release_target_training_lock`.

    Raises:
        OperatorJobError: If the target is already being trained.
    """
    locks_root = data_root / ".operator_handoff" / "target_locks"
    locks_root.mkdir(parents=True, exist_ok=True)
    lock_path = locks_root / f"{product}--{area}.lock"
    deadline = time.monotonic() + max(float(timeout_seconds), 0.0)
    while True:
        try:
            lock_path.mkdir()
            now = datetime.now(timezone.utc).isoformat()
            lease_id = uuid.uuid4().hex
            owner = {
                "schema_version": 2,
                "job_id": job_id,
                "lease_id": lease_id,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "process_identity": _process_identity(os.getpid()),
                "created_at": now,
                "heartbeat_at": now,
                "heartbeat_timeout_seconds": DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
            }
            _write_json_atomic(lock_path / "owner.json", owner)
            return TargetTrainingLock(
                path=lock_path,
                job_id=job_id,
                lease_id=lease_id,
            )
        except FileExistsError:
            if _remove_stale_lock(lock_path):
                continue
            if time.monotonic() >= deadline:
                owner = _read_json_mapping(lock_path / "owner.json")
                owner_job = str(owner.get("job_id") or "unknown")
                raise OperatorJobError(
                    f"{product}/{area} already has an active training job: {owner_job}"
                ) from None
            time.sleep(0.1)
        except OSError as exc:
            raise OperatorJobError(f"Unable to acquire training lock: {exc}") from exc


def release_target_training_lock(lock: TargetTrainingLock | None) -> None:
    """Release a target lock owned by the current operator job."""
    if lock is None:
        return
    owner_path = lock.path / "owner.json"
    owner = _read_json_mapping(owner_path)
    if owner and (
        str(owner.get("job_id") or "") != lock.job_id
        or str(owner.get("lease_id") or "") != lock.lease_id
    ):
        return
    try:
        owner_path.unlink(missing_ok=True)
        lock.path.rmdir()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise OperatorJobError(f"Unable to release training lock: {exc}") from exc


def refresh_operator_job_lease(
    status_path: Path | None,
    *,
    job_id: str,
    lock: TargetTrainingLock | None,
) -> OperatorControlRequest | None:
    """Refresh process/target leases and return an unhandled control request.

    The status and control files intentionally remain separate. The training
    process is the sole status writer while the inference process is the sole
    control writer, avoiding cross-process lost updates.
    """
    normalized_job_id = str(job_id or "").strip()
    if not normalized_job_id:
        raise OperatorJobError("Operator job_id is required for heartbeat refresh.")
    now = datetime.now(timezone.utc).isoformat()
    if lock is not None:
        owner_path = lock.path / "owner.json"
        owner = _read_json_mapping(owner_path)
        if (
            str(owner.get("job_id") or "") != lock.job_id
            or str(owner.get("lease_id") or "") != lock.lease_id
        ):
            raise OperatorJobError("Training lock ownership changed during execution.")
        owner.update(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "process_identity": _process_identity(os.getpid()),
                "heartbeat_at": now,
                "heartbeat_timeout_seconds": DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
            }
        )
        try:
            _write_json_atomic(owner_path, owner)
        except OSError as exc:
            raise OperatorJobError(f"Unable to refresh training lock: {exc}") from exc

    if status_path is None:
        return None
    path = _validated_status_path(status_path)
    payload = _read_json_mapping(path)
    recorded_job_id = str(payload.get("job_id") or "").strip()
    if recorded_job_id and recorded_job_id != normalized_job_id:
        raise OperatorJobError("Operator status identity changed during execution.")
    payload.update(
        {
            "schema_version": 1,
            "job_id": normalized_job_id,
            "training_process_id": os.getpid(),
            "training_process_host": socket.gethostname(),
            "heartbeat_at": now,
            "heartbeat_timeout_seconds": DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
        }
    )
    if lock is not None:
        payload["training_lease_id"] = lock.lease_id
    try:
        _write_json_atomic(path, payload)
    except OSError as exc:
        raise OperatorJobError(f"Unable to refresh operator heartbeat: {exc}") from exc
    return _read_pending_control_request(path, normalized_job_id, payload)


def clear_operator_job_process(status_path: Path | None, *, job_id: str) -> None:
    """Clear process ownership when a resumable non-training window closes."""
    if status_path is None:
        return
    path = _validated_status_path(status_path)
    payload = _read_json_mapping(path)
    if str(payload.get("job_id") or "") not in {"", str(job_id)}:
        raise OperatorJobError("Cannot clear a status owned by another job.")
    payload.update(
        {
            "training_process_id": 0,
            "training_process_host": "",
            "heartbeat_at": "",
            "training_lease_id": "",
        }
    )
    try:
        _write_json_atomic(path, payload)
    except OSError as exc:
        raise OperatorJobError(f"Unable to clear operator process lease: {exc}") from exc


def _remove_stale_lock(lock_path: Path) -> bool:
    owner_path = lock_path / "owner.json"
    owner = _read_json_mapping(owner_path)
    age_seconds = _lease_age_seconds(owner, owner_path)
    same_host = str(owner.get("host") or "") == socket.gethostname()
    try:
        owner_pid = int(owner.get("pid", 0))
    except (TypeError, ValueError):
        owner_pid = 0
    if same_host and owner_pid > 0:
        # A verified local PID is authoritative. If it has exited, recover the
        # lock immediately instead of treating the recent lock as active for
        # twelve hours after a hard crash.
        if _process_is_alive(owner_pid):
            recorded_identity = str(owner.get("process_identity") or "")
            current_identity = _process_identity(owner_pid)
            if not recorded_identity or recorded_identity == current_identity:
                return False
    elif owner and age_seconds < REMOTE_LOCK_RECOVERY_SECONDS:
        # A remote process cannot be queried safely. Its heartbeat lease is the
        # conservative recovery signal; a short network pause cannot unlock it.
        return False
    try:
        owner_path.unlink(missing_ok=True)
        lock_path.rmdir()
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _process_is_alive(pid: int) -> bool:
    if os.name == "nt":
        return bool(_process_identity(pid))
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _process_identity(pid: int) -> str:
    """Return a PID-reuse-safe process creation identity when available."""
    if pid <= 0:
        return ""
    if os.name == "nt":
        try:
            import ctypes

            process_query_limited_information = 0x1000
            windll = getattr(ctypes, "windll", None)
            if windll is None:
                return ""
            kernel32 = windll.kernel32
            handle = kernel32.OpenProcess(
                process_query_limited_information,
                False,
                pid,
            )
            if not handle:
                return ""
            try:
                creation = ctypes.c_ulonglong()
                exit_time = ctypes.c_ulonglong()
                kernel = ctypes.c_ulonglong()
                user = ctypes.c_ulonglong()
                if not kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(creation),
                    ctypes.byref(exit_time),
                    ctypes.byref(kernel),
                    ctypes.byref(user),
                ):
                    return ""
                return f"windows-filetime:{creation.value}"
            finally:
                kernel32.CloseHandle(handle)
        except (AttributeError, OSError, ValueError):
            return ""
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        _, separator, suffix = stat_text.rpartition(")")
        fields = suffix.split() if separator else []
        return f"proc-start:{fields[19]}" if len(fields) > 19 else ""
    except (OSError, IndexError):
        return ""


def _lease_age_seconds(owner: dict[str, Any], owner_path: Path) -> float:
    timestamp = str(owner.get("heartbeat_at") or owner.get("created_at") or "")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())
    except (TypeError, ValueError):
        try:
            return max(0.0, time.time() - owner_path.stat().st_mtime)
        except FileNotFoundError:
            return float("inf")


def _read_pending_control_request(
    status_path: Path,
    job_id: str,
    status: dict[str, Any],
) -> OperatorControlRequest | None:
    payload = _read_json_mapping(status_path.with_name("control.json"))
    if str(payload.get("job_id") or "") != job_id:
        return None
    request_id = str(payload.get("request_id") or "").strip()
    action = str(payload.get("action") or "").strip().lower()
    requested_at = str(payload.get("requested_at") or "").strip()
    if (
        not request_id
        or action != "cancel"
        or not requested_at
        or request_id == str(status.get("handled_control_request_id") or "")
    ):
        return None
    return OperatorControlRequest(
        request_id=request_id,
        action=action,
        requested_at=requested_at,
    )


def _validated_status_path(status_path: Path) -> Path:
    path = status_path.expanduser().resolve()
    if path.name != "status.json" or not path.parent.name:
        raise OperatorJobError(f"Invalid operator job status path: {path}")
    return path


def _read_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
