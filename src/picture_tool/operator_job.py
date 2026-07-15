"""Shared operator-training job status and target lifecycle locking."""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class OperatorJobError(RuntimeError):
    """Raised when an operator job cannot safely advance."""


@dataclass(frozen=True)
class TargetTrainingLock:
    """Exclusive product/station lock held for one training lifecycle."""

    path: Path
    job_id: str


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
            owner = {
                "job_id": job_id,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_json_atomic(lock_path / "owner.json", owner)
            return TargetTrainingLock(path=lock_path, job_id=job_id)
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
    if owner and str(owner.get("job_id") or "") != lock.job_id:
        return
    try:
        owner_path.unlink(missing_ok=True)
        lock.path.rmdir()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise OperatorJobError(f"Unable to release training lock: {exc}") from exc


def _remove_stale_lock(lock_path: Path) -> bool:
    owner_path = lock_path / "owner.json"
    owner = _read_json_mapping(owner_path)
    try:
        age_seconds = time.time() - lock_path.stat().st_mtime
    except FileNotFoundError:
        return True
    same_host = str(owner.get("host") or "") == socket.gethostname()
    try:
        owner_pid = int(owner.get("pid", 0))
    except (TypeError, ValueError):
        owner_pid = 0
    owner_alive = same_host and owner_pid > 0 and _process_is_alive(owner_pid)
    if owner_alive or (owner and age_seconds < 12 * 60 * 60):
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
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


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
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
