"""Cross-process serialization for mutable dataset manifests."""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator


_LOGGER = logging.getLogger(__name__)
_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[Path, threading.Lock] = {}


class DatasetManifestLockTimeoutError(TimeoutError):
    """Raised when another process owns a dataset manifest transaction."""


def _thread_lock_for(lock_path: Path) -> threading.Lock:
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(lock_path, threading.Lock())


def _try_lock_file(handle: BinaryIO) -> None:
    """Acquire one non-blocking byte/file lock on the current platform."""
    handle.seek(0)
    if os.name == "nt":
        msvcrt = importlib.import_module("msvcrt")
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    fcntl = importlib.import_module("fcntl")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        msvcrt = importlib.import_module("msvcrt")
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    fcntl = importlib.import_module("fcntl")
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _cross_process_lock(
    lock_path: Path,
    timeout_seconds: float,
) -> Iterator[None]:
    if timeout_seconds < 0:
        raise ValueError("Dataset manifest lock timeout cannot be negative.")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    thread_lock = _thread_lock_for(lock_path)
    if not thread_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
        raise DatasetManifestLockTimeoutError(
            f"Timed out waiting for dataset manifest lock: {lock_path}"
        )

    handle: BinaryIO | None = None
    file_locked = False
    try:
        handle = lock_path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        while not file_locked:
            try:
                _try_lock_file(handle)
                file_locked = True
            except OSError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DatasetManifestLockTimeoutError(
                        f"Timed out waiting for dataset manifest lock: {lock_path}"
                    ) from exc
                time.sleep(min(0.05, remaining))
        yield
    finally:
        if handle is not None:
            if file_locked:
                try:
                    _unlock_file(handle)
                except OSError as exc:
                    _LOGGER.warning(
                        "Dataset manifest lock release required handle cleanup: "
                        "path=%s error=%s",
                        lock_path,
                        exc,
                    )
            handle.close()
        thread_lock.release()


@contextmanager
def dataset_manifest_lock(
    dataset_root: str | Path,
    *,
    timeout_seconds: float = 15.0,
) -> Iterator[None]:
    """Serialize manifest read-modify-write transactions for one target.

    The persistent lock file is intentional. The operating system releases its
    byte/file lock when a process exits, so a crash cannot leave a stale owner.
    A process-local lock also protects platforms whose file locks do not
    serialize independent threads in the same process.
    """
    root = Path(dataset_root).expanduser().resolve()
    scope_identity = os.path.normcase(str(root)).encode("utf-8")
    scope_hash = hashlib.sha256(scope_identity).hexdigest()
    lock_path = (
        root.parents[1]
        / ".operator_handoff"
        / "dataset_locks"
        / f"{scope_hash}.lock"
    )
    with _cross_process_lock(lock_path, timeout_seconds):
        yield


@contextmanager
def portable_import_lock(
    data_root: str | Path,
    *,
    timeout_seconds: float = 15.0,
) -> Iterator[None]:
    """Serialize package imports without a stale lock after process failure."""
    root = Path(data_root).expanduser().resolve()
    lock_path = root / ".operator_handoff" / "portable_import.lock"
    with _cross_process_lock(lock_path, timeout_seconds):
        yield


@contextmanager
def master_manifest_lock(
    data_root: str | Path,
    *,
    timeout_seconds: float = 15.0,
) -> Iterator[None]:
    """Serialize rebuilds of dataset-wide derived manifest files."""
    root = Path(data_root).expanduser().resolve()
    lock_path = root / ".operator_handoff" / "master_manifest.lock"
    with _cross_process_lock(lock_path, timeout_seconds):
        yield
