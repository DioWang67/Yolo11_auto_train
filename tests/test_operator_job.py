from __future__ import annotations

import json
import socket
import time
from pathlib import Path

import pytest

from picture_tool.operator_job import (
    OperatorJobError,
    acquire_target_training_lock,
    clear_operator_job_process,
    refresh_operator_job_lease,
    release_target_training_lock,
    update_job_status,
)


def test_job_status_updates_atomically_and_preserves_identity(tmp_path: Path) -> None:
    status = tmp_path / "jobs" / "job-1" / "status.json"
    status.parent.mkdir(parents=True)
    status.write_text(
        json.dumps({"job_id": "job-1", "created_at": "2026-07-15T00:00:00Z"}),
        encoding="utf-8",
    )

    update_job_status(
        status,
        state="training",
        message="training",
        progress=35,
    )

    payload = json.loads(status.read_text(encoding="utf-8"))
    assert payload["job_id"] == "job-1"
    assert payload["state"] == "training"
    assert payload["progress"] == 35
    assert payload["updated_at"]
    assert payload["training_process_id"] > 0
    assert payload["training_process_host"]
    assert payload["heartbeat_at"]
    assert not list(status.parent.glob("*.tmp"))


def test_target_lock_blocks_second_job_and_releases_cleanly(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    first = acquire_target_training_lock(
        data_root,
        product="Cable1",
        area="A",
        job_id="job-1",
        timeout_seconds=0,
    )

    with pytest.raises(OperatorJobError, match="active training job"):
        acquire_target_training_lock(
            data_root,
            product="Cable1",
            area="A",
            job_id="job-2",
            timeout_seconds=0,
        )

    release_target_training_lock(first)
    second = acquire_target_training_lock(
        data_root,
        product="Cable1",
        area="A",
        job_id="job-2",
        timeout_seconds=0,
    )
    release_target_training_lock(second)
    assert not second.path.exists()


def test_recent_lock_from_dead_local_process_is_recovered_immediately(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    lock_path = (
        data_root
        / ".operator_handoff"
        / "target_locks"
        / "Cable1--A.lock"
    )
    lock_path.mkdir(parents=True)
    (lock_path / "owner.json").write_text(
        json.dumps(
            {
                "job_id": "crashed-job",
                "pid": 43210,
                "host": socket.gethostname(),
                "created_at": "2026-07-22T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "picture_tool.operator_job._process_is_alive",
        lambda process_id: process_id != 43210,
    )

    recovered = acquire_target_training_lock(
        data_root,
        product="Cable1",
        area="A",
        job_id="retry-job",
        timeout_seconds=0,
    )

    owner = json.loads((recovered.path / "owner.json").read_text(encoding="utf-8"))
    assert owner["job_id"] == "retry-job"
    release_target_training_lock(recovered)


def test_heartbeat_refreshes_lock_status_and_returns_cancel_once(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    status_path = data_root / ".operator_handoff" / "jobs" / "job-1" / "status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps({"job_id": "job-1", "state": "training"}),
        encoding="utf-8",
    )
    lock = acquire_target_training_lock(
        data_root,
        product="Cable1",
        area="A",
        job_id="job-1",
        timeout_seconds=0,
    )
    control_path = status_path.with_name("control.json")
    control_path.write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "request_id": "cancel-1",
                "action": "cancel",
                "requested_at": "2026-07-22T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    request = refresh_operator_job_lease(
        status_path,
        job_id="job-1",
        lock=lock,
    )

    assert request is not None
    assert request.request_id == "cancel-1"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    owner = json.loads((lock.path / "owner.json").read_text(encoding="utf-8"))
    assert status["training_lease_id"] == lock.lease_id
    assert status["heartbeat_at"]
    assert owner["heartbeat_at"]

    update_job_status(
        status_path,
        state="cancelling",
        message="stopping",
        handled_control_request_id=request.request_id,
    )
    assert (
        refresh_operator_job_lease(status_path, job_id="job-1", lock=lock)
        is None
    )
    release_target_training_lock(lock)


def test_clear_process_lease_keeps_resumable_job_state(tmp_path: Path) -> None:
    status_path = tmp_path / "jobs" / "job-1" / "status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "state": "waiting_annotation",
                "training_process_id": 123,
                "training_process_host": "host",
                "heartbeat_at": "2026-07-22T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    clear_operator_job_process(status_path, job_id="job-1")

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["state"] == "waiting_annotation"
    assert status["training_process_id"] == 0
    assert status["training_process_host"] == ""
    assert status["heartbeat_at"] == ""


def test_five_hundred_heartbeat_refreshes_remain_atomic_and_bounded(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    status_path = data_root / ".operator_handoff" / "jobs" / "stress-job" / "status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps({"job_id": "stress-job", "state": "training"}),
        encoding="utf-8",
    )
    lock = acquire_target_training_lock(
        data_root,
        product="Cable1",
        area="A",
        job_id="stress-job",
        timeout_seconds=0,
    )

    started = time.perf_counter()
    for _iteration in range(500):
        assert refresh_operator_job_lease(
            status_path,
            job_id="stress-job",
            lock=lock,
        ) is None
    elapsed_seconds = time.perf_counter() - started

    status = json.loads(status_path.read_text(encoding="utf-8"))
    owner = json.loads((lock.path / "owner.json").read_text(encoding="utf-8"))
    assert status["heartbeat_at"] == owner["heartbeat_at"]
    assert not list(status_path.parent.glob("*.tmp"))
    assert not list(lock.path.glob("*.tmp"))
    assert elapsed_seconds < 10.0
    release_target_training_lock(lock)
