from __future__ import annotations

import json
from pathlib import Path

import pytest

from picture_tool.operator_job import (
    OperatorJobError,
    acquire_target_training_lock,
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
