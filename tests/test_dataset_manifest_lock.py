from __future__ import annotations

import threading
from pathlib import Path

from picture_tool.dataset_manifest_lock import dataset_manifest_lock
from picture_tool.pending_annotations import _target_lock


def test_portable_and_annotation_transactions_share_target_lock(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "data" / "Cable1" / "A"
    owner_acquired = threading.Event()
    release_owner = threading.Event()
    contender_started = threading.Event()
    contender_acquired = threading.Event()
    failures: list[BaseException] = []

    def own_portable_transaction() -> None:
        try:
            with dataset_manifest_lock(dataset_root, timeout_seconds=2.0):
                owner_acquired.set()
                if not release_owner.wait(2.0):
                    raise TimeoutError("test did not release manifest owner")
        except BaseException as exc:  # pragma: no cover - surfaced below
            failures.append(exc)

    def enter_annotation_transaction() -> None:
        contender_started.set()
        try:
            with _target_lock(dataset_root, timeout_seconds=2.0):
                contender_acquired.set()
        except BaseException as exc:  # pragma: no cover - surfaced below
            failures.append(exc)

    owner = threading.Thread(target=own_portable_transaction)
    contender = threading.Thread(target=enter_annotation_transaction)
    owner.start()
    assert owner_acquired.wait(1.0)
    contender.start()
    assert contender_started.wait(1.0)
    assert not contender_acquired.wait(0.15)

    release_owner.set()
    owner.join(timeout=2.0)
    contender.join(timeout=2.0)

    assert not owner.is_alive()
    assert not contender.is_alive()
    assert failures == []
    assert contender_acquired.is_set()
