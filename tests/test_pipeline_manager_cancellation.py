from __future__ import annotations

from picture_tool.gui.pipeline_manager import PipelineManager


class _RunningWorker:
    def __init__(self) -> None:
        self.stop_requested = False

    def isRunning(self) -> bool:
        return True

    def request_stop(self) -> None:
        self.stop_requested = True


def test_cooperative_stop_request_does_not_wait_or_terminate() -> None:
    manager = PipelineManager()
    worker = _RunningWorker()
    manager.worker_thread = worker

    requested = manager.request_pipeline_stop()

    assert requested is True
    assert worker.stop_requested is True


def test_cooperative_stop_is_idempotent_when_no_worker_is_running() -> None:
    manager = PipelineManager()

    assert manager.request_pipeline_stop() is False
