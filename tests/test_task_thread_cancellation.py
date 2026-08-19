from picture_tool.gui import task_thread
from picture_tool.gui.task_thread import WorkerThread


def test_cancelled_worker_does_not_report_success(tmp_path, monkeypatch):
    """An OP stop must never be presented as a completed deployment."""
    monkeypatch.setattr(task_thread.pipeline, "run_pipeline", lambda *args, **kwargs: None)
    worker = WorkerThread(
        ["dataset_splitter"],
        {"pipeline": {"log_file": str(tmp_path / "pipeline.log")}},
    )
    worker.request_stop()
    completed: list[bool] = []
    errors: list[str] = []
    worker.finished_signal.connect(lambda: completed.append(True))
    worker.error_occurred.connect(errors.append)

    worker.run()

    assert completed == []
    assert errors == ["Training was stopped before deployment completed."]
