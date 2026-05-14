from __future__ import annotations

import subprocess
from pathlib import Path
from typing import IO, Any

from picture_tool.gui.labelimg_launcher import LabelImgLauncher


class FakeProcess:
    """Small subprocess stand-in for launch result checks."""

    def __init__(self, exit_code: int | None):
        self.pid = 12345
        self._exit_code = exit_code

    def poll(self) -> int | None:
        """Return the configured process state."""
        return self._exit_code


def make_launcher(executable: str = "labelImg") -> LabelImgLauncher:
    """Create a launcher without probing the host environment."""
    launcher = LabelImgLauncher.__new__(LabelImgLauncher)
    launcher.process = None
    launcher.labelimg_executable = executable
    launcher.last_error = None
    launcher.last_log_path = None
    return launcher


def test_launch_returns_false_when_input_dir_missing(tmp_path: Path) -> None:
    launcher = make_launcher()

    success = launcher.launch(
        tmp_path / "missing-images",
        tmp_path / "labels",
        None,
    )

    assert success is False
    assert "Input directory does not exist" in str(launcher.last_error)


def test_launch_reports_immediate_process_exit(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "labels"
    input_dir.mkdir()
    launcher = make_launcher()

    def fake_popen(*args: Any, stdout: IO[str], stderr: IO[str], **kwargs: Any) -> FakeProcess:
        stdout.write("stdout detail\n")
        stderr.write("stderr detail\n")
        return FakeProcess(exit_code=1)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("picture_tool.gui.labelimg_launcher.time.sleep", lambda _: None)

    success = launcher.launch(input_dir, output_dir, None)

    assert success is False
    assert launcher.last_log_path == tmp_path / "labelimg_launch.log"
    assert "exited immediately with code 1" in str(launcher.last_error)
    assert "stderr detail" in str(launcher.last_error)


def test_launch_returns_true_when_process_keeps_running(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "labels"
    input_dir.mkdir()
    launcher = make_launcher()

    def fake_popen(*args: Any, **kwargs: Any) -> FakeProcess:
        return FakeProcess(exit_code=None)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("picture_tool.gui.labelimg_launcher.time.sleep", lambda _: None)

    success = launcher.launch(input_dir, output_dir, None)

    assert success is True
    assert launcher.last_error is None
    assert launcher.last_log_path == tmp_path / "labelimg_launch.log"
