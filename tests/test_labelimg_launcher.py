from __future__ import annotations

import os
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


def test_launch_reports_process_exit_without_blocking_the_ui(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "labels"
    input_dir.mkdir()
    output_dir.mkdir()
    (output_dir / "classes.txt").write_text("part\n", encoding="utf-8")
    launcher = make_launcher()

    def fake_popen(*args: Any, stdout: IO[str], stderr: IO[str], **kwargs: Any) -> FakeProcess:
        stdout.write("stdout detail\n")
        stderr.write("stderr detail\n")
        return FakeProcess(exit_code=1)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    success = launcher.launch(input_dir, output_dir, None)

    assert success is True
    assert launcher.last_log_path == tmp_path / "labelimg_launch.log"
    exit_error = launcher.process_exit_error()
    assert "exited with code 1" in str(exit_error)
    assert "stderr detail" in str(exit_error)


def test_launch_returns_true_when_process_keeps_running(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "labels"
    input_dir.mkdir()
    output_dir.mkdir()
    (output_dir / "classes.txt").write_text("part\n", encoding="utf-8")
    launcher = make_launcher()

    def fake_popen(*args: Any, **kwargs: Any) -> FakeProcess:
        return FakeProcess(exit_code=None)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    success = launcher.launch(input_dir, output_dir, None)

    assert success is True
    assert launcher.last_error is None
    assert launcher.last_log_path == tmp_path / "labelimg_launch.log"


def test_completed_label_paths_detects_an_explicit_same_content_save(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "labels"
    input_dir.mkdir()
    output_dir.mkdir()
    (output_dir / "classes.txt").write_text("part\n", encoding="utf-8")
    label = output_dir / "sample.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    process = FakeProcess(exit_code=None)
    launcher = make_launcher()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)

    assert launcher.launch(input_dir, output_dir, None) is True
    original = label.read_text(encoding="utf-8")
    previous_mtime_ns = label.stat().st_mtime_ns
    label.write_text(original, encoding="utf-8")
    os.utime(
        label,
        ns=(label.stat().st_atime_ns, previous_mtime_ns + 1_000_000),
    )
    process._exit_code = 0

    assert launcher.completed_label_paths() == (label.resolve(),)


def test_prepare_environment_writes_both_labelimg_class_files(tmp_path: Path) -> None:
    launcher = make_launcher()
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "labels"
    input_dir.mkdir()

    success = launcher.prepare_environment(
        ["Black", "Green"], input_dir, output_dir
    )

    assert success is True
    expected = "Black\nGreen\n"
    assert (tmp_path / "predefined_classes.txt").read_text(
        encoding="utf-8"
    ) == expected
    assert (output_dir / "classes.txt").read_text(encoding="utf-8") == expected


def test_launch_rejects_missing_yolo_class_file(tmp_path: Path) -> None:
    launcher = make_launcher()
    input_dir = tmp_path / "images"
    input_dir.mkdir()

    success = launcher.launch(input_dir, tmp_path / "labels", None)

    assert success is False
    assert "YOLO class file is missing" in str(launcher.last_error)
