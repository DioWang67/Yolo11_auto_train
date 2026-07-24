import sys

import pytest

from picture_tool.gui import app as gui_app
from picture_tool.gui.main_window import _operator_epoch_status


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "[YOLO Lifecycle] Starting Epoch 1/20",
            ("模型訓練中：Epoch 1/20", 40),
        ),
        (
            "[YOLO Lifecycle] Finished Epoch 10/20",
            ("模型訓練中：Epoch 10/20", 58),
        ),
        (
            "[YOLO Lifecycle] Finished Epoch 20/20",
            ("模型訓練中：Epoch 20/20", 77),
        ),
        ("unrelated log message", None),
    ],
)
def test_operator_epoch_status_maps_training_into_full_pipeline_progress(
    message, expected
):
    assert _operator_epoch_status(message) == expected


def test_background_handoff_keeps_orchestrator_hidden(monkeypatch, tmp_path):
    handoff = tmp_path / "handoff.json"
    handoff.write_text("{}", encoding="utf-8")
    events = []

    class FakeApplication:
        def __init__(self, arguments):
            events.append(("application", arguments))

        def setFont(self, font):
            events.append(("font", font))

        def setQuitOnLastWindowClosed(self, enabled):
            events.append(("quit_on_last_window_closed", enabled))

        def exec_(self):
            events.append(("exec",))
            return 0

    class FakeWindow:
        def set_background_mode(self, enabled):
            events.append(("background", enabled))

        def show(self):
            events.append(("show",))

        def apply_operator_handoff(self, path):
            events.append(("handoff", path))

    monkeypatch.setattr(gui_app, "QApplication", FakeApplication)
    monkeypatch.setattr(gui_app, "MainWindow", FakeWindow)
    monkeypatch.setattr(gui_app.QtGui, "QFont", lambda *_args: "font")
    monkeypatch.setattr(
        gui_app.QTimer,
        "singleShot",
        lambda _delay, callback: callback(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["picture-tool", "--handoff", str(handoff), "--background"],
    )

    with pytest.raises(SystemExit) as exit_info:
        gui_app.main()

    assert exit_info.value.code == 0
    assert ("background", True) in events
    assert ("quit_on_last_window_closed", False) in events
    assert ("handoff", str(handoff)) in events
    assert ("show",) not in events


def test_visible_handoff_still_shows_window_for_manual_resume(
    monkeypatch, tmp_path
):
    handoff = tmp_path / "handoff.json"
    handoff.write_text("{}", encoding="utf-8")
    events = []

    class FakeApplication:
        def __init__(self, _arguments):
            pass

        def setFont(self, _font):
            pass

        def exec_(self):
            return 0

    class FakeWindow:
        def set_background_mode(self, enabled):
            events.append(("background", enabled))

        def show(self):
            events.append(("show",))

        def apply_operator_handoff(self, path):
            events.append(("handoff", path))

    monkeypatch.setattr(gui_app, "QApplication", FakeApplication)
    monkeypatch.setattr(gui_app, "MainWindow", FakeWindow)
    monkeypatch.setattr(gui_app.QtGui, "QFont", lambda *_args: "font")
    monkeypatch.setattr(
        gui_app.QTimer,
        "singleShot",
        lambda _delay, callback: callback(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["picture-tool", "--handoff", str(handoff)],
    )

    with pytest.raises(SystemExit):
        gui_app.main()

    assert ("background", False) in events
    assert ("show",) in events
