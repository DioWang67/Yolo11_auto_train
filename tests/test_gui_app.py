import os
import pytest

pytest.importorskip("pytestqt")

if (
    os.environ.get("DISPLAY", "") == ""
    and os.environ.get("QT_QPA_PLATFORM") != "offscreen"
):
    pytest.skip(
        "Qt GUI tests require DISPLAY or QT_QPA_PLATFORM=offscreen",
        allow_module_level=True,
    )

from picture_tool.gui import pipeline_controller
from picture_tool.gui.app import PictureToolGUI


@pytest.fixture()
def gui(qtbot, tmp_path, monkeypatch):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("pipeline: {}\n", encoding="utf-8")

    monkeypatch.setattr(
        pipeline_controller.PipelineControllerMixin,
        "_default_config_path",
        lambda self: config_path,
    )

    widget = PictureToolGUI()
    qtbot.addWidget(widget)
    yield widget
    widget.close()


def test_log_message_tracks_history(gui):
    gui.log_message("hello world")

    assert gui._log_history[-1] == "hello world"
    assert "hello world" in gui.log_text.toPlainText()

    first_item = gui.status_list.item(0)
    assert first_item is not None
    # Changed from "->" to "⚪" in modern UI update
    assert "⚪" in first_item.text()


def test_rebuild_status_items_resets_entries(gui):
    gui._set_task_status("format_conversion", "running")
    gui._rebuild_status_items(default_state="waiting")
    first_item = gui.status_list.item(0)
    assert first_item is not None
    assert first_item.text().endswith("waiting")
