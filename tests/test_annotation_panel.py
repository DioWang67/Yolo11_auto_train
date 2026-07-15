import pytest
from unittest.mock import MagicMock
from PyQt5.QtWidgets import QMessageBox
from picture_tool.gui.annotation_panel import AnnotationPanel

# Ensure we skip if no display (standard pattern in this repo)
pytest.importorskip("pytestqt")

@pytest.fixture
def manager_mock():
    m = MagicMock()
    m.config = {"yolo_training": {"class_names": ["cat", "dog"]}}
    return m

@pytest.fixture(autouse=True)
def empty_qsettings(monkeypatch):
    class EmptySettings:
        def value(self, key, default=None, **kwargs):
            return default

        def setValue(self, key, value):
            return None

    monkeypatch.setattr("picture_tool.gui.annotation_panel.QtCore.QSettings", EmptySettings)

@pytest.fixture
def panel(qtbot, manager_mock):
    widget = AnnotationPanel(manager=manager_mock)
    qtbot.addWidget(widget)
    return widget

def test_initial_state(panel):
    assert panel.annotation_classes == []
    assert panel.annotation_input_dir is None
    assert panel.annotation_output_dir is None


def test_operator_mode_hides_engineering_controls(panel):
    panel.set_operator_mode(True)

    assert panel.class_management_panel.isHidden()
    assert panel.annotation_input_edit.isHidden()
    assert panel.annotation_output_edit.isHidden()
    assert panel.validate_annotations_btn.isHidden()
    assert panel.operator_instruction_label.isHidden() is False


def test_operator_pending_schedules_annotation_tool_once(
    panel, monkeypatch, tmp_path
):
    callbacks = []
    monkeypatch.setattr(panel, "_scan_annotation_progress", MagicMock())
    monkeypatch.setattr(
        "picture_tool.gui.annotation_panel.QtCore.QTimer.singleShot",
        lambda _delay, callback: callbacks.append(callback),
    )

    panel.configure_operator_pending(
        tmp_path / "Cable1" / "A",
        ["defect"],
        tmp_path / "handoff.json",
    )
    panel.configure_operator_pending(
        tmp_path / "Cable1" / "A",
        ["defect"],
        tmp_path / "handoff.json",
    )

    assert panel.operator_mode_enabled is True
    assert callbacks == [panel._launch_labelimg]


def test_operator_annotation_close_triggers_automatic_validation(panel, monkeypatch):
    monkeypatch.setattr(panel.labelimg_launcher, "is_running", lambda: False)
    complete = MagicMock()
    monkeypatch.setattr(panel, "_complete_operator_pending", complete)

    panel._poll_operator_labelimg()

    complete.assert_called_once_with(automatic=True)

def test_add_class_success(panel, monkeypatch, qtbot):
    # Mock QInputDialog.getText
    monkeypatch.setattr(
        "PyQt5.QtWidgets.QInputDialog.getText",
        lambda *args, **kwargs: ("NewClass", True)
    )
    
    with qtbot.waitSignal(panel.message_logged):
        panel._add_annotation_class()
    
    assert "NewClass" in panel.annotation_classes
    assert panel.annotation_class_list.count() == 1
    assert panel.annotation_class_list.item(0).text() == "NewClass"

def test_add_duplicate_class_warning(panel, monkeypatch, qtbot):
    panel.annotation_classes = ["Existing"]
    
    monkeypatch.setattr(
        "PyQt5.QtWidgets.QInputDialog.getText",
        lambda *args, **kwargs: ("Existing", True)
    )
    
    # Mock QMessageBox.warning to avoid popup
    warning_mock = MagicMock()
    monkeypatch.setattr("PyQt5.QtWidgets.QMessageBox.warning", warning_mock)
    
    panel._add_annotation_class()
    
    # Validation
    warning_mock.assert_called_once()
    assert len(panel.annotation_classes) == 1

def test_import_classes_from_config(panel, qtbot, monkeypatch):
    # manager_mock has ["cat", "dog"]
    info_mock = MagicMock()
    monkeypatch.setattr("PyQt5.QtWidgets.QMessageBox.information", info_mock)
    
    with qtbot.waitSignal(panel.message_logged):
        panel._import_classes_from_config()
        
    assert "cat" in panel.annotation_classes
    assert "dog" in panel.annotation_classes
    assert panel.annotation_class_list.count() == 2

def test_read_classes_file_deduplicates_and_ignores_blank_lines(panel, tmp_path):
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("\ncat\n# comment\ndog\ncat\n\n", encoding="utf-8")

    assert panel._read_classes_file(classes_file) == ["cat", "dog"]

def test_read_classes_file_rejects_empty_file(panel, tmp_path):
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("\n# comment\n", encoding="utf-8")

    with pytest.raises(ValueError, match="沒有有效標籤"):
        panel._read_classes_file(classes_file)

def test_import_classes_from_file_dialog(panel, qtbot, monkeypatch, tmp_path):
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("cat\ndog\ncat\n", encoding="utf-8")
    info_mock = MagicMock()
    saved_settings = {}

    monkeypatch.setattr(
        "PyQt5.QtWidgets.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(classes_file), ""),
    )
    monkeypatch.setattr("PyQt5.QtWidgets.QMessageBox.information", info_mock)
    monkeypatch.setattr(
        panel,
        "_save_annotation_setting",
        lambda key, value: saved_settings.setdefault(key, value),
    )

    with qtbot.waitSignal(panel.message_logged):
        panel._import_classes_from_file_dialog()

    assert panel.annotation_classes == ["cat", "dog"]
    assert panel.annotation_class_list.count() == 2
    assert saved_settings["last_classes_file"] == str(classes_file)

def test_browse_annotation_paths_remembers_selection(panel, monkeypatch, tmp_path):
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "labels"
    input_dir.mkdir()
    output_dir.mkdir()
    saved_settings = {}

    monkeypatch.setattr(panel, "_scan_annotation_progress", MagicMock())
    monkeypatch.setattr(
        panel,
        "_save_annotation_setting",
        lambda key, value: saved_settings.setdefault(key, value),
    )
    monkeypatch.setattr(
        "PyQt5.QtWidgets.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(input_dir),
    )

    panel._browse_annotation_input()

    monkeypatch.setattr(
        "PyQt5.QtWidgets.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(output_dir),
    )

    panel._browse_annotation_output()

    assert panel.annotation_input_dir == input_dir
    assert panel.annotation_output_dir == output_dir
    assert saved_settings["input_dir"] == str(input_dir)
    assert saved_settings["output_dir"] == str(output_dir)

def test_delete_class(panel, monkeypatch, qtbot):
    panel.annotation_classes = ["ToDie"]
    panel._refresh_class_list()
    panel.annotation_class_list.setCurrentRow(0)
    
    # Mock Yes response
    monkeypatch.setattr(
        "PyQt5.QtWidgets.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.Yes
    )
    
    panel._delete_annotation_class()
    
    assert "ToDie" not in panel.annotation_classes
    assert "ToDie" not in panel.annotation_classes
    assert panel.annotation_class_list.count() == 0

def test_scan_starts_worker(panel, monkeypatch):
    panel.annotation_input_dir = MagicMock()
    panel.annotation_output_dir = MagicMock()
    
    # Mock Worker class
    worker_cls_mock = MagicMock()
    worker_instance = MagicMock()
    worker_cls_mock.return_value = worker_instance
    
    monkeypatch.setattr("picture_tool.gui.annotation_panel.AnnotationWorker", worker_cls_mock)
    
    panel._scan_annotation_progress()
    
    worker_cls_mock.assert_called_once()
    worker_instance.start.assert_called_once()

def test_on_scan_completed(panel, qtbot):
    stats = {
        "total_images": 100,
        "annotated_images": 50,
        "unannotated_images": [],
        "annotated_images_list": [],
        "progress_percent": 50.0
    }
    
    # Mock tracker to return empty distribution to avoid errors
    panel.annotation_tracker.get_class_distribution = MagicMock(return_value={})
    
    with qtbot.waitSignal(panel.message_logged):
        panel._on_scan_completed(stats)
        
    assert panel.annotation_progress_bar.value() == 50
    text = panel.annotation_stats_label.text()
    assert "總圖片：100" in text
    assert "已標註：50" in text

def test_on_scan_error(panel, qtbot, monkeypatch):
    # Mock QMessageBox to avoid blocking
    warning_mock = MagicMock()
    monkeypatch.setattr("PyQt5.QtWidgets.QMessageBox.warning", warning_mock)
    
    with qtbot.waitSignal(panel.message_logged) as blocker:
        panel._on_scan_error("Something went wrong")
        
    assert blocker.args[0] == "[ERROR] Scan failed: Something went wrong"
    warning_mock.assert_called_once()
