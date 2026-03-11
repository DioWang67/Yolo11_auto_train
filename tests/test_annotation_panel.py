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

@pytest.fixture
def panel(qtbot, manager_mock):
    widget = AnnotationPanel(manager=manager_mock)
    qtbot.addWidget(widget)
    return widget

def test_initial_state(panel):
    assert panel.annotation_classes == []
    assert panel.annotation_input_dir is None
    assert panel.annotation_output_dir is None

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
