import pytest
import yaml
from PyQt5.QtWidgets import QApplication

from picture_tool.gui.wizards import NewProjectWizard




def test_wizard_create_structure(qtbot, tmp_path):
    """Test standard project creation."""
    wizard = NewProjectWizard()

    project_name = "MyTestProject"
    project_root = tmp_path / project_name

    # Simulate creation
    wizard._create_structure(project_root)

    assert project_root.exists()
    assert (project_root / "data" / "raw" / "images").exists()
    assert (project_root / "config.yaml").exists()

    # Verify config content
    with (project_root / "config.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["yolo_training"]["name"] == "train"
    # Verify augmentation key exists to prevent KeyError
    assert "yolo_augmentation" in cfg
    # Verify splitter config
    assert ".jpg" in cfg["train_test_split"]["input_formats"]
    # Verify path style (forward slashes)
    img_dir = cfg["train_test_split"]["input"]["image_dir"]
    assert "/" in img_dir or "\\" not in img_dir  # Should use posix style
    assert project_name in img_dir
    assert "augmented" in img_dir


def test_wizard_chinese_path(qtbot, tmp_path):
    """Test project creation with Chinese characters in path."""
    wizard = NewProjectWizard()

    project_name = "測試專案"
    project_root = tmp_path / project_name

    wizard._create_structure(project_root)

    assert project_root.exists()
    assert (project_root / "config.yaml").exists()

    with (project_root / "config.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Verify Chinese path is preserved in config
    log_file = cfg["pipeline"]["log_file"]
    assert "測試專案" in log_file


def test_wizard_validation(qtbot, tmp_path, monkeypatch):
    """Test input validation."""
    wizard = NewProjectWizard()

    # Mock MessageBox to prevent blocking
    monkeypatch.setattr("PyQt5.QtWidgets.QMessageBox.warning", lambda *args: None)

    # 1. Missing info
    wizard.name_edit.setText("")
    wizard.location_edit.setText(str(tmp_path))
    wizard.create_project()
    assert not hasattr(wizard, "created_path")  # Should fail

    # 2. Existing directory
    (tmp_path / "Existing").mkdir()
    wizard.name_edit.setText("Existing")
    wizard.create_project()
    assert not hasattr(wizard, "created_path")  # Should fail (already exists)
