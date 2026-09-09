import yaml

from picture_tool.gui.wizards import NewProjectWizard




def test_wizard_create_structure(qtbot, tmp_path):
    """Test standard project creation."""
    wizard = NewProjectWizard()

    project_name = "MyTestProject"
    project_root = tmp_path / project_name

    # Simulate creation
    wizard._create_structure(project_root)

    assert project_root.exists()
    # Station level defaults to "A" when the dialog fields are untouched.
    assert (project_root / "data" / project_name / "A" / "raw" / "images").exists()
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
    assert "processed" in img_dir


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


def test_wizard_builds_product_area_layout_the_handoff_accepts(qtbot, tmp_path):
    """The dataset must be keyed by product/area, not by project name.

    operator_handoff validates a target against data/<product>/<area>, so a
    project laid out any other way cannot enter the retraining flow.
    """
    wizard = NewProjectWizard()
    wizard.product_edit.setText("Cable1")
    wizard.area_edit.setText("B")

    project_root = tmp_path / "StationProject"
    wizard._create_structure(project_root)

    dataset_root = project_root / "data" / "Cable1" / "B"
    assert (dataset_root / "raw" / "images").is_dir()
    assert (dataset_root / "raw" / "labels").is_dir()
    # Where review_dataset_manifest.csv will live; the handoff requires it.
    assert (dataset_root / "metadata").is_dir()
    assert (project_root / "runs" / "Cable1" / "B" / "train").is_dir()

    cfg = yaml.safe_load((project_root / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["product"] == "Cable1"
    assert cfg["area"] == "B"
    assert "/Cable1/B/" in cfg["train_test_split"]["output"]["output_dir"]


def test_wizard_rejects_path_segments_in_product_or_area(qtbot, tmp_path, monkeypatch):
    """A station name is a folder name; it must not redirect the dataset."""
    warnings: list[str] = []
    monkeypatch.setattr(
        "PyQt5.QtWidgets.QMessageBox.warning",
        lambda _parent, _title, message, *args: warnings.append(message),
    )
    wizard = NewProjectWizard()
    wizard.name_edit.setText("Escaping")
    wizard.location_edit.setText(str(tmp_path))
    wizard.product_edit.setText("../elsewhere")
    wizard.area_edit.setText("A")

    wizard.create_project()

    assert not hasattr(wizard, "created_path")
    assert not (tmp_path / "Escaping").exists()
    assert warnings and "separators" in warnings[0]
