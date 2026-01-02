
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from picture_tool.gui.wizards import NewProjectWizard
from picture_tool.tasks import augmentation, conversion, quality, training

from PyQt5.QtWidgets import QApplication

MODULES_TO_TEST = [augmentation, conversion, quality, training]

@pytest.fixture(scope="session")
def qapp():
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()

@pytest.fixture
def default_config(qapp, tmp_path):
    """Generate the exact config AND structure that the Wizard produces."""
    wizard = NewProjectWizard()
    # Actually create the folders so exist() checks pass
    project_dir = tmp_path / "TestProject"
    wizard._create_structure(project_dir)

    # Create dummy training run directory for position_validation
    (project_dir / "runs" / "detect" / "train").mkdir(parents=True, exist_ok=True)
    
    # Reload the generated config so paths are correct
    import yaml
    with (project_dir / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Assert Wizard generated class_names
    assert "class_names" in config["yolo_training"], "Wizard failed to add class_names"
    assert config["yolo_training"]["class_names"] == ["object"]

    # Assert Wizard generated artifact_bundle
    assert "artifact_bundle" in config["yolo_training"], "Wizard failed to add artifact_bundle"
    assert config["yolo_training"]["artifact_bundle"]["enabled"] is True

    # COMPREHENSIVE CHECKS for potentially missing keys
    assert "dataset_lint" in config, "Missing dataset_lint"
    assert "input_dir" in config["dataset_lint"]
    
    assert "color_inspection" in config, "Missing color_inspection"
    assert "enabled" in config["color_inspection"]
    
    assert "aug_preview" in config, "Missing aug_preview"
    assert "number" in config["aug_preview"]
    
    assert "yolo_evaluation" in config, "Missing yolo_evaluation"
    assert "conf" in config["yolo_evaluation"]
    
    yt = config["yolo_training"]
    assert "position_validation" in yt, "Missing position_validation in yolo_training"
    assert "export_onnx" in yt, "Missing export_onnx in yolo_training"
    assert yt["export_onnx"]["enabled"] is True

    return config

def test_full_pipeline_dryrun(default_config, tmp_path):
    """
    Iterate over ALL defined tasks and ensure they don't throw KeyError 
    when initialized with the default Wizard configuration.
    """
    
    # Mock args object
    args = MagicMock()
    args.force = False
    args.stop_event = MagicMock()
    args.stop_event.is_set.return_value = False
    args.input_format = None
    args.output_format = None

    # We need to mock the actual heavy lifting functions so we don't 
    # run real training/inference/io operations.
    # We ONLY want to check if config[key] access crashes.
    
    with patch("picture_tool.tasks.augmentation.YoloDataAugmentor"), \
         patch("picture_tool.tasks.augmentation.ImageAugmentor"), \
         patch("picture_tool.tasks.augmentation.preview_dataset"), \
         patch("picture_tool.tasks.conversion.convert_format"), \
         patch("picture_tool.tasks.quality.process_anomaly_detection"), \
         patch("picture_tool.tasks.quality.split_dataset"), \
         patch("picture_tool.tasks.quality.lint_dataset"), \
         patch("picture_tool.tasks.quality.generate_report"), \
         patch("picture_tool.tasks.quality.run_batch_inference"), \
         patch("picture_tool.tasks.quality.generate_qc_summary"), \
         patch("picture_tool.tasks.quality.color_verifier"), \
         patch("picture_tool.tasks.quality.subprocess.run"), \
         patch("picture_tool.tasks.training.train_yolo"), \
         patch("picture_tool.tasks.training.evaluate_yolo"), \
         patch("picture_tool.tasks.training.run_position_validation"), \
         patch("picture_tool.tasks.training.detect_existing_weights", return_value=(None, None)):

        for module in MODULES_TO_TEST:
            if not hasattr(module, "TASKS"):
                continue
                
            for task in module.TASKS:
                print(f"Testing Task: {task.name}")
                
                # 1. Test Skip Function (if exists)
                if task.skip_fn:
                    try:
                        task.skip_fn(default_config, args)
                    except Exception as e:
                        pytest.fail(f"Task '{task.name}' skip_fn CRASHED with config: {e}")

                # 2. Test Run Function
                try:
                    task.run(default_config, args)
                except Exception as e:
                    pytest.fail(f"Task '{task.name}' run CRASHED with config: {e}")
