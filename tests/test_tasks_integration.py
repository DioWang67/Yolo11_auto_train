import cv2
import numpy as np
import pytest
import yaml
from picture_tool.augment.image_augmentor import ImageAugmentor
from picture_tool.config_validation import validate_config_schema
from picture_tool.main_pipeline import build_task_registry
from picture_tool.tasks import run_yolo_train

def test_config_validation_schema():
    """Test that the config validation schema correctly accepts valid configs and rejects invalid ones."""
    # Valid config
    valid_config = {
        "yolo_training": {"class_names": ["cat", "dog"]},
        "augmentation": {"num_images": 5, "operations": {"flip": {"probability": 0.5}}},
        "processing": {"batch_size": 8}
    }
    assert validate_config_schema(valid_config, strict=True) == valid_config

    # Invalid config: empty class names
    invalid_config_1 = {
        "yolo_training": {"class_names": []},
    }
    with pytest.raises(Exception): # Pydantic ValidationError or _ManualConfigError
        validate_config_schema(invalid_config_1, strict=True)

    # Invalid config: negative num_images
    invalid_config_2 = {
        "augmentation": {"num_images": -1},
    }
    with pytest.raises(Exception):
        validate_config_schema(invalid_config_2, strict=True)

def test_image_augmentation_multiprocessing(tmp_path):
    """Test image augmentation with multiprocessing enabled."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Create a dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (255, 0, 0), -1)
    cv2.imwrite(str(input_dir / "test.jpg"), img)

    config_path = tmp_path / "config.yaml"
    config = {
        "input": {"image_dir": str(input_dir)},
        "output": {"image_dir": str(output_dir)},
        "augmentation": {
            "num_images": 2,
            "operations": {
                "flip": {"probability": 1.0},
                "rotate": {"angle": 90}
            }
        },
        "processing": {
            "use_process_pool": True,
            "num_workers": 2
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    augmentor = ImageAugmentor(str(config_path))
    augmentor.process_dataset()

    # Verify output
    files = list(output_dir.glob("*.png"))
    assert len(files) == 2, f"Expected 2 augmented images, found {len(files)}"

def test_task_handlers_registration():
    """Verify that task handlers are correctly registered and imported."""
    expected_tasks = [
        "format_conversion", "yolo_augmentation", "yolo_train", 
        "color_verification", "qc_summary"
    ]
    registry = build_task_registry({})
    for task in expected_tasks:
        assert task in registry, f"Missing task handler: {task}"
    
    # Verify mapping correctness
    assert registry["yolo_train"].run == run_yolo_train
