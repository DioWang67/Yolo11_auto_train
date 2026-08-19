import cv2
import numpy as np
import pytest
import yaml
from typing import Dict, Any
from picture_tool.augment.image_augmentor import ImageAugmentor
from picture_tool.config_validation import validate_config_schema
from picture_tool.main_pipeline import build_task_registry
from types import SimpleNamespace

from picture_tool.tasks import run_anomalib_package, run_anomalib_train, run_yolo_train


def test_config_validation_schema():
    """Test that the config validation schema correctly accepts valid configs and rejects invalid ones."""
    # Valid config
    valid_config = {
        "yolo_training": {"class_names": ["cat", "dog"]},
        "augmentation": {"num_images": 5, "operations": {"flip": {"probability": 0.5}}},
        "processing": {"batch_size": 8},
    }
    assert validate_config_schema(valid_config, strict=True) == valid_config

    # Invalid config: empty class names
    invalid_config_1: Dict[str, Any] = {
        "yolo_training": {"class_names": []},
    }
    with pytest.raises(Exception):  # Pydantic ValidationError or _ManualConfigError
        validate_config_schema(invalid_config_1, strict=True)

    # Invalid config: negative num_images
    invalid_config_2 = {
        "augmentation": {"num_images": -1},
    }
    with pytest.raises(Exception):
        validate_config_schema(invalid_config_2, strict=True)


def test_image_augmentation_multiprocessing(tmp_path):
    """Test image augmentation with multiprocessing enabled."""
    from picture_tool.augment import image_augmentor
    import pytest
    if image_augmentor.A is None:
        pytest.skip("albumentations depends on Torch and is bypassed during pytest")

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
            "operations": {"flip": {"probability": 1.0}, "rotate": {"angle": 90}},
        },
        "processing": {"use_process_pool": True, "num_workers": 2},
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
        "format_conversion",
        "yolo_augmentation",
        "yolo_train",
        "anomalib_train",
        "anomalib_package",
        "color_verification",
        "qc_summary",
    ]
    registry = build_task_registry({})
    for task in expected_tasks:
        assert task in registry, f"Missing task handler: {task}"

    # Verify mapping correctness
    assert registry["yolo_train"].run == run_yolo_train
    assert registry["anomalib_train"].run == run_anomalib_train
    assert registry["anomalib_package"].run == run_anomalib_package


def test_anomalib_package_task_uses_configured_run_dir(tmp_path, monkeypatch):
    """Anomalib package task should pass GUI/pipeline config to packager."""
    from picture_tool.tasks import training

    captured = {}
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "packages"

    def fake_package_anomalib_run(*args, **kwargs):
        captured["run_dir"] = args[0]
        captured.update(kwargs)
        return SimpleNamespace(
            zip_path=output_dir / "PCBA1_B_anomalib_padim_package.zip",
            baseline_only=True,
        )

    monkeypatch.setattr(training, "package_anomalib_run", fake_package_anomalib_run)

    training.run_anomalib_package(
        {
            "anomalib_training": {"name": "PCBA1_B"},
            "anomalib_package": {
                "run_dir": str(run_dir),
                "output_dir": str(output_dir),
                "product": "PCBA1",
                "area": "B",
                "threshold": 0.42,
                "force": True,
            },
        },
        SimpleNamespace(force=False),
    )

    assert captured["run_dir"] == run_dir
    assert captured["output_dir"] == output_dir
    assert captured["product"] == "PCBA1"
    assert captured["area"] == "B"
    assert captured["threshold"] == 0.42
    assert captured["force"] is True


def test_anomalib_package_task_autodetects_latest_run_from_product_override(
    tmp_path, monkeypatch
):
    """GUI Product override should avoid hard-coding anomalib_package.run_dir."""
    from picture_tool.tasks import training

    runs_root = tmp_path / "runs" / "anomalib"
    older_run = runs_root / "PCBA1" / "B" / "Padim" / "PCBA1_B" / "latest"
    newer_run = runs_root / "PCBA1" / "B" / "EfficientAd" / "PCBA1_B" / "latest"
    for index, run_dir in enumerate((older_run, newer_run), start=1):
        checkpoint = run_dir / "weights" / "lightning" / "model.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text(f"checkpoint {index}", encoding="utf-8")

    captured = {}

    def fake_package_anomalib_run(*args, **kwargs):
        captured["run_dir"] = args[0]
        captured.update(kwargs)
        return SimpleNamespace(
            zip_path=tmp_path / "packages" / "PCBA1_B_anomalib_efficientad_package.zip",
            baseline_only=False,
        )

    monkeypatch.setattr(training, "package_anomalib_run", fake_package_anomalib_run)

    training.run_anomalib_package(
        {
            "anomalib_training": {"project": str(runs_root)},
            "anomalib_package": {
                "run_dir": None,
                "output_dir": str(tmp_path / "packages"),
                "force": True,
            },
        },
        SimpleNamespace(force=False, product="PCBA1,B"),
    )

    assert captured["run_dir"] == newer_run
    assert captured["product"] == "PCBA1"
    assert captured["area"] == "B"
