import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

from picture_tool.picture_tool.augment.image_augmentor import ImageAugmentor


def test_image_augmentor_process_dataset(tmp_path, monkeypatch):
    input_dir = tmp_path / "input" / "images"
    output_dir = tmp_path / "output" / "images"
    input_dir.mkdir(parents=True)

    image = np.full((80, 120, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(input_dir / "sample.png"), image)

    config = {
        "input": {"image_dir": str(input_dir)},
        "output": {"image_dir": str(output_dir)},
        "augmentation": {
            "num_images": 2,
            "operations": {
                "flip": {"probability": 1.0},
                "rotate": {"angle": (-5, 5)},
                "blur": {"kernel": 3},
            },
        },
        "processing": {"batch_size": 1, "num_workers": 1},
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    def _setup_logger(name, log_file=None):
        logger = logging.getLogger(f"{name}_test")
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.INFO)
        return logger

    monkeypatch.setattr(
        "picture_tool.picture_tool.augment.image_augmentor.setup_module_logger",
        _setup_logger,
    )

    augmentor = ImageAugmentor(str(config_path))
    augmentor.process_dataset()

    generated = list(Path(output_dir).glob("*.png"))
    assert len(generated) == 2

    for path in generated:
        augmented = cv2.imread(str(path))
        assert augmented is not None
        assert augmented.shape == (640, 640, 3)
