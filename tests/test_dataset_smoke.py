import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from picture_tool.picture_tool.quality import dataset_linter


@pytest.fixture(autouse=True)
def matplotlib_agg(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    yield


@pytest.fixture()
def dataset(tmp_path):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()

    # create dummy image and label
    data = np.zeros((64, 64, 3), dtype=np.uint8)
    data[:, :] = (0, 255, 0)
    cv2.imwrite(str(img_dir / "sample.jpg"), data)
    (lbl_dir / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    return img_dir, lbl_dir


def test_lint_and_preview(tmp_path, dataset):
    img_dir, lbl_dir = dataset

    config = {
        "dataset_lint": {
            "image_dir": str(img_dir),
            "label_dir": str(lbl_dir),
            "output_dir": str(tmp_path / "lint"),
        },
        "aug_preview": {
            "image_dir": str(img_dir),
            "label_dir": str(lbl_dir),
            "output_dir": str(tmp_path / "preview"),
            "num_samples": 1,
            "cols": 1,
        },
        "yolo_training": {"class_names": ["dummy"]},
    }

    lint_dir = dataset_linter.lint_dataset(config, logger=logging.getLogger("test"))
    preview_path = dataset_linter.preview_dataset(config, logger=logging.getLogger("test"))

    assert (Path(lint_dir) / "lint.csv").exists()
    assert preview_path.exists()
