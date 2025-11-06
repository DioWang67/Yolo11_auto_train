import cv2
import numpy as np

from picture_tool.split.dataset_splitter import split_dataset


def _write_sample_pair(image_path, label_path, cls_id=0):
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((32, 32, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(image_path), img)
    label_path.write_text(f"{cls_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def test_split_dataset_creates_train_val_test_structure(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    output_dir = tmp_path / "split"

    for idx in range(6):
        _write_sample_pair(
            image_dir / f"img{idx}.jpg", label_dir / f"img{idx}.txt", cls_id=idx % 2
        )

    config = {
        "train_test_split": {
            "input": {"image_dir": str(image_dir), "label_dir": str(label_dir)},
            "output": {"output_dir": str(output_dir)},
            "split_ratios": {"train": 0.5, "val": 0.25, "test": 0.25},
            "input_formats": [".jpg"],
            "label_format": ".txt",
            "stratified": False,
        },
        "yolo_training": {"class_names": ["ok", "ng"]},
    }

    split_dataset(config)

    total_images = 0
    for split in ("train", "val", "test"):
        img_files = list((output_dir / split / "images").glob("*.jpg"))
        lbl_files = list((output_dir / split / "labels").glob("*.txt"))
        assert img_files, f"{split} split should contain at least one image"
        assert len(img_files) == len(lbl_files)
        total_images += len(img_files)

    assert total_images == 6
