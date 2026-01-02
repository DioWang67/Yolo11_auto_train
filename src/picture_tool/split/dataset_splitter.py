import logging
import shutil
from pathlib import Path
from typing import List
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

try:
    # Optional: iterative stratification for multi-label balance
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit  # type: ignore
except Exception:  # pragma: no cover
    MultilabelStratifiedShuffleSplit = None  # type: ignore


logger = logging.getLogger(__name__)


def _load_classes_from_label(label_path: Path) -> List[int]:
    try:
        lines = [
            ln.strip()
            for ln in label_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    except Exception:
        return []
    classes: List[int] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 5:
            try:
                cls = int(float(parts[0]))
                if cls not in classes:
                    classes.append(cls)
            except Exception:
                continue
    return classes


def _build_multilabel_matrix(
    label_paths: List[Path], num_classes: int
) -> List[List[int]]:
    Y: List[List[int]] = []
    for p in label_paths:
        cls_list = _load_classes_from_label(p)
        row = [0] * num_classes
        for c in cls_list:
            if 0 <= c < num_classes:
                row[c] = 1
        Y.append(row)
    return Y


def split_dataset(config, log_file=None, logger=None):
    """將影像與標註切割成訓練、驗證、測試集

    Args:
        config: 設定
        log_file: 選用的 log 檔案路徑
        logger: 傳入既有 logger 以便集中管理
    """
    logger = logger or logging.getLogger(__name__)
    handler = None
    if log_file:
        log_path = Path(log_file).resolve()
        exists = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path
            for h in logger.handlers
        )
        if not exists:
            handler = logging.FileHandler(log_path)
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(handler)
    split_config = config["train_test_split"]
    image_dir = Path(split_config["input"]["image_dir"])
    label_dir = Path(split_config["input"]["label_dir"])
    output_dir = Path(split_config["output"]["output_dir"])
    train_ratio = split_config["split_ratios"]["train"]
    val_ratio = split_config["split_ratios"]["val"]
    test_ratio = split_config["split_ratios"]["test"]
    input_formats = split_config.get("input_formats", [".jpg", ".jpeg", ".png", ".bmp"])
    label_format = split_config.get("label_format", ".txt")

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("split_ratios 必須等於 1")
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError("輸入影像/標註目錄不存在")

    image_dict = {
        p.stem: p for p in image_dir.glob("*") if p.suffix.lower() in input_formats
    }
    label_dict = {
        p.stem: p for p in label_dir.glob("*") if p.suffix.lower() == label_format
    }

    common_keys = image_dict.keys() & label_dict.keys()
    missing_images = label_dict.keys() - image_dict.keys()
    missing_labels = image_dict.keys() - label_dict.keys()

    for key in sorted(missing_images):
        logger.warning(f"缺少影像對應標註: {label_dict[key]}")
    for key in sorted(missing_labels):
        logger.warning(f"缺少標籤對應影像: {image_dict[key]}")

    paired_images = [image_dict[k] for k in sorted(common_keys)]
    paired_labels = [label_dict[k] for k in sorted(common_keys)]

    if len(paired_images) == 0:
        raise ValueError("找不到可分割的資料（影像/標註配對為 0）")

    # Optional multi-label stratified split if config enabled and lib available
    strat_cfg = split_config.get("stratified", True)
    num_classes = None
    yolo_cfg = config.get("yolo_training", {})
    class_names = yolo_cfg.get("class_names") or []
    if isinstance(class_names, list) and class_names:
        num_classes = len(class_names)
    else:
        # infer max class id + 1
        try:
            max_cls = -1
            for p in paired_labels:
                for ln in p.read_text(encoding="utf-8").splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    parts = ln.split()
                    if len(parts) >= 5:
                        cid = int(float(parts[0]))
                        if cid > max_cls:
                            max_cls = cid
            if max_cls >= 0:
                num_classes = max_cls + 1
        except Exception:
            pass

    if (
        strat_cfg
        and MultilabelStratifiedShuffleSplit
        and num_classes
        and num_classes > 1
    ):
        Y = _build_multilabel_matrix(paired_labels, num_classes)
        # If all-zero (no labels), fallback
        all_zero = all(sum(row) == 0 for row in Y)
        if not all_zero:
            msss = MultilabelStratifiedShuffleSplit(
                n_splits=1, test_size=val_ratio + test_ratio, random_state=42
            )
            idx = list(range(len(paired_images)))
            train_idx, temp_idx = next(msss.split(idx, Y))
            # Split temp into val/test
            temp_Y = [Y[i] for i in temp_idx]
            msss2 = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=42,
            )
            temp_idx2, test_idx2 = next(msss2.split(list(range(len(temp_idx))), temp_Y))
            val_idx = [temp_idx[i] for i in temp_idx2]
            test_idx = [temp_idx[i] for i in test_idx2]

            train_images = [paired_images[i] for i in train_idx]
            train_labels = [paired_labels[i] for i in train_idx]
            val_images = [paired_images[i] for i in val_idx]
            val_labels = [paired_labels[i] for i in val_idx]
            test_images = [paired_images[i] for i in test_idx]
            test_labels = [paired_labels[i] for i in test_idx]
        else:
            # Fallback to random
            train_images, temp_images, train_labels, temp_labels = train_test_split(
                paired_images,
                paired_labels,
                test_size=val_ratio + test_ratio,
                random_state=42,
            )
            val_images, test_images, val_labels, test_labels = train_test_split(
                temp_images,
                temp_labels,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=42,
            )
    else:
        train_images, temp_images, train_labels, temp_labels = train_test_split(
            paired_images,
            paired_labels,
            test_size=val_ratio + test_ratio,
            random_state=42,
        )
        val_images, test_images, val_labels, test_labels = train_test_split(
            temp_images,
            temp_labels,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=42,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    def copy_files(images, labels, split):
        for img, lbl in zip(images, labels):
            shutil.copy(img, output_dir / split / "images" / img.name)
            shutil.copy(lbl, output_dir / split / "labels" / lbl.name)

    copy_files(train_images, train_labels, "train")
    copy_files(val_images, val_labels, "val")
    copy_files(test_images, test_labels, "test")

    logger.info("檔案已完成分割並複製至訓練/驗證/測試目錄")

    # Copy classes.txt if exists (Crucial for trainer auto-detection)
    src_classes = label_dir / "classes.txt"
    dst_classes = output_dir / "classes.txt"
    if src_classes.exists():
        shutil.copy(src_classes, dst_classes)
        logger.info(f"Copied classes.txt to {dst_classes}")

    if handler:
        logger.removeHandler(handler)
        handler.close()
