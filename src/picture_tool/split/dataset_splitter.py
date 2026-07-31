import logging
import hashlib
import random
import re
import shutil
from pathlib import Path
from typing import List

try:
    # Optional: iterative stratification for multi-label balance
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit  # type: ignore
except ImportError:  # pragma: no cover
    MultilabelStratifiedShuffleSplit = None  # type: ignore


logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_group_key(image_path: Path) -> str:
    """Group augmented variants so one source cannot cross data splits."""
    return re.sub(r"(?:_aug_?\d+)$", "", image_path.stem, flags=re.IGNORECASE)


def _build_source_groups(
    images: List[Path], labels: List[Path], logger: logging.Logger
) -> List[List[tuple[Path, Path]]]:
    """Deduplicate identical images and group augmentation families."""
    by_source: dict[str, List[tuple[Path, Path]]] = {}
    by_digest: dict[str, tuple[str, Path]] = {}
    duplicate_count = 0
    for image_path, label_path in zip(images, labels):
        digest = _sha256_file(image_path)
        label_content = label_path.read_text(encoding="utf-8").strip()
        previous = by_digest.get(digest)
        if previous is not None:
            previous_label, previous_path = previous
            if previous_label != label_content:
                raise ValueError(
                    "Identical images have conflicting labels: "
                    f"{previous_path} and {image_path}"
                )
            duplicate_count += 1
            continue
        by_digest[digest] = (label_content, image_path)
        by_source.setdefault(_source_group_key(image_path), []).append(
            (image_path, label_path)
        )
    if duplicate_count:
        logger.warning(
            "Removed %d byte-identical duplicate image(s) before split.",
            duplicate_count,
        )
    return [by_source[key] for key in sorted(by_source)]


def _group_multilabel_matrix(
    groups: List[List[tuple[Path, Path]]], num_classes: int
) -> List[List[int]]:
    matrix: List[List[int]] = []
    for group in groups:
        row = [0] * num_classes
        for _, label_path in group:
            for class_id in _load_classes_from_label(label_path):
                if 0 <= class_id < num_classes:
                    row[class_id] = 1
        matrix.append(row)
    return matrix


def _flatten_groups(
    groups: List[List[tuple[Path, Path]]], indices
) -> tuple[List[Path], List[Path]]:
    pairs = [pair for index in indices for pair in groups[int(index)]]
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def _group_review_sample_ids(group: List[tuple[Path, Path]]) -> set[str]:
    """Return operator sample IDs represented by one source-image group."""
    sample_ids: set[str] = set()
    for image_path, _label_path in group:
        source_stem = _source_group_key(image_path)
        if source_stem.startswith("review_"):
            sample_id = source_stem[len("review_") :]
            if sample_id:
                sample_ids.add(sample_id)
    return sample_ids


def _deterministic_group_split(
    group_indices: List[int],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    has_forced_train: bool,
    has_forced_test: bool = False,
    minimum_val_groups: int = 0,
    minimum_test_groups: int = 0,
) -> tuple[List[int], List[int], List[int]]:
    """Split small group sets without producing an empty required cohort."""
    shuffled = list(group_indices)
    random.Random(42).shuffle(shuffled)
    minimums = {
        "val": max(1 if val_ratio > 0 else 0, minimum_val_groups),
        "test": max(
            1 if test_ratio > 0 and not has_forced_test else 0,
            minimum_test_groups,
        ),
    }
    val_count = max(minimums["val"], int(round(len(shuffled) * val_ratio)))
    test_count = max(minimums["test"], int(round(len(shuffled) * test_ratio)))
    required_dynamic_train = 0 if has_forced_train else (1 if train_ratio > 0 else 0)
    maximum_holdout = len(shuffled) - required_dynamic_train
    while val_count + test_count > maximum_holdout:
        if val_count > minimums["val"] and val_count >= test_count:
            val_count -= 1
        elif test_count > minimums["test"]:
            test_count -= 1
        else:
            raise ValueError(
                "Not enough independent historical source groups to keep "
                "train/val/test isolated. Continue collecting reviewed images."
            )
    val_indices = shuffled[:val_count]
    test_indices = shuffled[val_count : val_count + test_count]
    train_indices = shuffled[val_count + test_count :]
    return train_indices, val_indices, test_indices


def _load_classes_from_label(label_path: Path) -> List[int]:
    try:
        lines = [
            ln.strip()
            for ln in label_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    except (FileNotFoundError, UnicodeDecodeError, OSError):
        return []
    classes: List[int] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 5:
            try:
                cls = int(float(parts[0]))
                if cls not in classes:
                    classes.append(cls)
            except (ValueError, TypeError):
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
    train_ratio = float(split_config["split_ratios"]["train"])
    val_ratio = float(split_config["split_ratios"]["val"])
    test_ratio = float(split_config["split_ratios"]["test"])
    minimum_source_groups = split_config.get("minimum_source_groups", {}) or {}
    if not isinstance(minimum_source_groups, dict):
        raise ValueError("minimum_source_groups 必須是 split 對數量的設定")
    try:
        minimum_val_groups = int(minimum_source_groups.get("val", 0))
        minimum_test_groups = int(minimum_source_groups.get("test", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("minimum_source_groups 必須使用整數") from exc
    if minimum_val_groups < 0 or minimum_test_groups < 0:
        raise ValueError("minimum_source_groups 不可為負數")
    input_formats = split_config.get("input_formats", [".jpg", ".jpeg", ".png", ".bmp"])
    label_format = split_config.get("label_format", ".txt")

    if any(
        ratio < 0.0 or ratio > 1.0 for ratio in (train_ratio, val_ratio, test_ratio)
    ):
        raise ValueError("split_ratios 必須介於 0 與 1")
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
        except (ValueError, TypeError, UnicodeDecodeError, OSError):
            pass

    source_groups = _build_source_groups(paired_images, paired_labels, logger)
    raw_forced_sample_ids = split_config.get("force_train_sample_ids", [])
    if not isinstance(raw_forced_sample_ids, list):
        raise ValueError("force_train_sample_ids 必須是 sample ID 清單")
    raw_forced_test_sample_ids = split_config.get("force_test_sample_ids", [])
    if not isinstance(raw_forced_test_sample_ids, list):
        raise ValueError("force_test_sample_ids 必須是 sample ID 清單")
    forced_train_sample_ids = {
        str(sample_id).strip()
        for sample_id in raw_forced_sample_ids
        if str(sample_id).strip()
    }
    forced_test_sample_ids = {
        str(sample_id).strip()
        for sample_id in raw_forced_test_sample_ids
        if str(sample_id).strip()
    }
    overlapping_sample_ids = forced_train_sample_ids & forced_test_sample_ids
    if overlapping_sample_ids:
        raise ValueError(
            "Samples cannot be forced into both train and test: "
            + ", ".join(sorted(overlapping_sample_ids)[:5])
        )
    if forced_train_sample_ids and train_ratio <= 0:
        raise ValueError("有補訓樣本時，train split ratio 必須大於 0")
    if forced_test_sample_ids and test_ratio <= 0:
        raise ValueError("有位置黃金樣本時，test split ratio 必須大於 0")
    forced_train_idx = {
        index
        for index, group in enumerate(source_groups)
        if _group_review_sample_ids(group) & forced_train_sample_ids
    }
    forced_test_idx = {
        index
        for index, group in enumerate(source_groups)
        if _group_review_sample_ids(group) & forced_test_sample_ids
    }
    overlapping_group_indices = forced_train_idx & forced_test_idx
    if overlapping_group_indices:
        raise ValueError(
            "One augmented source family cannot be assigned to both train and test."
        )
    found_forced_train_ids = {
        sample_id
        for index in forced_train_idx
        for sample_id in _group_review_sample_ids(source_groups[index])
        if sample_id in forced_train_sample_ids
    }
    missing_forced_train_ids = forced_train_sample_ids - found_forced_train_ids
    if missing_forced_train_ids:
        raise ValueError(
            "Submitted feedback samples are missing from the split input: "
            + ", ".join(sorted(missing_forced_train_ids)[:5])
        )
    found_forced_test_ids = {
        sample_id
        for index in forced_test_idx
        for sample_id in _group_review_sample_ids(source_groups[index])
        if sample_id in forced_test_sample_ids
    }
    missing_forced_test_ids = forced_test_sample_ids - found_forced_test_ids
    if missing_forced_test_ids:
        raise ValueError(
            "Position golden samples are missing from the split input: "
            + ", ".join(sorted(missing_forced_test_ids)[:5])
        )
    if len(source_groups) < 3:
        raise ValueError(
            "At least three independent source-image groups are required for "
            "train/val/test splitting."
        )
    group_indices = [
        index
        for index in range(len(source_groups))
        if index not in forced_train_idx and index not in forced_test_idx
    ]
    dynamic_minimum_test_groups = max(
        0, minimum_test_groups - len(forced_test_idx)
    )
    required_dynamic_groups = int(val_ratio > 0) + int(
        test_ratio > 0 and not forced_test_idx
    )
    if not forced_train_idx and train_ratio > 0:
        required_dynamic_groups += 1
    if len(group_indices) < required_dynamic_groups:
        raise ValueError(
            "Not enough independent historical source groups after reserving "
            "submitted feedback for training. Continue collecting reviewed images."
        )
    required_holdout_groups = minimum_val_groups + dynamic_minimum_test_groups
    required_train_groups = 0 if forced_train_idx else int(train_ratio > 0)
    if len(group_indices) < required_holdout_groups + required_train_groups:
        raise ValueError(
            "Not enough independent historical source groups for safe validation: "
            f"available={len(group_indices)}, required_val={minimum_val_groups}, "
            f"required_dynamic_test={dynamic_minimum_test_groups}, "
            f"reserved_test={len(forced_test_idx)}. "
            "Continue collecting reviewed images."
        )
    train_idx = val_idx = test_idx = None
    if (
        strat_cfg
        and MultilabelStratifiedShuffleSplit
        and num_classes
        and num_classes > 1
        and val_ratio > 0
        and test_ratio > 0
    ):
        full_group_matrix = _group_multilabel_matrix(source_groups, num_classes)
        group_matrix = [full_group_matrix[index] for index in group_indices]
        if not all(sum(row) == 0 for row in group_matrix):
            try:
                effective_minimum_val = max(int(val_ratio > 0), minimum_val_groups)
                effective_minimum_test = max(
                    int(test_ratio > 0 and not forced_test_idx),
                    dynamic_minimum_test_groups,
                )
                holdout_count = max(
                    effective_minimum_val + effective_minimum_test,
                    int(round(len(group_indices) * (val_ratio + test_ratio))),
                )
                holdout_count = min(
                    holdout_count,
                    len(group_indices) - required_train_groups,
                )
                test_count = max(
                    effective_minimum_test,
                    int(round(holdout_count * test_ratio / (val_ratio + test_ratio))),
                )
                test_count = min(test_count, holdout_count - effective_minimum_val)
                splitter = MultilabelStratifiedShuffleSplit(
                    n_splits=1,
                    test_size=holdout_count,
                    random_state=42,
                )
                local_train, local_temp = next(
                    splitter.split(list(range(len(group_indices))), group_matrix)
                )
                train_idx = [group_indices[int(index)] for index in local_train]
                temp_idx = [group_indices[int(index)] for index in local_temp]
                temp_matrix = [full_group_matrix[int(index)] for index in temp_idx]
                temp_splitter = MultilabelStratifiedShuffleSplit(
                    n_splits=1,
                    test_size=test_count,
                    random_state=42,
                )
                local_val, local_test = next(
                    temp_splitter.split(list(range(len(temp_idx))), temp_matrix)
                )
                val_idx = [temp_idx[int(index)] for index in local_val]
                test_idx = [temp_idx[int(index)] for index in local_test]
            except ValueError as exc:
                logger.warning(
                    "Grouped stratified split unavailable (%s); using random groups.",
                    exc,
                )

    if train_idx is None or val_idx is None or test_idx is None:
        train_idx, val_idx, test_idx = _deterministic_group_split(
            group_indices,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            has_forced_train=bool(forced_train_idx),
            has_forced_test=bool(forced_test_idx),
            minimum_val_groups=minimum_val_groups,
            minimum_test_groups=dynamic_minimum_test_groups,
        )

    train_idx = sorted({int(index) for index in train_idx} | forced_train_idx)
    test_idx = sorted({int(index) for index in test_idx} | forced_test_idx)

    train_images, train_labels = _flatten_groups(source_groups, train_idx)
    val_images, val_labels = _flatten_groups(source_groups, val_idx)
    test_images, test_labels = _flatten_groups(source_groups, test_idx)

    staging_dir = output_dir.with_name(f".{output_dir.name}.staging")
    backup_dir = output_dir.with_name(f".{output_dir.name}.backup")
    for generated_dir in (staging_dir, backup_dir):
        if generated_dir.exists():
            shutil.rmtree(generated_dir)
    for split in ["train", "val", "test"]:
        (staging_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (staging_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    def copy_files(images, labels, split):
        for img, lbl in zip(images, labels):
            shutil.copy2(img, staging_dir / split / "images" / img.name)
            shutil.copy2(lbl, staging_dir / split / "labels" / lbl.name)

    copy_files(train_images, train_labels, "train")
    copy_files(val_images, val_labels, "val")
    copy_files(test_images, test_labels, "test")

    logger.info("檔案已完成分割並複製至訓練/驗證/測試目錄")

    # Copy classes.txt if exists (Crucial for trainer auto-detection)
    src_classes = label_dir / "classes.txt"
    dst_classes = staging_dir / "classes.txt"
    if src_classes.exists():
        shutil.copy2(src_classes, dst_classes)
        logger.info(f"Copied classes.txt to {dst_classes}")

    if output_dir.exists():
        output_dir.replace(backup_dir)
    try:
        staging_dir.replace(output_dir)
    except OSError:
        if backup_dir.exists() and not output_dir.exists():
            backup_dir.replace(output_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    if handler:
        logger.removeHandler(handler)
        handler.close()
