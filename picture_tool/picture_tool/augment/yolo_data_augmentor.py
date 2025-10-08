import cv2
try:
    import albumentations as A  # type: ignore[import]
except ImportError as exc:
    raise ImportError('Albumentations is required for DataAugmentor.') from exc

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from typing import Any, List, Optional, TYPE_CHECKING
from picture_tool.utils import list_images, DEFAULT_IMAGE_EXTS, setup_module_logger  # type: ignore[import]
import yaml  # type: ignore[import]
from pathlib import Path
from tqdm import tqdm  # type: ignore[import]
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


if TYPE_CHECKING:
    from albumentations.core.composition import Compose  # type: ignore[import]
else:
    Compose = Any


class DataAugmentor:
    def __init__(self, config_path: Optional[str] = None):
        self._setup_logging()
        self.config = (
            self._load_config(config_path) if config_path else self._default_config()
        )
        # 延後初始化，避免在 main_pipeline 覆蓋 config 前就建立多餘輸出資料夾
        self.augmentations: Optional[Compose] = None

    def _setup_logging(self):
        self.logger = setup_module_logger(__name__, "augmentation_fixed.log")

    def _setup_output_dirs(self):
        try:
            output_img_dir = Path(self.config["output"]["image_dir"])
            output_img_dir.mkdir(parents=True, exist_ok=True)

            output_label_dir = Path(self.config["output"]["label_dir"])
            output_label_dir.mkdir(parents=True, exist_ok=True)

            debug_dir = Path(
                self.config["output"].get("debug_dir", "debug_visualizations")
            )
            debug_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"已建輸出資料夾: {output_img_dir}, {output_label_dir}")
        except Exception as e:
            self.logger.error(f"建立輸出目錄失敗: {e}")
            raise

    def _load_config(self, config_path: Optional[str]) -> dict:
        try:
            if config_path and Path(config_path).exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"載入 {config_path} 設定")
                return config
            else:
                self.logger.warning(f"設定檔 {config_path} 不存在，使用預設設定")
                return self._default_config()
        except Exception as e:
            self.logger.error(f"讀取設定檔失敗: {e}")
            return self._default_config()

    def _default_config(self) -> dict:
        return {
            "input": {"image_dir": "A/img", "label_dir": "A/label"},
            "output": {
                "image_dir": "output/images",
                "label_dir": "output/labels",
                "debug_dir": "debug_visualizations",
            },
            "augmentation": {
                "num_images": 5,
                "target_size": 640,
                "operations": {
                    "flip": {"probability": 0.3},
                    "rotate": {"angle": (-5, 5)},
                    "multiply": {"range": (0.9, 1.1)},
                    "scale": {"range": (0.95, 1.05)},
                    "contrast": {"range": (0.9, 1.3)},
                    "hue": {"range": (-5, 5)},
                    "noise": {"scale": (0, 0.02)},
                    "perspective": {"scale": (0, 0)},
                    "blur": {"kernel": (0, 0)},
                },
            },
            "processing": {"batch_size": 10, "num_workers": None, "debug_mode": True},
        }

    def yolo_to_absolute(self, yolo_bbox, img_width, img_height):
        x_center, y_center, width, height = yolo_bbox
        abs_x_center = x_center * img_width
        abs_y_center = y_center * img_height
        abs_width = width * img_width
        abs_height = height * img_height
        x1 = abs_x_center - abs_width / 2
        y1 = abs_y_center - abs_height / 2
        x2 = abs_x_center + abs_width / 2
        y2 = abs_y_center + abs_height / 2
        return [x1, y1, x2, y2]

    def absolute_to_yolo(self, abs_bbox, img_width, img_height):
        x1, y1, x2, y2 = abs_bbox
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        x_center = np.clip(x_center, 0, 1)
        y_center = np.clip(y_center, 0, 1)
        width = np.clip(width, 0, 1)
        height = np.clip(height, 0, 1)
        return [x_center, y_center, width, height]

    def visualize_bboxes(self, image, bboxes, class_labels, save_path=None, title=""):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image)
        ax.set_title(title)
        img_height, img_width = image.shape[:2]
        for bbox, label in zip(bboxes, class_labels):
            if len(bbox) == 4 and all(0 <= coord <= 1 for coord in bbox):
                x_center, y_center, width, height = bbox
                x_center_px = x_center * img_width
                y_center_px = y_center * img_height
                width_px = width * img_width
                height_px = height * img_height
                x1 = x_center_px - width_px / 2
                y1 = y_center_px - height_px / 2
                rect = patches.Rectangle(
                    (x1, y1),
                    width_px,
                    height_px,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    x1,
                    y1 - 5,
                    f"Class {int(label)}",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                )
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def _create_augmentations(self) -> Compose:
        aug_config = self.config["augmentation"]
        ops_config = aug_config["operations"]
        aug_list = []
        if ops_config.get("flip", {}).get("probability", 0) > 0:
            aug_list.append(A.HorizontalFlip(p=ops_config["flip"]["probability"]))
        if ops_config.get("rotate") and ops_config["rotate"]["angle"] != (0, 0):
            angle = ops_config["rotate"]["angle"]
            if isinstance(angle, (list, tuple)):
                limit = max(abs(angle[0]), abs(angle[1]))
            else:
                limit = abs(angle)
            # 使用 border_value 以相容不同版本 Albumentations
            aug_list.append(
                A.Rotate(
                    limit=limit,
                    p=0.3,
                    border_mode=cv2.BORDER_CONSTANT,
                    border_value=(128, 128, 128),
                )
            )
        if ops_config.get("multiply"):
            multiply_range = ops_config["multiply"]["range"]
            brightness_limit = (multiply_range[0] - 1, multiply_range[1] - 1)
            aug_list.append(
                A.RandomBrightnessContrast(
                    brightness_limit=brightness_limit, contrast_limit=0, p=0.5
                )
            )
        if ops_config.get("contrast"):
            contrast_range = ops_config["contrast"]["range"]
            contrast_limit = (contrast_range[0] - 1, contrast_range[1] - 1)
            aug_list.append(
                A.RandomBrightnessContrast(
                    brightness_limit=0, contrast_limit=contrast_limit, p=0.5
                )
            )
        if ops_config.get("hue"):
            hue_range = ops_config["hue"]["range"]
            if isinstance(hue_range, (list, tuple)):
                hue_limit = hue_range
            else:
                hue_limit = (-abs(hue_range), abs(hue_range))
            aug_list.append(
                A.HueSaturationValue(
                    hue_shift_limit=hue_limit,
                    sat_shift_limit=(-10, 10),
                    val_shift_limit=(-10, 10),
                    p=0.3,
                )
            )
        if ops_config.get("noise") and ops_config["noise"]["scale"][1] > 0:
            noise_scale = ops_config["noise"]["scale"]
            # 兼容不同版本 Albumentations，若 var_limit 參數不可用則降級使用預設
            try:
                aug_list.append(
                    A.GaussNoise(var_limit=(0, noise_scale[1] * 255), p=0.2)
                )
            except TypeError:
                aug_list.append(A.GaussNoise(p=0.2))
        self.logger.info(
            f"增強管線包含 {len(aug_list)} 個操作 {[type(aug).__name__ for aug in aug_list]}"
        )
        return A.Compose(
            aug_list,
            bbox_params=A.BboxParams(
                format="yolo", label_fields=["class_labels"], min_visibility=0.3
            ),
        )

    def resize_with_padding(self, image, target_size=640):
        h, w = image.shape[:2]
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        result = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        transform_params = {
            "scale": scale,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "new_width": new_w,
            "new_height": new_h,
            "target_size": target_size,
        }
        return result, transform_params

    def transform_bboxes_after_resize(
        self, bboxes, original_width, original_height, transform_params
    ):
        transformed_bboxes = []
        for bbox in bboxes:
            x_center, y_center, width, height = bbox
            abs_x_center = x_center * original_width
            abs_y_center = y_center * original_height
            abs_width = width * original_width
            abs_height = height * original_height
            scaled_x_center = abs_x_center * transform_params["scale"]
            scaled_y_center = abs_y_center * transform_params["scale"]
            scaled_width = abs_width * transform_params["scale"]
            scaled_height = abs_height * transform_params["scale"]
            final_x_center = scaled_x_center + transform_params["x_offset"]
            final_y_center = scaled_y_center + transform_params["y_offset"]
            target_size = transform_params["target_size"]
            new_x_center = final_x_center / target_size
            new_y_center = final_y_center / target_size
            new_width = scaled_width / target_size
            new_height = scaled_height / target_size
            new_x_center = np.clip(new_x_center, 0, 1)
            new_y_center = np.clip(new_y_center, 0, 1)
            new_width = np.clip(new_width, 0, 1)
            new_height = np.clip(new_height, 0, 1)
            transformed_bboxes.append(
                [new_x_center, new_y_center, new_width, new_height]
            )
        return transformed_bboxes

    def _process_single_image(self, img_file: str) -> None:
        try:
            img_path = Path(self.config["input"]["image_dir"]) / img_file
            label_path = Path(self.config["input"]["label_dir"]) / (
                Path(img_file).stem + ".txt"
            )
            image = cv2.imread(str(img_path))
            if image is None:
                self.logger.error(f"無法讀取影像: {img_path}")
                return
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_height, original_width = image.shape[:2]
            self.logger.info(
                f"處理 {img_file}, 原始尺寸: {original_width}x{original_height}"
            )
            try:
                with open(label_path, "r") as f:
                    annotations = [
                        line.strip().split() for line in f.readlines() if line.strip()
                    ]
            except Exception as e:
                self.logger.error(f"讀取標註案失敗 {label_path}: {e}")
                return
            if not annotations:
                self.logger.warning(f"標註檔案為空: {label_path}")
                return
            bboxes = []
            class_labels = []
            for ann in annotations:
                if len(ann) >= 5:
                    cls, x_center, y_center, width, height = map(float, ann[:5])
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(int(cls))
            if not bboxes:
                self.logger.warning(f"沒有有效的標註: {label_path}")
                return
            if self.augmentations is None:
                self.augmentations = self._create_augmentations()
            augmentations = self.augmentations
            if augmentations is None:
                raise RuntimeError('Augmentation pipeline failed to initialize.')
            if self.config["processing"].get("debug_mode", False):
                debug_dir = Path(
                    self.config["output"].get("debug_dir", "debug_visualizations")
                )
                original_viz_path = debug_dir / f"{Path(img_file).stem}_original.png"
                self.visualize_bboxes(
                    image,
                    bboxes,
                    class_labels,
                    str(original_viz_path),
                    f"Original: {img_file}",
                )
            target_size = self.config["augmentation"].get("target_size", 640)
            for i in range(self.config["augmentation"]["num_images"]):
                try:
                    if len(augmentations.transforms) > 0:
                        transformed = augmentations(
                            image=image, bboxes=bboxes, class_labels=class_labels
                        )
                        augmented_image = transformed["image"]
                        augmented_bboxes = transformed["bboxes"]
                        augmented_labels = transformed["class_labels"]
                    else:
                        augmented_image = image.copy()
                        augmented_bboxes = bboxes.copy()
                        augmented_labels = class_labels.copy()
                    if len(augmented_bboxes) == 0:
                        self.logger.warning(
                            f"增強後沒有剩餘標註，跳過: {img_file}_aug_{i+1}"
                        )
                        continue
                    final_image, transform_params = self.resize_with_padding(
                        augmented_image, target_size
                    )
                    # 使用增強後影像尺寸作為座標基準
                    aug_h, aug_w = augmented_image.shape[:2]
                    final_bboxes = self.transform_bboxes_after_resize(
                        augmented_bboxes, aug_w, aug_h, transform_params
                    )
                    valid_bboxes = []
                    valid_labels = []
                    for bbox, label in zip(final_bboxes, augmented_labels):
                        if (
                            bbox[2] > 0.001
                            and bbox[3] > 0.001
                            and 0 <= bbox[0] <= 1
                            and 0 <= bbox[1] <= 1
                        ):
                            valid_bboxes.append(bbox)
                            valid_labels.append(label)
                    if len(valid_bboxes) == 0:
                        self.logger.warning(
                            f"沒有有效的最終標註，跳過 {img_file}_aug_{i+1}"
                        )
                        continue
                    if self.config["processing"].get("debug_mode", False):
                        debug_dir = Path(
                            self.config["output"].get(
                                "debug_dir", "debug_visualizations"
                            )
                        )
                        debug_viz_path = (
                            debug_dir / f"{Path(img_file).stem}_aug_{i+1}.png"
                        )
                        self.visualize_bboxes(
                            final_image,
                            valid_bboxes,
                            valid_labels,
                            str(debug_viz_path),
                            f"Augmented: {img_file}_aug_{i+1}",
                        )
                    aug_img_filename = f"{Path(img_file).stem}_aug_{i + 1}.png"
                    aug_img_path = (
                        Path(self.config["output"]["image_dir"]) / aug_img_filename
                    )
                    final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(aug_img_path), final_image_bgr)
                    aug_label_filename = f"{Path(img_file).stem}_aug_{i + 1}.txt"
                    aug_label_path = (
                        Path(self.config["output"]["label_dir"]) / aug_label_filename
                    )
                    with open(aug_label_path, "w") as f:
                        for bbox, label in zip(valid_bboxes, valid_labels):
                            f.write(
                                f"{int(label)} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                            )
                    self.logger.info(
                        f"輸出: {aug_img_filename} (含 {len(valid_bboxes)} 個框)"
                    )
                except Exception as e:
                    self.logger.error(f"套用增強發生錯誤 {img_file}_aug_{i+1}: {e}")
                    continue
        except Exception as e:
            self.logger.error(f"處理單張影像出錯 {img_file}: {e}")

    def process_dataset(self):
        # 確保已建立輸出資料夾與增強流程
        if self.augmentations is None:
            self._setup_output_dirs()
            self.augmentations = self._create_augmentations()
        start_time = time.time()
        input_img_dir = Path(self.config["input"]["image_dir"])
        if not input_img_dir.exists():
            self.logger.error(f"輸入資料夾不存在: {input_img_dir}")
            return
        img_files: List[str] = list_images(input_img_dir, DEFAULT_IMAGE_EXTS)
        if not img_files:
            self.logger.error("沒有找到影像檔案")
            return
        self.logger.info(f"待處理 {len(img_files)} 張影像")
        if self.config["processing"].get("debug_mode", False):
            self.logger.info("調試模式：僅處理第一張影像")
            self._process_single_image(img_files[0])
            self.logger.info(
                "請檢查 debug_visualizations 目錄中的結果，確認無誤再全量處理"
            )
            return
        num_workers = self.config["processing"].get("num_workers", None) or cpu_count()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(
                tqdm(
                    executor.map(self._process_single_image, img_files),
                    total=len(img_files),
                    desc="處理進度",
                    mininterval=0.2,
                )
            )
        elapsed_time = time.time() - start_time
        self.logger.info(f"完成! 花費: {elapsed_time:.2f} 秒")


if __name__ == "__main__":
    augmentor = DataAugmentor()
    augmentor.process_dataset()
