import cv2
import albumentations as A
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from tqdm import tqdm
import time
import yaml
import logging
import argparse
from typing import List

from picture_tool.utils import list_images, DEFAULT_IMAGE_EXTS, setup_module_logger


class DataAugmentor:
    def __init__(self, config_path: str = None):
        """YOLO 影像與標註資料增強器（維持相容的工具版本）"""
        self._setup_logging()
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self._set_seed(self.config.get('processing', {}).get('seed', None))
        self._setup_output_dirs()
        self.augmentations = self._create_augmentations()

    def _setup_logging(self):
        self.logger = setup_module_logger(__name__, 'augmentation.log')

    def _set_seed(self, seed):
        if seed is None:
            return
        try:
            import random
            random.seed(int(seed))
            np.random.seed(int(seed))
            self.logger.info(f"已設定隨機種子: {seed}")
        except Exception as e:
            self.logger.warning(f"設定隨機種子失敗: {e}")

    def _setup_output_dirs(self):
        try:
            output_img_dir = Path(self.config['output']['image_dir'])
            output_img_dir.mkdir(parents=True, exist_ok=True)

            output_label_dir = Path(self.config['output']['label_dir'])
            output_label_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"已建輸出資料夾: {output_img_dir}, {output_label_dir}")
        except Exception as e:
            self.logger.error(f"建立輸出目錄失敗: {e}")
            raise

    def _load_config(self, config_path: str) -> dict:
        try:
            if config_path and Path(config_path).exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"載入設定: {config_path}")
                return config
            else:
                self.logger.warning(f"設定檔 {config_path} 不存在，使用預設設定")
                return self._default_config()
        except Exception as e:
            self.logger.error(f"讀取設定檔失敗: {e}")
            return self._default_config()

    def _default_config(self) -> dict:
        return {
            'input': {
                'image_dir': 'A/img',
                'label_dir': 'A/label'
            },
            'output': {
                'image_dir': 'output/images',
                'label_dir': 'output/labels'
            },
            'augmentation': {
                'num_images': 5,
                'num_operations': (3, 5),
                'operations': {
                    'flip': {'probability': 0.5},
                    'rotate': {'angle': (-10, 10)},
                    'multiply': {'range': (0.8, 1.2)},
                    'scale': {'range': (0.8, 1.2)},
                    'contrast': {'range': (0.75, 1.5)},
                    'hue': {'range': (-10, 10)},
                    'noise': {'scale': (0, 0.05)},
                    'perspective': {'scale': (0.01, 0.05)},
                    'blur': {'kernel': (3, 5)}
                }
            },
            'processing': {
                'batch_size': 10,
                'num_workers': None,
                'seed': None
            }
        }

    def _create_augmentations(self) -> A.Compose:
        aug_config = self.config['augmentation']
        ops_config = aug_config['operations']

        aug_list = []

        # 先固定尺寸為 640x640（等比例縮放並填黑）
        aug_list.append(A.LongestMaxSize(max_size=640, interpolation=cv2.INTER_AREA, p=1.0))
        aug_list.append(A.PadIfNeeded(
            min_height=640,
            min_width=640,
            border_mode=cv2.BORDER_CONSTANT,
            border_value=(0, 0, 0),
            p=1.0
        ))

        if ops_config.get('flip'):
            aug_list.append(A.HorizontalFlip(p=ops_config['flip']['probability']))

        if ops_config.get('rotate'):
            angle = ops_config['rotate']['angle']
            limit = angle if isinstance(angle, (int, float)) else (angle[0], angle[1])
            aug_list.append(A.Rotate(limit=limit, p=0.5))

        if ops_config.get('multiply'):
            multiply_range = ops_config['multiply']['range']
            aug_list.append(A.RandomBrightnessContrast(
                brightness_limit=(multiply_range[0]-1, multiply_range[1]-1), p=0.5))

        if ops_config.get('scale'):
            scale_range = ops_config['scale']['range']
            aug_list.append(A.RandomScale(scale_limit=(scale_range[0]-1, scale_range[1]-1), p=0.5))

        if ops_config.get('contrast'):
            contrast_range = ops_config['contrast']['range']
            aug_list.append(A.RandomBrightnessContrast(
                contrast_limit=(contrast_range[0]-1, contrast_range[1]-1), p=0.5))

        if ops_config.get('hue'):
            hue_range = ops_config['hue']['range']
            if isinstance(hue_range, (int, float)):
                hue_limit = (-abs(int(hue_range)), abs(int(hue_range)))
            else:
                hue_limit = (int(hue_range[0]), int(hue_range[1]))
            aug_list.append(A.HueSaturationValue(hue_shift_limit=hue_limit, p=0.5))

        if ops_config.get('noise'):
            noise_scale = ops_config['noise']['scale']
            aug_list.append(A.GaussNoise(var_limit=(0, noise_scale[1]*255), p=0.5))

        if ops_config.get('perspective'):
            perspective_scale = ops_config['perspective']['scale']
            aug_list.append(A.Perspective(scale=perspective_scale[1], p=0.5))

        if ops_config.get('blur'):
            blur_kernel = ops_config['blur']['kernel']
            aug_list.append(A.MotionBlur(
                blur_limit=blur_kernel if isinstance(blur_kernel, int) else blur_kernel[1], p=0.5))

        return A.Compose(
            aug_list,
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)
        )

    def _process_single_image(self, img_file: str) -> bool:
        try:
            img_path = Path(self.config['input']['image_dir']) / img_file
            label_path = Path(self.config['input']['label_dir']) / (Path(img_file).stem + '.txt')

            image = cv2.imread(str(img_path))
            if image is None:
                self.logger.error(f"無法讀取影像: {img_path}")
                return False

            # 讀取 YOLO 標註
            try:
                with open(label_path, 'r') as f:
                    annotations = []
                    for line in f.readlines():
                        s = line.strip()
                        if not s or s.startswith('#'):
                            continue
                        parts = s.split()
                        if len(parts) < 5:
                            continue
                        annotations.append(parts[:5])
            except Exception as e:
                self.logger.error(f"讀取標註檔案失敗 {label_path}: {e}")
                return False

            bboxes = []
            class_labels = []
            for ann in annotations:
                try:
                    cls, x_center, y_center, width, height = map(float, ann)
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(int(cls))
                except ValueError:
                    continue

            saved_any = False
            for i in range(self.config['augmentation']['num_images']):
                try:
                    transformed = self.augmentations(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )

                    augmented_image = transformed['image']
                    augmented_bboxes = transformed['bboxes']
                    augmented_labels = transformed['class_labels']

                    if len(augmented_bboxes) != len(bboxes):
                        self.logger.warning(f"增強後框數不一致，跳過此回合: {img_file}")
                        continue

                    aug_img_filename = f"{Path(img_file).stem}_aug_{i + 1}.png"
                    aug_img_path = Path(self.config['output']['image_dir']) / aug_img_filename
                    cv2.imwrite(str(aug_img_path), augmented_image)

                    aug_label_filename = f"{Path(img_file).stem}_aug_{i + 1}.txt"
                    aug_label_path = Path(self.config['output']['label_dir']) / aug_label_filename
                    with open(aug_label_path, 'w') as f:
                        for bbox, label in zip(augmented_bboxes, augmented_labels):
                            x_center = np.clip(bbox[0], 0, 1)
                            y_center = np.clip(bbox[1], 0, 1)
                            width = np.clip(bbox[2], 0, 1)
                            height = np.clip(bbox[3], 0, 1)
                            f.write(f"{int(label)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                    saved_any = True
                except Exception as e:
                    self.logger.error(f"套用增強發生錯誤 {img_file}, 索引 {i}: {e}")
                    continue

            return saved_any

        except Exception as e:
            self.logger.error(f"處理單張影像發生錯誤 {img_file}: {e}")
            return False

    def process_dataset(self):
        start_time = time.time()

        input_img_dir = Path(self.config['input']['image_dir'])
        if not input_img_dir.exists():
            self.logger.error(f"輸入資料夾不存在: {input_img_dir}")
            return

        img_files: List[str] = list_images(input_img_dir, DEFAULT_IMAGE_EXTS)
        if not img_files:
            self.logger.error("沒有找到影像檔案")
            return

        self.logger.info(f"待處理 {len(img_files)} 張影像")

        num_workers = self.config['processing'].get('num_workers', None) or cpu_count()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(self._process_single_image, img_files),
                total=len(img_files),
                desc="處理進度",
                mininterval=0.2
            ))

        elapsed_time = time.time() - start_time
        ok = sum(1 for r in results if r)
        fail = len(results) - ok
        self.logger.info(f"完成! 花費: {elapsed_time:.2f} 秒，成功 {ok}，失敗 {fail}")


def main():
    parser = argparse.ArgumentParser(description='YOLO 標註增強工具')
    parser.add_argument('--config', type=str, default='config.yaml', help='設定檔路徑')
    args = parser.parse_args()

    augmentor = DataAugmentor(args.config)
    augmentor.process_dataset()


if __name__ == "__main__":
    main()

