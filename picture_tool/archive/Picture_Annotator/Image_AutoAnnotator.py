import cv2
import albumentations as A
import numpy as np
import os
from multiprocessing import Pool, cpu_count
import random
import logging
from pathlib import Path
from tqdm import tqdm
import time
import yaml
import argparse
from typing import List, Dict

class ImageAugmentor:
    def __init__(self, config_path: str = None):
        """初始化圖片增強器"""
        self._setup_logging()
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self._setup_output_dirs()
        self.augmentations = self._create_augmentations()

    def _setup_logging(self):
        """設置日誌系統"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('image_augmentation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_output_dirs(self):
        """建立輸出目錄"""
        try:
            output_img_dir = Path(self.config['output']['image_dir'])
            output_img_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"創建輸出目錄成功: {output_img_dir}")
        except Exception as e:
            self.logger.error(f"創建輸出目錄失敗: {e}")
            raise

    def _load_config(self, config_path: str) -> dict:
        """從YAML文件加載配置"""
        try:
            if config_path and Path(config_path).exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"成功從 {config_path} 加載配置")
                return config
            else:
                self.logger.warning(f"配置文件 {config_path} 不存在，使用默認配置")
                return self._default_config()
        except Exception as e:
            self.logger.error(f"加載配置文件失敗: {e}")
            return self._default_config()

    def _default_config(self) -> dict:
        """返回默認配置"""
        return {
            'input': {
                'image_dir': 'A/img'
            },
            'output': {
                'image_dir': 'output/images'
            },
            'augmentation': {
                'num_images': 5,
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
                'num_workers': None
            }
        }

    def _create_augmentations(self) -> A.Compose:
        """根據配置創建增強序列"""
        aug_config = self.config['augmentation']
        ops_config = aug_config['operations']
        
        aug_list = []

        # 等比例縮放並用黑邊填充
        aug_list.append(A.LongestMaxSize(max_size=640, interpolation=cv2.INTER_AREA, p=1.0))
        aug_list.append(A.PadIfNeeded(
            min_height=640,
            min_width=640,
            border_mode=cv2.BORDER_CONSTANT,
            border_value=(0, 0, 0),  # 黑色填充
            p=1.0
        ))

        # 建立增強操作列表
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
            aug_list.append(A.HueSaturationValue(
                hue_shift_limit=hue_range if isinstance(hue_range, int) else hue_range[1], p=0.5))
        
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

        return A.Compose(aug_list)

    def _process_single_image(self, img_file: str) -> None:
        """處理單張圖片的增強"""
        try:
            img_path = Path(self.config['input']['image_dir']) / img_file

            # 讀取圖片
            image = cv2.imread(str(img_path))
            if image is None:
                self.logger.error(f"無法讀取圖片: {img_path}")
                return

            # 應用增強
            for i in range(self.config['augmentation']['num_images']):
                try:
                    # 應用增強
                    transformed = self.augmentations(image=image)
                    augmented_image = transformed['image']

                    # 保存增強後的圖片
                    aug_img_filename = f"{Path(img_file).stem}_aug_{i + 1}.png"
                    aug_img_path = Path(self.config['output']['image_dir']) / aug_img_filename
                    cv2.imwrite(str(aug_img_path), augmented_image)
                    self.logger.info(f"生成增強圖片: {aug_img_path}")

                except Exception as e:
                    self.logger.error(f"處理增強時出錯 {img_file}, 索引 {i}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"處理圖片時出錯 {img_file}: {e}")

    def process_dataset(self):
        """處理整個數據集"""
        start_time = time.time()
        
        input_img_dir = Path(self.config['input']['image_dir'])
        if not input_img_dir.exists():
            self.logger.error(f"輸入圖片目錄不存在: {input_img_dir}")
            return
            
        img_files = [f for f in os.listdir(input_img_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
        
        if not img_files:
            self.logger.error("沒有找到圖片文件")
            return

        self.logger.info(f"開始處理 {len(img_files)} 張圖片")

        num_workers = self.config['processing'].get('num_workers', None) or cpu_count()
        
        with Pool(num_workers) as pool:
            list(tqdm(
                pool.imap(self._process_single_image, img_files),
                total=len(img_files),
                desc="處理進度"
            ))

        elapsed_time = time.time() - start_time
        self.logger.info(f"處理完成! 耗時: {elapsed_time:.2f} 秒")

def main():
    parser = argparse.ArgumentParser(description='圖片增強工具')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路徑')
    args = parser.parse_args()

    augmentor = ImageAugmentor(args.config)
    augmentor.process_dataset()

if __name__ == "__main__":
    main()