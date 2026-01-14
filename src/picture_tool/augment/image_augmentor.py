from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, List, Optional

import albumentations as A  # type: ignore[import-untyped]
import cv2
import yaml  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from picture_tool.utils import list_images, DEFAULT_IMAGE_EXTS, setup_module_logger

# Global variable for worker processes
_worker_augmentations = None


def _init_worker(ops_config: dict[str, Any]) -> None:
    """Initialize the augmentation pipeline in the worker process."""
    global _worker_augmentations
    _worker_augmentations = ImageAugmentor._build_augmentations_from_ops(ops_config)


class ImageAugmentor:
    def __init__(self, config_path: str | None = None) -> None:
        self._setup_logging()
        self.config: dict[str, Any] = (
            self._load_config(config_path) if config_path else self._default_config()
        )
        self._set_seed(self.config.get("processing", {}).get("seed"))
        self._setup_output_dirs()
        self.augmentations = self._create_augmentations()

    def _setup_logging(self) -> None:
        self.logger = setup_module_logger(__name__, "image_augmentation.log")

    def _set_seed(self, seed: Any) -> None:
        if seed is None:
            return
        try:
            import random
            import numpy as np

            random.seed(int(seed))
            np.random.seed(int(seed))
            self.logger.info(f"已設定隨機種子: {seed}")
        except Exception as e:
            self.logger.warning(f"設定隨機種子失敗: {e}")

    def _setup_output_dirs(self) -> None:
        try:
            output_img_dir = Path(self.config["output"]["image_dir"])
            output_img_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"已建輸出資料夾: {output_img_dir}")
        except (ValueError, TypeError, AttributeError) as e:
            self.logger.error(f"建立輸出目錄失敗: {e}")
            raise

    def _load_config(self, config_path: Optional[str]) -> dict[str, Any]:
        try:
            if config_path and Path(config_path).exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"載入 {config_path} 設定")
                return config
            else:
                self.logger.warning(f"設定檔 {config_path} 不存在，使用預設設定")
                return self._default_config()
        except (OSError, yaml.YAMLError) as e:
            self.logger.error(f"讀取設定檔失敗: {e}")
            return self._default_config()

    def _default_config(self) -> dict[str, Any]:
        return {
            "input": {"image_dir": "A/img"},
            "output": {"image_dir": "output/images"},
            "augmentation": {
                "num_images": 5,
                "operations": {
                    "flip": {"probability": 0.5},
                    "rotate": {"angle": (-10, 10)},
                    "multiply": {"range": (0.8, 1.2)},
                    "scale": {"range": (0.8, 1.2)},
                    "contrast": {"range": (0.75, 1.5)},
                    "hue": {"range": (-10, 10)},
                    "noise": {"scale": (0, 0.05)},
                    "perspective": {"scale": (0.01, 0.05)},
                    "blur": {"kernel": (3, 5)},
                },
            },
            "processing": {
                "batch_size": 10,
                "num_workers": None,
                "use_process_pool": False,
            },
        }

    @staticmethod
    def _build_augmentations_from_ops(ops_config: dict[str, Any]) -> Any:
        aug_list = []

        if ops_config.get("flip"):
            aug_list.append(A.HorizontalFlip(p=ops_config["flip"]["probability"]))

        if ops_config.get("rotate"):
            angle = ops_config["rotate"]["angle"]
            limit = angle if isinstance(angle, (int, float)) else (angle[0], angle[1])
            aug_list.append(A.Rotate(limit=limit, p=0.5))

        if ops_config.get("multiply"):
            multiply_range = ops_config["multiply"]["range"]
            aug_list.append(
                A.RandomBrightnessContrast(
                    brightness_limit=(multiply_range[0] - 1, multiply_range[1] - 1),
                    p=0.5,
                )
            )

        if ops_config.get("scale"):
            scale_range = ops_config["scale"]["range"]
            aug_list.append(
                A.RandomScale(
                    scale_limit=(scale_range[0] - 1, scale_range[1] - 1), p=0.5
                )
            )

        if ops_config.get("contrast"):
            contrast_range = ops_config["contrast"]["range"]
            aug_list.append(
                A.RandomBrightnessContrast(
                    contrast_limit=(contrast_range[0] - 1, contrast_range[1] - 1), p=0.5
                )
            )

        if ops_config.get("hue"):
            hue_range = ops_config["hue"]["range"]
            if isinstance(hue_range, (int, float)):
                hue_limit = (-abs(int(hue_range)), abs(int(hue_range)))
            else:
                hue_limit = (int(hue_range[0]), int(hue_range[1]))
            aug_list.append(A.HueSaturationValue(hue_shift_limit=hue_limit, p=0.5))

        if ops_config.get("noise"):
            noise_scale = ops_config["noise"]["scale"]
            aug_list.append(A.GaussNoise(var_limit=(0, noise_scale[1] * 255), p=0.5))

        if ops_config.get("perspective"):
            perspective_scale = ops_config["perspective"]["scale"]
            aug_list.append(A.Perspective(scale=perspective_scale[1], p=0.5))

        if ops_config.get("blur"):
            blur_kernel = ops_config["blur"]["kernel"]
            aug_list.append(
                A.MotionBlur(
                    blur_limit=(
                        blur_kernel if isinstance(blur_kernel, int) else blur_kernel[1]
                    ),
                    p=0.5,
                )
            )

        aug_list.append(
            A.LongestMaxSize(max_size=640, interpolation=cv2.INTER_AREA, p=1.0)
        )
        aug_list.append(
            A.PadIfNeeded(
                min_height=640,
                min_width=640,
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0,
            )
        )

        return A.Compose(aug_list)

    def _create_augmentations(self) -> Any:
        aug_config = self.config["augmentation"]
        ops_config = aug_config["operations"]
        return self._build_augmentations_from_ops(ops_config)

    def _process_single_image(self, img_file: str) -> bool:
        try:
            img_path = Path(self.config["input"]["image_dir"]) / img_file

            image = cv2.imread(str(img_path))
            if image is None:
                self.logger.error(f"無法讀取影像: {img_path}")
                return False

            saved_any = False
            for i in range(self.config["augmentation"]["num_images"]):
                try:
                    transformed = self.augmentations(image=image)
                    augmented_image = transformed["image"]

                    if augmented_image.shape[:2] != (640, 640):
                        self.logger.warning(
                            f"影像 {img_file} 增強後尺寸非 640x640: {augmented_image.shape[:2]}"
                        )

                    aug_img_filename = f"{Path(img_file).stem}_aug_{i + 1}.png"
                    aug_img_path = (
                        Path(self.config["output"]["image_dir"]) / aug_img_filename
                    )
                    cv2.imwrite(str(aug_img_path), augmented_image)
                    self.logger.info(f"輸出增強影像: {aug_img_path}")
                    saved_any = True

                except (ValueError, RuntimeError, OSError) as e:
                    self.logger.error(f"處理增強發生錯誤 {img_file}, 索引 {i}: {e}")
                    continue

        except (OSError, cv2.error, TypeError) as e:
            self.logger.error(f"處理單張影像發生錯誤 {img_file}: {e}")
            return False
        return saved_any

    def process_dataset(self) -> None:
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

        workers_value = self.config["processing"].get("num_workers")
        if isinstance(workers_value, (int, float)):
            num_workers = int(workers_value) if int(workers_value) > 0 else cpu_count()
        else:
            num_workers = cpu_count()

        use_process_pool = bool(
            self.config["processing"].get("use_process_pool", False)
        )

        if use_process_pool:
            # Pass only necessary data, not the full config which might contain non-picklable items
            ops_config = self.config["augmentation"]["operations"]
            job_args = [
                (
                    img_file,
                    self.config["input"]["image_dir"],
                    self.config["output"]["image_dir"],
                    self.config["augmentation"]["num_images"],
                )
                for img_file in img_files
            ]
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_worker,
                initargs=(ops_config,),
            ) as executor:
                results = list(
                    tqdm(
                        executor.map(_process_single_image_job, job_args),
                        total=len(img_files),
                        desc="處理進度",
                        mininterval=0.2,
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(
                    tqdm(
                        executor.map(self._process_single_image, img_files),
                        total=len(img_files),
                        desc="處理進度",
                        mininterval=0.2,
                    )
                )

        elapsed_time = time.time() - start_time
        ok = sum(1 for r in results if r)
        fail = len(results) - ok
        self.logger.info(f"完成! 花費: {elapsed_time:.2f} 秒，成功 {ok}，失敗 {fail}")


def _process_single_image_job(args: tuple[str, str, str, int]) -> bool:
    img_file, input_dir, output_dir, num_images = args
    try:
        img_path = Path(input_dir) / img_file
        image = cv2.imread(str(img_path))
        if image is None:
            return False

        # Use the global augmentations initialized in the worker
        global _worker_augmentations
        if _worker_augmentations is None:
            return False

        saved_any = False
        for i in range(num_images):
            transformed = _worker_augmentations(image=image)
            augmented_image = transformed["image"]
            aug_img_filename = f"{Path(img_file).stem}_aug_{i + 1}.png"
            aug_img_path = Path(output_dir) / aug_img_filename
            cv2.imwrite(str(aug_img_path), augmented_image)
            saved_any = True
        return saved_any
    except (OSError, ValueError, TypeError):
        return False
