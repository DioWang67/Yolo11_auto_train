# Augmentation API

## `ImageAugmentor`

General-purpose image augmentor (no YOLO labels). Uses Albumentations + `ThreadPoolExecutor` / `ProcessPoolExecutor`.

```python
from picture_tool.augment import ImageAugmentor

aug = ImageAugmentor(config_path="configs/augment.yaml")
aug.augment_images()
```

### Constructor

```python
ImageAugmentor(config_path: str | None = None)
```

| Parameter | Description |
|-----------|-------------|
| `config_path` | Path to a YAML config file. If `None`, uses built-in defaults. |

### Key Methods

| Method | Description |
|--------|-------------|
| `augment_images()` | Run augmentation on all images in `input.image_dir`, write results to `output.image_dir`. |

### Config Structure

```yaml
input:
  image_dir: ./data/raw/images

output:
  image_dir: ./data/augmented/images

augmentation:
  enabled: true
  num_images: 5          # augmented copies per source image

operations:
  horizontal_flip:
    probability: 0.5
  brightness_contrast:
    probability: 0.3
    brightness_limit: 0.2
    contrast_limit: 0.2
  rotate:
    probability: 0.4
    limit: 15

processing:
  seed: 42
  num_workers: 4         # 0 = auto
```

---

## `YoloDataAugmentor`

YOLO-label-aware augmentor. Augments image + bounding box labels together, preserving YOLO format.

```python
from picture_tool.augment import YoloDataAugmentor

aug = DataAugmentor(config_path="configs/yolo_augment.yaml")
aug.augment_dataset()
```

### Constructor

```python
DataAugmentor(config_path: str | None = None)
```

### Key Methods

| Method | Description |
|--------|-------------|
| `augment_dataset()` | Augment image-label pairs from `input.image_dir` / `input.label_dir`, write to `output.*`. |

### Config Structure

```yaml
input:
  image_dir: ./data/raw/images
  label_dir: ./data/raw/labels

output:
  image_dir: ./data/augmented/images
  label_dir: ./data/augmented/labels
  debug_dir: ./debug_visualizations   # optional bbox overlay previews

augmentation:
  num_images: 3

operations:
  horizontal_flip:
    probability: 0.5
  random_crop:
    probability: 0.3
    height: 512
    width: 512
  brightness_contrast:
    probability: 0.4
```

### `AugmentationError`

Raised when augmentation fails (e.g., missing input directory, incompatible label format).

```python
from picture_tool.augment.yolo_data_augmentor import AugmentationError
```

---

## Pipeline Task Keys

| Task key | Class used | Config block |
|----------|-----------|--------------|
| `image_augmentation` | `ImageAugmentor` | `image_augmentation` |
| `yolo_augmentation` | `YoloDataAugmentor` | `yolo_augmentation` |
