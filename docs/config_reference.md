# Configuration Reference

This document summarises the keys expected in `picture_tool/resources/default_pipeline.yaml`, the
default values, and typical usage patterns so new frontends or automation can
build valid configs without reading the entire YAML manually.

## Top-level structure

```
pipeline:                # Task orchestration and logging
format_conversion:       # Image format conversion settings
image_augmentation:      # Generic image augmentation
yolo_augmentation:       # YOLO specific augmentation
train_test_split:        # Dataset splitter settings
yolo_training:           # Ultralytics YOLO training / export
batch_inference:         # Bulk inference after training
dataset_lint (optional)  # Dataset lint/preview helpers
aug_preview (optional)   # Augmentation preview output
led_qc_enhanced (opt.)   # LED QC workflow configuration
```

Below sections outline the most frequently adjusted blocks. All paths are
interpreted relative to the project root unless stated otherwise.

---

## `pipeline`

| Key             | Type     | Description |
|-----------------|----------|-------------|
| `log_file`      | str      | Log file path used by CLI/GUI pipeline runs. |
| `tasks`         | list     | Canonical task definitions (`name`, `enabled`, `dependencies`). GUI uses this for checkboxes. |
| `task_groups`   | mapping  | Named task presets consumed by `get_tasks_from_groups` and GUI presets. |
| `force` (opt.)  | bool     | When true, `_should_skip` will always execute tasks even if outputs look fresh. |

---

## `format_conversion`

| Key               | Type        | Description |
|-------------------|-------------|-------------|
| `input_dir`       | str         | Source image directory for conversion. |
| `input_formats`   | list[str]   | Allowed input suffixes (e.g. `[".bmp"]`). |
| `output_dir`      | str         | Destination directory for converted images. |
| `output_format`   | str         | Target suffix (e.g. `.png`). |
| `quality` (opt.)  | int         | JPEG output quality when relevant. |

---

## `image_augmentation` / `yolo_augmentation`

| Key                 | Type                   | Description |
|---------------------|------------------------|-------------|
| `input.image_dir`   | str                    | Input image root. |
| `input.label_dir`   | str (YOLO only)        | Label directory (YOLO augmentation). |
| `output.image_dir`  | str                    | Output augmented image directory. |
| `output.label_dir`  | str (YOLO only)        | Output label directory. |
| `augmentation`      | mapping                | Augmentation definition: `num_images`, `num_operations`, and per-operation params (`blur.kernel`, `flip.probability`, etc.). |
| `processing.num_workers` | int              | Worker count. Keep `0` on Windows/GUI to avoid spawning issues. |

---

## `train_test_split`

| Key                           | Type        | Description |
|-------------------------------|-------------|-------------|
| `input.image_dir`             | str         | Source image directory. |
| `input.label_dir`             | str         | Source label directory. |
| `input_formats`               | list[str]   | Allowed suffixes. |
| `label_format`                | str         | Label suffix (e.g. `.txt`). |
| `output.output_dir`           | str         | Root directory for split dataset (`train/val/test`). |
| `split_ratios.train/val/test` | float       | Ratios that must sum to 1.0. |
| `stratified`                  | bool        | Whether to stratify by label. |

---

## `yolo_training`

| Key                        | Type        | Description |
|----------------------------|-------------|-------------|
| `dataset_dir`              | str         | Root directory containing `train/val/test`. |
| `class_names`              | list[str]   | List of class labels, required by trainer. |
| `model`                    | str         | Model path or Ultralytics weight name. |
| `epochs`                   | int         | Training epochs. |
| `imgsz`                    | int         | Image size. |
| `batch`                    | int         | Batch size. |
| `device`                   | str         | `cpu`, `0`, etc. |
| `project`                  | str         | Ultralytics run directory root (e.g. `./runs/detect`). |
| `name`                     | str         | Run name inside project. |
| `workers` (opt.)           | int         | DataLoader workers; default 0 on Windows via trainer helper. |
| `position_validation`      | mapping     | Optional position validation config (see below). |
| `export_detection_config`  | mapping     | Post-training export settings (thresholds, output dir, expected items). |
| `artifact_bundle`          | mapping     | Controls bundling of outputs (weights, configs, results). |

### `position_validation`
Common keys: `enabled` (bool), `product` (str), `area` (str), `imgsz`, `sample_dir`,
`config_path` or inline `config`. When enabled, GUI/mixin validates required
fields before allowing runs.

### `export_detection_config`
Important keys: `enabled`, `output_path`, `weights_name`, `conf_thres`, `iou_thres`,
`current_product`, `area`, `expected_items`, `include_all_products`.

### `artifact_bundle`
Controls the optional `bundle/` directory created after training. Keys include
`enabled`, `dir_name`, `base_dir`, and toggles such as `include_weights`,
`include_detection_config`, etc.

---

## `batch_inference`

| Key            | Type   | Description |
|----------------|--------|-------------|
| `input_dir`    | str    | Images to run inference on. |
| `output_dir`   | str    | Destination for predicted CSV/visualisations. |
| `weights`      | str    | Path to YOLO weights. Defaults to latest run if unset. |
| `imgsz`        | int    | Inference image size. |
| `device`       | str    | Device string. |
| `conf`, `iou`  | float  | Prediction thresholds. |

---

## LED QC (`led_qc_enhanced`)

This block is only required when using the LED QC sidebar. Important keys:

- `colors`, `color_aliases`: Define colour mapping and alternative names.
- `color_conf_min_per_color`: Per-colour minimum confidence thresholds.
- `color_hue_range_margin`: Margin applied when matching hues.
- `build` / `detect` / `detect_dir` / `analyze`: Sub-configurations controlling
  template building, single-image detection, directory detection, and analysis.
- Paths inside each block follow the same relative-path convention as other sections.

---

## Console entry points

Once installed (`pip install -e .`), the following interfaces are
available:

| Command | Description |
|---------|-------------|
| `picture-tool-pipeline --config CONFIG --tasks TASK1 TASK2` | Run the pipeline from the CLI. |
| `python -m picture_tool.main_pipeline --config CONFIG ...` | Module form of the same CLI. |
| `picture-tool-gui` or `python -m picture_tool.gui.app` | Launch the Qt GUI. |

---

## Tips

- Keep `yolo_training.workers` at `0` on Windows/GUI to avoid multiprocessing
  crashes.
- Use `pipeline.task_groups` to define higher-level presets for GUI/CLI reuse.
- After editing your config file, the GUI can reload the file via the **?頛**
  button; CLI will re-read before each task execution via `load_config_if_updated`.
