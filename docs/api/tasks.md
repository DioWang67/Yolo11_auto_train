# Tasks API

## Training Tasks (`picture_tool.tasks.training`)

### `run_yolo_train(config, args)`

Runs YOLO training and post-training steps:
1. `train_yolo()` — trains the model, writes versioned run directory (`train`, `train2`, …)
2. `PositionConfigGenerator.generate()` — generates `auto_position_config.yaml`
3. `OnnxExporter.export()` — exports `best.onnx` if `export_onnx.enabled`
4. `DetectionConfigExporter.export()` — writes `detection_config.yaml` for use by yolo11_inference

### `skip_yolo_train(config, args) -> str | None`

Skips if the **latest** versioned run directory has matching dataset + config hashes and a `best.pt` exists.
Falls back to mtime comparison if `last_run_metadata.json` is missing or corrupt.
Always returns `None` (no skip) when `args.force is True`.

### `run_yolo_evaluation(config, args)`

Runs YOLO evaluation on the validation set using existing weights. Auto-detects the latest `best.pt` if no explicit weights are configured.

### `run_position_validation_task(config, args)`

Runs offline position validation using trained weights and sample images. Reads `auto_position_config.yaml` from the run directory if no explicit config is provided.

---

## Task Registry

All training tasks are registered in `TASKS`:

| Task name | `run` function | `skip_fn` | Dependencies |
|-----------|---------------|-----------|--------------|
| `yolo_train` | `run_yolo_train` | `skip_yolo_train` | `dataset_splitter` |
| `yolo_evaluation` | `run_yolo_evaluation` | — | `yolo_train` |
| `position_validation` | `run_position_validation_task` | — | — |
| `artifact_bundle` | `run_artifact_bundle` | — | — |
| `deploy` | `run_deploy` | — | `yolo_train` |

---

## Conversion Tasks (`picture_tool.tasks.conversion`)

### `run_format_conversion(config, args)`

Bulk-converts images between formats (JPEG, PNG, BMP, TIFF, WebP).

Config block: `format_conversion`

| Key | Type | Description |
|-----|------|-------------|
| `input_dir` | str | Source directory |
| `output_dir` | str | Destination directory |
| `output_format` | str | e.g. `"jpg"`, `"png"` |
| `quality` | int | JPEG quality 1–100 (default `95`) |

---

## Quality Tasks (`picture_tool.tasks.quality`)

| Task name | Description | Config block |
|-----------|-------------|--------------|
| `dataset_splitter` | Split images+labels into train/val/test | `train_test_split` |
| `dataset_lint` | Check label format, missing files, class IDs | `dataset_lint` |
| `aug_preview` | Generate augmentation preview images | `aug_preview` |
| `batch_inference` | Run batch YOLO inference on a folder | `batch_inference` |
| `qc_summary` | Aggregate colour/position/inference QC results | `qc_summary` |

---

## Augmentation Tasks (`picture_tool.tasks.augmentation`)

| Task name | Description | Config block |
|-----------|-------------|--------------|
| `yolo_augmentation` | Augment image-label pairs (YOLO format) | `yolo_augmentation` |
| `image_augmentation` | Augment images only (no labels) | `image_augmentation` |
