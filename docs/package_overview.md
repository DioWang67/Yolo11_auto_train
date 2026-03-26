# Picture Tool — Package Overview

## Overview

A structured toolkit for image pipelines: format conversion, augmentation (image/YOLO), anomaly masks, dataset splitting, quality control, and a PyQt5 GUI orchestrator.

## Package Layout

```
picture_tool/
├── augment/          # Image and YOLO label augmentation (Albumentations)
├── anomaly/          # Anomaly mask generation
├── color/            # LED colour inspection and verification
├── eval/             # YOLO evaluation helpers
├── format/           # Image format conversion
├── infer/            # Batch inference
├── pipeline/         # Pipeline core (Task DAG, executor)
├── position/         # Offline position validation
├── quality/          # Dataset linting, augmentation preview
├── report/           # Report generation and QC summary
├── split/            # train/val/test dataset splitter
├── tasks/            # Task implementations (training, conversion, …)
├── train/            # YOLO trainer wrapper
├── tracking/         # Experiment tracker (MLflow / local YAML)
├── utils/            # IO, hashing, logging, ONNX export helpers
├── gui/              # PyQt5 GUI (app.py, widgets, log viewer)
├── cli.py            # CLI entry points
├── config_validation.py  # Pydantic config validation
├── constants.py      # Project-wide constants
└── exceptions.py     # Domain exception hierarchy
```

## Public API

```python
# Augmentation
from picture_tool.augment import ImageAugmentor, YoloDataAugmentor

# Anomaly mask generation
from picture_tool.anomaly import process_anomaly_detection

# Image format conversion
from picture_tool.format import ImageFormatConverter

# Dataset splitting
from picture_tool.split import split_dataset

# Pipeline execution
from picture_tool.pipeline.core import Pipeline, Task

# Config validation
from picture_tool.config_validation import validate_config_schema

# Training
from picture_tool.train.yolo_trainer import train_yolo

# Evaluation
from picture_tool.eval.yolo_evaluator import evaluate_yolo
```

## CLI Entry Points

After `pip install -e .`:

| Command | Description |
|---------|-------------|
| `picture-tool-pipeline` | Run the full pipeline or selected tasks |
| `picture-tool-gui` | Launch the PyQt5 GUI |
| `picture-tool-color-verify` | Standalone LED colour verification |
| `picture-tool-doctor` | Environment health check |

```bash
# Full pipeline
picture-tool-pipeline --config configs/default_pipeline.yaml --tasks full

# Selected tasks only
picture-tool-pipeline --config configs/default_pipeline.yaml --tasks yolo_train

# Force retrain even when data/config is unchanged
picture-tool-pipeline --config configs/default_pipeline.yaml --tasks yolo_train --force

# List available tasks
picture-tool-pipeline --list-tasks

# Describe a task
picture-tool-pipeline --describe-task yolo_train
```

## Notes

- All task implementations live in `picture_tool/tasks/`; the pipeline resolves dependencies automatically via a DAG.
- `yolo_train` creates versioned run directories (`train`, `train2`, `train3`, …); use `--force` to trigger a new run when data/config is unchanged.
- Albumentations + OpenCV operations use `ThreadPoolExecutor` by default for Windows compatibility.
- For configuration details see `docs/config_reference.md`.
- For the training → inference deployment flow see `docs/INTEGRATION_GUIDE.md`.
