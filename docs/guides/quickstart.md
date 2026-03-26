# Quick Start Guide

## Installation

```bash
# Clone repository
git clone <repo-url>
cd Yolo11_auto_train

# Install dependencies
pip install -r requirements.txt

# Optional: Install GUI support
pip install -e ".[gui]"
```

## Basic Usage

### 1. Prepare Configuration

Create or edit `configs/my_project.yaml`:

```yaml
pipeline:
  default_tasks:
    - dataset_splitter
    - yolo_train
    - yolo_evaluation

train_test_split:
  input:
    image_dir: "./data/raw/images"
    label_dir: "./data/raw/labels"
  output:
    output_dir: "./data/split"
  split_ratios:
    train: 0.7
    val: 0.2
    test: 0.1

yolo_training:
  model: "yolov11n.pt"
  class_names: ["red", "green", "blue"]
  epochs: 50
  imgsz: 640
  batch: 16
  device: "0"  # GPU index, or "cpu"
```

### 2. Run Pipeline

#### CLI
```bash
python -m picture_tool.main_pipeline configs/my_project.yaml
```

#### GUI
```bash
python -m picture_tool.gui.app
```

### 3. Check Results

Training outputs (versioned — each run creates a new directory):
```
runs/detect/
  ├── train/          ← first run
  │   ├── weights/
  │   │   ├── best.pt
  │   │   └── last.pt
  │   ├── results.csv
  │   └── confusion_matrix.png
  ├── train2/         ← second run (--force)
  └── train3/         ← third run (--force)
```

> **Skip behaviour**: if the dataset and config are unchanged from the last run, `yolo_train` is skipped automatically. Use `--force` to bypass and create a new versioned run directory.

## Common Tasks

### Train a Model
```bash
python -m picture_tool.main_pipeline configs/train.yaml --tasks yolo_train
```

To force retrain even when data/config is unchanged (creates `train2`, `train3`, …):
```bash
python -m picture_tool.main_pipeline configs/train.yaml --tasks yolo_train --force
```

### Run Inference
```bash
python -m picture_tool.infer.batch_infer --config configs/infer.yaml
```

### Export to ONNX
Enable in config:
```yaml
yolo_training:
  export_onnx:
    enabled: true
    simplify: true
```

## Next Steps

- See [Architecture](ARCHITECTURE.md)
- See [Config Reference](config_reference.md)
- See [Integration Guide (Train → Deploy)](INTEGRATION_GUIDE.md)
