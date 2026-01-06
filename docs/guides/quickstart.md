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

Training outputs:
```
runs/detect/train/
  ├── weights/
  │   ├── best.pt
  │   └── last.pt
  ├── results.csv
  └── confusion_matrix.png
```

## Common Tasks

### Train a Model
```bash
python -m picture_tool.main_pipeline configs/train.yaml --tasks yolo_train
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

- Read [Training Guide](guides/training.md)
- Read [Testing Guide](guides/testing.md)
- See [Architecture](ARCHITECTURE.md)
