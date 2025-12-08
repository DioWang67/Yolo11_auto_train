# Welcome to Picture Tool Documentation

Picture Tool is a comprehensive pipeline for image processing, augmentation, and YOLO model training.

## Features

- **Format Conversion**: Bulk convert images between formats.
- **Augmentation**: Powerful image and YOLO label augmentation using Albumentations.
- **Training**: Automated YOLOv8/v11 training and evaluation.
- **Quality Control**: Dataset linting, anomaly detection, and color verification.
- **CLI**: Easy-to-use command line interface.

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Usage

Run the pipeline using the CLI:

```bash
picture-tool-pipeline --help
```

Or run specific tasks:

```bash
picture-tool-pipeline run --tasks format_conversion yolo_augmentation
```
