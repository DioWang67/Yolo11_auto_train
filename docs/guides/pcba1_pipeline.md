# PCBA1 YOLO Pipeline Guide

This guide records the current PCBA1 workflow used by this repository:

- LabelImg annotations are imported from `data/PCBA1/raw`.
- YOLO training uses `configs/pcba1_pipeline.yaml`.
- The deployable runtime config uses ONNX by default.
- Color checking and left-to-right sequence checking are disabled for PCBA1.
- Count checking and position checking remain enabled.

## Task Classification

This workflow is a Class C computer-vision pipeline task. The priority is a clear, repeatable local training and deployment flow rather than an enterprise-style service architecture. Configuration and paths are kept explicit so the pipeline can be adjusted quickly when PCBA1 data changes.

## Directory Layout

Expected raw dataset layout:

```text
data/PCBA1/raw/
  images/
    *.jpg
    *.png
  labels/
    *.txt
    classes.txt
```

`classes.txt` should contain the LabelImg class names in the same order as the YOLO labels. Current PCBA1 classes are:

```text
J5-1
J5-2
C22B
J6
J7
```

The annotation validator intentionally ignores `classes.txt`. It is metadata, not a YOLO bounding-box label file.

## Main Config

Use this config for PCBA1:

```text
configs/pcba1_pipeline.yaml
```

Important settings:

```yaml
yolo_augmentation:
  augmentation:
    num_images: 120

yolo_training:
  epochs: 80

  export_onnx:
    enabled: true
    weights_name: best.pt

  export_detection_config:
    enabled: true
    weights_name: best.onnx
    pipeline:
      - count_check
      - save_results
    enable_color_check: false
```

Meaning:

- `export_onnx.weights_name: best.pt` means ONNX is exported from the trained PyTorch weight.
- `export_detection_config.weights_name: best.onnx` means the generated inference config loads ONNX by default.
- `color_check` is not in `pipeline`.
- `sequence_check` is not in `pipeline`.

## Running PCBA1

From the repository root:

```powershell
D:\miniconda\envs\yolo_anomalib\python.exe -m picture_tool.cli --config configs\pcba1_pipeline.yaml --tasks full
```

If using the GUI, select:

- Product: `PCBA1`
- Config: `configs/pcba1_pipeline.yaml`
- Task group: the default YOLO training + position workflow

## Outputs

Training output:

```text
runs/PCBA1/train/
  weights/
    best.pt
    last.pt
    best.onnx
  detection_config.yaml
  auto_position_config.yaml
  results.csv
```

Deployable folder:

```text
runs/PCBA1/train/PCBA1/A/yolo/
  config.yaml
  args.yaml
  results.csv
  weights/
    best.onnx
    best.pt
    last.pt
```

Deployable zip:

```text
runs/PCBA1/train/PCBA1_bundle.zip
```

The deployable `config.yaml` should contain:

```yaml
weights: models/PCBA1/A/yolo/weights/best.onnx
pipeline:
- count_check
- save_results
enable_color_check: false
```

It should not contain:

```yaml
- color_check
- sequence_check
```

## Checks Still Enabled

### Count Check

`count_check` is enabled and strict:

```yaml
steps:
  count_check:
    strict: true
```

The expected PCBA1/A items are:

```yaml
expected_items:
  PCBA1:
    A:
      - J5-1
      - J5-2
      - C22B
      - J6
      - J7
```

### Position Check

Position checking is included through `position_config`. The pipeline auto-generates position statistics after training and writes them into the exported detection config. This is the preferred PCBA1 layout check because component order can be misleading when using only left-to-right sorting.

## Disabled Checks

### Color Check

Color checking is disabled:

```yaml
enable_color_check: false
```

Reason: current PCBA1 deployment does not include `color_stats.json`, and the required check is component detection/position rather than color classification.

### Sequence Check

Sequence checking is disabled by removing `sequence_check` from `pipeline`.

Reason: the PCBA1 component positions are better verified by `position_config`. A simple left-to-right sequence can fail or misrepresent the board layout.

## When To Rerun Training

Rerun training when:

- New PCBA1 raw images or labels are added.
- Label names or class order changes.
- `epochs`, `num_images`, augmentation policy, image size, or model changes.
- You want a new model after improving real-world data coverage.

You do not need to rerun training when only changing:

- Deploy config from `best.pt` to `best.onnx`.
- `enable_color_check`.
- `pipeline` entries such as disabling `sequence_check`.

For config-only changes, regenerate or edit the deployable config/bundle instead.

## Validation Checklist

After a successful PCBA1 run, verify:

```powershell
Test-Path runs\PCBA1\train\weights\best.onnx
Test-Path runs\PCBA1\train\detection_config.yaml
Test-Path runs\PCBA1\train\PCBA1\A\yolo\config.yaml
```

Check the deploy config:

```powershell
Get-Content runs\PCBA1\train\PCBA1\A\yolo\config.yaml | Select-String "weights:|pipeline:|sequence_check|color_check|enable_color_check"
```

Expected result:

```text
weights: models/PCBA1/A/yolo/weights/best.onnx
pipeline:
enable_color_check: false
```

There should be no `sequence_check` or `color_check` line.

Run focused tests after code/config changes:

```powershell
D:\miniconda\envs\yolo_anomalib\python.exe -m pytest tests\test_detection_config_full.py tests\test_tasks_and_utils_features.py tests\test_path_resolver.py tests\test_annotation_tracker.py
```

## Deployment Notes

For zip deployment, use:

```text
runs/PCBA1/train/PCBA1_bundle.zip
```

Extract it into the inference project's `models/` directory so the final layout becomes:

```text
models/
  PCBA1/
    A/
      yolo/
        config.yaml
        weights/
          best.onnx
```

For direct deployment through the `deploy` task, the deploy logic respects the weight selected in `detection_config.yaml`. If `detection_config.yaml` says `weights: best.onnx`, the deployed versioned weight will also be `.onnx`.

## Common Issues

### Annotation validation reports `classes.txt` as invalid

This should be fixed. `classes.txt` is now treated as metadata and ignored by annotation validation.

### Detection config exports default color names

Use `configs/pcba1_pipeline.yaml` and set Product to `PCBA1`. The resolver reads PCBA1 class names from:

```text
data/PCBA1/raw/labels/classes.txt
```

### ONNX file exists but config uses `best.pt`

Check:

```yaml
yolo_training:
  export_detection_config:
    weights_name: best.onnx
```

`export_onnx.weights_name` should stay `best.pt`; only `export_detection_config.weights_name` should be `best.onnx`.

### Metrics are very high

High metrics can be valid for a pipeline sanity check, but if the raw dataset has only a few real images and many augmentations, the result may overfit. Add more real PCBA photos across lighting, focus, board placement, rotation, and production variation before treating the model as production-ready.
