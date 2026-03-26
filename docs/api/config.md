# Config Validation API

## `validate_config_schema`

```python
from picture_tool.config_validation import validate_config_schema

config = validate_config_schema(raw_dict, logger=logger, strict=False)
```

Validates the pipeline configuration dictionary using Pydantic v2 when available, falling back to manual checks.

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `dict` | Raw configuration dictionary (from YAML). |
| `logger` | `logging.Logger \| None` | Logger for validation warnings. Defaults to `None`. |
| `strict` | `bool` | If `True`, raises `ValueError` on any validation error instead of logging a warning. Defaults to `False`. |

**Returns:** The original `config` dict (pass-through; validation is side-effectful via warnings).

**Raises:** `ValueError` when `strict=True` and validation fails.

---

## Pydantic Schemas

The following schemas are used internally. They accept unknown keys via `extra="ignore"` unless noted.

### `YoloTrainingSchema`

| Field | Type | Default | Constraint |
|-------|------|---------|------------|
| `dataset_dir` | `Path \| None` | `None` | Must exist on disk if set |
| `class_names` | `list[str] \| None` | `None` | Must not be empty if set |
| `model` | `str \| Path` | `"yolo11n.pt"` | — |
| `epochs` | `int` | `100` | `> 0` |
| `imgsz` | `int` | `640` | `> 0` |
| `batch` | `int` | `16` | `> 0` |
| `device` | `str` | `"cpu"` | — |
| `project` | `Path \| None` | `None` | — |
| `name` | `str` | `"train"` | — |
| `position_validation` | `PositionValidationSchema \| None` | `None` | — |

Extra keys (`export_onnx`, `export_detection_config`, etc.) are forwarded without error.

### `PositionValidationSchema`

| Field | Type | Default | Constraint |
|-------|------|---------|------------|
| `enabled` | `bool` | `False` | — |
| `auto_generate` | `bool` | `True` | — |
| `product` | `str \| None` | `None` | — |
| `area` | `str \| None` | `None` | — |
| `tolerance` | `float` | `0.0` | — |
| `sample_dir` | `Path \| None` | `None` | Must exist on disk if set |

### `AugmentationSchema`

| Field | Type | Default | Constraint |
|-------|------|---------|------------|
| `enabled` | `bool` | `True` | — |
| `num_images` | `int` | `0` | `>= 0` |
| `operations` | `dict` | `{}` | Each op must have `probability` ∈ [0, 1] |

### `ProcessingSchema`

| Field | Type | Default | Constraint |
|-------|------|---------|------------|
| `batch_size` | `int` | `16` | `> 0` |
| `num_workers` | `int` | `4` | `>= 0` |
| `device` | `str` | `"cpu"` | — |

---

## Example

```python
import yaml
from picture_tool.config_validation import validate_config_schema

with open("configs/my_project.yaml") as f:
    raw = yaml.safe_load(f)

# Non-strict: logs warnings, continues
config = validate_config_schema(raw, strict=False)

# Strict: raises ValueError if dataset_dir doesn't exist
config = validate_config_schema(raw, strict=True)
```
