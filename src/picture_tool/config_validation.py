from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationError
except ImportError:
    BaseModel = None  # type: ignore


class _ManualConfigError(Exception):
    """Fallback error when Pydantic is missing."""

    def __init__(self, messages: List[str]) -> None:
        super().__init__("\n".join(messages))
        self.messages = messages


# --- Models ---

if BaseModel is not None:

    class BaseSchema(BaseModel):
        model_config = ConfigDict(
            extra="ignore"
        )  # Ignore unknown keys by default for forward compatibility

    class AugmentationOperation(BaseSchema):
        probability: float = Field(..., ge=0.0, le=1.0)
        # Allow flexible extra fields for specific agumentations (e.g. angle, noise_limit)
        model_config = ConfigDict(extra="allow")

    class AugmentationSchema(BaseSchema):
        enabled: bool = True
        num_images: int = Field(default=0, ge=0)
        operations: Dict[str, Union[AugmentationOperation, Dict[str, Any]]] = Field(
            default_factory=dict
        )

    class ProcessingSchema(BaseSchema):
        batch_size: int = Field(default=16, gt=0)
        num_workers: int = Field(default=4, ge=0)
        device: str = "cpu"

    class PositionValidationSchema(BaseSchema):
        enabled: bool = False
        auto_generate: bool = True
        product: Optional[str] = None
        area: Optional[str] = None
        tolerance: float = 0.0
        sample_dir: Optional[Path] = None

        @field_validator("sample_dir")
        @classmethod
        def _path_exists(cls, v: Optional[Path]) -> Optional[Path]:
            if v and not v.exists():
                raise ValueError(f"sample_dir does not exist: {v}")
            return v

    class YoloTrainingSchema(BaseSchema):
        dataset_dir: Optional[Path] = None
        class_names: Optional[List[str]] = None
        model: Union[str, Path] = "yolo11n.pt"
        epochs: int = Field(default=100, gt=0)
        imgsz: int = Field(default=640, gt=0)
        batch: int = Field(default=16, gt=0)
        device: str = "cpu"
        project: Optional[Path] = None
        name: str = "train"
        position_validation: Optional[PositionValidationSchema] = None

        model_config = ConfigDict(
            extra="allow"
        )  # Allow export_onnx, export_detection_config etc.

        @field_validator("class_names")
        @classmethod
        def _non_empty_list(cls, v: Optional[List[str]]) -> Optional[List[str]]:
            if v is not None and len(v) == 0:
                raise ValueError("class_names must not be empty")
            return v

        @field_validator("dataset_dir")
        @classmethod
        def _dir_exists(cls, v: Optional[Path]) -> Optional[Path]:
            if v and not v.exists():
                raise ValueError(f"dataset_dir does not exist: {v}")
            return v

    class PipelineSchema(BaseSchema):
        yolo_training: Optional[YoloTrainingSchema] = None
        augmentation: Optional[AugmentationSchema] = None
        processing: Optional[ProcessingSchema] = None


def _manual_validate(config: Dict[str, Any]) -> None:
    """Fallback manual validation."""
    errors = []
    ycfg = config.get("yolo_training") or {}

    # Dataset Check
    ddir = ycfg.get("dataset_dir")
    if ddir and not Path(str(ddir)).exists():
        errors.append(f"yolo_training.dataset_dir not found: {ddir}")

    # Class Names Check
    names = ycfg.get("class_names")
    if names is not None and not names:
        errors.append("yolo_training.class_names cannot be empty")

    # Augmentation Check
    aug = config.get("augmentation") or {}
    ops = aug.get("operations") or {}
    for op, params in ops.items():
        prob = params.get("probability")
        if prob is not None and not (0 <= prob <= 1):
            errors.append(f"augmentation.operations.{op}.probability must be 0-1")

    if errors:
        raise _ManualConfigError(errors)


def validate_config_schema(
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Validate configuration using Pydantic (if available) or manual checks.

    Args:
        config: The configuration dictionary.
        logger: Logger for warnings.
        strict: If True, raise exception on ANY validation error.
    """
    if BaseModel is not None:
        try:
            # Pydantic validation
            PipelineSchema.model_validate(config)
            return config
        except ValidationError as e:
            msg = f"Config validation failed:\n{e}"
            if strict:
                raise ValueError(msg) from e
            if logger:
                logger.warning(msg)
            return config
    else:
        # Fallback
        try:
            _manual_validate(config)
        except _ManualConfigError as e:
            if strict:
                raise ValueError(f"Config validation failed:\n{e}") from e
            if logger:
                logger.warning(f"Config validation warnings:\n{e}")
        return config
