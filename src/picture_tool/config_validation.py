from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # Optional: use pydantic when available
    from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
except Exception:  # pragma: no cover
    BaseModel = None  # type: ignore


class _ManualConfigError(Exception):
    """Lightweight validation error when pydantic is unavailable."""

    def __init__(self, messages: List[str]) -> None:
        super().__init__("\n".join(messages))
        self.messages = messages


def _manual_validate(config: Dict[str, Any]) -> None:
    errors: List[str] = []
    yolo_cfg = config.get("yolo_training") or {}
    class_names = yolo_cfg.get("class_names")
    if class_names is not None and len(class_names) == 0:
        errors.append("yolo_training.class_names must not be empty when provided")
    dataset_dir = yolo_cfg.get("dataset_dir")
    if dataset_dir:
        path = Path(str(dataset_dir))
        if not path.exists():
            errors.append(f"yolo_training.dataset_dir does not exist: {path}")
    if errors:
        raise _ManualConfigError(errors)


def validate_config_schema(
    config: Dict[str, Any], logger: Optional[logging.Logger] = None, *, strict: bool = False
) -> Dict[str, Any]:
    """
    Validate the config structure.

    - Uses pydantic if installed; otherwise falls back to lightweight checks.
    - strict=True raises on validation errors; otherwise logs warnings and continues.
    """
    if BaseModel is None:
        try:
            _manual_validate(config)
        except _ManualConfigError as exc:
            if strict:
                raise
            if logger:
                logger.warning("Config validation warnings: " + "; ".join(exc.messages))
        return config

    class YoloTrainingSchema(BaseModel):
        model_config = ConfigDict(extra="allow")
        dataset_dir: Optional[Path] = None
        class_names: Optional[List[str]] = None

        @field_validator("class_names")
        @classmethod
        def _non_empty(cls, value: Optional[List[str]]) -> Optional[List[str]]:
            if value is not None and len(value) == 0:
                raise ValueError("class_names must not be empty when provided")
            return value

        @field_validator("dataset_dir")
        @classmethod
        def _dataset_exists(cls, value: Optional[Path]) -> Optional[Path]:
            if value and not value.exists():
                raise ValueError(f"dataset_dir does not exist: {value}")
            return value

    class PipelineSchema(BaseModel):
        model_config = ConfigDict(extra="allow")
        yolo_training: Optional[YoloTrainingSchema] = None

    try:
        PipelineSchema.model_validate(config)
    except ValidationError as exc:
        if strict:
            raise
        if logger:
            logger.warning(f"Config validation warnings:\n{exc}")
    return config
