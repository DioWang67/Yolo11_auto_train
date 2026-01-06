"""
Input validation utilities for picture_tool.

Provides reusable validation functions for common input types.
"""
from pathlib import Path
from typing import Any, Dict, List, Iterable

from picture_tool.exceptions import ValidationError


def validate_required_keys(config: Dict[str, Any], required_keys: Iterable[str], context: str = "config") -> None:
    """
    Validate that all required keys exist in config.
    
    Args:
        config: Configuration dictionary
        required_keys: Keys that must exist
        context: Context name for error message
        
    Raises:
        ValidationError: If any required key is missing
    """
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValidationError(
            f"{context} is missing required keys: {missing}"
        )


def validate_path_exists(path: Path, must_be_file: bool = False, must_be_dir: bool = False) -> None:
    """
    Validate that a path exists and optionally check its type.
    
    Args:
        path: Path to validate
        must_be_file: If True, path must be a file
        must_be_dir: If True, path must be a directory
        
    Raises:
        ValidationError: If path doesn't exist or wrong type
    """
    if not path.exists():
        raise ValidationError(f"Path does not exist: {path}")
    
    if must_be_file and not path.is_file():
        raise ValidationError(f"Path must be a file: {path}")
    
    if must_be_dir and not path.is_dir():
        raise ValidationError(f"Path must be a directory: {path}")


def validate_positive_int(value: Any, name: str) -> int:
    """
    Validate that value is a positive integer.
    
    Args:
        value: Value to validate
        name: Name for error message
        
    Returns:
        Validated integer value
        
    Raises:
        ValidationError: If value is not a positive integer
    """
    try:
        int_value = int(value)
    except (ValueError, TypeError):
        raise ValidationError(
            f"{name} must be an integer, got {type(value).__name__}"
        )
    
    if int_value <= 0:
        raise ValidationError(
            f"{name} must be positive, got {int_value}"
        )
    
    return int_value


def validate_ratio(value: Any, name: str) -> float:
    """
    Validate that value is a ratio between 0 and 1.
    
    Args:
        value: Value to validate
        name: Name for error message
        
    Returns:
        Validated float value
        
    Raises:
        ValidationError: If value is not a valid ratio
    """
    try:
        float_value = float(value)
    except (ValueError, TypeError):
        raise ValidationError(
            f"{name} must be a number, got {type(value).__name__}"
        )
    
    if not 0 <= float_value <= 1:
        raise ValidationError(
            f"{name} must be between 0 and 1, got {float_value}"
        )
    
    return float_value


def validate_class_names(class_names: Any) -> List[str]:
    """
    Validate class names list.
    
    Args:
        class_names: Value to validate
        
    Returns:
        Validated list of class names
        
    Raises:
        ValidationError: If class_names is invalid
    """
    if not class_names:
        raise ValidationError("class_names cannot be empty")
    
    if not isinstance(class_names, (list, tuple)):
        raise ValidationError(
            f"class_names must be a list, got {type(class_names).__name__}"
        )
    
    names = list(class_names)
    if not all(isinstance(n, str) for n in names):
        raise ValidationError("All class names must be strings")
    
    if len(names) != len(set(names)):
        raise ValidationError("class_names contains duplicates")
    
    return names
