"""Utilities for normalizing configuration values."""
from typing import Any, List

def normalize_imgsz(value: Any) -> List[int] | None:
    """
    Normalize imgsz to [width, height].
    
    Args:
        value: Can be int, str, list, tuple
        
    Returns:
        [width, height] or None if invalid
    """
    if value is None:
        return None
    
    if isinstance(value, (list, tuple)):
        ints = []
        for v in value:
            try:
                ints.append(int(float(v)))
            except (TypeError, ValueError):
                pass
        if len(ints) >= 2:
            return [ints[0], ints[1]]
        elif len(ints) == 1:
            return [ints[0], ints[0]]
        else:
            return None
    
    # Single value (int or string)
    try:
        val = int(float(value))
        return [val, val]
    except (TypeError, ValueError):
        return None

def normalize_name_sequence(value: Any) -> List[str]:
    """
    Normalize class names to a list of strings.
    
    Args:
        value: Can be list, tuple, dict, or single value
        
    Returns:
        List of string names
    """
    if value is None:
        return []
    
    # Dict mapping {id: name}
    if isinstance(value, dict):
        sorted_items = sorted(value.items())
        return [str(v) for k, v in sorted_items if v is not None]
    
    # List or tuple
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None]
    
    # Single value - return empty list (invalid type)
    return []
