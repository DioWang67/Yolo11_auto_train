from typing import Any, List, Sequence, Mapping

def normalize_imgsz(value: Any) -> Any:
    """Normalize imgsz representations to [width, height]."""
    if value in (None, "", []):
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        ints: List[int] = []
        for item in value:
            if item in (None, ""):
                continue
            try:
                ints.append(int(float(item)))
            except (TypeError, ValueError):
                continue
        if not ints:
            return None
        if len(ints) == 1:
            ints *= 2
        elif len(ints) >= 2:
            ints = ints[:2]
            if len(ints) == 1:
                ints.append(ints[0])
        if len(ints) < 2:
            ints.append(ints[0])
        return [ints[0], ints[1]]
    try:
        val = int(float(value))
    except (TypeError, ValueError):
        value = None # Fallback
    if isinstance(val, int):
        return [val, val]
    return None

def normalize_name_sequence(value: Any) -> List[str]:
    """Convert various name representations (list or dict) to a string list."""
    if isinstance(value, Mapping):
        items = []
        for key, val in value.items():
            try:
                sort_key = int(key)
            except (TypeError, ValueError):
                sort_key = key
            items.append((sort_key, str(val)))
        try:
            items.sort(key=lambda item: item[0])
        except TypeError:
            items.sort(key=lambda item: str(item[0]))
        return [name for _, name in items if name]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in value if item not in (None, "")]
    if value not in (None, ""):
        return [str(value)]
    return []
