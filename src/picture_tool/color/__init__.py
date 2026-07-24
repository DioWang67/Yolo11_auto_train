from importlib import import_module
from types import ModuleType

__all__ = ["color_inspection", "color_verifier"]


def __getattr__(name: str) -> ModuleType:
    """Load OpenCV-heavy color tools only when a caller requests them."""
    if name not in __all__:
        raise AttributeError(name)
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module
