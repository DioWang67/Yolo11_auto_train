from typing import Dict, Optional
import importlib
import pkgutil
from types import ModuleType

from picture_tool.color.strategies.base import ColorStrategy

class ColorStrategyRegistry:
    """Registry to manage and retrieve color-specific strategies dynamically."""
    
    _strategies: Dict[str, ColorStrategy] = {}
    _fallback: Optional[ColorStrategy] = None
    _initialized: bool = False

    @classmethod
    def register(cls, *names: str):
        """Decorator to register a ColorStrategy."""
        def decorator(strategy_class):
            instance = strategy_class()
            for name in names:
                cls._strategies[name.lower()] = instance
            return strategy_class
        return decorator

    @classmethod
    def register_fallback(cls):
        """Decorator to register the fallback GenericStrategy."""
        def decorator(strategy_class):
            cls._fallback = strategy_class()
            return strategy_class
        return decorator

    @classmethod
    def initialize(cls):
        """Dynamically load all strategies in the package to trigger decorators."""
        if cls._initialized:
            return
            
        import picture_tool.color.strategies as strategies_pkg
        
        for _, module_name, _ in pkgutil.iter_modules(strategies_pkg.__path__):
            if module_name not in ('base', 'registry'):
                before = (frozenset(cls._strategies), cls._fallback)
                module = importlib.import_module(
                    f"picture_tool.color.strategies.{module_name}"
                )
                after = (frozenset(cls._strategies), cls._fallback)
                # A reset registry plus an already-cached module otherwise
                # stays silently incomplete because decorators only ran on the
                # first import. Reload only when importing produced no
                # registration; normal startup still imports every module once.
                should_restore_cached_registration = (
                    before == after
                    and isinstance(module, ModuleType)
                    and (module_name != "generic" or cls._fallback is None)
                )
                if should_restore_cached_registration:
                    importlib.reload(module)
                
        cls._initialized = True

    @classmethod
    def get_strategy(cls, color_name: str) -> ColorStrategy:
        """Get the strategy for a color name, falling back to Generic if not found."""
        cls.initialize()
        
        color_lower = color_name.lower()
        if color_lower in cls._strategies:
            return cls._strategies[color_lower]
            
        # Fuzzy/Keyword match
        for key, strategy in cls._strategies.items():
            if key in color_lower:
                return strategy
                
        if cls._fallback is None:
            raise RuntimeError("Fallback GenericStrategy implicitly missing. Was generic.py not loaded?")
            
        return cls._fallback

    @classmethod
    def all_strategies(cls) -> Dict[str, ColorStrategy]:
        """Returns all registered strategies."""
        cls.initialize()
        return cls._strategies
