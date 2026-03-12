from typing import Dict, Type
from picture_tool.color.strategies.base import ColorStrategy
from picture_tool.color.strategies.generic import GenericStrategy
from picture_tool.color.strategies.black import BlackStrategy
from picture_tool.color.strategies.yellow import YellowStrategy
from picture_tool.color.strategies.red_orange import RedOrangeStrategy
from picture_tool.color.strategies.green import GreenStrategy

class ColorStrategyRegistry:
    """Registry to manage and retrieve color-specific strategies."""
    
    _strategies: Dict[str, ColorStrategy] = {}
    _fallback = GenericStrategy()

    @classmethod
    def initialize(cls):
        """Initialize the default mapping of color names to strategy instances."""
        # Instances are stateless or share-able across lookups
        black = BlackStrategy()
        yellow = YellowStrategy()
        red_orange = RedOrangeStrategy()
        green = GreenStrategy()
        
        cls._strategies = {
            "Black": black,
            "Yellow": yellow,
            "Red": red_orange,
            "Orange": red_orange,
            "Green": green
        }

    @classmethod
    def get_strategy(cls, color_name: str) -> ColorStrategy:
        """Get the strategy for a color name, falling back to Generic if not found."""
        if not cls._strategies:
            cls.initialize()
        
        # 1. Exact match
        if color_name in cls._strategies:
            return cls._strategies[color_name]
            
        # 2. Fuzzy/Keyword match (e.g., "ProductA_Black" -> BlackStrategy)
        name_lower = color_name.lower()
        if "black" in name_lower:
            return cls._strategies["Black"]
        if "yellow" in name_lower:
            return cls._strategies["Yellow"]
        if "red" in name_lower or "orange" in name_lower:
            return cls._strategies["Red"]
        if "green" in name_lower:
            return cls._strategies["Green"]
            
        # 3. Last fallback
        return cls._fallback

    @classmethod
    def all_strategies(cls) -> Dict[str, ColorStrategy]:
        """Returns all registered strategies."""
        if not cls._strategies:
            cls.initialize()
        return cls._strategies
