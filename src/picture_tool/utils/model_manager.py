from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Optional, Protocol, Callable

logger = logging.getLogger(__name__)


class ModelProtocol(Protocol):
    """Protocol for models managed by ModelManager."""
    
    def to(self, device: str) -> None: ...


class ModelManager:
    """
    Manages loaded models with an LRU (Least Recently Used) cache policy.
    Ensures that we don't run out of memory when switching between multiple tasks/models.
    """

    def __init__(self, capacity: int = 2) -> None:
        """
        Args:
            capacity: Maximum number of models to keep in memory.
        """
        self.capacity = capacity
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.logger = logger

    def get(self, model_path: str, loader_fn: Optional[Callable[[str], Any]] = None) -> Any:
        """
        Retrieve a model from cache or load it if not present.

        Args:
            model_path: Absolute path or identifier for the model.
            loader_fn: Function to load the model if it's not in cache. 
                       Must accept (model_path) as argument.

        Returns:
            The loaded model object.
        """
        key = str(model_path)

        if key in self.cache:
            self.logger.info(f"Model cache hit: {key}")
            self.cache.move_to_end(key)  # Mark as recently used
            return self.cache[key]

        if loader_fn is None:
            raise ValueError(f"Model {key} not in cache and no loader_fn provided.")

        self.logger.info(f"Model cache miss: {key}. Loading...")
        try:
            model = loader_fn(key)
        except Exception as e:
            self.logger.error(f"Failed to load model {key}: {e}")
            raise e

        # Evict if full
        if len(self.cache) >= self.capacity:
            self._evict()

        self.cache[key] = model
        self.logger.info(f"Model loaded and cached: {key}")
        return model

    def _evict(self) -> None:
        """Removes the least recently used model from cache."""
        oldest_key, _ = self.cache.popitem(last=False)
        self.logger.info(f"Evicted model from cache: {oldest_key}")
        # Explicit cleanup suggestions (optional, depending on backend)
        # import gc; gc.collect() 
        # import torch; torch.cuda.empty_cache()

    def clear(self) -> None:
        """Clears the entire cache."""
        self.cache.clear()
        self.logger.info("Model cache cleared.")

    def set_capacity(self, capacity: int) -> None:
        """Updates the capacity and evicts excess items if necessary."""
        self.capacity = capacity
        while len(self.cache) > self.capacity:
            self._evict()

    def __contains__(self, key: str) -> bool:
        return str(key) in self.cache

    def __len__(self) -> int:
        return len(self.cache)

# Global instance for easy access
_shared_manager = ModelManager()

def get_shared_model_manager() -> ModelManager:
    return _shared_manager
