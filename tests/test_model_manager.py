import pytest
from picture_tool.utils.model_manager import ModelManager


class DummyModel:
    def __init__(self, name):
        self.name = name


def test_model_manager_lru():
    """Test standard LRU eviction behavior."""
    manager = ModelManager(capacity=2)

    def loader(path):
        return DummyModel(path)

    # Load 1
    manager.get("model1", loader)
    assert "model1" in manager
    assert len(manager) == 1

    # Load 2
    manager.get("model2", loader)
    assert "model2" in manager
    assert len(manager) == 2

    # Access 1 (makes 1 most recently used)
    manager.get("model1", loader)

    # Load 3 (should evict 2, because 1 was just used)
    manager.get("model3", loader)

    assert "model1" in manager
    assert "model3" in manager
    assert "model2" not in manager
    assert len(manager) == 2


def test_model_manager_capacity_update():
    """Test that reducing capacity triggers eviction."""
    manager = ModelManager(capacity=5)

    def loader(path):
        return DummyModel(path)

    for i in range(5):
        manager.get(f"model{i}", loader)

    assert len(manager) == 5

    # Reduce capacity
    manager.set_capacity(2)
    assert len(manager) == 2
    # Should keep model3 and model4 (most recently inserted)
    assert "model3" in manager
    assert "model4" in manager
    assert "model0" not in manager


def test_model_manager_loader_error():
    """Test error handling when loader fails."""
    manager = ModelManager(capacity=1)

    def failing_loader(path):
        raise RuntimeError("Load failed")

    with pytest.raises(RuntimeError):
        manager.get("bad_path", failing_loader)

    assert len(manager) == 0


def test_model_manager_clear():
    manager = ModelManager(capacity=2)

    def loader(path):
        return DummyModel(path)

    manager.get("m1", loader)
    manager.clear()
    assert len(manager) == 0
