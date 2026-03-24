"""Tests for picture_tool.config_loader module."""

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from picture_tool.config_loader import (
    load_config,
    load_config_if_updated,
)


@pytest.fixture
def tmp_config(tmp_path):
    """Create a temporary YAML config file."""
    cfg = {"pipeline": {"log_file": "test.log"}, "yolo_training": {"epochs": 10}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path, cfg


class TestLoadConfig:
    def test_load_existing_config(self, tmp_config):
        path, expected = tmp_config
        result = load_config(path)
        assert result == expected

    def test_load_nonexistent_falls_back_to_packaged(self):
        # Should not raise — falls back to packaged default
        result = load_config("nonexistent_config_xyz.yaml")
        assert isinstance(result, dict)

    def test_load_empty_yaml_returns_empty_dict(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        result = load_config(path)
        assert result == {}


class TestLoadConfigIfUpdated:
    def test_returns_same_config_when_no_change(self, tmp_config):
        path, cfg = tmp_config
        logger = logging.getLogger("test")

        # Reset mtime tracking
        if hasattr(load_config_if_updated, "last_mtime"):
            delattr(load_config_if_updated, "last_mtime")

        # First call: initialises mtime
        result = load_config_if_updated(path, cfg, logger)
        assert result == cfg

        # Second call: no change
        result = load_config_if_updated(path, cfg, logger)
        assert result == cfg

    def test_returns_new_config_when_file_changed(self, tmp_config, tmp_path):
        path, cfg = tmp_config
        logger = logging.getLogger("test")

        if hasattr(load_config_if_updated, "last_mtime"):
            delattr(load_config_if_updated, "last_mtime")

        load_config_if_updated(path, cfg, logger)

        # Modify file
        import time
        time.sleep(0.05)
        new_cfg = {"pipeline": {"log_file": "new.log"}, "yolo_training": {"epochs": 50}}
        path.write_text(yaml.dump(new_cfg), encoding="utf-8")

        # Force a new mtime by updating the file's timestamp
        import os
        os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 1))

        result = load_config_if_updated(path, cfg, logger)
        assert result["yolo_training"]["epochs"] == 50

    def test_returns_config_when_file_missing(self):
        logger = logging.getLogger("test")
        cfg = {"key": "value"}

        if hasattr(load_config_if_updated, "last_mtime"):
            delattr(load_config_if_updated, "last_mtime")

        result = load_config_if_updated("nonexistent.yaml", cfg, logger)
        assert result == cfg
