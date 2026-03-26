"""Tests for TrainingMetricsParser."""

import pytest

from picture_tool.gui.training_metrics import TrainingMetricsParser


class TestTrainingMetricsParser:
    def test_parse_epoch_line_basic(self):
        line = "      5/100      3.66G    0.6543    0.5432    0.1234       320       640"
        result = TrainingMetricsParser.parse_epoch_line(line)
        assert result is not None
        assert result["epoch"] == 5
        assert result["total_epochs"] == 100
        assert abs(result["box_loss"] - 0.6543) < 1e-5
        assert abs(result["cls_loss"] - 0.5432) < 1e-5
        assert abs(result["dfl_loss"] - 0.1234) < 1e-5

    def test_parse_epoch_line_with_log_wrapper(self):
        line = "2026-03-24 10:30:00 - INFO -       10/50      2.1G    0.4321    0.3210    0.0987       128       640"
        result = TrainingMetricsParser.parse_epoch_line(line)
        assert result is not None
        assert result["epoch"] == 10
        assert result["total_epochs"] == 50

    def test_parse_non_epoch_line_returns_none(self):
        assert TrainingMetricsParser.parse_epoch_line("Starting training...") is None
        assert TrainingMetricsParser.parse_epoch_line("") is None
        assert TrainingMetricsParser.parse_epoch_line("Epoch    GPU_mem   box_loss") is None

    def test_parse_epoch_zero_gpu_mem(self):
        line = "      1/200      0    0.9999    0.8888    0.7777       64       320"
        result = TrainingMetricsParser.parse_epoch_line(line)
        assert result is not None
        assert result["epoch"] == 1
        assert result["total_epochs"] == 200

    def test_parse_final_epoch(self):
        line = "    100/100      4.2G    0.0123    0.0456    0.0078       256       640"
        result = TrainingMetricsParser.parse_epoch_line(line)
        assert result is not None
        assert result["epoch"] == 100
        assert result["total_epochs"] == 100
