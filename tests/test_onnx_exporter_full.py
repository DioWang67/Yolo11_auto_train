"""
Comprehensive tests for utils/onnx_exporter.py module.
Coverage target: 0% → 70%+

Note: Using monkeypatch instead of mocker for compatibility.
"""

import logging
from unittest.mock import MagicMock
import pytest

from picture_tool.utils.onnx_exporter import OnnxExporter


class TestOnnxExporterExport:
    """Test OnnxExporter.export method."""

    def test_skips_when_export_disabled(self, tmp_path):
        """Should return None when export_onnx.enabled=False."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        config = {"yolo_training": {"export_onnx": {"enabled": False}}}

        logger = logging.getLogger("test")
        result = OnnxExporter.export(config, run_dir, logger)

        assert result is None

    def test_skips_when_export_config_missing(self, tmp_path):
        """Should return None when export_onnx not configured."""
        config = {"yolo_training": {}}
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        logger = logging.getLogger("test")
        result = OnnxExporter.export(config, run_dir, logger)

        assert result is None

    def test_skips_when_yolo_training_invalid(self, tmp_path):
        """Should return None when yolo_training is not a dict."""
        config = {"yolo_training": "invalid"}
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        logger = logging.getLogger("test")
        result = OnnxExporter.export(config, run_dir, logger)

        assert result is None

    def test_skips_when_weights_not_found(self, tmp_path, monkeypatch):
        """Should return None when weights file doesn't exist."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "weights").mkdir()

        config = {
            "yolo_training": {
                "export_onnx": {"enabled": True, "weights_name": "best.pt"}
            }
        }

        # Mock onnx package existence
        monkeypatch.setattr("importlib.util.find_spec", lambda x: MagicMock())

        logger = logging.getLogger("test")
        result = OnnxExporter.export(config, run_dir, logger)

        assert result is None

    def test_raises_import_error_when_onnx_missing(self, tmp_path, monkeypatch):
        """Should raise ImportError when onnx package not found."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        weights_dir = run_dir / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_text("weights")

        config = {"yolo_training": {"export_onnx": {"enabled": True}}}

        # Mock onnx not found
        monkeypatch.setattr("importlib.util.find_spec", lambda x: None)

        logger = logging.getLogger("test")

        with pytest.raises(ImportError, match="onnx"):
            OnnxExporter.export(config, run_dir, logger)

    def test_skips_when_ultralytics_not_available(self, tmp_path, monkeypatch):
        """Should return None when YOLO is None (ultralytics not installed)."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        weights_dir = run_dir / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_text("weights")

        config = {"yolo_training": {"export_onnx": {"enabled": True}}}

        # Mock onnx available but YOLO=None
        monkeypatch.setattr("importlib.util.find_spec", lambda x: MagicMock())
        monkeypatch.setattr("picture_tool.utils.onnx_exporter.YOLO", None)

        logger = logging.getLogger("test")
        result = OnnxExporter.export(config, run_dir, logger)

        assert result is None

    def test_resolves_onnx_path_from_weights_path(self, tmp_path, monkeypatch):
        """Should resolve ONNX path from weights path when export succeeds."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        weights_dir = run_dir / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_text("weights")
        onnx_file = weights_dir / "best.onnx"
        onnx_file.write_text("onnx_content" * 100)  # Make sure it's not empty

        config = {"yolo_training": {"export_onnx": {"enabled": True}}}

        # Mock dependencies
        monkeypatch.setattr("importlib.util.find_spec", lambda x: MagicMock())

        mock_model = MagicMock()
        mock_model.export.return_value = str(onnx_file)
        mock_yolo_class = MagicMock(return_value=mock_model)
        monkeypatch.setattr("picture_tool.utils.onnx_exporter.YOLO", mock_yolo_class)

        # Mock validation functions
        monkeypatch.setattr(
            "picture_tool.utils.onnx_validation.validate_onnx_structure", lambda x: None
        )
        monkeypatch.setattr(
            "picture_tool.utils.onnx_validation.validate_onnx_runtime",
            lambda x, **kwargs: None,
        )

        logger = logging.getLogger("test")
        result = OnnxExporter.export(config, run_dir, logger)

        assert result == onnx_file.resolve()

    def test_uses_default_imgsz_when_not_configured(self, tmp_path, monkeypatch):
        """Should default to [640, 640] when imgsz not in config."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        weights_dir = run_dir / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_text("weights")
        onnx_file = weights_dir / "best.onnx"
        onnx_file.write_text("onnx" * 100)

        config = {"yolo_training": {"export_onnx": {"enabled": True}}}

        # Setup mocks
        monkeypatch.setattr("importlib.util.find_spec", lambda x: MagicMock())

        mock_model = MagicMock()
        mock_model.export.return_value = str(onnx_file)
        mock_yolo_class = MagicMock(return_value=mock_model)
        monkeypatch.setattr("picture_tool.utils.onnx_exporter.YOLO", mock_yolo_class)
        monkeypatch.setattr(
            "picture_tool.utils.onnx_validation.validate_onnx_structure", lambda x: None
        )
        monkeypatch.setattr(
            "picture_tool.utils.onnx_validation.validate_onnx_runtime",
            lambda x, **kwargs: None,
        )

        logger = logging.getLogger("test")
        result = OnnxExporter.export(config, run_dir, logger)

        # Verify export was called
        assert mock_model.export.called
        call_kwargs = mock_model.export.call_args.kwargs
        # Default should be scalar 640 for [640, 640]
        assert call_kwargs["imgsz"] == 640
        assert result is not None

    def test_handles_export_errors_gracefully(self, tmp_path, monkeypatch):
        """Should return None when export raises exception."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        weights_dir = run_dir / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_text("weights")

        config = {"yolo_training": {"export_onnx": {"enabled": True}}}

        # Mock to raise error
        monkeypatch.setattr("importlib.util.find_spec", lambda x: MagicMock())

        mock_model = MagicMock()
        mock_model.export.side_effect = RuntimeError("Export failed")
        mock_yolo_class = MagicMock(return_value=mock_model)
        monkeypatch.setattr("picture_tool.utils.onnx_exporter.YOLO", mock_yolo_class)

        logger = logging.getLogger("test")
        result = OnnxExporter.export(config, run_dir, logger)

        assert result is None

    def test_returns_none_when_onnx_file_not_found_after_export(
        self, tmp_path, monkeypatch
    ):
        """Should return None when ONNX file cannot be located after export."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        weights_dir = run_dir / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_text("weights")
        # No ONNX file created

        config = {"yolo_training": {"export_onnx": {"enabled": True}}}

        # Mock successful export but file doesn't exist
        monkeypatch.setattr("importlib.util.find_spec", lambda x: MagicMock())

        mock_model = MagicMock()
        mock_model.export.return_value = None  # No path returned
        mock_yolo_class = MagicMock(return_value=mock_model)
        monkeypatch.setattr("picture_tool.utils.onnx_exporter.YOLO", mock_yolo_class)

        logger = logging.getLogger("test")
        result = OnnxExporter.export(config, run_dir, logger)

        assert result is None

    def test_validates_onnx_after_export(self, tmp_path, monkeypatch):
        """Should call validation functions on exported ONNX."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        weights_dir = run_dir / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_text("weights")
        onnx_file = weights_dir / "best.onnx"
        onnx_file.write_text("onnx" * 100)

        config = {
            "yolo_training": {
                "imgsz": 800,
                "device": "cpu",
                "export_onnx": {"enabled": True},
            }
        }

        # Track validation calls
        structure_called = []
        runtime_called = []

        def mock_structure(path):
            structure_called.append(path)

        def mock_runtime(path, **kwargs):
            runtime_called.append((path, kwargs))

        # Setup mocks
        monkeypatch.setattr("importlib.util.find_spec", lambda x: MagicMock())

        mock_model = MagicMock()
        mock_model.export.return_value = str(onnx_file)
        mock_yolo_class = MagicMock(return_value=mock_model)
        monkeypatch.setattr("picture_tool.utils.onnx_exporter.YOLO", mock_yolo_class)
        monkeypatch.setattr(
            "picture_tool.utils.onnx_validation.validate_onnx_structure", mock_structure
        )
        monkeypatch.setattr(
            "picture_tool.utils.onnx_validation.validate_onnx_runtime", mock_runtime
        )

        logger = logging.getLogger("test")
        result = OnnxExporter.export(config, run_dir, logger)

        assert len(structure_called) == 1
        assert len(runtime_called) == 1
        assert result == onnx_file.resolve()

    def test_returns_none_when_validation_fails(self, tmp_path, monkeypatch):
        """Should return None when validation raises exception."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        weights_dir = run_dir / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_text("weights")
        onnx_file = weights_dir / "best.onnx"
        onnx_file.write_text("onnx" * 100)

        config = {"yolo_training": {"export_onnx": {"enabled": True}}}

        # Setup mocks
        monkeypatch.setattr("importlib.util.find_spec", lambda x: MagicMock())

        mock_model = MagicMock()
        mock_model.export.return_value = str(onnx_file)
        mock_yolo_class = MagicMock(return_value=mock_model)
        monkeypatch.setattr("picture_tool.utils.onnx_exporter.YOLO", mock_yolo_class)

        # Validation fails
        def fail_validation(path):
            raise RuntimeError("Validation failed")

        monkeypatch.setattr(
            "picture_tool.utils.onnx_validation.validate_onnx_structure", fail_validation
        )

        logger = logging.getLogger("test")
        result = OnnxExporter.export(config, run_dir, logger)

        assert result is None

    def test_disables_simplify_when_onnxsim_missing(self, tmp_path, monkeypatch):
        """Should fall back to simplify=False when onnxsim not available."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        weights_dir = run_dir / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_text("weights")
        onnx_file = weights_dir / "best.onnx"
        onnx_file.write_text("onnx" * 100)

        config = {"yolo_training": {"export_onnx": {"enabled": True, "simplify": True}}}

        # Mock: onnx available, onnxsim not
        def custom_find_spec(name):
            if name == "onnxsim":
                return None
            return MagicMock()

        monkeypatch.setattr("importlib.util.find_spec", custom_find_spec)

        mock_model = MagicMock()
        mock_model.export.return_value = str(onnx_file)
        mock_yolo_class = MagicMock(return_value=mock_model)
        monkeypatch.setattr("picture_tool.utils.onnx_exporter.YOLO", mock_yolo_class)
        monkeypatch.setattr(
            "picture_tool.utils.onnx_validation.validate_onnx_structure", lambda x: None
        )
        monkeypatch.setattr(
            "picture_tool.utils.onnx_validation.validate_onnx_runtime",
            lambda x, **kwargs: None,
        )

        logger = logging.getLogger("test")
        result = OnnxExporter.export(config, run_dir, logger)

        # Should have disabled simplify
        call_kwargs = mock_model.export.call_args.kwargs
        assert call_kwargs["simplify"] is False
        assert result is not None
