
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest
from picture_tool.train import yolo_trainer
from picture_tool.utils import onnx_validation

# Mock YOLO class
@pytest.fixture
def mock_yolo():
    with patch("picture_tool.train.yolo_trainer.YOLO") as mock:
        yield mock

@pytest.fixture
def mock_onnx_validation():
    # Patch where they are defined, since they are locally imported in yolo_trainer
    with patch("picture_tool.utils.onnx_validation.validate_onnx_structure") as mock_struct, \
         patch("picture_tool.utils.onnx_validation.validate_onnx_runtime") as mock_run:
        yield mock_struct, mock_run

def test_missing_onnx_fails(tmp_path):
    config = {"yolo_training": {"export_onnx": {"enabled": True}}}
    logger = logging.getLogger("test")
    # Clean sys.modules to ensure import check runs
    with patch.dict("sys.modules", {"onnx": None}):
        with pytest.raises(ImportError, match="requires package onnx"):
             yolo_trainer._maybe_export_onnx(config, tmp_path, logger)

def test_missing_onnxsim_fallback(mock_yolo, tmp_path, mock_onnx_validation):
    # Setup
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").touch()

    config = {
        "yolo_training": {
            "export_onnx": {"enabled": True, "simplify": True}
        }
    }
    logger = MagicMock()

    # Mock onnxsim missing
    with patch.dict("sys.modules", {"onnxsim": None}):
        yolo_trainer._maybe_export_onnx(config, run_dir, logger)

    # Verify fallback
    mock_yolo.return_value.export.assert_called_once()
    assert mock_yolo.return_value.export.call_args[1]["simplify"] is False
    # Verify warning logged
    logger.warning.assert_any_call(
        "ONNX export: simplify=True requested but 'onnxsim' not found. Falling back to simplify=False."
    )

def test_export_runtime_error_logged(mock_yolo, tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").touch()
    
    config = {"yolo_training": {"export_onnx": {"enabled": True}}}
    logger = MagicMock()
    
    mock_yolo.return_value.export.side_effect = RuntimeError("Torch failure")
    
    # Ensure onnx is present
    with patch.dict("sys.modules", {"onnx": MagicMock()}):
        # Should log error and return None (caught in outer try/except)
        result = yolo_trainer._maybe_export_onnx(config, run_dir, logger)
        assert result is None
    
    # Check log
    logger.error.assert_any_call("ONNX export runtime error: Torch failure")

def test_path_resolution_strategies(mock_yolo, tmp_path, mock_onnx_validation):
    run_dir = tmp_path / "run"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    best_pt = weights_dir / "best.pt"
    best_pt.touch()
    
    config = {"yolo_training": {"export_onnx": {"enabled": True}}}
    logger = MagicMock()

    # Ensure onnx is present for all steps
    with patch.dict("sys.modules", {"onnx": MagicMock()}):
        # Case A: Export returns valid string path
        expected_onnx_a = weights_dir / "returned.onnx"
        expected_onnx_a.touch()
        # Mock size > 0
        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 100
            mock_stat.return_value.st_mtime = 2000
            mock_stat.return_value.st_mode = 33188
            
            mock_yolo.return_value.export.return_value = str(expected_onnx_a)
            
            res = yolo_trainer._maybe_export_onnx(config, run_dir, logger)
            assert res == expected_onnx_a

        # Case B: Export returns None, but default path exists
        mock_yolo.return_value.export.return_value = None
        default_onnx = weights_dir / "best.onnx"
        default_onnx.touch()
        
        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 100
            mock_stat.return_value.st_mtime = 2000
            mock_stat.return_value.st_mode = 33188
            res = yolo_trainer._maybe_export_onnx(config, run_dir, logger)
            assert res == default_onnx

        with patch("pathlib.Path.glob") as mock_glob, \
             patch("pathlib.Path.stat", autospec=True) as mock_stat, \
             patch("pathlib.Path.exists", autospec=True) as mock_exists:
            
            # Setup returns
            mock_yolo.return_value.export.return_value = None
            
            fallback_onnx = weights_dir / "fallback.onnx"
            
            # Configure glob to return our fallback path
            mock_glob.return_value = [fallback_onnx]
            
            # Configure exists
            def exists_side_effect(self):
                # fallback.onnx exists
                if self.name == "fallback.onnx":
                    return True
                # weights dir exists
                if self.name == "weights":
                    return True
                # best.pt exists
                if self.name == "best.pt":
                     return True
                return False
            
            # Since Path.exists is called on instance, we need autospec=True?
            # Or just set side_effect. patch("pathlib.Path.exists") works on the class method.
            # But wait, Path.exists is dynamic?
            # It should work.
            mock_exists.side_effect = exists_side_effect

            # Configure stat for the fallback path ONLY (others won't be called if exists returns False)
            stat_obj = MagicMock()
            stat_obj.st_size = 100
            stat_obj.st_mtime = 1000
            stat_obj.st_mode = 33188
            
            def stat_side_effect(self):
                if self.name == "fallback.onnx":
                    return stat_obj
                # Should not be reached for non-existing files if exists() works
                raise FileNotFoundError(f"{self} not found")
            
            mock_stat.side_effect = stat_side_effect
            
            res = yolo_trainer._maybe_export_onnx(config, run_dir, logger)
            assert res == fallback_onnx

def test_runtime_validation_success():
    onnx_path = Path("test.onnx")
    logger = MagicMock()
    
    # Mock ort
    with patch("picture_tool.utils.onnx_validation._is_package_available", return_value=True), \
         patch("picture_tool.utils.onnx_validation.importlib.util.find_spec"), \
         patch.dict("sys.modules", {"onnxruntime": MagicMock(), "numpy": MagicMock()}):
         
        import onnxruntime as ort
        session_mock = MagicMock()
        ort.InferenceSession.return_value = session_mock
        
        # Setup inputs
        input_meta = MagicMock()
        input_meta.name = "images"
        input_meta.shape = [1, 3, 640, 640]
        input_meta.type = "tensor(float)"
        session_mock.get_inputs.return_value = [input_meta]
        
        # Mock outputs
        output_meta = MagicMock()
        output_meta.name = "output0"
        output_meta.shape = [1, 84, 8400]
        output_meta.type = "tensor(float)"
        session_mock.get_outputs.return_value = [output_meta]
        
        session_mock.run.return_value = ["output"]
        
        res = onnx_validation.validate_onnx_runtime(onnx_path, imgsz=640)
        assert res is True
        session_mock.run.assert_called_once()

def test_runtime_validation_failure():
    onnx_path = Path("test.onnx")
    
    with patch("picture_tool.utils.onnx_validation._is_package_available", return_value=True), \
         patch.dict("sys.modules", {"onnxruntime": MagicMock(), "numpy": MagicMock()}):
         
        import onnxruntime as ort
        session_mock = MagicMock()
        ort.InferenceSession.return_value = session_mock
        session_mock.run.side_effect = RuntimeError("Inference failed")
        
        # We need to ensure helper raises RuntimeError
        with pytest.raises(RuntimeError, match="Inference failed"):
             onnx_validation.validate_onnx_runtime(onnx_path)
