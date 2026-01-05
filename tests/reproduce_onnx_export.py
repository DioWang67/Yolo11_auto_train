
import logging
from unittest.mock import patch
import pytest
from picture_tool.train import yolo_trainer

# Mock YOLO class
@pytest.fixture
def mock_yolo():
    with patch("picture_tool.train.yolo_trainer.YOLO") as mock:
        yield mock

def test_export_onnx_success(mock_yolo, tmp_path):
    # Setup
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    weights_dir = run_dir / "weights"
    weights_dir.mkdir()
    (weights_dir / "best.pt").touch()

    config = {
        "yolo_training": {
            "imgsz": 640,
            "device": "cpu",
            "export_onnx": {
                "enabled": True,
                "weights_name": "best.pt",
                "simplify": True,
                "opset": 12
            }
        }
    }
    logger = logging.getLogger("test")

    # Mock export return value
    expected_onnx = weights_dir / "best.onnx"
    expected_onnx.touch() # Simulate file creation
    
    instance = mock_yolo.return_value
    instance.export.return_value = str(expected_onnx)

    # Execution
    result = yolo_trainer._maybe_export_onnx(config, run_dir, logger)

    # Assertion
    assert result == expected_onnx.resolve()
    instance.export.assert_called_once()
    call_args = instance.export.call_args[1]
    assert call_args["format"] == "onnx"
    assert call_args["simplify"] is True
    assert call_args["opset"] == 12

def test_export_onnx_defaults(mock_yolo, tmp_path):
    # Setup
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    weights_dir = run_dir / "weights"
    weights_dir.mkdir()
    (weights_dir / "best.pt").touch()

    config = {
        "yolo_training": {
            "imgsz": 640,
            "export_onnx": {
                "enabled": True
            }
        }
    }
    logger = logging.getLogger("test")

    # Mock export return value
    expected_onnx = weights_dir / "best.onnx"
    expected_onnx.touch() 
    
    instance = mock_yolo.return_value
    instance.export.return_value = str(expected_onnx)

    # Execution
    result = yolo_trainer._maybe_export_onnx(config, run_dir, logger)

    # Assertion
    assert result == expected_onnx.resolve()
    call_args = instance.export.call_args[1]
    assert call_args["dynamic"] is False
    assert call_args["simplify"] is False
    assert "opset" not in call_args

def test_export_onnx_failure(mock_yolo, tmp_path):
    # Setup
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    weights_dir = run_dir / "weights"
    weights_dir.mkdir()
    (weights_dir / "best.pt").touch()

    config = {
        "yolo_training": {
            "export_onnx": {"enabled": True}
        }
    }
    logger = logging.getLogger("test")

    instance = mock_yolo.return_value
    instance.export.side_effect = Exception("Export Error")

    # Execution
    result = yolo_trainer._maybe_export_onnx(config, run_dir, logger)

    # Assertion
    assert result is None
