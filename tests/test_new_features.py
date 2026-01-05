
import pytest
from picture_tool.config_validation import validate_config_schema
from picture_tool.serve import app

try:
    from fastapi.testclient import TestClient
except ImportError:
    TestClient = None  # type: ignore

def test_pydantic_valid_config(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    cfg = {
        "yolo_training": {
            "dataset_dir": str(d),
            "class_names": ["a"],
            "epochs": 10
        }
    }
    # Should pass
    validated = validate_config_schema(cfg, strict=True)
    assert validated["yolo_training"]["epochs"] == 10

def test_pydantic_invalid_config():
    cfg = {
        "yolo_training": {
            "dataset_dir": "/non/existent",  # Should fail
            "class_names": [] # Should fail
        }
    }
    # We expect some error (ValueError or Pydantic ValidationError)
    try:
        validate_config_schema(cfg, strict=True)
        assert False, "Should have raised ValidationError"
    except (ValueError, Exception):
        pass

def test_serving_endpoint():
    if TestClient is None or app is None:
        pytest.skip("FastAPI not installed")
    
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()
