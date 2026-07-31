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
    cfg = {"yolo_training": {"dataset_dir": str(d), "class_names": ["a"], "epochs": 10}}
    # Should pass
    validated = validate_config_schema(cfg, strict=True)
    assert validated["yolo_training"]["epochs"] == 10


def test_pydantic_invalid_config():
    cfg = {
        "yolo_training": {
            "dataset_dir": "/non/existent",  # Should fail
            "class_names": [],  # Should fail
        }
    }
    with pytest.raises(ValueError, match="Config validation failed"):
        validate_config_schema(cfg, strict=True)


@pytest.mark.parametrize(
    "position_validation",
    [
        {"calibration_source": "challenger_predictions"},
        {"calibration_min_samples": 0},
        {"gate": {"max_ok_false_reject_rate": 1.1}},
        {"gate": {"min_ng_recall": -0.1}},
        {"mode": "rectangle"},
        {"tolerance_unit": "millimeter"},
        {"tolerance": -1},
        {"conf": 1.1},
    ],
)
def test_position_gate_config_rejects_unsafe_values(position_validation):
    cfg = {
        "yolo_training": {
            "position_validation": position_validation,
        }
    }

    with pytest.raises(ValueError, match="Config validation failed"):
        validate_config_schema(cfg, strict=True)


def test_serving_endpoint():
    if TestClient is None or app is None:
        pytest.skip("FastAPI not installed")

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()
