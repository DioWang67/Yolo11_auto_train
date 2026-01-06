import io
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
from fastapi.testclient import TestClient

# Import the app object from serve.py
# We handle the case where imports might strictly fail if dependencies missing,
# although for this test file to run, dev deps should be installed.
from picture_tool import serve


@pytest.fixture
def client():
    # Ensure app is created
    if serve.app is None:
        pytest.skip("FastAPI app not initialized (missing dependencies?)")
    return TestClient(serve.app)


@pytest.fixture
def mock_yolo():
    with patch("picture_tool.serve.YOLO") as mock:
        # Create a mock model instance
        mock_instance = MagicMock()
        mock.return_value = mock_instance

        import numpy as np

        # Mock prediction results
        # Result structure: list of Results, each has .boxes
        # Box has .cls, .conf, .xyxy
        mock_box = MagicMock()
        mock_box.cls = 0
        mock_box.conf = 0.95
        # In real usage xyxy is a tensor/array. We use numpy for .tolist() support
        mock_box.xyxy = np.array([[10, 10, 50, 50]])

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]

        mock_instance.return_value = [mock_result]

        yield mock


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["model_loaded"] is False  # Initially False


def test_load_model_not_found(client):
    response = client.post("/load_model", params={"path": "non_existent.pt"})
    assert response.status_code == 404


def test_load_model_success(client, mock_yolo, tmp_path):
    # Create a dummy model file
    msg = b"dummy model"
    p = tmp_path / "model.pt"
    p.write_bytes(msg)

    response = client.post("/load_model", params={"path": str(p)})
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # Verify health check is now True (needs patching MODEL_INSTANCE global if not covered by mock_yolo return)
    # Since load_model sets global MODEL_INSTANCE, and we mocked YOLO class,
    # serve.py's load_model does `MODEL_INSTANCE = YOLO(path)`.
    # Our mock_yolo patches the class, so MODEL_INSTANCE becomes the MagicMock returned by YOLO().

    response = client.get("/health")
    assert response.json()["model_loaded"] is True


def test_predict_no_model(client):
    # Ensure model is unloaded
    serve.MODEL_INSTANCE = None

    # Create dummy image
    img_byte_arr = io.BytesIO()
    Image.new("RGB", (100, 100)).save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    response = client.post(
        "/predict", files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded"


def test_predict_success(client, mock_yolo, tmp_path):
    # Load model first
    p = tmp_path / "model.pt"
    p.write_bytes(b"dummy")
    client.post("/load_model", params={"path": str(p)})

    # Create dummy image
    img_byte_arr = io.BytesIO()
    Image.new("RGB", (100, 100), color="red").save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")},
        params={"conf": 0.5},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.jpg"
    assert len(data["detections"]) == 1
    det = data["detections"][0]
    assert det["class"] == 0
    assert det["conf"] == 0.95
    assert det["bbox"] == [10, 10, 50, 50]


def test_main_cli_args_parsing():
    # Simple test to check if main runs without crashing provided existing args
    # We patch sys.argv and uvicorn.run
    import sys

    with (
        patch.object(sys, "argv", ["serve.py", "--port", "9000"]),
        patch("uvicorn.run") as mock_run,
    ):
        serve.main()
        mock_run.assert_called_once()
        assert mock_run.call_args[1]["port"] == 9000
