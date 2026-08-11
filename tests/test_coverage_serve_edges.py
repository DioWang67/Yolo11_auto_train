from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from picture_tool import serve


def test_load_model_rejects_missing_dependency_and_propagates_loader_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"weights")
    monkeypatch.setattr(serve, "YOLO", None)
    with pytest.raises(RuntimeError, match="ultralytics not installed"):
        serve.load_model(str(model_path))

    for error in (RuntimeError("invalid weights"), OSError("unreadable")):
        monkeypatch.setattr(
            serve,
            "YOLO",
            lambda path, error=error: (_ for _ in ()).throw(error),
        )
        with pytest.raises(type(error), match=str(error)):
            serve.load_model(str(model_path))


def test_load_model_replaces_global_instance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"weights")
    model = object()
    monkeypatch.setattr(serve, "YOLO", lambda _: model)
    monkeypatch.setattr(serve, "MODEL_INSTANCE", None)
    serve.load_model(str(model_path))
    assert serve.MODEL_INSTANCE is model


def test_lifespan_missing_successful_and_failed_default_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    loader = MagicMock()
    monkeypatch.setattr(serve, "load_model", loader)
    monkeypatch.setattr(serve, "MODEL_INSTANCE", None)

    async def enter_lifespan() -> None:
        async with serve.lifespan(serve.app):
            pass

    asyncio.run(enter_lifespan())
    loader.assert_not_called()

    default_model = tmp_path / "runs" / "detect" / "train" / "weights" / "best.pt"
    default_model.parent.mkdir(parents=True)
    default_model.write_bytes(b"weights")
    asyncio.run(enter_lifespan())
    loader.assert_called_once_with(str(Path("runs/detect/train/weights/best.pt")))

    loader.reset_mock()
    loader.side_effect = OSError("cannot load")
    asyncio.run(enter_lifespan())
    loader.assert_called_once()

    loader.reset_mock(side_effect=True)
    monkeypatch.setattr(serve, "MODEL_INSTANCE", object())
    asyncio.run(enter_lifespan())
    loader.assert_not_called()


def test_health_and_api_load_model_error_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve, "MODEL_INSTANCE", object())
    assert serve.health_check() == {"status": "ok", "model_loaded": True}

    for error, status in ((FileNotFoundError("missing"), 404), (RuntimeError("bad"), 500), (OSError("io"), 500)):
        monkeypatch.setattr(
            serve,
            "load_model",
            lambda path, error=error: (_ for _ in ()).throw(error),
        )
        with pytest.raises(serve.HTTPException) as exc_info:
            serve.api_load_model("model.pt")
        assert exc_info.value.status_code == status


class _Upload:
    filename = "sample.png"

    def __init__(self, content: bytes = b"image") -> None:
        self.content = content

    async def read(self) -> bytes:
        return self.content


def test_predict_dependency_and_input_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(serve.HTTPException) as exc_info:
        asyncio.run(serve.predict(_Upload(), conf=-0.1))
    assert exc_info.value.status_code == 400

    monkeypatch.setattr(serve, "MODEL_INSTANCE", object())
    monkeypatch.setattr(serve, "Image", None)
    with pytest.raises(serve.HTTPException) as exc_info:
        asyncio.run(serve.predict(_Upload()))
    assert exc_info.value.status_code == 500

    monkeypatch.setattr(serve, "Image", SimpleNamespace())
    monkeypatch.setattr(serve, "run_in_threadpool", None)
    with pytest.raises(serve.HTTPException) as exc_info:
        asyncio.run(serve.predict(_Upload()))
    assert exc_info.value.status_code == 500


def test_predict_translates_bad_images_and_inference_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(serve, "MODEL_INSTANCE", object())
    monkeypatch.setattr(
        serve,
        "Image",
        SimpleNamespace(open=lambda _: (_ for _ in ()).throw(OSError("bad image"))),
    )
    monkeypatch.setattr(serve, "run_in_threadpool", object())
    with pytest.raises(serve.HTTPException) as exc_info:
        asyncio.run(serve.predict(_Upload()))
    assert exc_info.value.status_code == 400

    converted = object()
    image = SimpleNamespace(convert=lambda mode: converted)
    monkeypatch.setattr(serve, "Image", SimpleNamespace(open=lambda _: image))

    async def fail_inference(*args: object, **kwargs: object) -> object:
        raise RuntimeError("GPU failed")

    monkeypatch.setattr(serve, "run_in_threadpool", fail_inference)
    with pytest.raises(serve.HTTPException) as exc_info:
        asyncio.run(serve.predict(_Upload(), conf=0.4))
    assert exc_info.value.status_code == 500
    assert "GPU failed" in exc_info.value.detail

    async def malformed_result(*args: object, **kwargs: object) -> list[object]:
        return [SimpleNamespace()]

    monkeypatch.setattr(serve, "run_in_threadpool", malformed_result)
    with pytest.raises(serve.HTTPException) as exc_info:
        asyncio.run(serve.predict(_Upload()))
    assert exc_info.value.status_code == 500


def test_main_reports_missing_fastapi_and_loads_initial_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(serve, "FastAPI", None)
    serve.main()
    assert "FastAPI/Uvicorn not installed" in capsys.readouterr().out

    model = tmp_path / "model.pt"
    model.write_bytes(b"weights")
    runner = MagicMock()
    loader = MagicMock()
    monkeypatch.setattr(serve, "FastAPI", object())
    monkeypatch.setattr(serve.uvicorn, "run", runner)
    monkeypatch.setattr(serve, "load_model", loader)
    monkeypatch.setattr(
        sys,
        "argv",
        ["serve.py", "--host", "127.0.0.1", "--port", "9001", "--model", str(model)],
    )
    serve.main()
    loader.assert_called_once_with(str(model))
    runner.assert_called_once_with(serve.app, host="127.0.0.1", port=9001)


@pytest.mark.parametrize(
    "error",
    (
        FileNotFoundError("missing model"),
        RuntimeError("invalid model"),
        OSError("unreadable model"),
    ),
)
def test_main_fails_before_starting_server_when_initial_model_cannot_load(
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = MagicMock()
    monkeypatch.setattr(serve, "FastAPI", object())
    monkeypatch.setattr(serve.uvicorn, "run", runner)
    monkeypatch.setattr(serve, "load_model", MagicMock(side_effect=error))
    monkeypatch.setattr(sys, "argv", ["serve.py", "--model", "bad.pt"])

    with pytest.raises(SystemExit) as exc_info:
        serve.main()

    assert exc_info.value.code == 2
    runner.assert_not_called()
