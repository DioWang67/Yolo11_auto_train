from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from picture_tool import config_validation
from picture_tool import validation
from picture_tool.exceptions import ValidationError
from picture_tool.utils import onnx_validation


def test_generic_validation_helpers_cover_valid_and_invalid_boundaries(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("ok", encoding="utf-8")
    directory = tmp_path / "directory"
    directory.mkdir()

    validation.validate_required_keys({"a": 1, "b": 2}, ["a"])
    validation.validate_path_exists(file_path, must_be_file=True)
    validation.validate_path_exists(directory, must_be_dir=True)
    assert validation.validate_positive_int("3", "workers") == 3
    assert validation.validate_ratio("0.5", "ratio") == 0.5
    assert validation.validate_class_names(("ok", "ng")) == ["ok", "ng"]

    invalid_calls = [
        lambda: validation.validate_required_keys({}, ["a"], "pipeline"),
        lambda: validation.validate_path_exists(tmp_path / "missing"),
        lambda: validation.validate_path_exists(directory, must_be_file=True),
        lambda: validation.validate_path_exists(file_path, must_be_dir=True),
        lambda: validation.validate_positive_int("bad", "workers"),
        lambda: validation.validate_positive_int(0, "workers"),
        lambda: validation.validate_ratio(object(), "ratio"),
        lambda: validation.validate_ratio(1.01, "ratio"),
        lambda: validation.validate_class_names([]),
        lambda: validation.validate_class_names("ok"),
        lambda: validation.validate_class_names(["ok", 1]),
        lambda: validation.validate_class_names(["ok", "ok"]),
    ]
    for call in invalid_calls:
        with pytest.raises(ValidationError):
            call()


@pytest.mark.parametrize(
    ("value", "minimum"),
    [
        (True, 0),
        (1.5, 0),
        ("bad", 0),
        (-1, 0),
        (0, 1),
    ],
)
def test_manual_integer_validation_rejects_unsafe_values(
    value: object,
    minimum: int,
) -> None:
    errors: list[str] = []
    config_validation._manual_non_negative_integer(
        errors,
        value,
        "field",
        minimum=minimum,
    )
    assert errors == [f"field must be an integer >= {minimum}"]


@pytest.mark.parametrize("value", [True, "bad", float("nan"), -0.1])
def test_manual_number_validation_rejects_non_finite_or_negative_values(
    value: object,
) -> None:
    errors: list[str] = []
    config_validation._manual_non_negative_number(errors, value, "field")
    assert errors == ["field must be a finite number >= 0"]


@pytest.mark.parametrize("value", [True, object(), float("inf"), -0.1, 1.1])
def test_manual_rate_validation_rejects_values_outside_unit_interval(
    value: object,
) -> None:
    errors: list[str] = []
    config_validation._manual_rate(errors, value, "field")
    assert errors == ["field must be between 0 and 1"]


def test_manual_config_validation_accepts_complete_valid_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_validation, "BaseModel", None)
    config = {
        "yolo_training": {
            "dataset_dir": str(tmp_path),
            "class_names": ["part"],
            "position_validation": {
                "mode": "iou",
                "tolerance_unit": "pixel",
                "tolerance": 2.5,
                "tolerance_override": 1,
                "conf": 0.5,
                "calibration_min_samples": 2,
                "gate": {
                    "min_ok_samples": 1,
                    "min_ng_samples": 1,
                    "max_ok_false_reject_rate": 0.2,
                    "min_ng_recall": 0.8,
                    "max_ok_false_reject_regression": 0.1,
                    "max_ng_recall_regression": 0.1,
                },
            },
        },
        "anomalib_training": {
            "model": "efficient-ad",
            "image_size": 64,
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "max_epochs": 1,
        },
        "augmentation": {
            "operations": {"flip": {"probability": 0.5}},
        },
    }

    assert config_validation.validate_config_schema(config, strict=True) is config


def test_manual_config_validation_reports_all_invalid_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(config_validation, "BaseModel", None)
    config = {
        "yolo_training": {
            "dataset_dir": str(tmp_path / "missing"),
            "class_names": [],
            "position_validation": {
                "calibration_source": "camera",
                "mode": "polygon",
                "tolerance_unit": "mm",
                "tolerance": -1,
                "tolerance_override": "bad",
                "conf": 2,
                "calibration_min_samples": 0,
                "gate": "invalid",
            },
        },
        "anomalib_training": {
            "model": "unknown",
            "image_size": 0,
            "train_batch_size": -1,
            "eval_batch_size": 0,
            "max_epochs": -2,
        },
        "augmentation": {
            "operations": {"flip": {"probability": 1.5}},
        },
    }

    logger = logging.getLogger("manual-config-test")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        assert config_validation.validate_config_schema(config, logger=logger) is config
    assert "Config validation warnings" in caplog.text

    with pytest.raises(ValueError, match="Config validation failed"):
        config_validation.validate_config_schema(config, strict=True)


@pytest.mark.parametrize(
    "position_validation",
    ["invalid", {"gate": {"min_ok_samples": -1, "min_ng_samples": 1.5}}],
)
def test_manual_position_mapping_validation(
    position_validation: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_validation, "BaseModel", None)
    with pytest.raises(ValueError, match="Config validation failed"):
        config_validation.validate_config_schema(
            {"yolo_training": {"position_validation": position_validation}},
            strict=True,
        )


def test_pydantic_validation_warns_in_non_strict_mode(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    if config_validation.BaseModel is None:
        pytest.skip("Pydantic is unavailable")
    logger = logging.getLogger("pydantic-config-test")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        config = {"yolo_training": {"dataset_dir": tmp_path / "missing"}}
        assert config_validation.validate_config_schema(config, logger=logger) is config
    assert "Config validation failed" in caplog.text


def test_onnx_structure_validates_existing_non_empty_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"onnx")
    model = object()
    fake_onnx = SimpleNamespace(
        load=lambda value: model,
        checker=SimpleNamespace(check_model=lambda value: None),
    )
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    monkeypatch.setattr(onnx_validation, "_is_package_available", lambda _: True)

    onnx_validation.validate_onnx_structure(path)


def test_onnx_structure_rejects_missing_dependency_and_invalid_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(onnx_validation, "_is_package_available", lambda _: False)
    with pytest.raises(ImportError, match="requires 'onnx'"):
        onnx_validation.validate_onnx_structure(tmp_path / "model.onnx")

    monkeypatch.setattr(onnx_validation, "_is_package_available", lambda _: True)
    monkeypatch.setitem(
        sys.modules,
        "onnx",
        SimpleNamespace(load=lambda _: object(), checker=SimpleNamespace()),
    )
    with pytest.raises(FileNotFoundError):
        onnx_validation.validate_onnx_structure(tmp_path / "missing.onnx")

    empty_path = tmp_path / "empty.onnx"
    empty_path.touch()
    with pytest.raises(ValueError, match="empty"):
        onnx_validation.validate_onnx_structure(empty_path)


def test_onnx_structure_propagates_checker_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"onnx")

    def fail_load(_: str) -> object:
        raise RuntimeError("invalid graph")

    monkeypatch.setattr(onnx_validation, "_is_package_available", lambda _: True)
    monkeypatch.setitem(
        sys.modules,
        "onnx",
        SimpleNamespace(load=fail_load, checker=SimpleNamespace()),
    )
    with pytest.raises(RuntimeError, match="invalid graph"):
        onnx_validation.validate_onnx_structure(path)


class _FakeOrtSession:
    def __init__(
        self,
        path: str,
        *,
        providers: list[str],
        input_shape: list[object],
        input_type: str,
        outputs: list[object],
    ) -> None:
        self.path = path
        self.providers = providers
        self._input = SimpleNamespace(
            name="images",
            shape=input_shape,
            type=input_type,
        )
        self._output_meta = [
            SimpleNamespace(name="detections", shape=[1, 6], type="tensor(float)")
        ]
        self._outputs = outputs
        self.received: np.ndarray | None = None

    def get_inputs(self) -> list[object]:
        return [self._input]

    def get_outputs(self) -> list[object]:
        return self._output_meta

    def run(self, _: object, inputs: dict[str, np.ndarray]) -> list[object]:
        self.received = inputs["images"]
        return self._outputs


@pytest.mark.parametrize(
    ("imgsz", "shape", "input_type", "expected_shape", "expected_dtype"),
    [
        (None, [None, "channels", "height", "width", "extra"], "tensor(float)", (1, 3, 640, 640, 1), np.float32),
        (32, [1, 3, None, None], "tensor(float16)", (1, 3, 32, 32), np.float16),
        ([24], [1, 3, None, None], "tensor(uint8)", (1, 3, 24, 24), np.uint8),
        ((20, 30), [1.0, 3.0, None, None], "tensor(float)", (1, 3, 20, 30), np.float32),
        (object(), [1, 3, None, None], "tensor(float)", (1, 3, 640, 640), np.float32),
    ],
)
def test_onnx_runtime_builds_inputs_for_static_and_dynamic_shapes(
    imgsz: object,
    shape: list[object],
    input_type: str,
    expected_shape: tuple[int, ...],
    expected_dtype: type[np.generic],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions: list[_FakeOrtSession] = []

    def build_session(path: str, *, providers: list[str]) -> _FakeOrtSession:
        session = _FakeOrtSession(
            path,
            providers=providers,
            input_shape=shape,
            input_type=input_type,
            outputs=[np.zeros((1, 6))],
        )
        sessions.append(session)
        return session

    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider"],
        InferenceSession=build_session,
    )
    monkeypatch.setattr(onnx_validation, "_is_package_available", lambda _: True)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    assert onnx_validation.validate_onnx_runtime(Path("model.onnx"), imgsz, "cuda")
    assert sessions[0].providers == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert sessions[0].received is not None
    assert sessions[0].received.shape == expected_shape
    assert sessions[0].received.dtype == expected_dtype


def test_onnx_runtime_skip_and_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(onnx_validation, "_is_package_available", lambda _: False)
    assert onnx_validation.validate_onnx_runtime(Path("missing.onnx"))

    fake_ort = SimpleNamespace(
        get_available_providers=lambda: [],
        InferenceSession=lambda path, *, providers: _FakeOrtSession(
            path,
            providers=providers,
            input_shape=[1, 3, 8, 8],
            input_type="tensor(float)",
            outputs=[],
        ),
    )
    monkeypatch.setattr(onnx_validation, "_is_package_available", lambda _: True)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    with pytest.raises(RuntimeError, match="returned no outputs"):
        onnx_validation.validate_onnx_runtime(Path("model.onnx"), 8, "cpu")


def test_onnx_runtime_wraps_session_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_session(path: str, *, providers: list[str]) -> object:
        raise OSError(f"cannot open {path} with {providers}")

    monkeypatch.setattr(onnx_validation, "_is_package_available", lambda _: True)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: [],
            InferenceSession=fail_session,
        ),
    )
    with pytest.raises(RuntimeError, match="cannot open"):
        onnx_validation.validate_onnx_runtime(Path("model.onnx"))
