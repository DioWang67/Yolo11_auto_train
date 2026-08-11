from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from picture_tool.augment import image_augmentor
from picture_tool.augment.image_augmentor import ImageAugmentor


class _FakeAlbumentations:
    calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def _make(cls, name: str, **kwargs: object) -> tuple[str, dict[str, object]]:
        cls.calls.append((name, kwargs))
        return name, kwargs

    @classmethod
    def HorizontalFlip(cls, **kwargs: object) -> object:
        return cls._make("HorizontalFlip", **kwargs)

    @classmethod
    def Rotate(cls, **kwargs: object) -> object:
        return cls._make("Rotate", **kwargs)

    @classmethod
    def RandomBrightnessContrast(cls, **kwargs: object) -> object:
        return cls._make("RandomBrightnessContrast", **kwargs)

    @classmethod
    def RandomScale(cls, **kwargs: object) -> object:
        return cls._make("RandomScale", **kwargs)

    @classmethod
    def HueSaturationValue(cls, **kwargs: object) -> object:
        return cls._make("HueSaturationValue", **kwargs)

    @classmethod
    def GaussNoise(cls, **kwargs: object) -> object:
        return cls._make("GaussNoise", **kwargs)

    @classmethod
    def Perspective(cls, **kwargs: object) -> object:
        return cls._make("Perspective", **kwargs)

    @classmethod
    def MotionBlur(cls, **kwargs: object) -> object:
        return cls._make("MotionBlur", **kwargs)

    @classmethod
    def LongestMaxSize(cls, **kwargs: object) -> object:
        return cls._make("LongestMaxSize", **kwargs)

    @classmethod
    def PadIfNeeded(cls, **kwargs: object) -> object:
        return cls._make("PadIfNeeded", **kwargs)

    @classmethod
    def Compose(cls, operations: list[object]) -> object:
        cls.calls.append(("Compose", {"operations": operations}))
        return SimpleNamespace(operations=operations)


@pytest.mark.parametrize("rotate_angle", [5, (-10, 10)])
@pytest.mark.parametrize("hue_range", [8, (-4, 6)])
@pytest.mark.parametrize("blur_kernel", [3, (3, 7)])
def test_build_augmentations_maps_every_supported_operation(
    rotate_angle: object,
    hue_range: object,
    blur_kernel: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeAlbumentations.calls.clear()
    monkeypatch.setattr(image_augmentor, "A", _FakeAlbumentations)
    ops = {
        "flip": {"probability": 0.7},
        "rotate": {"angle": rotate_angle},
        "multiply": {"range": (0.8, 1.2)},
        "scale": {"range": (0.9, 1.1)},
        "contrast": {"range": (0.6, 1.4)},
        "hue": {"range": hue_range},
        "noise": {"scale": (0, 0.05)},
        "perspective": {"scale": (0.01, 0.03)},
        "blur": {"kernel": blur_kernel},
    }

    pipeline = ImageAugmentor._build_augmentations_from_ops(ops)

    assert len(pipeline.operations) == 11
    names = [name for name, _ in _FakeAlbumentations.calls]
    assert names[-3:] == ["LongestMaxSize", "PadIfNeeded", "Compose"]
    assert set(names[:-3]) == {
        "HorizontalFlip",
        "Rotate",
        "RandomBrightnessContrast",
        "RandomScale",
        "HueSaturationValue",
        "GaussNoise",
        "Perspective",
        "MotionBlur",
    }


def test_build_augmentations_supports_no_optional_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeAlbumentations.calls.clear()
    monkeypatch.setattr(image_augmentor, "A", _FakeAlbumentations)

    pipeline = ImageAugmentor._build_augmentations_from_ops({})

    assert len(pipeline.operations) == 2


def _bare_augmentor(config: dict[str, object]) -> ImageAugmentor:
    augmentor = object.__new__(ImageAugmentor)
    augmentor.config = config
    augmentor.logger = MagicMock()
    augmentor.augmentations = lambda *, image: {"image": image}
    return augmentor


def test_init_uses_defaults_and_creates_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    default_config = {
        "input": {"image_dir": str(tmp_path / "input")},
        "output": {"image_dir": str(output_dir)},
        "augmentation": {"num_images": 0, "operations": {}},
        "processing": {"seed": 11},
    }
    logger = MagicMock()
    monkeypatch.setattr(ImageAugmentor, "_default_config", lambda self: default_config)
    monkeypatch.setattr(image_augmentor, "setup_module_logger", lambda *args: logger)
    monkeypatch.setattr(
        ImageAugmentor,
        "_build_augmentations_from_ops",
        staticmethod(lambda ops: SimpleNamespace(operations=ops)),
    )

    augmentor = ImageAugmentor()

    assert augmentor.config is default_config
    assert output_dir.is_dir()
    assert augmentor.augmentations.operations == {}


def test_load_config_valid_missing_and_yaml_error_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    augmentor = _bare_augmentor({})
    fallback = {"fallback": True}
    monkeypatch.setattr(augmentor, "_default_config", lambda: fallback)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("value: 3\n", encoding="utf-8")

    assert augmentor._load_config(str(config_path)) == {"value": 3}
    assert augmentor._load_config(str(tmp_path / "missing.yaml")) is fallback

    monkeypatch.setattr(yaml, "safe_load", lambda _: (_ for _ in ()).throw(yaml.YAMLError("bad")))
    assert augmentor._load_config(str(config_path)) is fallback


def test_seed_and_output_setup_report_invalid_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    augmentor = _bare_augmentor({"output": {"image_dir": object()}})
    augmentor._set_seed(None)
    augmentor._set_seed("17")
    augmentor._set_seed("invalid")
    assert augmentor.logger.warning.called

    with pytest.raises(TypeError):
        augmentor._setup_output_dirs()
    assert augmentor.logger.error.called


def test_process_single_image_writes_each_successful_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    augmentor = _bare_augmentor(
        {
            "input": {"image_dir": str(input_dir)},
            "output": {"image_dir": str(output_dir)},
            "augmentation": {"num_images": 2},
        }
    )
    writes: list[Path] = []
    monkeypatch.setattr(image_augmentor.cv2, "imread", lambda _: image)
    monkeypatch.setattr(
        image_augmentor.cv2,
        "imwrite",
        lambda path, value: writes.append(Path(path)) or True,
    )

    assert augmentor._process_single_image("sample.jpg")
    assert [path.name for path in writes] == ["sample_aug_1.png", "sample_aug_2.png"]
    assert augmentor.logger.warning.call_count == 2


def test_process_single_image_handles_decode_transform_and_path_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    augmentor = _bare_augmentor(
        {
            "input": {"image_dir": str(tmp_path)},
            "output": {"image_dir": str(tmp_path)},
            "augmentation": {"num_images": 1},
        }
    )
    monkeypatch.setattr(image_augmentor.cv2, "imread", lambda _: None)
    assert not augmentor._process_single_image("missing.jpg")

    monkeypatch.setattr(
        image_augmentor.cv2,
        "imread",
        lambda _: np.zeros((640, 640, 3), dtype=np.uint8),
    )
    augmentor.augmentations = lambda **_: (_ for _ in ()).throw(ValueError("bad op"))
    assert not augmentor._process_single_image("bad.jpg")

    augmentor.config["input"] = {"image_dir": object()}
    assert not augmentor._process_single_image("bad-path.jpg")


class _InlineExecutor:
    instances: list[_InlineExecutor] = []

    def __init__(self, *, max_workers: int, initializer=None, initargs=()) -> None:
        self.max_workers = max_workers
        self.initializer = initializer
        self.initargs = initargs
        self.mapped_function = None
        self.mapped_values = []
        type(self).instances.append(self)
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self) -> _InlineExecutor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def map(self, function, values):
        self.mapped_function = function
        self.mapped_values = list(values)
        return [function(value) for value in self.mapped_values]


class _ThreadInlineExecutor(_InlineExecutor):
    instances: list[_InlineExecutor] = []


class _ProcessInlineExecutor(_InlineExecutor):
    instances: list[_InlineExecutor] = []


@pytest.mark.parametrize(
    ("workers", "use_process_pool"),
    [(1, False), (0, False), (None, True)],
)
def test_process_dataset_runs_thread_and_process_modes(
    tmp_path: Path,
    workers: object,
    use_process_pool: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    augmentor = _bare_augmentor(
        {
            "input": {"image_dir": str(input_dir)},
            "output": {"image_dir": str(output_dir)},
            "augmentation": {"num_images": 1, "operations": {}},
            "processing": {
                "num_workers": workers,
                "use_process_pool": use_process_pool,
            },
        }
    )
    monkeypatch.setattr(_ThreadInlineExecutor, "instances", [])
    monkeypatch.setattr(_ProcessInlineExecutor, "instances", [])
    monkeypatch.setattr(image_augmentor, "list_images", lambda *args: ["a.jpg", "b.jpg"])
    monkeypatch.setattr(image_augmentor, "ThreadPoolExecutor", _ThreadInlineExecutor)
    monkeypatch.setattr(image_augmentor, "ProcessPoolExecutor", _ProcessInlineExecutor)
    monkeypatch.setattr(image_augmentor, "cpu_count", lambda: 2)
    monkeypatch.setattr(image_augmentor, "tqdm", lambda values, **kwargs: values)
    monkeypatch.setattr(augmentor, "_process_single_image", lambda name: name == "a.jpg")
    worker_initializations = []
    monkeypatch.setattr(
        image_augmentor,
        "_init_worker",
        lambda ops: worker_initializations.append(ops),
    )
    monkeypatch.setattr(
        image_augmentor,
        "_process_single_image_job",
        lambda args: args[0] == "a.jpg",
    )

    augmentor.process_dataset()

    selected_instances = (
        _ProcessInlineExecutor.instances
        if use_process_pool
        else _ThreadInlineExecutor.instances
    )
    unselected_instances = (
        _ThreadInlineExecutor.instances
        if use_process_pool
        else _ProcessInlineExecutor.instances
    )
    assert len(selected_instances) == 1
    assert unselected_instances == []
    executor = selected_instances[0]
    assert executor.max_workers == (1 if workers == 1 else 2)
    assert len(executor.mapped_values) == 2
    if use_process_pool:
        assert executor.initializer is image_augmentor._init_worker
        assert executor.initargs == ({},)
        assert worker_initializations == [{}]
        assert executor.mapped_function is image_augmentor._process_single_image_job
        assert executor.mapped_values[0][0] == "a.jpg"
    else:
        assert executor.initializer is None
        assert worker_initializations == []
        assert executor.mapped_function == augmentor._process_single_image
        assert executor.mapped_values == ["a.jpg", "b.jpg"]
    final_message = augmentor.logger.info.call_args_list[-1].args[0]
    assert "成功 1，失敗 1" in final_message


def test_process_dataset_returns_for_missing_or_empty_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    augmentor = _bare_augmentor(
        {
            "input": {"image_dir": str(tmp_path / "missing")},
            "processing": {},
        }
    )
    augmentor.process_dataset()
    assert augmentor.logger.error.called

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    augmentor.config["input"] = {"image_dir": str(input_dir)}
    monkeypatch.setattr(image_augmentor, "list_images", lambda *args: [])
    augmentor.process_dataset()
    assert augmentor.logger.error.call_count == 2


def test_worker_job_requires_image_and_initialized_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = ("sample.jpg", str(tmp_path), str(tmp_path), 2)
    monkeypatch.setattr(image_augmentor.cv2, "imread", lambda _: None)
    assert not image_augmentor._process_single_image_job(args)

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(image_augmentor.cv2, "imread", lambda _: image)
    monkeypatch.setattr(image_augmentor, "_worker_augmentations", None)
    assert not image_augmentor._process_single_image_job(args)

    monkeypatch.setattr(
        image_augmentor,
        "_worker_augmentations",
        lambda *, image: {"image": image},
    )
    monkeypatch.setattr(image_augmentor.cv2, "imwrite", lambda *args: True)
    assert image_augmentor._process_single_image_job(args)

    monkeypatch.setattr(
        image_augmentor,
        "_worker_augmentations",
        lambda **_: (_ for _ in ()).throw(ValueError("bad")),
    )
    assert not image_augmentor._process_single_image_job(args)
