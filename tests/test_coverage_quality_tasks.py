from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from picture_tool.quality.dataset_readiness import DatasetReadinessError
from picture_tool.tasks import quality


def _split_config(image_dir: Path, label_dir: Path, output_dir: Path) -> dict:
    return {
        "train_test_split": {
            "input": {"image_dir": str(image_dir), "label_dir": str(label_dir)},
            "output": {"output_dir": str(output_dir)},
        }
    }


@pytest.mark.parametrize(
    "path",
    [Path("processed/not-images"), Path("other/images"), Path("images")],
)
def test_raw_input_fallback_rejects_noncanonical_paths(path: Path) -> None:
    assert quality._raw_input_fallback(path) is None


def test_split_input_resolution_prefers_existing_then_raw_fallback(tmp_path: Path) -> None:
    processed_images = tmp_path / "data" / "processed" / "images"
    processed_labels = tmp_path / "data" / "processed" / "labels"
    raw_images = tmp_path / "data" / "raw" / "images"
    raw_labels = tmp_path / "data" / "raw" / "labels"
    raw_images.mkdir(parents=True)
    raw_labels.mkdir(parents=True)
    config = _split_config(processed_images, processed_labels, tmp_path / "split")

    assert quality.resolve_split_input_dirs(config) == (raw_images, raw_labels, True)
    effective = quality._config_with_effective_split_inputs(config)
    assert effective is not config
    assert effective["train_test_split"]["input"] == {
        "image_dir": str(raw_images),
        "label_dir": str(raw_labels),
    }

    processed_images.mkdir(parents=True)
    processed_labels.mkdir(parents=True)
    assert quality.resolve_split_input_dirs(config) == (
        processed_images,
        processed_labels,
        False,
    )
    assert quality._config_with_effective_split_inputs(config) is config


def test_split_input_resolution_returns_missing_configured_paths_without_raw_pair(
    tmp_path: Path,
) -> None:
    images = tmp_path / "custom" / "images"
    labels = tmp_path / "custom" / "labels"
    config = _split_config(images, labels, tmp_path / "split")
    assert quality.resolve_split_input_dirs(config) == (images, labels, False)


def test_simple_quality_task_adapters_forward_config(monkeypatch: pytest.MonkeyPatch) -> None:
    config = {"key": "value"}
    anomaly = MagicMock()
    linter = MagicMock()
    reporter = MagicMock()
    monkeypatch.setattr(quality, "process_anomaly_detection", anomaly)
    monkeypatch.setattr(quality, "lint_dataset", linter)
    monkeypatch.setattr(quality, "generate_report", reporter)

    quality.run_anomaly_detection(config, SimpleNamespace())
    quality.run_dataset_lint(config, SimpleNamespace())
    quality.run_generate_report(config, SimpleNamespace())

    anomaly.assert_called_once_with(config)
    linter.assert_called_once_with(config)
    reporter.assert_called_once_with(config)


def test_run_dataset_splitter_writes_cache_for_effective_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    output = tmp_path / "split"
    images.mkdir()
    labels.mkdir()
    config = _split_config(images, labels, output)
    splitter = MagicMock()
    cache = MagicMock()
    monkeypatch.setattr(quality, "split_dataset", splitter)
    monkeypatch.setattr(quality, "write_task_cache", cache)

    quality.run_dataset_splitter(config, SimpleNamespace())

    splitter.assert_called_once_with(config)
    cache.assert_called_once_with(
        output,
        "dataset_splitter",
        config["train_test_split"],
        [images, labels],
    )


def test_skip_dataset_splitter_handles_missing_bad_and_readiness_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert quality.skip_dataset_splitter({}, SimpleNamespace()) is None
    assert quality.skip_dataset_splitter({"train_test_split": {}}, SimpleNamespace()) is None

    images = tmp_path / "images"
    labels = tmp_path / "labels"
    output = tmp_path / "split"
    images.mkdir()
    labels.mkdir()
    for split in ("train", "val", "test"):
        (output / split / "images").mkdir(parents=True)
    config = _split_config(images, labels, output)
    config["yolo_training"] = {"class_names": ["part"], "dataset_dir": str(output)}
    monkeypatch.setattr(
        quality,
        "validate_training_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(DatasetReadinessError("unsafe")),
    )
    assert quality.skip_dataset_splitter(config, SimpleNamespace()) is None


def test_skip_dataset_splitter_cache_and_mtime_decisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    output = tmp_path / "split"
    images.mkdir()
    labels.mkdir()
    for split in ("train", "val", "test"):
        (output / split / "images").mkdir(parents=True)
    config = _split_config(images, labels, output)
    monkeypatch.setattr(quality, "task_cache_matches", lambda *args: True)
    assert "cache matches" in quality.skip_dataset_splitter(config, SimpleNamespace())

    monkeypatch.setattr(quality, "task_cache_matches", lambda *args: False)
    monkeypatch.setattr(quality, "task_cache_exists", lambda *args: True)
    assert quality.skip_dataset_splitter(config, SimpleNamespace()) is None

    monkeypatch.setattr(quality, "task_cache_exists", lambda *args: False)
    monkeypatch.setattr(quality, "mtime_latest", lambda paths: 10 if "split" in str(paths[0]) else 1)
    assert "up-to-date" in quality.skip_dataset_splitter(config, SimpleNamespace())


def test_dataset_lint_skip_fresh_old_and_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images = tmp_path / "images"
    output = tmp_path / "lint"
    images.mkdir()
    output.mkdir()
    config = {"dataset_lint": {"image_dir": str(images), "output_dir": str(output)}}
    assert quality.skip_dataset_lint(config, SimpleNamespace()) is None
    lint_csv = output / "lint.csv"
    lint_csv.write_text("result", encoding="utf-8")
    monkeypatch.setattr(quality, "mtime_latest", lambda paths: 0)
    assert "newer" in quality.skip_dataset_lint(config, SimpleNamespace())
    monkeypatch.setattr(quality, "mtime_latest", lambda paths: lint_csv.stat().st_mtime + 1)
    assert quality.skip_dataset_lint(config, SimpleNamespace()) is None


def test_batch_inference_run_and_skip_cache_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    config = {"batch_inference": {"input_dir": str(input_dir), "output_dir": str(output_dir)}}
    monkeypatch.setattr(quality, "run_batch_inference", lambda cfg: output_dir)
    cache = MagicMock()
    monkeypatch.setattr(quality, "write_task_cache", cache)
    quality.run_batch_infer(config, SimpleNamespace())
    cache.assert_called_once_with(output_dir, "batch_inference", config["batch_inference"], [input_dir])

    assert quality.skip_batch_infer(config, SimpleNamespace()) is None
    csv_path = output_dir / "predictions.csv"
    csv_path.write_text("result", encoding="utf-8")
    monkeypatch.setattr(quality, "task_cache_matches", lambda *args: True)
    assert "cache matches" in quality.skip_batch_infer(config, SimpleNamespace())
    monkeypatch.setattr(quality, "task_cache_matches", lambda *args: False)
    monkeypatch.setattr(quality, "task_cache_exists", lambda *args: True)
    assert quality.skip_batch_infer(config, SimpleNamespace()) is None
    monkeypatch.setattr(quality, "task_cache_exists", lambda *args: False)
    monkeypatch.setattr(quality, "mtime_latest", lambda paths: 0)
    assert "newer" in quality.skip_batch_infer(config, SimpleNamespace())


def test_qc_summary_default_and_explicit_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    generator = MagicMock()
    monkeypatch.setattr(quality, "generate_qc_summary", generator)
    quality.run_qc_summary({}, SimpleNamespace())
    assert generator.call_args.kwargs["output_path"] is None
    quality.run_qc_summary(
        {"qc_summary": {"output_path": f" {tmp_path / 'qc.json'} "}},
        SimpleNamespace(),
    )
    assert generator.call_args.kwargs["output_path"] == tmp_path / "qc.json"


def test_section_enabled_defaults_and_explicit_values() -> None:
    assert not quality._section_enabled(None)
    assert quality._section_enabled({})
    assert not quality._section_enabled({"enabled": False})


def test_color_inspection_missing_disabled_invalid_and_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = MagicMock()
    monkeypatch.setattr(quality.subprocess, "run", runner)
    quality.run_color_inspection({}, SimpleNamespace())
    quality.run_color_inspection(
        {
            "color_inspection": {
                "enabled": False,
                "sam": {"checkpoint": "sam.pt"},
            }
        },
        SimpleNamespace(),
    )
    runner.assert_not_called()
    with pytest.raises(ValueError, match="Invalid color_inspection"):
        quality.run_color_inspection({"color_inspection": {"enabled": True}}, SimpleNamespace())

    quality.run_color_inspection(
        {
            "color_inspection": {
                "enabled": True,
                "input_dir": str(tmp_path / "samples"),
                "output_json": str(tmp_path / "stats.json"),
                "colors": ["Red", "Green"],
                "sam": {
                    "checkpoint": str(tmp_path / "sam.pt"),
                    "model_type": "vit_b",
                    "device": "cpu",
                },
                "max_side": 1024,
            }
        },
        SimpleNamespace(),
    )
    command = runner.call_args.args[0]
    assert command[-3:] == ["--colors", "Red", "Green"]
    assert runner.call_args.kwargs == {"check": True}


def test_color_inspection_without_color_filter_omits_colors_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = MagicMock()
    monkeypatch.setattr(quality.subprocess, "run", runner)
    quality.run_color_inspection(
        {"color_inspection": {"sam": {"checkpoint": "sam.pt"}}},
        SimpleNamespace(),
    )
    assert "--colors" not in runner.call_args.args[0]


def test_color_verification_missing_disabled_invalid_and_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = MagicMock()
    verifier.StripOptions.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setattr(quality, "color_verifier", verifier)
    quality.run_color_verification({}, SimpleNamespace())
    quality.run_color_verification(
        {"color_verification": {"enabled": False}},
        SimpleNamespace(),
    )
    verifier.verify_directory.assert_not_called()
    with pytest.raises(ValueError, match="Invalid color_verification"):
        quality.run_color_verification(
            {"color_verification": {"hsv_margin": [1, 2]}},
            SimpleNamespace(),
        )

    quality.run_color_verification(
        {
            "color_verification": {
                "enabled": True,
                "input_dir": str(tmp_path / "samples"),
                "color_stats": str(tmp_path / "stats.json"),
                "segments": 7,
                "orientation": "horizontal",
                "ratio_threshold": 0.7,
                "strip_sampling": {
                    "enabled": True,
                    "segments": 3,
                    "orientation": "vertical",
                    "threshold": 0.2,
                },
            }
        },
        SimpleNamespace(),
    )
    call = verifier.verify_directory.call_args.kwargs
    assert call["segments"] == 7
    assert call["orientation"] == "horizontal"
    assert call["ratio_threshold"] == 0.7
    assert call["strip_options"].segments == 3


def test_color_verification_uses_strip_defaults_for_all_optional_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = MagicMock()
    verifier.StripOptions.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setattr(quality, "color_verifier", verifier)
    quality.run_color_verification(
        {"color_verification": {"enabled": True}},
        SimpleNamespace(),
    )
    call = verifier.verify_directory.call_args.kwargs
    strip = call["strip_options"]
    for key in (
        "segments",
        "orientation",
        "min_strip_ratio",
        "ratio_threshold",
        "edge_margin",
        "sat_threshold",
        "val_threshold",
        "min_sat_ratio",
        "max_edge_ratio",
        "black_s_threshold",
        "black_v_threshold",
    ):
        assert call[key] == getattr(strip, key)
