from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from picture_tool.pipeline.preflight import PreflightChecker, PreflightIssue, Severity
from picture_tool.quality import dataset_linter


def test_preflight_issue_blocking_property() -> None:
    assert PreflightIssue(Severity.ERROR, "task", "error").is_blocking
    assert not PreflightIssue(Severity.WARNING, "task", "warning").is_blocking


def test_preflight_reports_label_model_color_and_deploy_problems(tmp_path: Path) -> None:
    labels = tmp_path / "dataset" / "train" / "labels"
    labels.mkdir(parents=True)
    (labels / "sample.txt").write_text(
        "bad line\n3 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    config = {
        "yolo_training": {
            "dataset_dir": str(tmp_path / "dataset"),
            "class_names": ["only"],
            "model": str(tmp_path / "missing.pt"),
            "project": str(tmp_path / "runs"),
            "name": "train",
            "deploy": {"enabled": False},
        },
        "color_inspection": {"sam_checkpoint": str(tmp_path / "missing-sam.pt")},
        "color_verification": {"color_stats": str(tmp_path / "missing-stats.json")},
    }

    issues = PreflightChecker().run(
        ["yolo_train", "color_inspection", "color_verification", "deploy"],
        config,
    )

    assert {issue.task for issue in issues} == {
        "yolo_train",
        "color_inspection",
        "color_verification",
        "deploy",
    }
    assert any(issue.severity is Severity.ERROR and issue.task == "yolo_train" for issue in issues)
    assert any(issue.severity is Severity.WARNING and issue.task == "deploy" for issue in issues)
    assert not any("best.pt" in issue.message for issue in issues)


def test_preflight_reports_missing_class_names_and_deployment_details(
    tmp_path: Path,
) -> None:
    config = {
        "yolo_training": {
            "dataset_dir": "",
            "class_names": [],
            "model": "",
            "project": str(tmp_path / "runs"),
            "name": "candidate",
            "deploy": {"enabled": True},
        }
    }
    checker = PreflightChecker()

    issues = checker.run(["yolo_train", "deploy"], config)
    assert any(issue.task == "yolo_train" and issue.severity is Severity.ERROR for issue in issues)
    assert any(issue.task == "deploy" and issue.severity is Severity.ERROR for issue in issues)

    config["yolo_training"]["deploy"] = {
        "enabled": True,
        "inference_models_dir": "relative-models",
    }
    issues = checker.run(["deploy"], config)
    assert sum(issue.task == "deploy" and issue.severity is Severity.WARNING for issue in issues) == 2


def test_preflight_accepts_existing_model_artifacts_and_optional_empty_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / "base.pt").write_bytes(b"weights")
    inference_models = tmp_path / "inference-models"
    inference_models.mkdir()
    best = tmp_path / "runs" / "candidate" / "weights" / "best.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"weights")
    config = {
        "yolo_training": {
            "class_names": ["part"],
            "model": "base.pt",
            "project": str(tmp_path / "runs"),
            "name": "candidate",
            "deploy": {
                "enabled": True,
                "inference_models_dir": str(inference_models),
            },
        },
        "color_inspection": {"sam_checkpoint": ""},
        "color_verification": {"color_stats": ""},
    }

    assert PreflightChecker().run(
        ["yolo_train", "color_inspection", "color_verification", "deploy"],
        config,
    ) == []


def test_preflight_label_scan_ignores_empty_invalid_and_io_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = PreflightChecker()
    labels = tmp_path / "dataset" / "train" / "labels"
    labels.mkdir(parents=True)
    label = labels / "sample.txt"
    label.write_text("not-a-class 0 0 0 0\n", encoding="utf-8")
    config = {
        "yolo_training": {
            "dataset_dir": str(tmp_path / "dataset"),
            "class_names": ["part"],
        }
    }
    assert checker.run([], config) == []

    label.write_text("\n", encoding="utf-8")
    assert checker.run([], config) == []

    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("io")))
    assert checker.run([], config) == []


def test_linter_label_helpers_cover_file_and_geometry_boundaries(tmp_path: Path) -> None:
    assert dataset_linter._read_labels(tmp_path / "missing.txt") == []
    label_path = tmp_path / "labels.txt"
    label_path.write_text(
        "0 0.5 0.5 0.2 0.2\nshort row\n1 0.1 0.2 0.3 0.4 extra\n",
        encoding="utf-8",
    )
    assert dataset_linter._read_labels(label_path) == [
        [0, 0.5, 0.5, 0.2, 0.2],
        [1, 0.1, 0.2, 0.3, 0.4],
    ]

    assert dataset_linter._validate_labels([], 1) == ["empty_labels"]
    issues = dataset_linter._validate_labels(
        [
            [-1, -0.1, 1.1, 0.0, -0.1],
            [3, 0.5, 0.5, 0.001, 0.001],
            [0, 0.5, 0.5, 0.95, 0.95],
        ],
        2,
    )
    assert "class_out_of_range:-1" in issues
    assert "class_out_of_range:3" in issues
    assert "x_out_of_bounds:-0.1000" in issues
    assert "y_out_of_bounds:1.1000" in issues
    assert "non_positive_area" in issues
    assert "tiny_box" in issues
    assert "huge_box" in issues


def test_list_files_filters_extensions_and_missing_directories(tmp_path: Path) -> None:
    assert dataset_linter._list_files(tmp_path / "missing", (".jpg",)) == {}
    (tmp_path / "one.JPG").write_bytes(b"image")
    (tmp_path / "two.txt").write_text("label", encoding="utf-8")
    (tmp_path / "nested.jpg").mkdir()
    assert dataset_linter._list_files(tmp_path, (".jpg",)) == {
        "one": tmp_path / "one.JPG"
    }


def test_lint_dataset_writes_missing_pair_geometry_and_histogram_reports(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    output_dir = tmp_path / "lint"
    image_dir.mkdir()
    label_dir.mkdir()
    for name in ("paired", "image-only"):
        cv2.imwrite(str(image_dir / f"{name}.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    (label_dir / "paired.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (label_dir / "label-only.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    config = {
        "dataset_lint": {
            "image_dir": str(image_dir),
            "label_dir": str(label_dir),
            "output_dir": str(output_dir),
        },
        "yolo_training": {"class_names": ["part"]},
    }

    assert dataset_linter.lint_dataset(config) == output_dir
    with (output_dir / "lint.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert ["image-only", "missing_label", str(image_dir / "image-only.jpg|png|...")] in rows
    assert ["label-only", "missing_image", str(label_dir / "label-only.txt")] in rows
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Images scanned: 3" in summary
    assert "part (0): 1" in summary


def test_preview_dataset_renders_pairs_and_marks_decode_failures(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    output_dir = tmp_path / "preview"
    image_dir.mkdir()
    label_dir.mkdir()
    cv2.imwrite(str(image_dir / "valid.png"), np.zeros((12, 16, 3), dtype=np.uint8))
    (image_dir / "broken.png").write_bytes(b"not an image")
    for name in ("valid", "broken"):
        (label_dir / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    config = {
        "aug_preview": {
            "image_dir": str(image_dir),
            "label_dir": str(label_dir),
            "output_dir": str(output_dir),
            "num_samples": 2,
            "cols": 2,
            "seed": 1,
        }
    }

    output_path = dataset_linter.preview_dataset(config)
    assert output_path.is_file()
    assert output_path.name == "preview.png"


def test_preview_dataset_rejects_unpaired_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No paired"):
        dataset_linter.preview_dataset(
            {
                "aug_preview": {
                    "image_dir": str(tmp_path / "images"),
                    "label_dir": str(tmp_path / "labels"),
                    "output_dir": str(tmp_path / "output"),
                }
            }
        )


def test_draw_boxes_handles_title_and_empty_title() -> None:
    axis = MagicMock()
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    dataset_linter._draw_boxes(axis, image, [[0, 0.5, 0.5, 0.2, 0.4]], "sample")
    axis.set_title.assert_called_once()
    axis.add_patch.assert_called_once()

    axis.reset_mock()
    dataset_linter._draw_boxes(axis, image, [], "")
    axis.set_title.assert_not_called()
