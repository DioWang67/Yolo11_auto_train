import os
import time

import pytest

from picture_tool.eval.yolo_evaluator import (
    _candidate_runs,
    _resolve_weights,
)
from picture_tool.train.yolo_trainer import (
    _ensure_data_yaml,
)


def test_candidate_runs_prefers_latest_best_weight(tmp_path):
    project = tmp_path / "runs"
    run1 = project / "train"
    run2 = project / "train2"
    for run in (run1, run2):
        (run / "weights").mkdir(parents=True, exist_ok=True)
        (run / "weights" / "best.pt").write_bytes(b"fake")

    older = time.time() - 60
    newer = time.time()
    os.utime(run1 / "weights" / "best.pt", (older, older))
    os.utime(run2 / "weights" / "best.pt", (newer, newer))

    runs = _candidate_runs(project, "train")
    assert runs and runs[0] == run2


def test_resolve_weights_uses_evaluation_override(tmp_path):
    custom = tmp_path / "custom.pt"
    custom.write_bytes(b"weights")

    config = {
        "yolo_training": {"project": str(tmp_path / "runs"), "name": "train"},
        "yolo_evaluation": {"weights": str(custom)},
    }

    resolved = _resolve_weights(config)
    assert resolved == custom.resolve()


def test_resolve_weights_falls_back_to_latest_run(tmp_path):
    project = tmp_path / "runs"
    run = project / "train"
    (run / "weights").mkdir(parents=True, exist_ok=True)
    best = run / "weights" / "best.pt"
    best.write_bytes(b"weights")

    config = {
        "yolo_training": {"project": str(project), "name": "train"},
        "yolo_evaluation": {},
    }

    resolved = _resolve_weights(config)
    assert resolved == best.resolve()


def test_ensure_data_yaml_creates_file(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "train").mkdir(parents=True)
    (dataset / "val").mkdir(parents=True)
    (dataset / "test").mkdir(parents=True)

    path = _ensure_data_yaml(dataset, ["cls1", "cls2"])
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "names:" in content
