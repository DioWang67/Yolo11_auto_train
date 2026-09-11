"""Champion versus challenger evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

from picture_tool.autotrain import golden
from picture_tool.autotrain.evaluator import (
    COMPLETED,
    INVALIDATED,
    EvaluationError,
    evaluate_candidate,
    golden_data_yaml,
)


class _FakeBox:
    def __init__(self, recall):
        self.mp = 0.85
        self.mr = recall
        self.map50 = recall
        self.map = 0.55
        self.ap_class_index = [0, 1]
        self.p = [0.9, 0.8]
        self.r = [recall, recall - 0.05]
        self.ap50 = [recall, recall - 0.05]
        self.ap = [[0.6], [0.5]]


class _FakeResults:
    def __init__(self, recall):
        self.box = _FakeBox(recall)
        self.names = {0: "Red", 1: "Orange"}
        self.results_dict = {}
        self.confusion_matrix = None


class _Recorder:
    """Injected validator; records the kwargs each model was measured with."""

    def __init__(self, scores: dict[str, float], on_call=None):
        self.scores = scores
        self.calls: list[tuple[str, dict]] = []
        self.on_call = on_call

    def __call__(self, weights, **kwargs):
        self.calls.append((weights, dict(kwargs)))
        if self.on_call is not None:
            self.on_call(weights, kwargs)
        for key, score in self.scores.items():
            if key in weights:
                return _FakeResults(score)
        return _FakeResults(0.5)


@pytest.fixture()
def artifacts(tmp_path):
    champion = tmp_path / "champion.pt"
    champion.write_bytes(b"champion")
    challenger = tmp_path / "challenger.pt"
    challenger.write_bytes(b"challenger")
    data = tmp_path / "data.yaml"
    data.write_text("path: .\ntrain: train\nval: val\n", encoding="utf-8")
    return champion, challenger, data


def _evaluate(artifacts, validator, **overrides):
    champion, challenger, data = artifacts
    options = {
        "champion_weights": champion,
        "challenger_weights": challenger,
        "data_yaml": data,
        "golden_status": golden.GoldenStatus(status=golden.NOT_CONFIGURED),
        "validator": validator,
    }
    options.update(overrides)
    return evaluate_candidate(**options)


# ---------------------------------------------------------------------------
# The comparison itself


def test_both_models_are_measured_on_the_same_split_and_confidence(artifacts):
    recorder = _Recorder({"champion": 0.80, "challenger": 0.85})

    _evaluate(artifacts, recorder, split="test", confidence=0.4, imgsz=640)

    assert len(recorder.calls) == 2
    (_, champion_kwargs), (_, challenger_kwargs) = recorder.calls
    assert champion_kwargs == challenger_kwargs
    assert champion_kwargs["split"] == "test"
    assert champion_kwargs["conf"] == 0.4


def test_the_report_carries_the_comparison(artifacts):
    report = _evaluate(artifacts, _Recorder({"champion": 0.80, "challenger": 0.85}))

    assert report.status == COMPLETED
    assert report.is_valid is True
    assert report.comparison.overall_delta("map50") == pytest.approx(0.05)
    assert report.comparison.recall_delta("Red") == pytest.approx(0.05)


def test_the_report_serialises(artifacts):
    report = _evaluate(artifacts, _Recorder({"champion": 0.80, "challenger": 0.85}))

    payload = report.to_dict()

    assert payload["status"] == COMPLETED
    assert payload["comparison"]["overall"][0]["name"] == "precision"
    assert payload["golden"]["status"] == golden.NOT_CONFIGURED


# ---------------------------------------------------------------------------
# The champion-swap guard


def test_a_champion_swapped_mid_comparison_invalidates_the_result(artifacts):
    champion, _, _ = artifacts

    def swap(weights, kwargs):
        if "challenger" in weights:
            champion.write_bytes(b"a-different-champion")

    report = _evaluate(
        artifacts, _Recorder({"champion": 0.80, "challenger": 0.99}, on_call=swap)
    )

    assert report.status == INVALIDATED
    assert report.is_valid is False
    assert "baseline is not the model that was measured" in report.detail


def test_an_untouched_champion_completes(artifacts):
    report = _evaluate(artifacts, _Recorder({"champion": 0.80, "challenger": 0.85}))

    assert report.status == COMPLETED
    assert report.champion_sha256


def test_evaluation_does_not_modify_the_champion_weights(artifacts):
    champion, _, _ = artifacts
    before = champion.read_bytes()

    _evaluate(artifacts, _Recorder({"champion": 0.80, "challenger": 0.85}))

    assert champion.read_bytes() == before


# ---------------------------------------------------------------------------
# Refusals


def test_a_missing_challenger_is_an_error(artifacts, tmp_path):
    with pytest.raises(EvaluationError, match="Challenger weights not found"):
        _evaluate(artifacts, _Recorder({}), challenger_weights=tmp_path / "nope.pt")


def test_a_missing_champion_is_an_error(artifacts, tmp_path):
    with pytest.raises(EvaluationError, match="without a baseline"):
        _evaluate(artifacts, _Recorder({}), champion_weights=tmp_path / "nope.pt")


def test_a_missing_dataset_descriptor_is_an_error(artifacts, tmp_path):
    with pytest.raises(EvaluationError, match="dataset descriptor not found"):
        _evaluate(artifacts, _Recorder({}), data_yaml=tmp_path / "nope.yaml")


# ---------------------------------------------------------------------------
# The golden comparison


def _registered_golden(tmp_path, *, with_data_yaml: bool):
    root = tmp_path / "golden"
    root.mkdir()
    (root / "a.jpg").write_bytes(b"golden-a")
    if with_data_yaml:
        (root / "data.yaml").write_text("path: .\nval: images\n", encoding="utf-8")
    dataset = golden.register(root, registered_by="engineer")
    return golden.resolve(str(root), dataset.manifest_sha256)


def test_the_golden_set_is_measured_when_it_is_evaluable(artifacts, tmp_path):
    status = _registered_golden(tmp_path, with_data_yaml=True)
    recorder = _Recorder({"champion": 0.80, "challenger": 0.85})

    report = _evaluate(artifacts, recorder, golden_status=status)

    assert report.golden_comparison is not None
    assert len(recorder.calls) == 4  # two models, two datasets
    assert any("golden" in str(kwargs["data"]) for _, kwargs in recorder.calls)


def test_a_golden_set_without_a_data_yaml_is_skipped_not_faked(artifacts, tmp_path):
    """Registering a directory of images does not make it an evaluable dataset."""
    status = _registered_golden(tmp_path, with_data_yaml=False)
    recorder = _Recorder({"champion": 0.80, "challenger": 0.85})

    report = _evaluate(artifacts, recorder, golden_status=status)

    assert report.golden_comparison is None
    assert len(recorder.calls) == 2
    assert golden_data_yaml(status) is None


def test_a_failing_golden_status_is_never_measured(artifacts, tmp_path):
    status = golden.GoldenStatus(status=golden.MISMATCH, detail="changed")
    recorder = _Recorder({"champion": 0.80, "challenger": 0.85})

    report = _evaluate(artifacts, recorder, golden_status=status)

    assert report.golden_comparison is None
    assert report.golden.status == golden.MISMATCH


def test_a_golden_evaluation_failure_does_not_lose_the_main_comparison(
    artifacts, tmp_path
):
    status = _registered_golden(tmp_path, with_data_yaml=True)

    def explode(weights, kwargs):
        if Path(str(kwargs.get("data", ""))).parent.name == "golden":
            raise RuntimeError("golden eval blew up")

    report = _evaluate(
        artifacts,
        _Recorder({"champion": 0.80, "challenger": 0.85}, on_call=explode),
        golden_status=status,
    )

    assert report.status == COMPLETED
    assert report.golden_comparison is None
    assert report.comparison.overall_delta("map50") == pytest.approx(0.05)


def test_ultralytics_is_never_loaded_under_pytest(artifacts):
    """The repository's own guard against torch DLL crashes on Windows."""
    with pytest.raises(EvaluationError, match="Refusing to load ultralytics"):
        _evaluate(artifacts, None)
