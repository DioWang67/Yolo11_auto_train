"""Champion versus challenger evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from picture_tool.autotrain import golden
from picture_tool.autotrain.class_schema import normalize_class_names
from picture_tool.autotrain.evaluator import (
    COMPLETED,
    DATA_YAML_NAME,
    GROUP_FAILED,
    GROUP_INSUFFICIENT,
    GROUP_MEASURED,
    GROUP_NO_LABELS,
    INVALIDATED,
    EvaluationError,
    evaluate_candidate,
    golden_data_yaml,
)
from picture_tool.autotrain.golden_candidates import HARD_CASE, REPRESENTATIVE

#: The real Cable1/A contract. A golden set is registered against one,
#: so that a later evaluation can refuse a model whose ids mean something
#: else rather than silently comparing different classes.
SCHEMA = normalize_class_names(
    ["Black", "Green", "Orange", "Red", "Yellow"], source="test"
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
    dataset = golden.register(root, class_schema=SCHEMA, registered_by="engineer")
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


# ---------------------------------------------------------------------------
# The representative / hard_case split
#
# The fixture below mirrors a real YOLO evaluation set --- images/ and labels/
# beside a data.yaml carrying the class names --- rather than a shape invented
# here. A fake runner will happily score a layout ultralytics could never
# load, which is how three defects survived a green suite once already.


def _split_golden(
    tmp_path,
    *,
    hard: int = 12,
    representative: int = 12,
    flat: bool = False,
    data_yaml: str | None = None,
):
    import hashlib

    root = tmp_path / "golden"
    images_dir = root if flat else root / "images" / "val"
    labels_dir = root / "labels" / "val"
    images_dir.mkdir(parents=True, exist_ok=True)
    if not flat:
        labels_dir.mkdir(parents=True, exist_ok=True)

    groups = {}
    for label, count in ((HARD_CASE, hard), (REPRESENTATIVE, representative)):
        for index in range(count):
            payload = f"{label}-{index}".encode("utf-8")
            name = f"{label}_{index:02d}"
            (images_dir / f"{name}.jpg").write_bytes(payload)
            if not flat:
                (labels_dir / f"{name}.txt").write_text(
                    "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                )
            groups[hashlib.sha256(payload).hexdigest()] = label

    if data_yaml != "":
        (root / "data.yaml").write_text(
            data_yaml
            if data_yaml is not None
            else (
                "path: .\ntrain: images/train\nval: images/val\n"
                "names:\n  0: Black\n  1: Green\n  2: Orange\n"
                "  3: Red\n  4: Yellow\n"
            ),
            encoding="utf-8",
        )
    dataset = golden.register(
        root, class_schema=SCHEMA, registered_by="engineer", groups=groups
    )
    return golden.resolve(str(root), dataset.manifest_sha256)


def _group_calls(recorder):
    """Validator calls whose descriptor is a generated subset, not the set."""
    return [
        (weights, kwargs)
        for weights, kwargs in recorder.calls
        if Path(str(kwargs["data"])).name != DATA_YAML_NAME
    ]


def test_each_declared_group_is_measured_separately(artifacts, tmp_path):
    status = _split_golden(tmp_path)
    recorder = _Recorder({"champion": 0.80, "challenger": 0.85})

    report = _evaluate(artifacts, recorder, golden_status=status)

    measured = {item.group: item for item in report.golden_groups}
    assert set(measured) == {HARD_CASE, REPRESENTATIVE}
    assert all(item.is_measured for item in measured.values())
    assert measured[HARD_CASE].sample_count == 12
    # two models x (main split + golden overall + two groups)
    assert len(recorder.calls) == 8
    assert len(_group_calls(recorder)) == 4


def test_a_group_is_scored_only_on_its_own_images(artifacts, tmp_path):
    """The subset is a list of paths; the golden set is never rearranged."""
    status = _split_golden(tmp_path)
    seen: dict[str, list[str]] = {}

    def capture(weights, kwargs):
        descriptor = Path(str(kwargs["data"]))
        if descriptor.name == DATA_YAML_NAME:
            return
        payload = yaml.safe_load(descriptor.read_text(encoding="utf-8"))
        seen[descriptor.stem] = (
            Path(payload["val"]).read_text(encoding="utf-8").split()
        )

    _evaluate(
        artifacts,
        _Recorder({"champion": 0.80, "challenger": 0.85}, on_call=capture),
        golden_status=status,
    )

    assert set(seen) == {HARD_CASE, REPRESENTATIVE}
    assert len(seen[HARD_CASE]) == 12
    assert all(HARD_CASE in name for name in seen[HARD_CASE])
    assert not any(REPRESENTATIVE in name for name in seen[HARD_CASE])


def test_the_subset_inherits_the_class_contract_verbatim(artifacts, tmp_path):
    """Re-deriving names here would be a second place for it to be wrong."""
    status = _split_golden(tmp_path)
    captured: list[dict] = []

    def capture(weights, kwargs):
        descriptor = Path(str(kwargs["data"]))
        if descriptor.name != DATA_YAML_NAME:
            captured.append(
                yaml.safe_load(descriptor.read_text(encoding="utf-8"))
            )

    _evaluate(
        artifacts,
        _Recorder({"champion": 0.80, "challenger": 0.85}, on_call=capture),
        golden_status=status,
    )

    assert captured
    for payload in captured:
        assert payload["names"] == {
            0: "Black",
            1: "Green",
            2: "Orange",
            3: "Red",
            4: "Yellow",
        }


def test_the_subset_keeps_a_train_key_pointing_at_the_subset(artifacts, tmp_path):
    """Found by the real run, invisible to a fake one.

    ultralytics' ``check_det_dataset`` raises ``SyntaxError`` unless both
    ``train`` and ``val`` are present, so an earlier version that dropped
    ``train`` failed every group against the real library while every test
    here passed --- a fake validator never parses this file. ``train`` points
    at the same subset list, so the descriptor still cannot name the real
    training images.
    """
    status = _split_golden(tmp_path)
    captured: list[dict] = []

    def capture(weights, kwargs):
        descriptor = Path(str(kwargs["data"]))
        if descriptor.name != DATA_YAML_NAME:
            captured.append(
                yaml.safe_load(descriptor.read_text(encoding="utf-8"))
            )

    _evaluate(
        artifacts,
        _Recorder({"champion": 0.80, "challenger": 0.85}, on_call=capture),
        golden_status=status,
    )

    assert captured
    for payload in captured:
        assert payload["train"] == payload["val"]
        assert Path(payload["val"]).suffix == ".txt"
        # Never the original split paths from the golden descriptor.
        assert payload["train"] != "images/train"
        assert "test" not in payload


def test_a_group_below_the_floor_is_not_given_a_number(artifacts, tmp_path):
    status = _split_golden(tmp_path, hard=4, representative=4)
    recorder = _Recorder({"champion": 0.80, "challenger": 0.85})

    report = _evaluate(
        artifacts, recorder, golden_status=status, min_group_samples=10
    )

    assert [item.status for item in report.golden_groups] == [
        GROUP_INSUFFICIENT,
        GROUP_INSUFFICIENT,
    ]
    assert all(item.comparison is None for item in report.golden_groups)
    assert _group_calls(recorder) == []


def test_a_group_whose_labels_cannot_be_located_is_not_scored(artifacts, tmp_path):
    """A flat directory has no images/ segment, so no ground truth resolves.

    Scoring it anyway would count every object as missed and publish that as
    a recall collapse.
    """
    status = _split_golden(tmp_path, flat=True)
    recorder = _Recorder({"champion": 0.80, "challenger": 0.85})

    report = _evaluate(artifacts, recorder, golden_status=status)

    assert {item.status for item in report.golden_groups} == {GROUP_NO_LABELS}
    assert _group_calls(recorder) == []


def test_a_failing_group_does_not_lose_the_main_comparison(artifacts, tmp_path):
    status = _split_golden(tmp_path)

    def explode(weights, kwargs):
        if Path(str(kwargs["data"])).name != DATA_YAML_NAME:
            raise RuntimeError("group eval blew up")

    report = _evaluate(
        artifacts,
        _Recorder({"champion": 0.80, "challenger": 0.85}, on_call=explode),
        golden_status=status,
    )

    assert report.status == COMPLETED
    assert report.comparison.overall_delta("map50") == pytest.approx(0.05)
    assert {item.status for item in report.golden_groups} == {GROUP_FAILED}
    assert report.group(HARD_CASE).detail


def test_an_unsplit_golden_set_produces_no_group_results(artifacts, tmp_path):
    status = _registered_golden(tmp_path, with_data_yaml=True)
    recorder = _Recorder({"champion": 0.80, "challenger": 0.85})

    report = _evaluate(artifacts, recorder, golden_status=status)

    assert report.golden_groups == ()
    assert len(recorder.calls) == 4


def test_building_a_subset_writes_nothing_into_the_golden_directory(
    artifacts, tmp_path
):
    """The subset is a list of paths, not a rearrangement of the set.

    Scoped to *this code*: a real ultralytics run additionally drops its own
    ``labels/<split>.cache`` beside the labels, which no injected validator
    reproduces. That is the library's, not ours --- see the note in
    :mod:`picture_tool.autotrain.golden`.
    """
    import hashlib

    status = _split_golden(tmp_path)
    root = status.dataset.root

    def snapshot():
        return {
            path.relative_to(root).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    before = snapshot()
    _evaluate(
        artifacts,
        _Recorder({"champion": 0.80, "challenger": 0.85}),
        golden_status=status,
    )

    assert snapshot() == before


def test_a_split_set_without_a_data_yaml_reports_every_group_as_failed(
    artifacts, tmp_path
):
    """Declaring a split does not make the directory evaluable."""
    status = _split_golden(tmp_path, data_yaml="")
    recorder = _Recorder({"champion": 0.80, "challenger": 0.85})

    report = _evaluate(artifacts, recorder, golden_status=status)

    assert {item.status for item in report.golden_groups} == {GROUP_FAILED}
    assert {item.group for item in report.golden_groups} == {
        HARD_CASE,
        REPRESENTATIVE,
    }
    assert "no data.yaml" in report.group(HARD_CASE).detail
    assert _group_calls(recorder) == []


def test_an_unparseable_descriptor_fails_the_groups_rather_than_the_cycle(
    artifacts, tmp_path
):
    status = _split_golden(tmp_path, data_yaml="names: [1,\n")
    recorder = _Recorder({"champion": 0.80, "challenger": 0.85})

    report = _evaluate(artifacts, recorder, golden_status=status)

    assert report.status == COMPLETED
    assert {item.status for item in report.golden_groups} == {GROUP_FAILED}
    assert "could not be parsed" in report.group(HARD_CASE).detail


def test_images_that_vanish_after_the_check_are_reported_not_scored(
    artifacts, tmp_path
):
    """The set is verified once, then validated; files can go in between."""
    status = _split_golden(tmp_path)
    root = status.dataset.root
    for sample_id in status.dataset.sample_ids_in_group(HARD_CASE):
        (root / status.dataset.images[sample_id]).unlink()

    report = _evaluate(
        artifacts,
        _Recorder({"champion": 0.80, "challenger": 0.85}),
        golden_status=status,
    )

    assert report.group(HARD_CASE).status == GROUP_FAILED
    assert "are on disk" in report.group(HARD_CASE).detail
    # The intact group is still measured; one missing group is not a reason
    # to discard the other.
    assert report.group(REPRESENTATIVE).status == GROUP_MEASURED


def test_the_groups_serialise_into_the_report(artifacts, tmp_path):
    status = _split_golden(tmp_path)

    payload = _evaluate(
        artifacts,
        _Recorder({"champion": 0.80, "challenger": 0.85}),
        golden_status=status,
    ).to_dict()

    groups = {item["group"]: item for item in payload["golden_groups"]}
    assert groups[HARD_CASE]["status"] == GROUP_MEASURED
    assert groups[HARD_CASE]["sample_count"] == 12
    assert groups[HARD_CASE]["comparison"]["overall"][0]["name"] == "precision"
