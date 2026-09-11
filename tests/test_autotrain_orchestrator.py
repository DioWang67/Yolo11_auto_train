"""The training cycle, end to end.

Everything is faked below the pipeline boundary: no YOLO is trained and no
model is loaded. What these tests actually pin down is that the cycle
produces the promised report, survives failure, resumes, and --- above all
--- never writes anything into the production inference project.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from picture_tool.autotrain import AutoTrainDisabledError, golden
from picture_tool.autotrain.candidate_pool import CandidatePool, CandidateSample
from picture_tool.autotrain.config import AutoTrainConfig
from picture_tool.autotrain.orchestrator import (
    BLOCKED,
    COMPLETED,
    STEP_DATASET,
    STEP_TRAIN,
    CycleError,
    TrainingCycle,
    run_training_cycle,
)
from picture_tool.autotrain.paths import AutoTrainPaths
from picture_tool.autotrain.promotion import PROMOTION_CANDIDATE, REJECTED
from picture_tool.autotrain.registry import CandidateRegistry
from picture_tool.workspace_paths import WorkspacePaths

PRODUCT = "Cable1"
AREA = "A"
CLASSES = ["Red", "Orange"]


# ---------------------------------------------------------------------------
# Fixtures


@pytest.fixture()
def paths(tmp_path) -> AutoTrainPaths:
    """A private paired workspace per test.

    conftest's isolated workspace is created once per session, so pools,
    dataset versions and registries would accumulate across tests and the
    version numbers alone would make assertions order-dependent.
    """
    root = tmp_path / "workspace"
    training = root / "Yolo11_auto_train"
    inference = root / "yolo11_inference"
    for relative in (
        training / "data",
        inference / "models",
        root / "station_data" / "yolo11_inference",
        root / "Result",
        root / "release_artifacts" / "yolo11_inference",
    ):
        relative.mkdir(parents=True, exist_ok=True)
    workspace = WorkspacePaths(
        workspace_root=root.resolve(),
        training_project=training.resolve(),
        inference_project=inference.resolve(),
        training_data=(training / "data").resolve(),
        inference_models=(inference / "models").resolve(),
        station_data=(root / "station_data" / "yolo11_inference").resolve(),
        inference_results=(root / "Result").resolve(),
        inference_artifacts=(root / "release_artifacts" / "yolo11_inference").resolve(),
        manifest_path=None,
    )
    return AutoTrainPaths.from_workspace(workspace)


@pytest.fixture()
def champion(paths) -> Path:
    """A deployed champion in the (isolated) production models tree."""
    model_dir = paths.production_model_dir(PRODUCT, AREA)
    model_dir.mkdir(parents=True, exist_ok=True)
    weights = model_dir / "weights"
    weights.mkdir(exist_ok=True)
    (weights / "best.training.pt").write_bytes(b"champion-weights")
    (model_dir / "deployment_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "deployed_version": "Cable1_A_v1.2.0",
                "deployed_weight_file": "weights/best.onnx",
                "training_weight_file": "weights/best.training.pt",
                "weight_sha256": "abc",
                "evaluation_metrics": {"map50": 0.80},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "config.yaml").write_text(
        yaml.safe_dump({"class_names": CLASSES}), encoding="utf-8"
    )
    return model_dir


@pytest.fixture()
def config(tmp_path) -> AutoTrainConfig:
    path = tmp_path / "autonomous_training.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "autonomous_training": {
                    "enabled": True,
                    "station": {"product": PRODUCT, "area": AREA},
                    "collector": {"lookback_days": 7, "compute_image_quality": False},
                    "selectors": {"low_confidence": {"enabled": True}},
                    "training": {"epochs": 1, "imgsz": 64, "batch": 1, "device": "cpu"},
                    "promotion": {"require_golden_pass": False},
                }
            }
        ),
        encoding="utf-8",
    )
    return AutoTrainConfig.load(path)


def _seed_verified_pool(paths, tmp_path, count=2) -> CandidatePool:
    """A pool whose candidates already carry human-verified labels."""
    pool = CandidatePool(paths.pool_dir(PRODUCT, AREA))
    samples = []
    for index in range(count):
        source = tmp_path / "src" / f"img{index}.jpg"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"pixels-{index}".encode("utf-8"))
        samples.append(
            CandidateSample(
                sample_id=f"sample{index}",
                inspection_id=f"insp{index}",
                product=PRODUCT,
                area=AREA,
                selector="low_confidence",
                selected_reason="uncertain",
                score=0.3,
                source_model="best.onnx",
                source_model_version="1.2.0",
                selected_at=f"2026-09-10T00:0{index}:00+00:00",
                production_timestamp="",
                image_path="",
                source_image_path=str(source),
            )
        )
    pool.add(samples)
    pool.labels_dir.mkdir(parents=True, exist_ok=True)
    for sample in pool.load():
        label = pool.labels_dir / f"{sample.sample_id}.txt"
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        pool.update_label_state(sample.sample_id, "VERIFIED", label_path=str(label))
    return pool


class _FakeRunner:
    """Writes what a real pipeline run would leave behind."""

    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls: list[list[str]] = []

    def __call__(self, tasks, config, logger, args):
        self.calls.append(list(tasks))
        if self.fail:
            raise RuntimeError("training blew up")
        project = Path(config["yolo_training"]["project"])
        weights = project / "candidate" / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        (weights / "best.pt").write_bytes(b"challenger-weights")
        split = Path(config["yolo_training"]["dataset_dir"])
        split.mkdir(parents=True, exist_ok=True)
        (split / "data.yaml").write_text(
            yaml.safe_dump({"path": str(split), "val": "val/images", "names": CLASSES}),
            encoding="utf-8",
        )


class _FakeBox:
    def __init__(self, score):
        self.mp = score
        self.mr = score
        self.map50 = score
        self.map = score - 0.2
        self.ap_class_index = [0, 1]
        self.p = [score, score]
        self.r = [score, score]
        self.ap50 = [score, score]
        self.ap = [[score], [score]]


class _FakeResults:
    def __init__(self, score):
        self.box = _FakeBox(score)
        self.names = {0: "Red", 1: "Orange"}
        self.results_dict = {}
        self.confusion_matrix = None


def _validator(champion_dir, champion_score=0.80, challenger_score=0.85):
    """Score by which weight file is being measured.

    Matched on the resolved champion path rather than a substring: the
    champion lives at an arbitrary temp path, and a loose match silently
    scored both models the same, which made an improvement test pass without
    there being an improvement.
    """
    champion_weights = (champion_dir / "weights" / "best.training.pt").resolve()

    def run(weights, **kwargs):
        is_champion = Path(weights).resolve() == champion_weights
        return _FakeResults(champion_score if is_champion else challenger_score)

    return run


# ---------------------------------------------------------------------------
# The feature flag


def test_a_cycle_cannot_start_while_the_feature_is_disabled(paths, tmp_path):
    disabled = AutoTrainConfig.disabled()

    with pytest.raises(AutoTrainDisabledError):
        TrainingCycle(disabled, paths)
    with pytest.raises(AutoTrainDisabledError):
        run_training_cycle(disabled, paths)

    assert not paths.cycles_root.exists()


# ---------------------------------------------------------------------------
# The happy path


def test_a_full_cycle_recommends_an_improved_challenger(
    paths, config, champion, tmp_path
):
    _seed_verified_pool(paths, tmp_path)
    runner = _FakeRunner()

    result = run_training_cycle(
        config,
        paths,
        cycle_id="cycle_test_001",
        runner=runner,
        validator=_validator(champion, 0.80, 0.85),
    )

    assert result.status == COMPLETED
    assert result.decision.decision == PROMOTION_CANDIDATE
    assert result.deployed is False
    assert "deploy" not in runner.calls[0]


def test_a_regressed_challenger_is_rejected(paths, config, champion, tmp_path):
    _seed_verified_pool(paths, tmp_path)

    result = run_training_cycle(
        config,
        paths,
        cycle_id="cycle_test_regress",
        runner=_FakeRunner(),
        validator=_validator(champion, 0.80, 0.60),
    )

    assert result.decision.decision == REJECTED
    assert result.deployed is False


def test_the_report_has_the_requested_shape(paths, config, champion, tmp_path):
    _seed_verified_pool(paths, tmp_path)

    result = run_training_cycle(
        config,
        paths,
        cycle_id="cycle_test_report",
        runner=_FakeRunner(),
        validator=_validator(champion, 0.80, 0.85),
    )

    text = result.report_text
    assert "Training Cycle: cycle_test_report" in text
    assert "Champion:" in text
    assert "Challenger:" in text
    assert "Dataset:     dataset_v001" in text
    assert "Red Recall" in text
    assert "Golden Dataset:" in text
    assert f"Decision:        {PROMOTION_CANDIDATE}" in text
    assert "Nothing has been deployed" in text


def test_the_json_report_records_that_nothing_was_deployed(
    paths, config, champion, tmp_path
):
    _seed_verified_pool(paths, tmp_path)

    result = run_training_cycle(
        config,
        paths,
        cycle_id="cycle_test_json",
        runner=_FakeRunner(),
        validator=_validator(champion),
    )

    payload = json.loads(
        (result.report_path.parent / "report.json").read_text(encoding="utf-8")
    )
    assert payload["deployed"] is False
    assert payload["dataset_version"] == "dataset_v001"
    assert payload["decision"]["decision"] == PROMOTION_CANDIDATE
    assert set(payload["steps"]) >= {"collect", "select", "dataset", "train"}


def test_the_challenger_is_registered_as_a_candidate_not_production(
    paths, config, champion, tmp_path
):
    _seed_verified_pool(paths, tmp_path)

    run_training_cycle(
        config,
        paths,
        cycle_id="cycle_test_registry",
        runner=_FakeRunner(),
        validator=_validator(champion),
    )

    registry = CandidateRegistry(paths.candidates_root / PRODUCT / AREA)
    models = registry.list_models()
    assert len(models) == 1
    assert models[0].status == PROMOTION_CANDIDATE
    assert models[0].dataset_version == "dataset_v001"


# ---------------------------------------------------------------------------
# Production is never touched


def test_a_full_cycle_leaves_the_production_tree_byte_identical(
    paths, config, champion, tmp_path
):
    _seed_verified_pool(paths, tmp_path)
    before = {
        path: path.read_bytes()
        for path in sorted(paths.workspace.inference_project.rglob("*"))
        if path.is_file()
    }

    run_training_cycle(
        config,
        paths,
        cycle_id="cycle_test_untouched",
        runner=_FakeRunner(),
        validator=_validator(champion),
    )

    after = {
        path: path.read_bytes()
        for path in sorted(paths.workspace.inference_project.rglob("*"))
        if path.is_file()
    }
    assert after == before


def test_candidate_weights_never_land_under_the_production_models_tree(
    paths, config, champion, tmp_path
):
    _seed_verified_pool(paths, tmp_path)

    run_training_cycle(
        config,
        paths,
        cycle_id="cycle_test_paths",
        runner=_FakeRunner(),
        validator=_validator(champion),
    )

    weights = list(paths.candidates_root.rglob("*.pt"))
    assert weights
    for path in weights:
        assert not path.is_relative_to(paths.workspace.inference_models)
        assert not path.is_relative_to(paths.workspace.inference_project)


# ---------------------------------------------------------------------------
# Blocking conditions are results, not crashes


def test_a_cycle_with_nothing_labelled_blocks_and_says_so(
    paths, config, champion, tmp_path
):
    """Waiting for annotation is the healthy state of a young cycle."""
    result = run_training_cycle(
        config, paths, cycle_id="cycle_test_unlabelled", runner=_FakeRunner()
    )

    assert result.status == BLOCKED
    assert "human-verified label" in result.blocked_reason
    assert result.state.step(STEP_DATASET)["status"] == "BLOCKED"
    assert result.report_path.is_file()


def test_a_bare_station_blocks_before_cutting_a_dataset_version(
    paths, config, tmp_path
):
    """Nothing deployed and nothing collected means no class contract.

    It stops at the dataset step rather than at training because a dataset
    version is immutable: one cut here would permanently record class ids
    whose meaning nothing states.
    """
    _seed_verified_pool(paths, tmp_path)

    result = run_training_cycle(
        config, paths, cycle_id="cycle_test_noschema", runner=_FakeRunner()
    )

    assert result.status == BLOCKED
    assert "No class schema is available" in result.blocked_reason
    assert result.state.step(STEP_DATASET)["status"] == "BLOCKED"


def test_a_station_with_no_champion_blocks_before_training(paths, config, tmp_path):
    """With the contract known but nothing deployed, training is what blocks."""
    model_dir = paths.production_model_dir(PRODUCT, AREA)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.yaml").write_text(
        yaml.safe_dump({"class_names": CLASSES}), encoding="utf-8"
    )
    _seed_verified_pool(paths, tmp_path)

    result = run_training_cycle(
        config, paths, cycle_id="cycle_test_nochampion", runner=_FakeRunner()
    )

    assert result.status == BLOCKED
    assert "No champion is deployed" in result.blocked_reason
    assert result.state.step(STEP_TRAIN)["status"] == "BLOCKED"


# ---------------------------------------------------------------------------
# Failure containment and resume


def test_a_training_failure_is_recorded_and_leaves_production_untouched(
    paths, config, champion, tmp_path
):
    _seed_verified_pool(paths, tmp_path)
    before = {
        path: path.read_bytes()
        for path in sorted(paths.workspace.inference_project.rglob("*"))
        if path.is_file()
    }

    with pytest.raises(CycleError, match="failed"):
        run_training_cycle(
            config,
            paths,
            cycle_id="cycle_test_fail",
            runner=_FakeRunner(fail=True),
        )

    after = {
        path: path.read_bytes()
        for path in sorted(paths.workspace.inference_project.rglob("*"))
        if path.is_file()
    }
    assert after == before

    state = json.loads(
        (paths.cycle_dir("cycle_test_fail") / "cycle.json").read_text(encoding="utf-8")
    )
    assert state["steps"]["train"]["status"] == "FAILED"
    assert (paths.cycle_dir("cycle_test_fail") / "report.md").is_file()


def test_a_failed_cycle_marks_its_candidate_failed_not_rejected(
    paths, config, champion, tmp_path
):
    """A crash and a measured regression must not look the same afterwards."""
    _seed_verified_pool(paths, tmp_path)

    with pytest.raises(CycleError):
        run_training_cycle(
            config, paths, cycle_id="cycle_test_fail2", runner=_FakeRunner(fail=True)
        )

    models = CandidateRegistry(paths.candidates_root / PRODUCT / AREA).list_models()
    assert [model.status for model in models] == ["FAILED"]


def test_a_resumed_cycle_skips_the_steps_it_already_finished(
    paths, config, champion, tmp_path
):
    _seed_verified_pool(paths, tmp_path)
    first = TrainingCycle(config, paths, cycle_id="cycle_test_resume")
    first.collect()
    first.select()
    collected_at = first.state.step("collect")["finished_at"]

    result = run_training_cycle(
        config,
        paths,
        cycle_id="cycle_test_resume",
        runner=_FakeRunner(),
        validator=_validator(champion),
    )

    assert result.status == COMPLETED
    assert result.state.step("collect")["finished_at"] == collected_at


def test_steps_can_be_run_individually(paths, config, champion, tmp_path):
    _seed_verified_pool(paths, tmp_path)
    cycle = TrainingCycle(
        config, paths, cycle_id="cycle_test_steps", runner=_FakeRunner(),
        validator=_validator(champion),
    )

    cycle.collect()
    cycle.select()
    version = cycle.build_dataset()
    model_version = cycle.train()
    report = cycle.evaluate()
    decision = cycle.decide(report)

    assert version == "dataset_v001"
    assert model_version.endswith("_candidate")
    assert decision.decision == PROMOTION_CANDIDATE
    assert all(cycle.state.is_done(step) for step in ("collect", "select", "dataset"))


# ---------------------------------------------------------------------------
# The golden gate inside a real cycle


def test_an_unconfigured_golden_set_blocks_promotion_when_required(
    paths, champion, tmp_path
):
    config_path = tmp_path / "autonomous_training.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "autonomous_training": {
                    "enabled": True,
                    "station": {"product": PRODUCT, "area": AREA},
                    "collector": {"compute_image_quality": False},
                    "selectors": {"low_confidence": {"enabled": True}},
                    "training": {"epochs": 1, "imgsz": 64, "batch": 1},
                    "promotion": {"require_golden_pass": True},
                }
            }
        ),
        encoding="utf-8",
    )
    strict = AutoTrainConfig.load(config_path)
    _seed_verified_pool(paths, tmp_path)

    result = run_training_cycle(
        strict,
        paths,
        cycle_id="cycle_test_golden",
        runner=_FakeRunner(),
        validator=_validator(champion, 0.80, 0.99),
    )

    assert result.decision.decision == REJECTED
    assert result.decision.golden_status == golden.NOT_CONFIGURED
    assert any("golden dataset status" in r for r in result.decision.reasons)
