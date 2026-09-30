"""Challenger training.

No test here trains a real YOLO model: the pipeline runner is injected, so
these exercise the wiring, the refusals and the artifact placement, which is
where the safety properties actually live.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import yaml

from picture_tool.autotrain.dataset_versions import DatasetVersionStore, LabelledSample
from picture_tool.autotrain.trainer import (
    CANDIDATE_TASKS,
    FORBIDDEN_TASKS,
    CandidateTrainingError,
    TrainingResult,
    assert_no_forbidden_tasks,
    build_candidate_config,
    find_interrupted_run,
    prepare_work_dir,
    resolve_device,
    train_candidate,
)
from picture_tool.autotrain.class_schema import (
    ClassSchemaError,
    normalize_class_names,
)

#: The real Cable1/A contract, in the order the station's champion and
#: every handoff manifest record it.
SCHEMA = normalize_class_names(
    ["Black", "Green", "Orange", "Red", "Yellow"], source="test"
)

BASE_CONFIG = {
    "pipeline": {"tasks": [{"name": "yolo_train", "enabled": True}]},
    "yolo_augmentation": {
        "input": {"image_dir": "./data/project/raw/images"},
        "augmentation": {"operations": {"hue": {"range": [0.9, 1.1]}}},
    },
    "train_test_split": {"input": {}, "output": {}},
    "yolo_training": {
        "dataset_dir": "./data/project/split",
        "model": "./models/yolo11n.pt",
        "device": "cpu",
        "deploy": {"enabled": True, "inference_models_dir": "/production/models"},
        "export_onnx": {"enabled": True},
        # The packaged default, pointing at a directory a challenger cycle
        # never builds.
        "position_validation": {
            "enabled": True,
            "sample_dir": "./data/project/split/test/images",
        },
    },
    "yolo_evaluation": {"gate": {"enabled": True, "min_map50": 0.8}},
}


def _dataset_version(tmp_path: Path, ids=("a", "b")):
    store = DatasetVersionStore(tmp_path / "datasets", product="Cable1", area="A")
    samples = []
    for sample_id in ids:
        image = tmp_path / "src" / f"{sample_id}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(sample_id.encode("utf-8"))
        label = tmp_path / "src" / f"{sample_id}.txt"
        label.write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")
        samples.append(
            LabelledSample(sample_id=sample_id, image_path=image, label_path=label)
        )
    return store.create(
        samples,
        source="pool",
        label_source="operator",
        class_schema=SCHEMA,
    )


class _FakeRunner:
    """Stands in for run_pipeline, writing the weights a real run would."""

    def __init__(self, *, fail: bool = False, run_name: str = "candidate"):
        self.fail = fail
        self.run_name = run_name
        self.calls: list[tuple] = []

    def __call__(self, tasks, config, logger, args):
        self.calls.append((list(tasks), config, args))
        if self.fail:
            raise RuntimeError("ultralytics exploded")
        project = Path(config["yolo_training"]["project"])
        weights = project / self.run_name / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        (weights / "best.pt").write_bytes(b"trained-weights")


def _train(tmp_path, runner=None, **overrides):
    version = overrides.pop("dataset_version", None) or _dataset_version(tmp_path)
    options = {
        "dataset_version": version,
        "base_config": BASE_CONFIG,
        "candidate_dir": tmp_path / "candidates" / "v1.3.0",
        "work_dir": tmp_path / "cycle" / "work",
        "model_version": "Cable1_A_v1.3.0-candidate",
        "base_model": str(tmp_path / "champion.pt"),
        "class_schema": SCHEMA,
        "runner": runner or _FakeRunner(),
    }
    options.update(overrides)
    return train_candidate(**options)


# ---------------------------------------------------------------------------
# Deploy can never happen


def test_publishing_tasks_are_refused_outright():
    for task in sorted(FORBIDDEN_TASKS):
        with pytest.raises(CandidateTrainingError, match="never run publishing"):
            assert_no_forbidden_tasks(["yolo_train", task])


def test_the_default_task_list_contains_no_publishing_task():
    assert set(CANDIDATE_TASKS) & FORBIDDEN_TASKS == set()
    assert "deploy" not in CANDIDATE_TASKS


def test_training_refuses_a_task_list_containing_deploy(tmp_path):
    with pytest.raises(CandidateTrainingError, match="never run publishing"):
        _train(tmp_path, tasks=["yolo_train", "deploy"])


def test_the_runner_never_receives_deploy(tmp_path):
    runner = _FakeRunner()

    _train(tmp_path, runner=runner)

    tasks, _, args = runner.calls[0]
    assert "deploy" not in tasks
    assert set(args.exclude_tasks) >= FORBIDDEN_TASKS


def test_the_generated_config_disables_every_publishing_block(tmp_path):
    runner = _FakeRunner()

    _train(tmp_path, runner=runner)

    _, config, _ = runner.calls[0]
    assert config["yolo_training"]["deploy"]["enabled"] is False
    assert config["yolo_training"]["export_onnx"]["enabled"] is False
    assert config["autotrain"]["deploy_forbidden"] is True


def test_nothing_is_written_outside_the_candidate_and_work_directories(tmp_path):
    production = tmp_path / "production" / "models"
    production.mkdir(parents=True)

    _train(tmp_path)

    assert list(production.rglob("*")) == []


# ---------------------------------------------------------------------------
# Config derivation


def test_the_station_config_is_preserved_and_redirected(tmp_path):
    version = _dataset_version(tmp_path)

    config = build_candidate_config(
        BASE_CONFIG,
        dataset_version=version,
        work_dir=tmp_path / "work",
        run_project=tmp_path / "work" / "runs",
        run_name="candidate",
        base_model="champion.pt",
        class_names=["Red"],
        epochs=7,
        imgsz=512,
        batch=2,
        device="cpu",
    )

    assert config["pipeline"] == BASE_CONFIG["pipeline"]  # untouched
    assert config["yolo_training"]["epochs"] == 7
    assert config["yolo_training"]["model"] == "champion.pt"
    assert str(tmp_path / "work") in config["yolo_training"]["dataset_dir"]
    assert str(tmp_path / "work") in config["yolo_augmentation"]["input"]["image_dir"]


def test_the_source_config_is_not_mutated(tmp_path):
    before = yaml.safe_dump(BASE_CONFIG, sort_keys=True)

    build_candidate_config(
        BASE_CONFIG,
        dataset_version=_dataset_version(tmp_path),
        work_dir=tmp_path / "work",
        run_project=tmp_path / "runs",
        run_name="candidate",
        base_model="champion.pt",
        class_names=["Red"],
        epochs=1,
        imgsz=64,
        batch=1,
        device="cpu",
    )

    assert yaml.safe_dump(BASE_CONFIG, sort_keys=True) == before


def test_colour_destroying_augmentation_is_forced_off(tmp_path):
    """Wire colour is the label here; hue shifts would teach the wrong thing."""
    config = build_candidate_config(
        BASE_CONFIG,
        dataset_version=_dataset_version(tmp_path),
        work_dir=tmp_path / "work",
        run_project=tmp_path / "runs",
        run_name="candidate",
        base_model="champion.pt",
        class_names=["Red"],
        epochs=1,
        imgsz=64,
        batch=1,
        device="cpu",
    )

    operations = config["yolo_augmentation"]["augmentation"]["operations"]
    # Removed, not narrowed: the augmentor reads a hue range as a shift, so
    # the [1, 1] this test once asserted shifted hue by one step and switched
    # on saturation/value jitter with it.
    assert "hue" not in operations
    assert operations["flip"]["probability"] == 0.0
    assert operations["perspective"]["scale"] == [0.0, 0.0]


def test_the_challenger_trains_on_the_champions_recipe(tmp_path):
    """Measured against the champion, so trained the way the champion was.

    Before this shared the operator recipe, the challenger ran on Ultralytics'
    generic defaults: lr0 0.01, nothing frozen, patience 100 against 50
    epochs, mosaic and mirroring on, and no originals among its images.
    """
    config = build_candidate_config(
        BASE_CONFIG,
        dataset_version=_dataset_version(tmp_path),
        work_dir=tmp_path / "work",
        run_project=tmp_path / "runs",
        run_name="candidate",
        base_model="champion.pt",
        class_names=["Red"],
        epochs=50,
        imgsz=640,
        batch=4,
        device="cpu",
    )

    training = config["yolo_training"]
    assert {
        key: training[key]
        for key in (
            "epochs", "batch", "imgsz", "optimizer", "lr0", "lrf", "patience",
            "freeze", "mosaic", "fliplr", "close_mosaic", "cos_lr", "warmup_epochs",
        )
    } == {
        "epochs": 50, "batch": 4, "imgsz": 640, "optimizer": "AdamW",
        "lr0": 0.0005, "lrf": 0.1, "patience": 20, "freeze": 10, "mosaic": 0.0,
        "fliplr": 0.0, "close_mosaic": 0, "cos_lr": True, "warmup_epochs": 1.0,
    }
    for key in ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "erasing"):
        assert training[key] == 0.0, key
    policy = config["yolo_augmentation"]["augmentation"]
    # Without originals an empty label file --- a verified negative ---
    # yields no variants and never reaches the split.
    assert policy["include_originals"] is True
    assert policy["target_size"] == 640


def test_position_calibration_stays_out_of_a_challenger_run(tmp_path):
    config = build_candidate_config(
        BASE_CONFIG,
        dataset_version=_dataset_version(tmp_path),
        work_dir=tmp_path / "work",
        run_project=tmp_path / "runs",
        run_name="candidate",
        base_model="champion.pt",
        class_names=["Red"],
        epochs=1,
        imgsz=64,
        batch=1,
        device="cpu",
    )

    position = config["yolo_training"]["position_validation"]
    assert position["enabled"] is False
    # Disabled is not enough: the inherited sample_dir names a directory this
    # cycle never builds, and the schema check reports it before every task.
    assert "sample_dir" not in position


# ---------------------------------------------------------------------------
# The class contract


def test_a_schema_mismatch_stops_the_trainer_before_it_starts(tmp_path):
    """Not "fails during"; the pipeline is never invoked at all.

    Training under a different name order costs an epoch and produces a model
    whose every class id means something else, with metrics that look fine.
    """
    version = _dataset_version(tmp_path)
    reordered = normalize_class_names(
        ["Green", "Black", "Orange", "Red", "Yellow"], source="other"
    )
    runner = _FakeRunner()

    with pytest.raises(CandidateTrainingError, match="disagree"):
        _train(tmp_path, dataset_version=version, class_schema=reordered, runner=runner)

    assert runner.calls == []


def test_a_label_outside_the_schema_stops_the_trainer(tmp_path):
    """Checked on the working copy the pipeline actually reads.

    The version was cut against five classes and carries a ``3``. Training it
    under a one-class contract would hand ultralytics an id with no name.
    """
    store = DatasetVersionStore(tmp_path / "datasets", product="Cable1", area="A")
    image = tmp_path / "src" / "a.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"pixels")
    label = tmp_path / "src" / "a.txt"
    label.write_text("3 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    version = store.create(
        [LabelledSample(sample_id="a", image_path=image, label_path=label)],
        source="pool",
        label_source="operator",
        class_schema=SCHEMA,
    )

    narrow = normalize_class_names(["Black"], source="narrow")
    # Aligned with the version so that only the label range can object.
    object.__setattr__(version, "class_schema", narrow)
    runner = _FakeRunner()

    with pytest.raises(ClassSchemaError, match="outside 0..0"):
        _train(tmp_path, dataset_version=version, class_schema=narrow, runner=runner)

    assert runner.calls == []


def test_the_challenger_records_the_contract_it_trained_under(tmp_path):
    result = _train(tmp_path)

    assert result.class_schema is not None
    assert result.class_schema.names == SCHEMA.names
    assert result.to_dict()["class_schema"]["schema_hash"] == SCHEMA.schema_hash


# ---------------------------------------------------------------------------
# The immutable dataset stays immutable


def test_the_dataset_version_is_copied_not_used_in_place(tmp_path):
    version = _dataset_version(tmp_path)
    before = sorted(p.name for p in version.root.rglob("*"))

    prepare_work_dir(version, tmp_path / "work")

    assert sorted(p.name for p in version.root.rglob("*")) == before
    assert (tmp_path / "work" / "raw" / "images" / "a.jpg").is_file()
    assert (tmp_path / "work" / "raw" / "labels" / "a.txt").is_file()


def test_the_working_copy_is_writable(tmp_path):
    """The point of copying is to get a workspace the pipeline may write in.

    The version store marks its own files read-only to enforce immutability,
    and ``copy2`` carries permission bits, so the copy inherits that bit unless
    it is cleared deliberately.
    """
    version = _dataset_version(tmp_path)

    prepare_work_dir(version, tmp_path / "work")

    for relative in ("raw/images/a.jpg", "raw/labels/a.txt"):
        assert os.access(tmp_path / "work" / relative, os.W_OK), relative


def test_a_retry_can_prepare_the_same_work_dir_again(tmp_path):
    """A failed train step is retried into the cycle's existing work dir.

    Cycle steps are resumable and the work directory is keyed on the cycle, so
    the second attempt copies over the first attempt's files. If those are
    read-only the retry dies on PermissionError -- masking whatever actually
    failed the first time.
    """
    version = _dataset_version(tmp_path)
    with pytest.raises(CandidateTrainingError, match="Challenger training failed"):
        _train(tmp_path, dataset_version=version, runner=_FakeRunner(fail=True))

    result = _train(tmp_path, dataset_version=version, runner=_FakeRunner())

    assert result.weights_path.is_file()
    assert result.dataset_version == version.version


def test_an_unlabelled_image_stops_training(tmp_path):
    version = _dataset_version(tmp_path)
    label = version.labels_dir / "a.txt"
    label.chmod(0o666)
    label.unlink()

    with pytest.raises(CandidateTrainingError, match="missing a label"):
        prepare_work_dir(version, tmp_path / "work")


# ---------------------------------------------------------------------------
# Results and failures


def test_the_challenger_weights_land_in_the_candidate_directory(tmp_path):
    result = _train(tmp_path)

    assert isinstance(result, TrainingResult)
    assert result.weights_path.is_file()
    assert result.weights_path.parent.parent == tmp_path / "candidates" / "v1.3.0"
    assert result.weight_sha256 == hashlib.sha256(b"trained-weights").hexdigest()
    assert result.dataset_version == "dataset_v001"


def test_a_real_run_reports_that_it_trained(tmp_path):
    result = _train(tmp_path)

    assert result.trained_this_run is True
    assert result.to_dict()["trained_this_run"] is True


def test_weights_reused_by_the_skip_cache_are_reported_as_not_trained(tmp_path):
    """The pipeline's skip cache can satisfy a run without training at all.

    On a retry with an unchanged dataset and config the pipeline reuses the
    earlier attempt's weights. They are still correct -- that is what the
    cache checks -- but they are not this attempt's work, and a cycle report
    recommending a promotion should be able to say which of the two happened.
    """
    version = _dataset_version(tmp_path)
    first = _train(tmp_path, dataset_version=version)

    class Skipping(_FakeRunner):
        """Writes nothing, the way a fully cached pipeline run does."""

        def __call__(self, tasks, config, logger, args):
            self.calls.append((list(tasks), config, args))

    second = _train(tmp_path, dataset_version=version, runner=Skipping())

    assert first.trained_this_run is True
    assert second.trained_this_run is False
    assert second.weight_sha256 == first.weight_sha256


def test_a_training_failure_is_reported_without_touching_the_champion(tmp_path):
    champion = tmp_path / "champion.pt"
    champion.write_bytes(b"champion-weights")

    with pytest.raises(CandidateTrainingError, match="Challenger training failed"):
        _train(tmp_path, runner=_FakeRunner(fail=True))

    assert champion.read_bytes() == b"champion-weights"


def test_training_without_a_base_model_is_refused(tmp_path):
    with pytest.raises(CandidateTrainingError, match="No base model"):
        _train(tmp_path, base_model="")


def test_a_run_that_produced_no_weights_is_an_error(tmp_path):
    class Empty(_FakeRunner):
        def __call__(self, tasks, config, logger, args):
            Path(config["yolo_training"]["project"], "candidate").mkdir(parents=True)

    with pytest.raises(CandidateTrainingError, match="no weights"):
        _train(tmp_path, runner=Empty())


def test_an_incremented_run_directory_is_found(tmp_path):
    """ultralytics uses exist_ok=False, so reruns become candidate2, candidate3."""
    result = _train(tmp_path, runner=_FakeRunner(run_name="candidate3"))

    assert result.run_dir.name == "candidate3"


def test_the_generated_config_is_written_for_traceability(tmp_path):
    result = _train(tmp_path)

    assert result.config_path.is_file()
    written = yaml.safe_load(result.config_path.read_text(encoding="utf-8"))
    assert written["autotrain"]["dataset_version"] == "dataset_v001"


# ---------------------------------------------------------------------------
# Surviving a crash, and the shipped device default


def _leave_crashed_run(work_dir: Path, *, epochs: int, completed: int, name="candidate"):
    """What a process killed mid-training leaves: last.pt, no success marker."""
    run_dir = work_dir / "runs" / name
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "last.pt").write_bytes(b"mid-training")
    (run_dir / "args.yaml").write_text(
        yaml.safe_dump({"data": str(work_dir / "split" / "data.yaml"), "epochs": epochs}),
        encoding="utf-8",
    )
    rows = "".join(f"{epoch},0.5\n" for epoch in range(1, completed + 1))
    (run_dir / "results.csv").write_text("epoch,loss\n" + rows, encoding="utf-8")
    return run_dir


def test_an_interrupted_run_is_resumed_natively(tmp_path):
    work_dir = tmp_path / "cycle" / "work"
    crashed = _leave_crashed_run(work_dir, epochs=5, completed=3)
    runner = _FakeRunner()

    _train(tmp_path, runner=runner, epochs=5)

    training = runner.calls[0][1]["yolo_training"]
    assert training["resume_checkpoint"] == str(crashed / "weights" / "last.pt")
    assert training["resume_native"] is True
    assert training["resume_completed_epochs"] == 3


@pytest.mark.parametrize(
    "spoil",
    [
        "finished",  # the trainer wrote its success marker
        "other_data",  # trained on a different split
        "other_epochs",  # towards a different total
        "no_epoch_yet",  # died before finishing one epoch
    ],
)
def test_only_a_run_that_died_on_this_dataset_is_resumed(tmp_path, spoil):
    work_dir = tmp_path / "cycle" / "work"
    crashed = _leave_crashed_run(work_dir, epochs=5, completed=0 if spoil == "no_epoch_yet" else 3)
    if spoil == "finished":
        (crashed / "last_run_metadata.json").write_text("{}", encoding="utf-8")
    if spoil in ("other_data", "other_epochs"):
        (crashed / "args.yaml").write_text(
            yaml.safe_dump(
                {
                    "data": str(tmp_path / "elsewhere" / "data.yaml")
                    if spoil == "other_data"
                    else str(work_dir / "split" / "data.yaml"),
                    "epochs": 5 if spoil == "other_data" else 9,
                }
            ),
            encoding="utf-8",
        )
    runner = _FakeRunner()

    _train(tmp_path, runner=runner, epochs=5)

    assert "resume_checkpoint" not in runner.calls[0][1]["yolo_training"]


def test_the_newest_interrupted_attempt_wins(tmp_path):
    import os

    work_dir = tmp_path / "cycle" / "work"
    older = _leave_crashed_run(work_dir, epochs=5, completed=1)
    newer = _leave_crashed_run(work_dir, epochs=5, completed=2, name="candidate2")
    stamp = (older / "weights" / "last.pt").stat().st_mtime_ns
    os.utime(newer / "weights" / "last.pt", ns=(stamp + 10**9, stamp + 10**9))

    found = find_interrupted_run(
        work_dir / "runs", "candidate", data_yaml=work_dir / "split" / "data.yaml", epochs=5
    )

    assert found == (newer / "weights" / "last.pt", 2)


@pytest.mark.parametrize(("cuda", "expected"), [(False, "cpu"), (True, "0")])
def test_the_shipped_auto_device_becomes_one_ultralytics_accepts(monkeypatch, cuda, expected):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)

    assert resolve_device("auto") == expected
    assert resolve_device("cpu") == "cpu"


def test_the_generated_config_never_carries_auto(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runner = _FakeRunner()

    _train(tmp_path, runner=runner, device="auto")

    assert runner.calls[0][1]["yolo_training"]["device"] == "cpu"
