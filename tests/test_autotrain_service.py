"""The agent-facing service facade.

The most important assertion in this file is negative: there is no method
that deploys a model, and there never should be.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import yaml

from picture_tool.autotrain import AutoTrainDisabledError, AutoTrainError
from picture_tool.autotrain.candidate_pool import CandidatePool, CandidateSample
from picture_tool.autotrain.config import AutoTrainConfig
from picture_tool.autotrain.paths import AutoTrainPaths
from picture_tool.autotrain.service import TOOL_SURFACE, AutoTrainService
from picture_tool.workspace_paths import WorkspacePaths

PRODUCT = "Cable1"
AREA = "A"
CLASSES = ("Black", "Green", "Orange", "Red", "Yellow")


@pytest.fixture()
def paths(tmp_path) -> AutoTrainPaths:
    root = tmp_path / "workspace"
    training = root / "Yolo11_auto_train"
    inference = root / "yolo11_inference"
    for relative in (
        training / "data",
        inference / "models",
        root / "station_data" / "yolo11_inference",
        root / "Result",
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
        inference_artifacts=root.resolve(),
        manifest_path=None,
    )
    resolved = AutoTrainPaths.from_workspace(workspace)
    # A station that states its class contract. Without one, every write path
    # here fails closed on purpose --- see the dedicated test below.
    model_dir = resolved.production_model_dir(PRODUCT, AREA)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.yaml").write_text(
        yaml.safe_dump({"class_names": list(CLASSES)}), encoding="utf-8"
    )
    return resolved


@pytest.fixture()
def config(tmp_path) -> AutoTrainConfig:
    path = tmp_path / "autonomous_training.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "autonomous_training": {
                    "enabled": True,
                    "station": {"product": PRODUCT, "area": AREA},
                    "selectors": {"low_confidence": {"enabled": True}},
                }
            }
        ),
        encoding="utf-8",
    )
    return AutoTrainConfig.load(path)


@pytest.fixture()
def service(config, paths) -> AutoTrainService:
    return AutoTrainService(config, paths)


def _seed_pool(paths, tmp_path, *, verified=True):
    pool = CandidatePool(paths.pool_dir(PRODUCT, AREA))
    source = tmp_path / "src" / "img.jpg"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"pixels")
    pool.add(
        [
            CandidateSample(
                sample_id="sample0",
                inspection_id="insp0",
                product=PRODUCT,
                area=AREA,
                selector="low_confidence",
                selected_reason="uncertain",
                score=0.3,
                source_model="best.onnx",
                source_model_version="1.2.0",
                selected_at="2026-09-10T00:00:00+00:00",
                production_timestamp="",
                image_path="",
                source_image_path=str(source),
            )
        ]
    )
    if verified:
        label = pool.labels_dir
        label.mkdir(parents=True, exist_ok=True)
        path = label / "sample0.txt"
        path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        pool.update_label_state("sample0", "VERIFIED", label_path=str(path))
    return pool


# ---------------------------------------------------------------------------
# The boundary


def test_the_facade_exposes_no_way_to_deploy(service):
    """Not an oversight to fix later; this is the boundary."""
    for name in dir(service):
        assert "deploy" not in name.lower()
        assert "promote" not in name.lower()
        assert "activate" not in name.lower()


def test_the_documented_tool_surface_all_exists(service):
    for name in TOOL_SURFACE:
        assert callable(getattr(service, name)), name


def test_no_llm_or_network_client_is_imported_by_the_package():
    """An agent belongs above this interface, not inside the training core."""
    import picture_tool.autotrain as package

    root = Path(package.__file__).resolve().parent
    banned = ("import anthropic", "import openai", "from anthropic", "from openai")
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in banned:
            assert needle not in text, f"{path} contains {needle}"


def test_reads_work_while_the_feature_is_disabled(paths):
    service = AutoTrainService(AutoTrainConfig.disabled(), paths)

    health = service.get_model_health()

    assert health["enabled"] is False
    assert health["champion"] is None


def test_writes_refuse_while_the_feature_is_disabled(paths):
    service = AutoTrainService(AutoTrainConfig.disabled(), paths)

    with pytest.raises(AutoTrainDisabledError):
        service.create_dataset_version()
    with pytest.raises(AutoTrainDisabledError):
        service.run_training_cycle()


# ---------------------------------------------------------------------------
# Reads


def test_model_health_reports_the_whole_picture(service, paths, tmp_path):
    _seed_pool(paths, tmp_path)

    health = service.get_model_health()

    assert health["enabled"] is True
    assert health["product"] == PRODUCT
    assert health["pool"]["total"] == 1
    assert health["golden"]["status"] == "NOT_CONFIGURED"
    assert health["candidates_by_status"] == {}


def test_candidate_samples_can_be_filtered_by_label_state(service, paths, tmp_path):
    _seed_pool(paths, tmp_path, verified=True)

    verified = service.get_candidate_samples(label_state="VERIFIED")
    pending = service.get_candidate_samples(label_state="NEEDS_LABEL")

    assert len(verified) == 1
    assert pending == []


def test_dataset_statistics_list_versions(service, paths, tmp_path):
    _seed_pool(paths, tmp_path)
    service.create_dataset_version(description="first")

    stats = service.get_dataset_statistics()

    assert stats["latest"] == "dataset_v001"
    assert stats["versions"][0]["sample_count"] == 1
    assert stats["versions"][0]["parent_version"] == ""


def test_training_history_is_empty_before_any_cycle(service):
    assert service.get_training_history() == []


def test_training_history_reads_cycle_reports(service, paths):
    directory = paths.cycle_dir("cycle_a")
    directory.mkdir(parents=True)
    (directory / "report.json").write_text(
        json.dumps(
            {
                "cycle_id": "cycle_a",
                "dataset_version": "dataset_v001",
                "challenger": "cand-1",
                "decision": {"decision": "REJECTED"},
                "deployed": False,
            }
        ),
        encoding="utf-8",
    )

    history = service.get_training_history()

    assert history[0]["cycle_id"] == "cycle_a"
    assert history[0]["decision"] == "REJECTED"
    assert history[0]["deployed"] is False


def test_an_unreadable_cycle_report_is_skipped(service, paths):
    directory = paths.cycle_dir("cycle_bad")
    directory.mkdir(parents=True)
    (directory / "report.json").write_text("{oops", encoding="utf-8")

    assert service.get_training_history() == []


def test_the_registry_view_separates_production_from_candidates(service):
    view = service.get_model_registry()

    assert set(view) == {"production", "candidates"}
    assert view["production"] is None
    assert view["candidates"] == []


# ---------------------------------------------------------------------------
# Writes


def test_creating_a_dataset_version_requires_verified_labels(
    service, paths, tmp_path
):
    _seed_pool(paths, tmp_path, verified=False)

    with pytest.raises(AutoTrainError, match="human-verified label"):
        service.create_dataset_version()


def test_compare_models_needs_no_inference(service):
    champion = {"overall": {"map50": 0.80}, "per_class": {}}
    challenger = {"overall": {"map50": 0.85}, "per_class": {}}

    comparison = service.compare_models(champion, challenger)

    deltas = {item["name"]: item["delta"] for item in comparison["overall"]}
    assert deltas["map50"] == pytest.approx(0.05)


def test_every_public_method_returns_serialisable_data(service, paths, tmp_path):
    _seed_pool(paths, tmp_path)

    for name in ("get_model_health", "get_dataset_statistics", "get_model_registry"):
        payload = getattr(service, name)()
        json.dumps(payload)  # must not raise


def test_the_tool_surface_is_documented_in_order(service):
    """A stable list is what a future agent binds its tools to."""
    signature_names = [
        name
        for name, value in inspect.getmembers(service, predicate=inspect.ismethod)
        if not name.startswith("_")
    ]

    assert set(TOOL_SURFACE) <= set(signature_names)


# ---------------------------------------------------------------------------
# Production reads


def _write_snapshot(paths, inspection_id, status, *, image=None):
    directory = (
        paths.production_results_root()
        / "20260910"
        / PRODUCT
        / AREA
        / status
        / "metadata"
        / "yolo"
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{inspection_id}_config_snapshot.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "inspection_id": inspection_id,
                "timestamp": "2026-09-10T12:00:00",
                "status": status,
                "detector": "yolo",
                "product": PRODUCT,
                "area": AREA,
                "model_info": {"model_version": "1.2.0", "class_names": ["Red"]},
                "detections": [
                    {"class": "Red", "confidence": 0.3, "bbox": [0, 0, 1, 1]}
                ],
                "artifacts": {"original_path": str(image or "")},
                "config": {},
            }
        ),
        encoding="utf-8",
    )


def test_recent_failures_report_what_the_line_saw(service, paths, tmp_path):
    image = tmp_path / "img.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"pixels")
    _write_snapshot(paths, "good", "PASS", image=image)
    _write_snapshot(paths, "bad", "DETECTION_FAIL", image=image)

    failures = service.get_recent_failures()

    assert [record["inspection_id"] for record in failures] == ["bad"]


def test_recent_failures_are_bounded(service, paths, tmp_path):
    for index in range(5):
        _write_snapshot(paths, f"bad{index}", "DETECTION_FAIL")

    assert len(service.get_recent_failures(limit=2)) == 2


def test_recent_failures_refuse_while_disabled(paths):
    service = AutoTrainService(AutoTrainConfig.disabled(), paths)

    with pytest.raises(AutoTrainDisabledError):
        service.get_recent_failures()


def test_running_a_cycle_through_the_facade_never_reports_a_deployment(
    service, paths, tmp_path
):
    payload = service.run_training_cycle(cycle_id="cycle_facade")

    assert payload["deployed"] is False
    assert payload["status"] == "BLOCKED"  # nothing labelled yet
    assert "human-verified label" in payload["blocked_reason"]


def test_training_a_candidate_without_a_dataset_version_is_refused(service):
    with pytest.raises(AutoTrainError, match="No dataset version"):
        service.train_candidate()


# ---------------------------------------------------------------------------
# Reserved selectors


def test_the_reserved_selectors_state_what_they_would_need():
    """Declared, not forgotten: the placeholder names its own prerequisite."""
    from picture_tool.autotrain.selectors._planned import (
        PLANNED_SELECTOR_NOTES,
        NotImplementedSelector,
    )

    selector = NotImplementedSelector("distribution_drift")

    with pytest.raises(NotImplementedError, match="baseline window"):
        selector.select([], options={})
    assert set(PLANNED_SELECTOR_NOTES) == {
        "model_disagreement",
        "class_imbalance",
        "embedding_novelty",
        "distribution_drift",
    }
