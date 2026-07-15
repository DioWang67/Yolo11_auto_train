import hashlib
import json
from pathlib import Path

import pytest

from picture_tool.gui.operator_handoff import (
    OperatorHandoffError,
    apply_handoff_to_config,
    load_operator_handoff,
    materialize_job_dataset_snapshot,
)
from picture_tool.path_resolver import resolve_project_paths


def _create_dataset(root: Path, product: str = "Cable1", area: str = "A") -> Path:
    dataset = root / "data" / product / area
    (dataset / "raw" / "images").mkdir(parents=True, exist_ok=True)
    (dataset / "raw" / "labels").mkdir(parents=True, exist_ok=True)
    metadata = dataset / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "review_dataset_manifest.csv").write_text(
        "output_image,output_label,annotation_status\n",
        encoding="utf-8",
    )
    return dataset


def _write_handoff(
    training_root: Path,
    *,
    product: str = "Cable1",
    area: str = "A",
    dataset_root: Path | None = None,
    models_dir: Path | None = None,
    targets: list[dict] | None = None,
) -> Path:
    training_root.mkdir(parents=True, exist_ok=True)
    (training_root / "data").mkdir(exist_ok=True)
    if dataset_root is not None:
        dataset = dataset_root
    elif targets:
        dataset = Path(targets[0]["dataset_root"])
    else:
        dataset = _create_dataset(training_root, product, area)
    inference_models = models_dir or training_root.parent / "inference" / "models"
    inference_models.mkdir(parents=True, exist_ok=True)
    deployed_weights = inference_models / product / area / "yolo" / "weights"
    deployed_weights.mkdir(parents=True, exist_ok=True)
    (deployed_weights / "best.pt").write_bytes(b"deployed-model")
    path = training_root / "handoff.json"
    payload = {
        "schema_version": 1,
        "data_root": str((training_root / "data").resolve()),
        "inference_models_dir": str(inference_models.resolve()),
        "ready_count": 3,
        "pending_count": 2,
        "targets": targets
        or [
            {
                "product": product,
                "area": area,
                "dataset_root": str(dataset.resolve()),
                "ready_count": 3,
                "pending_count": 2,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_handoff_prepares_split_train_and_safe_deploy(tmp_path, monkeypatch):
    training_root = tmp_path / "train"
    path = _write_handoff(training_root)
    base = {"yolo_training": {"epochs": 20}}
    monkeypatch.chdir(training_root)

    handoff = load_operator_handoff(path, training_root=training_root)
    prepared = apply_handoff_to_config(base, handoff)

    assert base == {"yolo_training": {"epochs": 20}}
    assert Path(prepared["yolo_training"]["dataset_dir"]).parts[-3:] == (
        "Cable1",
        "A",
        "split",
    )
    deploy = prepared["yolo_training"]["deploy"]
    assert deploy["enabled"] is True
    assert deploy["product"] == "Cable1"
    assert deploy["area"] == "A"
    assert deploy["preserve_station_settings"] is True
    assert deploy["force"] is False
    assert Path(prepared["yolo_training"]["model"]).name == "best.pt"
    assert prepared["dataset_readiness"]["require_review_manifest"] is True
    assert prepared["dataset_readiness"]["min_instances_per_class"] == 5
    assert prepared["dataset_readiness"]["min_images_per_split"] == {
        "train": 3,
        "val": 5,
        "test": 10,
    }
    assert prepared["dataset_readiness"]["min_test_instances_per_class"] == 5
    assert prepared["yolo_evaluation"]["split"] == "test"
    assert prepared["yolo_evaluation"]["conf"] == 0.4
    resolved = resolve_project_paths(prepared, "Cable1,A")
    dataset_root = training_root / "data" / "Cable1" / "A"
    assert Path(resolved["train_test_split"]["input"]["image_dir"]) == (
        dataset_root / "raw" / "images"
    )
    assert Path(resolved["train_test_split"]["input"]["label_dir"]) == (
        dataset_root / "raw" / "labels"
    )


def test_handoff_rejects_dataset_outside_training_data(tmp_path):
    training_root = tmp_path / "train"
    outside = tmp_path / "outside"
    path = _write_handoff(training_root, dataset_root=outside)

    with pytest.raises(OperatorHandoffError, match="outside handoff data root"):
        load_operator_handoff(path, training_root=training_root)


def test_handoff_rejects_missing_inference_models_directory(tmp_path):
    training_root = tmp_path / "train"
    path = _write_handoff(training_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["inference_models_dir"] = str(tmp_path / "missing-models")
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OperatorHandoffError, match="models directory not found"):
        load_operator_handoff(path, training_root=training_root)


def test_handoff_rejects_incomplete_trainable_dataset(tmp_path):
    training_root = tmp_path / "train"
    incomplete = training_root / "data" / "Cable1" / "A"
    incomplete.mkdir(parents=True)
    path = _write_handoff(training_root, dataset_root=incomplete)

    with pytest.raises(OperatorHandoffError, match="dataset is incomplete"):
        load_operator_handoff(path, training_root=training_root)


def test_handoff_requires_exactly_one_trainable_target(tmp_path):
    training_root = tmp_path / "train"
    first = _create_dataset(training_root, "Cable1", "A")
    second = _create_dataset(training_root, "Cable1", "B")
    targets = [
        {
            "product": "Cable1",
            "area": area,
            "dataset_root": str(dataset.resolve()),
            "ready_count": 1,
            "pending_count": 0,
        }
        for area, dataset in (("A", first), ("B", second))
    ]
    path = _write_handoff(training_root, targets=targets)

    with pytest.raises(OperatorHandoffError, match="exactly one trainable"):
        load_operator_handoff(path, training_root=training_root)


def test_handoff_rejects_training_class_order_mismatch(tmp_path, monkeypatch):
    training_root = tmp_path / "train"
    path = _write_handoff(training_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    payload["targets"][0].update(
        {
            "class_names": ["Black", "Red"],
            "observed_class_map": {"0": "Black", "1": "Red"},
            "class_contract_required": True,
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.chdir(training_root)

    handoff = load_operator_handoff(path, training_root=training_root)

    with pytest.raises(OperatorHandoffError, match="class order"):
        apply_handoff_to_config(
            {"yolo_training": {"class_names": ["Red", "Black"]}}, handoff
        )


def test_handoff_enables_evaluation_gate_for_matching_classes(tmp_path, monkeypatch):
    training_root = tmp_path / "train"
    path = _write_handoff(training_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    payload["targets"][0].update(
        {
            "class_names": ["Black", "Red"],
            "observed_class_map": {"0": "Black", "1": "Red"},
            "class_contract_required": True,
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.chdir(training_root)

    handoff = load_operator_handoff(path, training_root=training_root)
    prepared = apply_handoff_to_config(
        {"yolo_training": {"class_names": ["Black", "Red"]}}, handoff
    )

    assert prepared["yolo_evaluation"]["gate"]["enabled"] is True
    assert prepared["yolo_evaluation"]["gate"]["min_recall"] == 0.9
    assert prepared["yolo_evaluation"]["gate"]["require_baseline"] is True
    assert prepared["yolo_evaluation"]["gate"]["compare_on_same_dataset"] is True


def _write_schema3_handoff(
    training_root: Path,
    *,
    pending_sample_ids: list[str] | None = None,
) -> Path:
    dataset = _create_dataset(training_root)
    image = dataset / "raw" / "images" / "sample.jpg"
    label = dataset / "raw" / "labels" / "sample.txt"
    image.write_bytes(b"original-image")
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (dataset / "metadata" / "review_dataset_manifest.csv").write_text(
        "sample_id,output_image,output_label,annotation_status\n"
        f"ready1,{image},{label},verified_annotation\n",
        encoding="utf-8",
    )
    models = training_root.parent / "inference" / "models"
    weights = models / "Cable1" / "A" / "yolo" / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    (weights / "best.pt").write_bytes(b"model")
    job_id = "20260715T120000Z-testjob"
    job_dir = training_root / "data" / ".operator_handoff" / "jobs" / job_id
    job_dir.mkdir(parents=True)
    handoff = job_dir / "handoff.json"
    status = job_dir / "status.json"
    payload = {
        "schema_version": 3,
        "job_id": job_id,
        "created_at": "2026-07-15T12:00:00+00:00",
        "data_root": str((training_root / "data").resolve()),
        "status_path": str(status.resolve()),
        "inference_models_dir": str(models.resolve()),
        "ready_count": 1,
        "pending_count": len(pending_sample_ids or []),
        "targets": [
            {
                "product": "Cable1",
                "area": "A",
                "dataset_root": str(dataset.resolve()),
                "ready_count": 1,
                "total_ready_count": 1,
                "pending_count": len(pending_sample_ids or []),
                "sample_ids": ["ready1", *(pending_sample_ids or [])],
                "pending_sample_ids": pending_sample_ids or [],
                "class_names": ["Black"],
                "class_schema_hash": hashlib.sha256(
                    json.dumps(
                        ["Black"], ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest(),
                "class_contract_required": True,
            }
        ],
    }
    handoff.write_text(json.dumps(payload), encoding="utf-8")
    status.write_text("{}", encoding="utf-8")
    return handoff


def test_schema3_handoff_rejects_corrupted_class_contract_hash(tmp_path: Path) -> None:
    training_root = tmp_path / "train"
    handoff_path = _write_schema3_handoff(training_root)
    payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    payload["targets"][0]["class_schema_hash"] = "0" * 64
    handoff_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OperatorHandoffError, match="checksum"):
        load_operator_handoff(handoff_path, training_root=training_root)


def test_schema3_handoff_counts_only_its_own_pending_samples(tmp_path: Path) -> None:
    training_root = tmp_path / "train"
    handoff_path = _write_schema3_handoff(
        training_root, pending_sample_ids=["job-pending"]
    )
    pending_manifest = (
        training_root / "data" / "Cable1" / "A" / "review_pending" / "manifest.csv"
    )
    pending_manifest.parent.mkdir(parents=True)
    pending_manifest.write_text(
        "sample_id\njob-pending\nother-job-pending\n", encoding="utf-8"
    )

    first = load_operator_handoff(handoff_path, training_root=training_root)
    pending_manifest.write_text("sample_id\nother-job-pending\n", encoding="utf-8")
    ready_manifest = (
        training_root
        / "data"
        / "Cable1"
        / "A"
        / "metadata"
        / "review_dataset_manifest.csv"
    )
    with ready_manifest.open("a", encoding="utf-8") as handle:
        handle.write("job-pending,,,verified_annotation\n")
    completed = load_operator_handoff(handoff_path, training_root=training_root)

    assert first.selected_target.pending_count == 1
    assert completed.selected_target.pending_count == 0


def test_schema3_handoff_rejects_unaccounted_job_sample(tmp_path: Path) -> None:
    training_root = tmp_path / "train"
    handoff_path = _write_schema3_handoff(
        training_root, pending_sample_ids=["job-pending"]
    )
    pending_manifest = (
        training_root / "data" / "Cable1" / "A" / "review_pending" / "manifest.csv"
    )
    pending_manifest.parent.mkdir(parents=True)
    pending_manifest.write_text("sample_id\n", encoding="utf-8")

    with pytest.raises(OperatorHandoffError, match="missing from both"):
        load_operator_handoff(handoff_path, training_root=training_root)


def test_schema3_snapshot_is_immutable_and_used_by_training_config(
    tmp_path: Path,
) -> None:
    training_root = tmp_path / "train"
    handoff_path = _write_schema3_handoff(training_root)
    original_handoff = handoff_path.read_bytes()
    handoff = load_operator_handoff(handoff_path, training_root=training_root)

    snapshot_handoff = materialize_job_dataset_snapshot(handoff)
    source_image = (
        handoff.selected_target.dataset_root / "raw" / "images" / "sample.jpg"
    )
    source_label = (
        handoff.selected_target.dataset_root / "raw" / "labels" / "sample.txt"
    )
    source_image.write_bytes(b"changed-after-snapshot")
    source_label.write_text("", encoding="utf-8")
    snapshot = snapshot_handoff.selected_target.dataset_root
    prepared = apply_handoff_to_config(
        {"yolo_training": {"class_names": ["Black"]}}, snapshot_handoff
    )

    assert (
        snapshot / "raw" / "images" / "sample.jpg"
    ).read_bytes() == b"original-image"
    assert (
        (snapshot / "raw" / "labels" / "sample.txt")
        .read_text(encoding="utf-8")
        .startswith("0 ")
    )
    assert Path(prepared["train_test_split"]["input"]["image_dir"]).is_relative_to(
        handoff_path.parent / "dataset"
    )
    assert prepared["yolo_training"]["epochs"] == 20
    assert prepared["yolo_training"]["optimizer"] == "AdamW"
    assert prepared["yolo_training"]["lr0"] == 0.0005
    assert prepared["yolo_training"]["freeze"] == 10
    assert prepared["yolo_training"]["mosaic"] == 0.0
    assert prepared["yolo_training"]["fliplr"] == 0.0
    assert prepared["train_test_split"]["minimum_source_groups"] == {
        "val": 5,
        "test": 10,
    }
    assert handoff_path.read_bytes() == original_handoff


def test_handoff_rejects_confirmation_only_feedback(tmp_path: Path) -> None:
    training_root = tmp_path / "train"
    handoff_path = _write_schema3_handoff(training_root)
    manifest = (
        training_root
        / "data"
        / "Cable1"
        / "A"
        / "metadata"
        / "review_dataset_manifest.csv"
    )
    manifest.write_text(
        "sample_id,output_image,output_label,annotation_status,review_label\n"
        "ready1,,,verified_snapshot,confirmed_ng\n",
        encoding="utf-8",
    )
    handoff = load_operator_handoff(handoff_path, training_root=training_root)

    with pytest.raises(OperatorHandoffError, match="operator_feedback_not_actionable"):
        apply_handoff_to_config({"yolo_training": {"class_names": ["Black"]}}, handoff)


def test_handoff_rejects_runtime_onnx_without_exact_paired_pt(tmp_path) -> None:
    training_root = tmp_path / "train"
    handoff_path = _write_handoff(training_root)
    handoff = load_operator_handoff(handoff_path, training_root=training_root)
    station = handoff.inference_models_dir / "Cable1" / "A" / "yolo"
    runtime = station / "weights" / "best.onnx"
    runtime.write_bytes(b"runtime")
    (station / "config.yaml").write_text(
        "weights: models/Cable1/A/yolo/weights/best.onnx\n",
        encoding="utf-8",
    )

    with pytest.raises(OperatorHandoffError, match="deployed_training_pair_missing"):
        apply_handoff_to_config({"yolo_training": {}}, handoff)


def test_handoff_accepts_checksum_verified_runtime_and_training_pair(
    tmp_path,
) -> None:
    import yaml

    training_root = tmp_path / "train"
    handoff_path = _write_handoff(training_root)
    handoff = load_operator_handoff(handoff_path, training_root=training_root)
    station = handoff.inference_models_dir / "Cable1" / "A" / "yolo"
    training_weight = station / "weights" / "best.pt"
    runtime = station / "weights" / "Cable1_A_v1.0.0_20260715.onnx"
    runtime.write_bytes(b"runtime")
    (station / "config.yaml").write_text(
        f"weights: models/Cable1/A/yolo/weights/{runtime.name}\n",
        encoding="utf-8",
    )
    (station / "deployment_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "deployed_file": runtime.name,
                "weight_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
                "training_weight_file": training_weight.name,
                "training_weight_sha256": hashlib.sha256(
                    training_weight.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    prepared = apply_handoff_to_config({"yolo_training": {}}, handoff)

    assert Path(prepared["yolo_training"]["model"]) == training_weight.resolve()


def test_schema3_snapshot_merges_only_matching_legacy_raw_pairs(tmp_path: Path) -> None:
    training_root = tmp_path / "train"
    handoff_path = _write_schema3_handoff(training_root)
    legacy_root = training_root / "data" / "Cable1" / "raw"
    legacy_images = legacy_root / "images"
    legacy_labels = legacy_root / "labels"
    legacy_images.mkdir(parents=True)
    legacy_labels.mkdir(parents=True)
    matching_stem = "old-yolo_Cable1_A_120000"
    other_area_stem = "old-yolo_Cable1_B_120001"
    (legacy_images / f"{matching_stem}.jpg").write_bytes(b"area-a")
    (legacy_labels / f"{matching_stem}.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (legacy_images / f"{other_area_stem}.jpg").write_bytes(b"area-b")
    (legacy_labels / f"{other_area_stem}.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    handoff = load_operator_handoff(handoff_path, training_root=training_root)

    snapshot_handoff = materialize_job_dataset_snapshot(handoff)

    snapshot = snapshot_handoff.selected_target.dataset_root
    assert (snapshot / "raw" / "images" / f"{matching_stem}.jpg").is_file()
    assert not (snapshot / "raw" / "images" / f"{other_area_stem}.jpg").exists()
    snapshot_manifest = json.loads(
        (handoff_path.parent / "dataset_snapshot.json").read_text(encoding="utf-8")
    )
    assert snapshot_manifest["legacy_image_count"] == 1
