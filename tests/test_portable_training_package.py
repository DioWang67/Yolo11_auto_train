import hashlib
import json
import zipfile
from pathlib import Path

import pytest

import picture_tool.portable_training_package as portable_package_module
from picture_tool.gui.operator_handoff import (
    find_deployed_runtime_weight,
    load_operator_handoff,
)
from picture_tool.operator_job import update_job_status
from picture_tool.portable_training_package import (
    PortableTrainingImportError,
    import_portable_training_package,
)


def _portable_package(
    tmp_path: Path,
    *,
    invalid_checksum: bool = False,
    invalid_class_contract: bool = False,
    metadata_pending_sample_ids: list[str] | None = None,
    metadata_sample_ids: list[str] | None = None,
    omit_training_options: bool = False,
    runtime_config_path: str | None = None,
    source_schema_version: int = 4,
) -> Path:
    package = tmp_path / "portable.zip"
    image_name = "review_sample.jpg"
    label_name = "review_sample.txt"
    original_dataset = Path("D:/source/training/data/Cable1/A")
    class_names = ["Black"]
    class_hash = hashlib.sha256(
        json.dumps(class_names, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    handoff = {
        "schema_version": source_schema_version,
        "job_id": "source-job",
        "created_at": "2026-07-17T00:00:00+00:00",
        "submission_hash": hashlib.sha256(b"submission").hexdigest(),
        "source_manifest": "D:/source/selected.csv",
        "data_root": "D:/source/training/data",
        "status_path": "D:/source/training/data/.operator_handoff/jobs/source-job/status.json",
        "inference_models_dir": "D:/source/inference/models",
        "ready_count": 1,
        "total_ready_count": 1,
        "pending_count": 0,
        "training_options": {
            "epochs": 20,
            "augmentations_per_image": 20,
            "batch": 8,
            "imgsz": 640,
            "position_training_mode": "auto",
            "position_activation": "preserve",
        },
        "targets": [
            {
                "product": "Cable1",
                "area": "A",
                "dataset_root": str(original_dataset),
                "ready_count": 1,
                "total_ready_count": 1,
                "pending_count": 0,
                "sample_ids": ["sample"],
                "pending_sample_ids": [],
                "class_names": class_names,
                "class_schema_hash": class_hash,
                "class_contract_required": True,
            }
        ],
    }
    if source_schema_version >= 6:
        handoff["inference_station_data_dir"] = "D:/source/station_data"
        handoff["inference_project_root"] = "D:/source/inference"
    if invalid_class_contract:
        handoff["targets"][0]["class_schema_hash"] = "0" * 64
    if omit_training_options:
        handoff.pop("training_options")
    if runtime_config_path is None:
        runtime_config_path = (
            "../shared_models/Cable1/A/yolo/weights/best.pt"
            if source_schema_version >= 6
            else "models/Cable1/A/yolo/weights/best.pt"
        )
    files = {
        "README.txt": b"portable package\n",
        f"payload/data/Cable1/A/raw/images/{image_name}": b"image",
        f"payload/data/Cable1/A/raw/labels/{label_name}": b"0 0.5 0.5 0.2 0.2\n",
        "payload/data/Cable1/A/metadata/review_dataset_manifest.csv": (
            "sample_id,image_sha256,output_image,output_label,annotation_status,review_label\n"
            f"sample,{hashlib.sha256(b'image').hexdigest()},"
            f"{original_dataset / 'raw/images' / image_name},"
            f"{original_dataset / 'raw/labels' / label_name},"
            "verified_annotation,false_positive\n"
        ).encode(),
        "payload/models/Cable1/A/yolo/config.yaml": (
            f"weights: {runtime_config_path}\n"
        ).encode(),
        "payload/models/Cable1/A/yolo/weights/best.pt": b"model",
        "payload/job/handoff.json": json.dumps(handoff).encode(),
    }
    inventory = {
        name: {
            "sha256": (
                "0" * 64
                if invalid_checksum and name.endswith(image_name)
                else hashlib.sha256(content).hexdigest()
            ),
            "size": len(content),
        }
        for name, content in files.items()
    }
    metadata = {
        "schema_version": 1,
        "package_id": "source-job-abcdef123456",
        "source_job_id": "source-job",
        "product": "Cable1",
        "area": "A",
        "handoff_path": "payload/job/handoff.json",
        "dataset_path": "payload/data/Cable1/A",
        "models_path": "payload/models",
        "sample_ids": (
            ["sample"] if metadata_sample_ids is None else metadata_sample_ids
        ),
        "pending_sample_ids": (
            []
            if metadata_pending_sample_ids is None
            else metadata_pending_sample_ids
        ),
        "files": inventory,
    }
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("package.json", json.dumps(metadata))
        for name, content in files.items():
            archive.writestr(name, content)
    return package


@pytest.mark.parametrize("source_schema_version", (3, 4, 6))
def test_import_portable_package_rewrites_paths_and_is_idempotent(
    tmp_path,
    source_schema_version,
):
    package = _portable_package(
        tmp_path,
        omit_training_options=source_schema_version == 3,
        source_schema_version=source_schema_version,
    )
    training_root = tmp_path / "remote-training"

    report = import_portable_training_package(package, training_root)
    loaded = load_operator_handoff(report.handoff_path, training_root=training_root)
    local_payload = json.loads(report.handoff_path.read_text(encoding="utf-8"))
    imported_target = loaded.selected_target
    second = import_portable_training_package(package, training_root)

    assert imported_target.dataset_root == (
        training_root / "data" / "Cable1" / "A"
    ).resolve()
    assert loaded.inference_models_dir.is_relative_to(
        (training_root / "data" / ".portable_models").resolve()
    )
    assert local_payload["schema_version"] == 4
    assert local_payload["portable_source_runtime_weight"] == (
        "../shared_models/Cable1/A/yolo/weights/best.pt"
        if source_schema_version >= 6
        else "models/Cable1/A/yolo/weights/best.pt"
    )
    assert local_payload["portable_runtime_weight"] == (
        "models/Cable1/A/yolo/weights/best.pt"
    )
    assert local_payload["training_options"] == {
        "epochs": 20,
        "augmentations_per_image": 20,
        "batch": 8,
        "imgsz": 640,
        "position_training_mode": "auto",
        "position_activation": "preserve",
    }
    assert "inference_station_data_dir" not in local_payload
    assert "inference_project_root" not in local_payload
    assert loaded.inference_station_data_dir is None
    assert loaded.inference_project_root is None
    assert find_deployed_runtime_weight(loaded) == (
        loaded.inference_models_dir
        / "Cable1"
        / "A"
        / "yolo"
        / "weights"
        / "best.pt"
    ).resolve()
    assert imported_target.pending_count == 0
    assert imported_target.total_ready_count == 1
    assert second.reused_existing is True
    receipt = json.loads(
        (
            training_root
            / "data"
            / ".portable_imports"
            / "source-job-abcdef123456"
            / "import.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["schema_version"] == 3
    assert receipt["state"] == "committed"
    assert "status_identity_sha256" in receipt
    assert "status_sha256" not in receipt
    manifest = imported_target.dataset_root / "metadata" / "review_dataset_manifest.csv"
    manifest_text = manifest.read_text(encoding="utf-8")
    assert str(imported_target.dataset_root.resolve()) in manifest_text
    assert "D:/source/training/data/Cable1/A/raw/images" not in manifest_text


def test_existing_portable_import_remains_idempotent_after_status_updates(tmp_path):
    package = _portable_package(tmp_path, source_schema_version=6)
    training_root = tmp_path / "training"
    first = import_portable_training_package(package, training_root)
    status_path = first.handoff_path.parent / "status.json"

    update_job_status(
        status_path,
        state="running",
        message="Training is active.",
        progress=35,
    )
    second = import_portable_training_package(package, training_root)

    assert second.reused_existing is True
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["state"] == "running"
    assert status["progress"] == 35


def test_existing_portable_import_rejects_changed_status_identity(tmp_path):
    package = _portable_package(tmp_path, source_schema_version=6)
    training_root = tmp_path / "training"
    report = import_portable_training_package(package, training_root)
    status_path = report.handoff_path.parent / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["created_at"] = "2026-08-18T00:00:00+00:00"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(PortableTrainingImportError, match="status identity"):
        import_portable_training_package(package, training_root)


def test_existing_portable_import_rejects_boolean_status_schema(tmp_path):
    package = _portable_package(tmp_path, source_schema_version=6)
    training_root = tmp_path / "training"
    report = import_portable_training_package(package, training_root)
    status_path = report.handoff_path.parent / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["schema_version"] = True
    status_path.write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(PortableTrainingImportError, match="status"):
        import_portable_training_package(package, training_root)


def test_schema_two_receipt_migrates_after_legitimate_status_update(tmp_path):
    package = _portable_package(tmp_path, source_schema_version=6)
    training_root = tmp_path / "training"
    report = import_portable_training_package(package, training_root)
    status_path = report.handoff_path.parent / "status.json"
    receipt_path = (
        training_root
        / "data"
        / ".portable_imports"
        / "source-job-abcdef123456"
        / "import.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["schema_version"] = 2
    receipt["status_sha256"] = hashlib.sha256(status_path.read_bytes()).hexdigest()
    receipt.pop("status_identity_sha256")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    update_job_status(status_path, state="running", message="Training is active.")

    reused = import_portable_training_package(package, training_root)

    migrated = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert reused.reused_existing is True
    assert migrated["schema_version"] == 3
    assert "status_identity_sha256" in migrated
    assert "status_sha256" not in migrated


def test_portable_import_locks_the_target_manifest_transaction(
    tmp_path,
    monkeypatch,
):
    package = _portable_package(tmp_path, source_schema_version=6)
    training_root = tmp_path / "training"
    observed_roots: list[Path] = []
    real_lock = portable_package_module.dataset_manifest_lock

    def recording_lock(dataset_root, **kwargs):
        observed_roots.append(Path(dataset_root).resolve())
        return real_lock(dataset_root, **kwargs)

    monkeypatch.setattr(
        portable_package_module,
        "dataset_manifest_lock",
        recording_lock,
    )

    import_portable_training_package(package, training_root)

    assert observed_roots == [
        (training_root / "data" / "Cable1" / "A").resolve()
    ]


@pytest.mark.parametrize(
    "untrusted_runtime_path",
    ("{absolute}", "../../../../outside/weights/best.pt"),
)
def test_runtime_resolver_never_substitutes_station_file_by_basename(
    tmp_path,
    untrusted_runtime_path,
):
    package = _portable_package(tmp_path, source_schema_version=6)
    training_root = tmp_path / "remote-training"
    report = import_portable_training_package(package, training_root)
    loaded = load_operator_handoff(report.handoff_path, training_root=training_root)
    outside_weight = tmp_path / "outside" / "weights" / "best.pt"
    outside_weight.parent.mkdir(parents=True)
    outside_weight.write_bytes(b"actual-live-runtime")
    station_weight = (
        loaded.inference_models_dir
        / "Cable1"
        / "A"
        / "yolo"
        / "weights"
        / "best.pt"
    )
    assert station_weight.read_bytes() == b"model"
    raw_path = (
        str(outside_weight.resolve())
        if untrusted_runtime_path == "{absolute}"
        else untrusted_runtime_path
    )
    config_path = station_weight.parent.parent / "config.yaml"
    config_path.write_text(f"weights: {raw_path}\n", encoding="utf-8")

    assert find_deployed_runtime_weight(loaded) is None


def test_import_portable_package_resolves_packaged_weight_for_absolute_source_path(
    tmp_path,
):
    source_runtime_path = str(
        (tmp_path.parent / "unavailable-source" / "weights" / "best.pt").resolve()
    )
    package = _portable_package(
        tmp_path,
        runtime_config_path=source_runtime_path,
        source_schema_version=6,
    )
    training_root = tmp_path / "remote-training"

    report = import_portable_training_package(package, training_root)
    loaded = load_operator_handoff(report.handoff_path, training_root=training_root)

    assert not Path(source_runtime_path).exists()
    assert find_deployed_runtime_weight(loaded) == (
        loaded.inference_models_dir
        / "Cable1"
        / "A"
        / "yolo"
        / "weights"
        / "best.pt"
    ).resolve()


@pytest.mark.parametrize("source_schema_version", (2, 7, 999))
def test_import_portable_package_rejects_unsupported_source_handoff_schema(
    tmp_path,
    source_schema_version,
):
    package = _portable_package(
        tmp_path,
        source_schema_version=source_schema_version,
    )
    training_root = tmp_path / "training"

    with pytest.raises(PortableTrainingImportError, match="schema is unsupported"):
        import_portable_training_package(package, training_root)

    data_root = training_root / "data"
    assert not (data_root / "Cable1").exists()
    assert not (data_root / ".portable_models").exists()
    assert not (
        data_root / ".portable_imports" / "source-job-abcdef123456"
    ).exists()


@pytest.mark.parametrize(
    ("package_options", "message"),
    (
        ({"omit_training_options": True}, "training options are invalid"),
        ({"invalid_class_contract": True}, "Class contract checksum"),
    ),
)
def test_import_portable_package_rejects_unloadable_handoff_before_publication(
    tmp_path,
    package_options,
    message,
):
    package = _portable_package(
        tmp_path,
        source_schema_version=6,
        **package_options,
    )
    training_root = tmp_path / "training"

    with pytest.raises(PortableTrainingImportError, match=message):
        import_portable_training_package(package, training_root)

    data_root = training_root / "data"
    assert not (data_root / "Cable1").exists()
    assert not (data_root / ".portable_models").exists()
    assert not (data_root / ".operator_handoff" / "latest.json").exists()
    assert not (
        data_root / ".portable_imports" / "source-job-abcdef123456" / "import.json"
    ).exists()


@pytest.mark.parametrize(
    "package_options",
    (
        {"metadata_sample_ids": []},
        {"metadata_pending_sample_ids": ["sample"]},
    ),
)
def test_import_portable_package_binds_metadata_samples_to_handoff(
    tmp_path,
    package_options,
):
    package = _portable_package(
        tmp_path,
        source_schema_version=6,
        **package_options,
    )
    training_root = tmp_path / "training"

    with pytest.raises(PortableTrainingImportError, match="do not match"):
        import_portable_training_package(package, training_root)

    data_root = training_root / "data"
    assert not (data_root / "Cable1").exists()
    assert not (data_root / ".portable_models").exists()
    assert not (data_root / ".operator_handoff" / "latest.json").exists()


def test_import_portable_package_rejects_checksum_mismatch(tmp_path):
    package = _portable_package(tmp_path, invalid_checksum=True)

    with pytest.raises(PortableTrainingImportError, match="Checksum mismatch"):
        import_portable_training_package(package, tmp_path / "training")


def test_existing_portable_receipt_revalidates_immutable_handoff(tmp_path):
    package = _portable_package(tmp_path, source_schema_version=6)
    training_root = tmp_path / "training"
    report = import_portable_training_package(package, training_root)
    report.handoff_path.write_text("{", encoding="utf-8")

    with pytest.raises(PortableTrainingImportError, match="Unable to read JSON"):
        import_portable_training_package(package, training_root)


def test_existing_portable_receipt_rejects_handoff_path_escape(tmp_path):
    package = _portable_package(tmp_path, source_schema_version=6)
    training_root = tmp_path / "training"
    import_portable_training_package(package, training_root)
    receipt_path = (
        training_root
        / "data"
        / ".portable_imports"
        / "source-job-abcdef123456"
        / "import.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    outside_handoff = tmp_path / "outside-handoff.json"
    outside_handoff.write_text("{}", encoding="utf-8")
    receipt["handoff_path"] = str(outside_handoff.resolve())
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(PortableTrainingImportError, match="outside its immutable job"):
        import_portable_training_package(package, training_root)


def test_receipt_write_failure_never_publishes_latest(
    tmp_path,
    monkeypatch,
):
    package = _portable_package(tmp_path, source_schema_version=6)
    training_root = tmp_path / "training"
    real_write = portable_package_module._write_json_atomic

    def deny_receipt(path, payload):
        if Path(path).name == "import.json":
            raise PermissionError("simulated receipt denial")
        real_write(Path(path), payload)

    monkeypatch.setattr(
        portable_package_module,
        "_write_json_atomic",
        deny_receipt,
    )

    with pytest.raises(PermissionError, match="simulated receipt denial"):
        import_portable_training_package(package, training_root)

    data_root = training_root / "data"
    assert not (data_root / ".operator_handoff" / "latest.json").exists()
    assert not (
        data_root / ".portable_imports" / "source-job-abcdef123456" / "import.json"
    ).exists()


def test_latest_failure_retains_prepared_receipt_and_retry_fails_closed(
    tmp_path,
    monkeypatch,
):
    package = _portable_package(tmp_path, source_schema_version=6)
    training_root = tmp_path / "training"
    data_root = training_root / "data"
    latest_path = data_root / ".operator_handoff" / "latest.json"
    receipt_path = (
        data_root
        / ".portable_imports"
        / "source-job-abcdef123456"
        / "import.json"
    )
    real_write = portable_package_module._write_json_atomic
    latest_path.parent.mkdir(parents=True)
    latest_path.write_text(
        json.dumps({"schema_version": 4, "job_id": "previous-job"}),
        encoding="utf-8",
    )

    real_rollback = portable_package_module._ImportMutationJournal.rollback

    with monkeypatch.context() as patch_context:

        def deny_latest(path, payload):
            if Path(path).name == "latest.json":
                raise PermissionError("simulated latest denial")
            real_write(Path(path), payload)

        def report_rollback_failure(journal):
            real_rollback(journal)
            raise PortableTrainingImportError(
                "simulated rollback diagnostic failure"
            )

        patch_context.setattr(
            portable_package_module,
            "_write_json_atomic",
            deny_latest,
        )
        patch_context.setattr(
            portable_package_module._ImportMutationJournal,
            "rollback",
            report_rollback_failure,
        )
        with pytest.raises(
            PermissionError,
            match="simulated latest denial",
        ) as error:
            import_portable_training_package(package, training_root)

    assert any(
        "simulated rollback diagnostic failure" in note
        for note in getattr(error.value, "__notes__", ())
    )

    assert json.loads(latest_path.read_text(encoding="utf-8"))["job_id"] == (
        "previous-job"
    )
    assert receipt_path.is_file()
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["state"] == (
        "prepared"
    )
    assert not (data_root / "Cable1").exists()
    assert not (data_root / ".portable_models").exists()

    with pytest.raises(
        PortableTrainingImportError,
        match="prepared portable import receipt has no exact latest-pointer commit",
    ):
        import_portable_training_package(package, training_root)


def test_receipt_finalize_failure_keeps_commit_and_retry_finalizes(
    tmp_path,
    monkeypatch,
):
    package = _portable_package(tmp_path, source_schema_version=6)
    training_root = tmp_path / "training"
    data_root = training_root / "data"
    receipt_path = (
        data_root
        / ".portable_imports"
        / "source-job-abcdef123456"
        / "import.json"
    )
    latest_path = data_root / ".operator_handoff" / "latest.json"
    real_write = portable_package_module._write_json_atomic

    with monkeypatch.context() as patch_context:

        def deny_committed_receipt(path, payload):
            if (
                Path(path).name == "import.json"
                and payload.get("state") == "committed"
            ):
                raise PermissionError("simulated receipt finalization denial")
            real_write(Path(path), payload)

        patch_context.setattr(
            portable_package_module,
            "_write_json_atomic",
            deny_committed_receipt,
        )
        report = import_portable_training_package(package, training_root)

    assert json.loads(latest_path.read_text(encoding="utf-8")) == json.loads(
        report.handoff_path.read_text(encoding="utf-8")
    )
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["state"] == (
        "prepared"
    )

    retried = import_portable_training_package(package, training_root)

    assert retried.reused_existing is True
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["state"] == (
        "committed"
    )


def test_prepared_receipt_post_replace_error_is_reconciled(
    tmp_path,
    monkeypatch,
):
    package = _portable_package(tmp_path, source_schema_version=6)
    training_root = tmp_path / "training"
    receipt_path = (
        training_root
        / "data"
        / ".portable_imports"
        / "source-job-abcdef123456"
        / "import.json"
    )
    real_write = portable_package_module._write_json_atomic
    injected = False

    def report_error_after_replace(path, payload):
        nonlocal injected
        real_write(Path(path), payload)
        if (
            not injected
            and Path(path).name == "import.json"
            and payload.get("state") == "prepared"
        ):
            injected = True
            raise PermissionError("simulated prepared post-replace error")

    monkeypatch.setattr(
        portable_package_module,
        "_write_json_atomic",
        report_error_after_replace,
    )

    import_portable_training_package(package, training_root)

    assert injected is True
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["state"] == (
        "committed"
    )


def test_latest_post_replace_error_is_reconciled(tmp_path, monkeypatch):
    package = _portable_package(tmp_path, source_schema_version=6)
    training_root = tmp_path / "training"
    data_root = training_root / "data"
    latest_path = data_root / ".operator_handoff" / "latest.json"
    real_write = portable_package_module._write_json_atomic
    injected = False

    def report_error_after_replace(path, payload):
        nonlocal injected
        real_write(Path(path), payload)
        if not injected and Path(path).name == "latest.json":
            injected = True
            raise PermissionError("simulated latest post-replace error")

    monkeypatch.setattr(
        portable_package_module,
        "_write_json_atomic",
        report_error_after_replace,
    )

    report = import_portable_training_package(package, training_root)

    assert injected is True
    assert json.loads(latest_path.read_text(encoding="utf-8")) == json.loads(
        report.handoff_path.read_text(encoding="utf-8")
    )
    receipt_path = (
        data_root
        / ".portable_imports"
        / "source-job-abcdef123456"
        / "import.json"
    )
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["state"] == (
        "committed"
    )


def test_package_handle_rejects_path_bytes_changed_after_zip_extraction(
    tmp_path,
    monkeypatch,
):
    package = _portable_package(tmp_path, source_schema_version=6)
    replacement_root = tmp_path / "replacement"
    replacement_root.mkdir()
    replacement = _portable_package(replacement_root, source_schema_version=4)
    training_root = tmp_path / "training"
    real_extract = portable_package_module._extract_verified_files

    def extract_then_replace_package(archive, inventory, staging):
        real_extract(archive, inventory, staging)
        package.write_bytes(replacement.read_bytes())

    monkeypatch.setattr(
        portable_package_module,
        "_extract_verified_files",
        extract_then_replace_package,
    )

    with pytest.raises(
        PortableTrainingImportError,
        match="changed while its ZIP payload was being verified",
    ):
        import_portable_training_package(package, training_root)

    data_root = training_root / "data"
    assert not (data_root / "Cable1").exists()
    assert not (data_root / ".portable_models").exists()
    assert not (data_root / ".operator_handoff" / "latest.json").exists()


def test_late_handoff_failure_rolls_back_all_shared_mutations(
    tmp_path,
    monkeypatch,
):
    package = _portable_package(tmp_path, source_schema_version=6)
    training_root = tmp_path / "training"
    data_root = training_root / "data"
    existing_manifest = (
        data_root
        / "Cable1"
        / "A"
        / "metadata"
        / "review_dataset_manifest.csv"
    )
    existing_manifest.parent.mkdir(parents=True)
    original_manifest = (
        "sample_id,image_sha256,annotation_status\n"
        f"existing,{hashlib.sha256(b'existing').hexdigest()},verified_annotation\n"
    ).encode()
    existing_manifest.write_bytes(original_manifest)

    def reject_local_handoff(*args, **kwargs):
        raise portable_package_module.OperatorHandoffError(
            "simulated late handoff conflict"
        )

    monkeypatch.setattr(
        portable_package_module,
        "load_operator_handoff",
        reject_local_handoff,
    )

    with pytest.raises(
        PortableTrainingImportError,
        match="simulated late handoff conflict",
    ):
        import_portable_training_package(package, training_root)

    assert existing_manifest.read_bytes() == original_manifest
    assert not (
        data_root / "Cable1" / "A" / "raw" / "images" / "review_sample.jpg"
    ).exists()
    assert not (data_root / ".portable_models").exists()
    assert not (data_root / ".operator_handoff" / "latest.json").exists()
    assert not (
        data_root / ".portable_imports" / "source-job-abcdef123456" / "import.json"
    ).exists()


def test_model_conflict_is_preflighted_before_dataset_mutation(tmp_path):
    package = _portable_package(tmp_path, source_schema_version=6)
    training_root = tmp_path / "training"
    data_root = training_root / "data"
    conflicting_weight = (
        data_root
        / ".portable_models"
        / "source-job-abcdef123456"
        / "models"
        / "Cable1"
        / "A"
        / "yolo"
        / "weights"
        / "best.pt"
    )
    conflicting_weight.parent.mkdir(parents=True)
    conflicting_weight.write_bytes(b"conflict")

    with pytest.raises(
        PortableTrainingImportError,
        match="Imported file conflicts with existing data",
    ):
        import_portable_training_package(package, training_root)

    assert conflicting_weight.read_bytes() == b"conflict"
    assert not (data_root / "Cable1").exists()
    assert not (data_root / ".operator_handoff" / "latest.json").exists()
    assert not (
        data_root / ".portable_imports" / "source-job-abcdef123456" / "import.json"
    ).exists()


def test_legacy_schema_one_receipt_remains_committed_when_latest_is_superseded(
    tmp_path,
):
    package = _portable_package(tmp_path, source_schema_version=6)
    training_root = tmp_path / "training"
    first = import_portable_training_package(package, training_root)
    data_root = training_root / "data"
    receipt_path = (
        data_root
        / ".portable_imports"
        / "source-job-abcdef123456"
        / "import.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["schema_version"] = 1
    for schema_two_field in (
        "state",
        "committed_at",
        "handoff_sha256",
        "status_sha256",
        "status_identity_sha256",
    ):
        receipt.pop(schema_two_field, None)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    latest_path = data_root / ".operator_handoff" / "latest.json"
    superseding_latest = {"schema_version": 4, "job_id": "newer-job"}
    latest_path.write_text(json.dumps(superseding_latest), encoding="utf-8")

    reused = import_portable_training_package(package, training_root)

    assert reused.reused_existing is True
    assert reused.handoff_path == first.handoff_path
    assert json.loads(latest_path.read_text(encoding="utf-8")) == (
        superseding_latest
    )


def test_import_mutation_journal_rollback_restores_file_and_removes_new_tree(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    existing = data_root / "Cable1" / "A" / "metadata" / "manifest.csv"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"original")
    created = data_root / "Cable1" / "A" / "new" / "artifact.bin"
    journal = portable_package_module._ImportMutationJournal(data_root)
    backup_root = journal._backup_root

    journal.record_before_write(existing)
    journal.record_before_write(existing)
    journal.record_before_write(created)
    existing.write_bytes(b"mutated")
    created.parent.mkdir(parents=True)
    created.write_bytes(b"new")

    journal.rollback()
    journal.rollback()

    assert existing.read_bytes() == b"original"
    assert not created.exists()
    assert not created.parent.exists()
    assert not backup_root.exists()


def test_import_mutation_journal_commit_keeps_bytes_and_rejects_unsafe_targets(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    destination = data_root / "models" / "best.pt"
    destination.parent.mkdir()
    destination.write_bytes(b"before")
    outside = tmp_path / "outside.pt"
    directory_target = data_root / "directory-target"
    directory_target.mkdir()
    journal = portable_package_module._ImportMutationJournal(data_root)
    backup_root = journal._backup_root

    with pytest.raises(
        PortableTrainingImportError,
        match="destination escapes training data",
    ):
        journal.record_before_write(outside)
    with pytest.raises(
        PortableTrainingImportError,
        match="destination is not a file",
    ):
        journal.record_before_write(directory_target)
    journal.record_before_write(destination)
    destination.write_bytes(b"committed")
    journal.commit()
    journal.commit()

    assert destination.read_bytes() == b"committed"
    assert not outside.exists()
    assert directory_target.is_dir()
    assert not backup_root.exists()


def test_shared_import_plan_rejects_staged_source_changed_after_preflight(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    source = tmp_path / "staging" / "artifact.bin"
    source.parent.mkdir()
    source.write_bytes(b"verified")
    destination = data_root / "models" / "artifact.bin"
    source_sha256 = hashlib.sha256(b"verified").hexdigest()
    plan = portable_package_module._SharedImportPlan(
        copies=(
            portable_package_module._FileCopyPlan(
                source=source,
                destination=destination,
                source_sha256=source_sha256,
                write_required=True,
            ),
        ),
        csv_writes=(),
    )
    journal = portable_package_module._ImportMutationJournal(data_root)
    source.write_bytes(b"changed")

    with pytest.raises(
        PortableTrainingImportError,
        match="staged file changed after preflight",
    ):
        portable_package_module._apply_shared_import_plan(plan, journal)

    journal.rollback()
    assert not destination.exists()
    assert not journal._backup_root.exists()


def test_shared_import_plan_rechecks_existing_destination_and_manifest(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    source = tmp_path / "staging" / "artifact.bin"
    source.parent.mkdir()
    source.write_bytes(b"same")
    destination = data_root / "models" / "artifact.bin"
    destination.parent.mkdir()
    destination.write_bytes(b"same")
    manifest = data_root / "metadata" / "manifest.csv"
    manifest.parent.mkdir()
    manifest.write_text("sample_id\nold\n", encoding="utf-8")
    expected_manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    plan = portable_package_module._SharedImportPlan(
        copies=(
            portable_package_module._FileCopyPlan(
                source=source,
                destination=destination,
                source_sha256=source_sha256,
                write_required=False,
            ),
        ),
        csv_writes=(
            portable_package_module._CsvWritePlan(
                destination=manifest,
                rows=({"sample_id": "new"},),
                expected_destination_sha256=expected_manifest_sha256,
            ),
        ),
    )
    journal = portable_package_module._ImportMutationJournal(data_root)
    destination.write_bytes(b"changed")

    with pytest.raises(
        PortableTrainingImportError,
        match="destination changed after preflight",
    ):
        portable_package_module._apply_shared_import_plan(plan, journal)

    destination.write_bytes(b"same")
    manifest.write_text("sample_id\nchanged\n", encoding="utf-8")
    with pytest.raises(
        PortableTrainingImportError,
        match="manifest changed after preflight",
    ):
        portable_package_module._apply_shared_import_plan(plan, journal)

    journal.rollback()
    assert destination.read_bytes() == b"same"
    assert manifest.read_text(encoding="utf-8") == "sample_id\nchanged\n"


def test_legacy_dataset_merge_normalizes_ready_and_pending_rows_idempotently(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "data" / "Cable1" / "A"
    for relative, content in {
        "raw/images/ready.jpg": b"ready-image",
        "raw/labels/ready.txt": b"0 0.5 0.5 0.2 0.2\n",
        "review_pending/images/pending.jpg": b"pending-image",
        "review_pending/labels/pending.txt": b"",
        "color_review/color.json": b"{}",
        "metadata/classes.json": b'{"names":["Cable"]}',
    }.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    ready_manifest = source / "metadata" / "review_dataset_manifest.csv"
    ready_manifest.write_text(
        "sample_id,image_sha256,source_image,output_image,output_label\n"
        "ready,ready-sha,D:/source.jpg,D:/ready.jpg,D:/ready.txt\n",
        encoding="utf-8",
    )
    pending_manifest = source / "review_pending" / "manifest.csv"
    pending_manifest.write_text(
        "sample_id,image_sha256,source_image,output_image,output_label\n"
        "pending,pending-sha,D:/pending-source.jpg,D:/pending.jpg,D:/pending.txt\n",
        encoding="utf-8",
    )

    portable_package_module._merge_dataset(
        source,
        destination,
        package_id="portable-job",
    )
    portable_package_module._merge_dataset(
        source,
        destination,
        package_id="portable-job",
    )

    ready_rows = portable_package_module._read_csv(
        destination / "metadata" / "review_dataset_manifest.csv"
    )
    pending_rows = portable_package_module._read_csv(
        destination / "review_pending" / "manifest.csv"
    )
    assert len(ready_rows) == 1
    assert ready_rows[0]["portable_package_id"] == "portable-job"
    assert ready_rows[0]["portable_original_source_image"] == "D:/source.jpg"
    assert Path(ready_rows[0]["output_image"]) == (
        destination / "raw" / "images" / "ready.jpg"
    ).resolve()
    assert len(pending_rows) == 1
    assert pending_rows[0]["detection_source_image"] == pending_rows[0][
        "output_image"
    ]
    assert Path(pending_rows[0]["output_label"]) == (
        destination / "review_pending" / "labels" / "pending.txt"
    ).resolve()
    assert (destination / "raw" / "images" / "ready.jpg").read_bytes() == (
        b"ready-image"
    )
    assert (
        destination / "review_pending" / "images" / "pending.jpg"
    ).read_bytes() == b"pending-image"
    assert (destination / "color_review" / "color.json").read_bytes() == b"{}"
    assert (destination / "metadata" / "classes.json").read_bytes() == (
        b'{"names":["Cable"]}'
    )


def test_legacy_manifest_merge_rejects_identity_conflict_without_overwrite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    destination = tmp_path / "destination.csv"
    source.write_text(
        "sample_id,image_sha256,output_image,output_label\n"
        "sample,new-sha,D:/new.jpg,D:/new.txt\n",
        encoding="utf-8",
    )
    original = (
        "sample_id,image_sha256,output_image,output_label\n"
        "sample,old-sha,D:/old.jpg,D:/old.txt\n"
    )
    destination.write_text(original, encoding="utf-8")

    with pytest.raises(
        PortableTrainingImportError,
        match="Conflicting manifest sample identity: sample",
    ):
        portable_package_module._merge_csv_manifest(
            source,
            destination,
            dataset_root=tmp_path / "dataset",
            pending=False,
            package_id="portable-job",
        )

    assert destination.read_text(encoding="utf-8") == original


def test_legacy_manifest_merge_requires_ready_manifest_but_allows_no_pending(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.csv"
    destination = tmp_path / "destination.csv"

    with pytest.raises(PortableTrainingImportError, match="manifest is missing"):
        portable_package_module._merge_csv_manifest(
            missing,
            destination,
            dataset_root=tmp_path / "dataset",
            pending=False,
            package_id="portable-job",
        )
    portable_package_module._merge_csv_manifest(
        missing,
        destination,
        dataset_root=tmp_path / "dataset",
        pending=True,
        package_id="portable-job",
    )

    assert not destination.exists()


def test_verified_portable_copy_checksum_failure_removes_temporary_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination" / "artifact.bin"
    source.write_bytes(b"source")

    with pytest.raises(
        PortableTrainingImportError,
        match="failed checksum verification",
    ):
        portable_package_module._copy_verified_with_sha(
            source,
            destination,
            expected_sha256="0" * 64,
        )

    assert not destination.exists()
    assert not list(destination.parent.glob(".*.tmp"))


def test_portable_import_missing_or_unopenable_package_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing.zip"
    training_root = tmp_path / "training"

    with pytest.raises(PortableTrainingImportError, match="Training package not found"):
        import_portable_training_package(missing, training_root)
    assert not (training_root / "data").exists()

    package = _portable_package(tmp_path, source_schema_version=6)
    real_open = Path.open

    def deny_package_open(path: Path, *args, **kwargs):
        if path.resolve() == package.resolve():
            raise PermissionError("simulated package open denial")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_package_open)
    with pytest.raises(PortableTrainingImportError, match="Unable to open"):
        import_portable_training_package(package, training_root)

    data_root = training_root / "data"
    assert not (data_root / ".operator_handoff" / "latest.json").exists()
    assert not (data_root / "Cable1").exists()


def test_source_contract_rejects_identity_class_and_sample_inconsistency() -> None:
    class_names = ["Cable", "Connector"]
    class_hash = portable_package_module._class_schema_hash(tuple(class_names))
    handoff = {
        "schema_version": 6,
        "job_id": "source-job",
        "training_options": {
            "epochs": 20,
            "augmentations_per_image": 10,
            "batch": 8,
            "imgsz": 640,
            "position_training_mode": "auto",
            "position_activation": "preserve",
        },
        "targets": [
            {
                "product": "Cable1",
                "area": "A",
                "sample_ids": ["ready", "pending"],
                "pending_sample_ids": ["pending"],
                "class_names": class_names,
                "class_schema_hash": class_hash,
                "class_contract_required": True,
            }
        ],
    }
    metadata = {
        "source_job_id": "source-job",
        "sample_ids": ["ready", "pending"],
        "pending_sample_ids": ["pending"],
    }

    options, sample_ids, pending_ids = portable_package_module._validate_source_contract(
        handoff,
        metadata,
        product="Cable1",
        area="A",
    )
    assert options["epochs"] == 20
    assert sample_ids == ("ready", "pending")
    assert pending_ids == ("pending",)

    def cloned_contract() -> tuple[dict, dict]:
        return json.loads(json.dumps(handoff)), json.loads(json.dumps(metadata))

    bad_handoff, bad_metadata = cloned_contract()
    bad_handoff["job_id"] = "different-job"
    with pytest.raises(PortableTrainingImportError, match="job_id does not match"):
        portable_package_module._validate_source_contract(
            bad_handoff, bad_metadata, product="Cable1", area="A"
        )

    bad_handoff, bad_metadata = cloned_contract()
    bad_handoff["targets"] = []
    with pytest.raises(PortableTrainingImportError, match="target is invalid"):
        portable_package_module._validate_source_contract(
            bad_handoff, bad_metadata, product="Cable1", area="A"
        )

    bad_handoff, bad_metadata = cloned_contract()
    bad_handoff["targets"][0]["area"] = "B"
    with pytest.raises(PortableTrainingImportError, match="does not match its metadata"):
        portable_package_module._validate_source_contract(
            bad_handoff, bad_metadata, product="Cable1", area="A"
        )

    bad_handoff, bad_metadata = cloned_contract()
    bad_handoff["targets"][0]["class_names"] = []
    bad_handoff["targets"][0].pop("class_schema_hash")
    with pytest.raises(PortableTrainingImportError, match="class contract is missing"):
        portable_package_module._validate_source_contract(
            bad_handoff, bad_metadata, product="Cable1", area="A"
        )

    for invalid_names, expected_message in (
        (["Cable", " "], "must not be empty"),
        (["Cable", "Cable"], "must be unique"),
    ):
        bad_handoff, bad_metadata = cloned_contract()
        bad_handoff["targets"][0]["class_names"] = invalid_names
        with pytest.raises(PortableTrainingImportError, match=expected_message):
            portable_package_module._validate_source_contract(
                bad_handoff, bad_metadata, product="Cable1", area="A"
            )

    bad_handoff, bad_metadata = cloned_contract()
    bad_handoff["targets"][0]["class_schema_hash"] = "0" * 64
    with pytest.raises(PortableTrainingImportError, match="checksum does not match"):
        portable_package_module._validate_source_contract(
            bad_handoff, bad_metadata, product="Cable1", area="A"
        )

    bad_handoff, bad_metadata = cloned_contract()
    bad_metadata["sample_ids"] = ["ready"]
    with pytest.raises(PortableTrainingImportError, match="sample_ids do not match"):
        portable_package_module._validate_source_contract(
            bad_handoff, bad_metadata, product="Cable1", area="A"
        )

    bad_handoff, bad_metadata = cloned_contract()
    bad_metadata["pending_sample_ids"] = ["outside"]
    bad_handoff["targets"][0]["pending_sample_ids"] = ["outside"]
    with pytest.raises(PortableTrainingImportError, match="must be a subset"):
        portable_package_module._validate_source_contract(
            bad_handoff, bad_metadata, product="Cable1", area="A"
        )


def test_staged_sample_contract_accepts_partition_and_rejects_overlap_or_duplicates(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    ready_manifest = dataset / "metadata" / "review_dataset_manifest.csv"
    pending_manifest = dataset / "review_pending" / "manifest.csv"
    ready_manifest.parent.mkdir(parents=True)
    pending_manifest.parent.mkdir(parents=True)
    ready_manifest.write_text("sample_id\nready\n", encoding="utf-8")
    pending_manifest.write_text("sample_id\npending\n", encoding="utf-8")

    portable_package_module._validate_staged_sample_contract(
        dataset,
        sample_ids=("ready", "pending"),
        pending_sample_ids=("pending",),
    )

    pending_manifest.write_text("sample_id\nready\n", encoding="utf-8")
    with pytest.raises(PortableTrainingImportError, match="both ready and pending"):
        portable_package_module._validate_staged_sample_contract(
            dataset,
            sample_ids=("ready",),
            pending_sample_ids=("ready",),
        )

    pending_manifest.write_text("sample_id\nunexpected\n", encoding="utf-8")
    with pytest.raises(PortableTrainingImportError, match="do not match"):
        portable_package_module._validate_staged_sample_contract(
            dataset,
            sample_ids=("ready", "pending"),
            pending_sample_ids=("pending",),
        )

    ready_manifest.write_text("sample_id\nready\nready\n", encoding="utf-8")
    with pytest.raises(PortableTrainingImportError, match="duplicate sample_ids"):
        portable_package_module._validated_manifest_sample_ids(
            ready_manifest,
            "ready",
        )
    ready_manifest.write_text("sample_id,other\n,value\n", encoding="utf-8")
    with pytest.raises(PortableTrainingImportError, match="empty sample_id"):
        portable_package_module._validated_manifest_sample_ids(
            ready_manifest,
            "ready",
        )


def test_sample_id_lists_reject_invalid_and_duplicate_values() -> None:
    for invalid in (None, "sample", [""], ["sample", 1]):
        with pytest.raises(PortableTrainingImportError, match="non-empty strings"):
            portable_package_module._validated_sample_id_list(invalid, "samples")
    with pytest.raises(PortableTrainingImportError, match="contains duplicates"):
        portable_package_module._validated_sample_id_list(
            ["sample", " sample "],
            "samples",
        )

    assert portable_package_module._validated_sample_id_list(
        [" sample ", "second"],
        "samples",
    ) == ("sample", "second")
