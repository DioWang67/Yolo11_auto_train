import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from picture_tool.gui.operator_handoff import load_operator_handoff
from picture_tool.portable_training_package import (
    PortableTrainingImportError,
    import_portable_training_package,
)


def _portable_package(tmp_path: Path, *, invalid_checksum: bool = False) -> Path:
    package = tmp_path / "portable.zip"
    image_name = "review_sample.jpg"
    label_name = "review_sample.txt"
    original_dataset = Path("D:/source/training/data/Cable1/A")
    class_names = ["Black"]
    class_hash = hashlib.sha256(
        json.dumps(class_names, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    handoff = {
        "schema_version": 4,
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
            "weights: models/Cable1/A/yolo/weights/best.pt\n"
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
        "sample_ids": ["sample"],
        "pending_sample_ids": [],
        "files": inventory,
    }
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("package.json", json.dumps(metadata))
        for name, content in files.items():
            archive.writestr(name, content)
    return package


def test_import_portable_package_rewrites_paths_and_is_idempotent(tmp_path):
    package = _portable_package(tmp_path)
    training_root = tmp_path / "remote-training"

    report = import_portable_training_package(package, training_root)
    loaded = load_operator_handoff(report.handoff_path, training_root=training_root)
    imported_target = loaded.selected_target
    second = import_portable_training_package(package, training_root)

    assert imported_target.dataset_root == (
        training_root / "data" / "Cable1" / "A"
    ).resolve()
    assert loaded.inference_models_dir.is_relative_to(
        (training_root / "data" / ".portable_models").resolve()
    )
    assert imported_target.pending_count == 0
    assert imported_target.total_ready_count == 1
    assert second.reused_existing is True
    manifest = imported_target.dataset_root / "metadata" / "review_dataset_manifest.csv"
    manifest_text = manifest.read_text(encoding="utf-8")
    assert str(imported_target.dataset_root.resolve()) in manifest_text
    assert "D:/source/training/data/Cable1/A/raw/images" not in manifest_text


def test_import_portable_package_rejects_checksum_mismatch(tmp_path):
    package = _portable_package(tmp_path, invalid_checksum=True)

    with pytest.raises(PortableTrainingImportError, match="Checksum mismatch"):
        import_portable_training_package(package, tmp_path / "training")
