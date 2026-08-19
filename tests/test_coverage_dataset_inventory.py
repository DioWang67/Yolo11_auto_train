from __future__ import annotations

import json
from pathlib import Path

import pytest

import picture_tool
from picture_tool import dataset_inventory
from picture_tool.dataset_inventory import DatasetInventoryItem


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        (Path("yolo_Cable1_B_sample.png"), ("Cable1", "B")),
        (Path("anomaly/sample.png"), ("unclassified", "")),
        (Path("Cable1/processed/images/sample.png"), ("Cable1", "A")),
        (Path("Cable1/LongAreaName/sample.png"), ("Cable1", "")),
    ],
)
def test_inventory_target_inference_conventions(
    relative: Path,
    expected: tuple[str, str],
) -> None:
    assert dataset_inventory._infer_target(relative) == expected


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        (Path("Cable1/A/split/train/images/a.png"), ("split", "train")),
        (Path("Cable1/A/qc/images/a.png"), ("qc", "")),
        (Path("anomaly/good/a.png"), ("anomaly", "")),
        (Path("original_picture/a.png"), ("raw", "")),
        (Path("Cable1/A/processed/a.png"), ("processed", "")),
        (Path("Cable1/A/review/a.png"), ("review", "")),
        (Path("augmented/a.png"), ("augmented", "")),
        (Path("legacy/a.png"), ("legacy", "")),
    ],
)
def test_inventory_role_inference_conventions(
    relative: Path,
    expected: tuple[str, str],
) -> None:
    assert dataset_inventory._infer_role(relative) == expected


def test_matching_label_paths_and_statuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "images" / "sample.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
    paired = tmp_path / "labels" / "sample.txt"
    assert dataset_inventory._matching_label_path(image) == paired
    assert dataset_inventory._label_status(paired) == "missing"
    paired.parent.mkdir()
    paired.write_text("", encoding="utf-8")
    assert dataset_inventory._label_status(paired) == "empty"
    paired.write_text("0 0.5 0.5 1 1", encoding="utf-8")
    assert dataset_inventory._label_status(paired) == "nonempty"

    sibling_image = tmp_path / "legacy.png"
    sibling_label = tmp_path / "legacy.txt"
    assert dataset_inventory._matching_label_path(sibling_image) is None
    sibling_label.write_text("label", encoding="utf-8")
    assert dataset_inventory._matching_label_path(sibling_image) == sibling_label

    original_read_text = Path.read_text

    def fail_selected(path: Path, *args: object, **kwargs: object) -> str:
        if path == paired:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_selected)
    assert dataset_inventory._label_status(paired) == "unreadable"


@pytest.mark.parametrize(
    ("product", "area", "role", "status", "duplicate", "expected"),
    [
        ("unclassified", "A", "raw", "nonempty", False, "review_target"),
        ("Cable1", "", "raw", "nonempty", False, "review_target"),
        ("Cable1", "A", "processed", "nonempty", False, "generated_do_not_resplit"),
        ("Cable1", "A", "split", "nonempty", False, "generated_do_not_resplit"),
        ("Cable1", "A", "qc", "nonempty", False, "qc_only"),
        ("Cable1", "A", "anomaly", "nonempty", False, "anomaly_training_only"),
        ("Cable1", "A", "raw", "nonempty", True, "review_duplicate"),
        ("Cable1", "A", "raw", "missing", False, "review_annotation"),
        ("Cable1", "A", "raw", "nonempty", False, "retain_raw"),
        ("Cable1", "A", "legacy", "nonempty", False, "review_role"),
    ],
)
def test_recommended_action_covers_each_classification_policy(
    product: str,
    area: str,
    role: str,
    status: str,
    duplicate: bool,
    expected: str,
) -> None:
    assert dataset_inventory._recommended_action(product, area, role, status, duplicate) == expected


def _record(path: Path, *, split: str, digest: str) -> DatasetInventoryItem:
    return DatasetInventoryItem(
        image_path=str(path),
        product="Cable1",
        area="A",
        role="split",
        split=split,
        label_path="",
        label_status="missing",
        sha256=digest,
        duplicate_of="",
        recommended_action="generated_do_not_resplit",
    )


def test_inventory_summary_detects_cross_split_exact_and_augmented_families(
    tmp_path: Path,
) -> None:
    records = [
        _record(tmp_path / "train" / "part_aug_1.png", split="train", digest="same"),
        _record(tmp_path / "val" / "part_aug2.png", split="val", digest="same"),
        DatasetInventoryItem(
            image_path=str(tmp_path / "raw.png"),
            product="Cable1",
            area="A",
            role="raw",
            split="",
            label_path="",
            label_status="missing",
            sha256="",
            duplicate_of="first.png",
            recommended_action="review_duplicate",
        ),
    ]

    _, summary_path = dataset_inventory.write_inventory(records, tmp_path / "inventory.csv")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["cross_split_exact_duplicate_groups"] == 1
    assert summary["cross_split_augmented_family_groups"] == 1
    assert summary["duplicates"] == 1


def test_inventory_missing_root_no_hash_and_cli_entrypoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset root not found"):
        dataset_inventory.inventory_dataset(tmp_path / "missing")

    image = tmp_path / "Cable1" / "A" / "raw" / "sample.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"large enough")
    records = dataset_inventory.inventory_dataset(tmp_path, compute_hashes=False)
    assert records[0].sha256 == ""
    assert dataset_inventory._sha256_file(image)

    output = tmp_path / "reports" / "inventory.csv"
    assert dataset_inventory.main(
        ["--data-root", str(tmp_path), "--output", str(output), "--no-hash"]
    ) == 0
    assert output.is_file()
    assert "Classified" in capsys.readouterr().out


@pytest.mark.parametrize(
    "name",
    [
        "process_anomaly_detection",
        "ImageAugmentor",
        "YoloDataAugmentor",
        "convert_format",
        "split_dataset",
        "setup_logging",
    ],
)
def test_public_package_lazy_exports_resolve(name: str) -> None:
    assert picture_tool.__getattr__(name) is not None


def test_public_package_lazy_export_rejects_unknown_name() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        picture_tool.__getattr__("unknown")
