import json

from picture_tool.dataset_inventory import inventory_dataset, write_inventory


def test_inventory_classifies_product_area_and_label(tmp_path):
    images = tmp_path / "PCBA1" / "B" / "raw" / "images"
    labels = tmp_path / "PCBA1" / "B" / "raw" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "board.png").write_bytes(b"board")
    (labels / "board.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    records = inventory_dataset(tmp_path)

    assert len(records) == 1
    assert records[0].product == "PCBA1"
    assert records[0].area == "B"
    assert records[0].role == "raw"
    assert records[0].label_status == "nonempty"
    assert records[0].recommended_action == "retain_raw"


def test_inventory_marks_duplicates_and_writes_summary(tmp_path):
    for name in ("one", "two"):
        images = tmp_path / "Cable1" / "A" / "raw" / "images"
        labels = tmp_path / "Cable1" / "A" / "raw" / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        (images / f"{name}.png").write_bytes(b"duplicate")
        (labels / f"{name}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )

    records = inventory_dataset(tmp_path)
    csv_path, summary_path = write_inventory(records, tmp_path / "report.csv")

    assert sum(bool(record.duplicate_of) for record in records) == 1
    assert csv_path.exists()
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["duplicates"] == 1
    assert summary["cross_split_exact_duplicate_groups"] == 0
