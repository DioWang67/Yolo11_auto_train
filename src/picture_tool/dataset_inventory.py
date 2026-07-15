"""Inventory and classify training images without moving source files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from picture_tool.utils.io_utils import DEFAULT_IMAGE_EXTS


TECHNICAL_ROOTS = {
    "anomaly",
    "augmented",
    "converter_output",
    "demo_doctor",
    "original_picture",
    "raw",
    "訓練用資料",
}


@dataclass(frozen=True)
class DatasetInventoryItem:
    """Classification record for one image in a training workspace."""

    image_path: str
    product: str
    area: str
    role: str
    split: str
    label_path: str
    label_status: str
    sha256: str
    duplicate_of: str
    recommended_action: str


def inventory_dataset(
    data_root: str | Path, *, compute_hashes: bool = True
) -> list[DatasetInventoryItem]:
    """Scan and classify image files without modifying the dataset.

    Args:
        data_root: Root directory that contains product and legacy datasets.
        compute_hashes: Compute SHA-256 identities for duplicate detection.

    Returns:
        Stable, path-sorted inventory records.

    Raises:
        FileNotFoundError: If ``data_root`` does not exist.
    """
    root = Path(data_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    records: list[DatasetInventoryItem] = []
    first_by_hash: dict[str, str] = {}
    images = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in DEFAULT_IMAGE_EXTS
    )
    for image_path in images:
        relative = image_path.relative_to(root)
        product, area = _infer_target(relative)
        role, split = _infer_role(relative)
        label_path = _matching_label_path(image_path)
        label_status = _label_status(label_path)
        digest = _sha256_file(image_path) if compute_hashes else ""
        duplicate_of = first_by_hash.get(digest, "") if digest else ""
        if digest and not duplicate_of:
            first_by_hash[digest] = str(image_path)
        records.append(
            DatasetInventoryItem(
                image_path=str(image_path),
                product=product,
                area=area,
                role=role,
                split=split,
                label_path=str(label_path) if label_path else "",
                label_status=label_status,
                sha256=digest,
                duplicate_of=duplicate_of,
                recommended_action=_recommended_action(
                    product, area, role, label_status, bool(duplicate_of)
                ),
            )
        )
    return records


def write_inventory(
    records: list[DatasetInventoryItem], csv_path: str | Path
) -> tuple[Path, Path]:
    """Write CSV details and a compact JSON summary."""
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(DatasetInventoryItem.__dataclass_fields__)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)

    split_hashes: dict[str, set[str]] = {}
    split_sources: dict[str, set[str]] = {}
    for record in records:
        if record.role != "split" or not record.split:
            continue
        if record.sha256:
            split_hashes.setdefault(record.sha256, set()).add(record.split)
        source_key = re.sub(
            r"(?:_aug_?\d+)$",
            "",
            Path(record.image_path).stem,
            flags=re.IGNORECASE,
        )
        split_sources.setdefault(source_key, set()).add(record.split)

    summary_path = output_path.with_suffix(".summary.json")
    summary = {
        "images": len(records),
        "products": dict(Counter(record.product for record in records)),
        "roles": dict(Counter(record.role for record in records)),
        "label_status": dict(Counter(record.label_status for record in records)),
        "recommended_actions": dict(
            Counter(record.recommended_action for record in records)
        ),
        "duplicates": sum(bool(record.duplicate_of) for record in records),
        "cross_split_exact_duplicate_groups": sum(
            len(splits) > 1 for splits in split_hashes.values()
        ),
        "cross_split_augmented_family_groups": sum(
            len(splits) > 1 for splits in split_sources.values()
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path, summary_path


def _infer_target(relative: Path) -> tuple[str, str]:
    parts = relative.parts
    match = re.search(r"(?:^|-)yolo_([^_]+)_([^_]+)_", relative.name)
    if match:
        return match.group(1), match.group(2)
    if not parts or parts[0].lower() in TECHNICAL_ROOTS:
        return "unclassified", ""
    product = parts[0]
    area = parts[1] if len(parts) > 1 and len(parts[1]) <= 8 else ""
    if area.lower() in {"raw", "processed", "split", "qc", "metadata"}:
        area = "A"
    return product, area


def _infer_role(relative: Path) -> tuple[str, str]:
    lowered = [part.lower() for part in relative.parts]
    for split in ("train", "val", "test"):
        if split in lowered and "split" in lowered:
            return "split", split
    if any("qc" in part for part in lowered) or lowered[0] == "訓練用資料":
        return "qc", ""
    if lowered[0] == "anomaly":
        return "anomaly", ""
    if lowered[0] == "original_picture":
        return "raw", ""
    for role in ("raw", "processed", "qc", "review"):
        if role in lowered:
            return role, ""
    if "augmented" in lowered:
        return "augmented", ""
    return "legacy", ""


def _matching_label_path(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    for index in range(len(parts) - 2, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    sibling = image_path.with_suffix(".txt")
    return sibling if sibling.exists() else None


def _label_status(label_path: Path | None) -> str:
    if label_path is None or not label_path.exists():
        return "missing"
    try:
        return "nonempty" if label_path.read_text(encoding="utf-8").strip() else "empty"
    except (OSError, UnicodeDecodeError):
        return "unreadable"


def _recommended_action(
    product: str,
    area: str,
    role: str,
    label_status: str,
    duplicate: bool,
) -> str:
    if product == "unclassified" or not area:
        return "review_target"
    if role in {"processed", "split", "augmented"}:
        return "generated_do_not_resplit"
    if role == "qc":
        return "qc_only"
    if role == "anomaly":
        return "anomaly_training_only"
    if duplicate:
        return "review_duplicate"
    if label_status in {"missing", "empty", "unreadable"}:
        return "review_annotation"
    if role == "raw":
        return "retain_raw"
    return "review_role"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for non-destructive dataset classification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="runs/dataset_inventory.csv")
    parser.add_argument("--no-hash", action="store_true")
    args = parser.parse_args(argv)
    records = inventory_dataset(args.data_root, compute_hashes=not args.no_hash)
    csv_path, summary_path = write_inventory(records, args.output)
    print(f"Classified {len(records)} images: {csv_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
