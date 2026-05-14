from pathlib import Path

from picture_tool.gui.annotation_tracker import AnnotationTracker


def test_validate_annotations_ignores_classes_txt(tmp_path: Path) -> None:
    """classes.txt should not be parsed as a YOLO annotation file."""
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    (label_dir / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")
    (label_dir / "image_001.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n",
        encoding="utf-8",
    )

    errors = AnnotationTracker().validate_annotations(label_dir, num_classes=2)

    assert errors == []


def test_class_distribution_ignores_classes_txt(tmp_path: Path) -> None:
    """Class distribution should count only real annotation files."""
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    (label_dir / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")
    (label_dir / "image_001.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n",
        encoding="utf-8",
    )

    distribution = AnnotationTracker().get_class_distribution(
        label_dir,
        ["cat", "dog"],
    )

    assert distribution == {"cat": 1, "dog": 1}
