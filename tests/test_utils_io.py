
from picture_tool.picture_tool.utils.io_utils import list_images


def test_list_images_returns_sorted_names(tmp_path):
    files = ["b.PNG", "a.jpg", "c.txt"]
    for name in files:
        path = tmp_path / name
        path.write_bytes(b"x")

    result = list_images(tmp_path)
    assert result == ["a.jpg", "b.PNG"]


def test_list_images_handles_missing_directory(tmp_path):
    missing = tmp_path / "missing"
    assert list_images(missing) == []
