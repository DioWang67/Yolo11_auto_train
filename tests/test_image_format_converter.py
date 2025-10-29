import cv2
import numpy as np
import pytest

from picture_tool.picture_tool.format.image_format_converter import convert_format


def _write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((32, 32, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_convert_format_creates_output_images(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    src_path = input_dir / "sample.jpg"

    _write_image(src_path)

    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_formats": [".jpg"],
        "output_format": ".png",
        "quality": 80,
        "png_compression": 1,
    }

    convert_format(config)

    generated = list(output_dir.glob("*.png"))
    assert len(generated) == 1
    assert generated[0].stem == "sample"
    img = cv2.imread(str(generated[0]))
    assert img is not None


def test_convert_format_missing_input_dir_raises(tmp_path):
    config = {
        "input_dir": str(tmp_path / "missing"),
        "output_dir": str(tmp_path / "output"),
        "input_formats": [".jpg"],
        "output_format": ".png",
    }

    with pytest.raises(FileNotFoundError):
        convert_format(config)
