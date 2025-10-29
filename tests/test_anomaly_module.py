import cv2
import numpy as np

from picture_tool.picture_tool.anomaly import anomaly_mask_generator as anomaly


def test_generate_anomaly_mask_detects_difference():
    ref = np.zeros((32, 32), dtype=np.uint8)
    test = ref.copy()
    test[10:22, 10:22] = 255

    mask = anomaly.generate_anomaly_mask(ref, test, threshold=30)

    assert mask.sum() > 0
    assert mask[16, 16] == 255


def test_process_anomaly_detection_creates_mask(tmp_path):
    ref_dir = tmp_path / "ref"
    test_dir = tmp_path / "test"
    out_dir = tmp_path / "out"
    ref_dir.mkdir()
    test_dir.mkdir()

    ref_img = np.zeros((64, 64), dtype=np.uint8)
    test_img = ref_img.copy()
    cv2.rectangle(test_img, (20, 20), (30, 30), 255, -1)

    ref_path = ref_dir / "ref.png"
    test_path = test_dir / "test.png"
    cv2.imwrite(str(ref_path), ref_img)
    cv2.imwrite(str(test_path), test_img)

    config = {
        "anomaly_detection": {
            "reference_folder": str(ref_dir),
            "test_folder": str(test_dir),
            "output_folder": str(out_dir),
            "threshold": 25,
            "z_threshold": 1.0,
            "align": "none",
            "open_ksize": 0,
            "close_ksize": 0,
            "dilate_ksize": 0,
            "min_area": 5,
            "input_formats": [".png"],
            "recursive": False,
            "save_overlay": False,
            "overlay_alpha": 0.5,
            "quiet": True,
        }
    }

    anomaly.process_anomaly_detection(config)

    mask_path = out_dir / "test.png"

    assert mask_path.is_file()
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert mask_image is not None
    assert mask_image.sum() > 0
