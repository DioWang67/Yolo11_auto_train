import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def convert_format(format_config):
    """Convert images in input_dir from input_formats to output_format.

    format_config keys:
    - input_dir: str
    - output_dir: str
    - input_formats: list[str]
    - output_format: str (e.g., '.png')
    - quality: int (for JPEG)
    - png_compression: int (for PNG)
    """
    input_dir = Path(format_config['input_dir'])
    output_dir = Path(format_config['output_dir'])
    input_formats = format_config['input_formats']
    output_format = format_config['output_format']
    quality = format_config.get('quality', 95)
    png_compression = format_config.get('png_compression', 3)

    if not input_dir.exists():
        raise FileNotFoundError(f"輸入資料夾 {input_dir} 不存在")

    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in input_dir.iterdir():
        if img_path.suffix.lower() in input_formats:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"讀取失敗: {img_path}")
                continue

            output_filename = img_path.stem + output_format
            output_path = output_dir / output_filename

            if output_format.lower() in ('.jpg', '.jpeg'):
                cv2.imwrite(str(output_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            elif output_format.lower() == '.png':
                cv2.imwrite(str(output_path), img, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])
            else:
                cv2.imwrite(str(output_path), img)

            logger.info(f"已輸出: {output_path}")
