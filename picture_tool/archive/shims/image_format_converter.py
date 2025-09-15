from picture_tool.format.image_format_converter import convert_format

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    sample = {
        "input_dir": "./pcba_fail",
        "output_dir": "./pcba_fail_png",
        "input_formats": [".bmp"],
        "output_format": ".png",
        "quality": 95,
        "png_compression": 3,
    }
    convert_format(sample)

