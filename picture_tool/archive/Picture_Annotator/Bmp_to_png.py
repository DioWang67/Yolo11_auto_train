import cv2
import logging
import os

# 設定你的來源目錄
input_dir = 'target\pcba_fail'
# 設定轉換後要放的目錄
output_dir = "target\pcba_fail"

os.makedirs(output_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(".bmp"):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"讀取失敗: {img_path}")
            continue

        # 轉成 png 的路徑
        png_filename = os.path.splitext(filename)[0] + ".png"
        png_path = os.path.join(output_dir, png_filename)

        # 儲存 png
        cv2.imwrite(png_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        logging.info(f"已轉換: {png_path}")

