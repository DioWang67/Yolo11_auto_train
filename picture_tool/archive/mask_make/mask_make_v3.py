import cv2
import numpy as np
import os
from glob import glob

def annotate_folder(folder_path):
    image_paths = sorted(glob(os.path.join(folder_path, "*.*")))
    supported_ext = ['.png', '.jpg', '.jpeg', '.bmp']

    for img_path in image_paths:
        if not any(img_path.lower().endswith(ext) for ext in supported_ext):
            continue

        print(f"\n🔍 開啟：{img_path}")
        image = cv2.imread(img_path)
        if image is None:
            print("❌ 無法讀取圖片，跳過。")
            continue

        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        while True:
            roi = cv2.selectROI(f"[{os.path.basename(img_path)}] 按 Enter 確認，Esc 結束框選", image, showCrosshair=True)
            x, y, w, h = roi
            if w == 0 or h == 0:
                print("✅ 本圖框選結束")
                break
            mask[y:y+h, x:x+w] = 255
            print(f"📦 已新增框選區域：{x}, {y}, {w}, {h}")

        cv2.destroyAllWindows()

        # 儲存為 原圖名 + "_mask.png"
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_name = f"{base_name}_mask.png"
        mask_path = os.path.join(folder_path, mask_name)
        cv2.imwrite(mask_path, mask)
        print(f"✅ 已儲存 Mask：{mask_path}")

        # 顯示短暫預覽
        cv2.imshow("Mask 預覽", mask)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

    print("🎉 所有圖片處理完成！")

# ⚠️ 修改這個為你圖片資料夾路徑
folder = "D:\Git\robotlearning\mask_make\target"  # 例如：folder = "./images"
annotate_folder(folder)
