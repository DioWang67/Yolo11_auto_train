import cv2
import numpy as np
import os
from pathlib import Path
import json

def load_config(config_path="config.json"):
    """讀取 config.json 檔案"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到設定檔案：{config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"設定檔案 {config_path} 格式錯誤")

def load_reference_images(ref_folder):
    """讀取正常圖像資料夾並生成參考模板（平均圖像）"""
    ref_avg = None
    n = 0
    for file in os.listdir(ref_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(ref_folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if ref_avg is None:
                    ref_avg = np.zeros_like(img, dtype=np.float32)
                ref_avg = (ref_avg * n + img.astype(np.float32)) / (n + 1)
                n += 1

    if n == 0:
        raise ValueError("正常圖像資料夾中沒有可用的圖像！")

    return ref_avg.astype(np.uint8)

def generate_anomaly_mask(ref_img, test_img, threshold=30):
    """生成異常mask：比較測試圖像與參考圖像的差異"""
    # 確保圖像尺寸一致
    if ref_img.shape != test_img.shape:
        test_img = cv2.resize(test_img, (ref_img.shape[1], ref_img.shape[0]))
    
    # 計算差異圖像
    diff = cv2.absdiff(ref_img, test_img)
    
    # 應用閾值，生成二值化mask
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # 可選：形態學處理以減少噪點
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def process_test_images(ref_folder, test_folder, output_folder, threshold=30):
    """處理測試圖像資料夾中的所有圖像並生成異常mask"""
    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 載入參考模板
    ref_img = load_reference_images(ref_folder)
    
    # 掃描測試圖像資料夾
    test_images = [os.path.join(test_folder, file) for file in os.listdir(test_folder) 
                   if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not test_images:
        raise ValueError("測試圖像資料夾中沒有可用的圖像！")
    
    # 處理每張測試圖像
    for test_path in test_images:
        test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        if test_img is None:
            print(f"無法讀取測試圖像：{test_path}")
            continue
        
        # 生成異常mask
        mask = generate_anomaly_mask(ref_img, test_img, threshold)
        
        # 保存mask
        output_filename = os.path.join(output_folder, f"{Path(test_path).stem}.png")
        cv2.imwrite(output_filename, mask)
        print(f"已生成mask並保存至：{output_filename}")

def main():
    # 載入設定
    config = load_config()
    
    # 從 config.json 獲取參數
    ref_folder = config["reference_folder"]
    test_folder = config["test_folder"]
    output_folder = config["output_folder"]
    threshold = config["threshold"]
    
    # 執行異常mask生成
    process_test_images(ref_folder, test_folder, output_folder, threshold)

if __name__ == "__main__":
    main()
