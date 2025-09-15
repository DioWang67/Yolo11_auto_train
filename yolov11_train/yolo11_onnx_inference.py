from ultralytics import YOLO

# 載入 ONNX 模型
model = YOLO("best.onnx")
# 推理
results = model("102540.jpg")

for result in results:
    print("boxes:", result.boxes)          # 邊界框
    print("keypoints:", result.keypoints)  # 關鍵點（若有）
    print("masks:", result.masks)          # 分割遮罩（若有）
    print("names:", result.names)          # 類別名稱
    print("obb:", result.obb)              # OBB（若有）
    print("orig_img:", result.orig_img)    # 原始圖片 (np.ndarray)
    print("orig_shape:", result.orig_shape)  # 原圖尺寸
    print("path:", result.path)            # 圖片路徑
    print("probs:", result.probs)          # 分類概率（若有）
    print("save_dir:", result.save_dir)    # 儲存目錄
    print("speed:", result.speed)          # 推理速度資訊

    # 顯示檢測結果
    result.show()  # 於視窗顯示標註後圖像

    # 儲存檢測結果影像
    result.save(filename="result.jpg")  # 儲存至檔案