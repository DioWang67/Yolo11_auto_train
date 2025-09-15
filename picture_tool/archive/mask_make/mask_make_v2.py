import cv2
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import concurrent.futures
import warnings
from sklearn.ensemble import IsolationForest
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

# 關閉警告訊息
warnings.filterwarnings('ignore')

class AnomalyDetector:
    def __init__(
        self,
        input_shape=(256, 256, 3),
        contamination=0.05,
        n_estimators=100,
        min_anomaly_size=100,
        gaussian_kernel_size=(5, 5),
        morphology_kernel_size=(5, 5)
    ):
        """
        初始化異常檢測器
        
        參數:
            input_shape: 輸入圖片尺寸
            contamination: Isolation Forest 預期的異常比例
            n_estimators: Isolation Forest 決策樹數量
            min_anomaly_size: 最小異常區域大小
            gaussian_kernel_size: 高斯模糊核大小
            morphology_kernel_size: 形態學運算核大小
        """
        self.input_shape = input_shape
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.min_anomaly_size = min_anomaly_size
        self.gaussian_kernel_size = gaussian_kernel_size
        self.morphology_kernel_size = morphology_kernel_size
        
        # 初始化 VGG16 模型
        base_model = VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )
        self.feature_extractor = Model(
            inputs=base_model.input,
            outputs=base_model.output
        )
        
        # 初始化 Isolation Forest
        self.clf = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )

    def preprocess_image(self, img):
        """
        圖片預處理
        
        參數:
            img: 輸入圖片
        返回:
            預處理後的圖片
        """
        # 調整圖片大小
        img_resized = cv2.resize(img, self.input_shape[:2])
        
        # 正規化
        img_normalized = cv2.normalize(
            img_resized,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F
        )
        
        return img_normalized

    def extract_features(self, img):
        """
        提取圖片特徵
        
        參數:
            img: 預處理後的圖片
        返回:
            正規化後的特徵向量
        """
        # 擴展維度以符合模型輸入需求
        img_array = np.expand_dims(img, axis=0)
        
        # VGG16 預處理
        img_array = preprocess_input(img_array)
        
        # 提取特徵
        features = self.feature_extractor.predict(img_array, verbose=0)
        
        # 展平特徵
        features_flattened = features.flatten()
        
        # 特徵正規化
        features_normalized = (features_flattened - np.mean(features_flattened)) / np.std(features_flattened)
        
        return features_normalized

    def create_binary_mask(self, diff_gray, threshold):
        """
        創建二值遮罩
        
        參數:
            diff_gray: 灰度差異圖
            threshold: 閾值係數
        返回:
            二值遮罩
        """
        # 高斯模糊
        blurred = cv2.GaussianBlur(diff_gray, self.gaussian_kernel_size, 0)
        
        # 計算動態閾值
        mean_val = np.mean(blurred)
        std_val = np.std(blurred)
        dynamic_threshold = mean_val + (threshold * std_val)
        
        # 二值化
        _, mask = cv2.threshold(blurred, dynamic_threshold, 255, cv2.THRESH_BINARY)
        
        # 形態學運算
        kernel = np.ones(self.morphology_kernel_size, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 移除小區域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.min_anomaly_size:
                mask[labels == i] = 0
                
        return mask

    def fit(self, normal_images):
        """
        訓練模型
        
        參數:
            normal_images: 正常圖片列表
        """
        features_list = []
        for img in tqdm(normal_images, desc="提取正常樣本特徵"):
            img_processed = self.preprocess_image(img)
            features = self.extract_features(img_processed)
            features_list.append(features)
            
        self.clf.fit(features_list)
        self.normal_features = features_list
        self.normal_images = [self.preprocess_image(img) for img in normal_images]

    def detect(self, anomaly_image, threshold=2.0):
        """
        檢測異常並生成遮罩
        
        參數:
            anomaly_image: 待檢測圖片
            threshold: 遮罩生成閾值
        返回:
            mask: 異常遮罩
            score: 異常分數
        """
        # 預處理圖片
        anomaly_processed = self.preprocess_image(anomaly_image)
        
        # 提取特徵
        anomaly_features = self.extract_features(anomaly_processed)
        
        # 計算異常分數
        anomaly_score = self.clf.decision_function([anomaly_features])[0]
        
        # 計算異常閾值
        anomaly_threshold = np.percentile(
            self.clf.decision_function(self.normal_features),
            5
        )
        
        # 根據分數調整閾值
        if anomaly_score < anomaly_threshold:
            threshold = threshold * (1 + abs(anomaly_score / anomaly_threshold))
        
        # 生成遮罩
        diff = cv2.absdiff(self.normal_images[0], anomaly_processed)
        diff_gray = cv2.cvtColor((diff * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        mask = self.create_binary_mask(diff_gray, threshold)
        
        return mask, anomaly_score

def generate_anomaly_masks(
    normal_path,
    anomaly_path,
    output_path,
    threshold=2.0,
    contamination=0.05,
    min_anomaly_size=100
):
    """
    批次處理異常檢測
    
    參數:
        normal_path: 正常樣本資料夾路徑
        anomaly_path: 異常樣本資料夾路徑
        output_path: 輸出資料夾路徑
        threshold: 遮罩生成閾值
        contamination: 預期異常比例
        min_anomaly_size: 最小異常區域大小
    """
    # 創建輸出資料夾
    os.makedirs(output_path, exist_ok=True)
    
    # 獲取所有圖片路徑
    normal_files = list(Path(normal_path).glob('*.jpg')) + list(Path(normal_path).glob('*.png'))
    anomaly_files = list(Path(anomaly_path).glob('*.jpg')) + list(Path(anomaly_path).glob('*.png'))
    
    print(f"找到 {len(normal_files)} 張正常圖片和 {len(anomaly_files)} 張異常圖片")
    
    # 初始化檢測器
    detector = AnomalyDetector(
        contamination=contamination,
        min_anomaly_size=min_anomaly_size
    )
    
    # 讀取正常圖片
    normal_images = []
    for file in tqdm(normal_files, desc="讀取正常圖片"):
        img = cv2.imread(str(file))
        if img is not None:
            normal_images.append(img)
    
    # 訓練模型
    print("訓練模型...")
    detector.fit(normal_images)
    
    # 處理異常圖片
    print("處理異常圖片...")
    for file in tqdm(anomaly_files, desc="生成遮罩"):
        # 讀取圖片
        img = cv2.imread(str(file))
        if img is None:
            print(f"無法讀取圖片: {file}")
            continue
            
        # 檢測異常
        mask, score = detector.detect(img, threshold)
        
        # 儲存遮罩
        output_file = Path(output_path) / f"{file.stem}_mask.png"
        cv2.imwrite(str(output_file), mask)
        print(f"已處理: {file.name} (異常分數: {score:.2f})")

if __name__ == "__main__":
    # 設定路徑
    normal_dir = 'data/datasets/con1/train/good'    # 正常樣本資料夾
    anomaly_dir = 'data/datasets/con1/test/Reversed_Connection'   # 異常樣本資料夾
    output_dir = 'data/datasets/con1/ground_truth/Reversed_Connection'    # 輸出遮罩資料夾
    
    # 執行異常檢測
    generate_anomaly_masks(
        normal_dir,
        anomaly_dir,
        output_dir,
        threshold=0.7,
        contamination=0.1,
        min_anomaly_size=300
    )

# import cv2
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm
# import os

# def rotate_images(input_folder, output_folder=None):
#     """
#     將資料夾內的所有圖片旋轉180度
    
#     參數:
#     input_folder: 輸入資料夾路徑
#     output_folder: 輸出資料夾路徑，如果不指定則覆蓋原圖片
#     """
#     # 如果指定輸出資料夾，確保它存在
#     if output_folder:
#         os.makedirs(output_folder, exist_ok=True)
    
#     # 取得所有圖片檔案
#     image_files = list(Path(input_folder).glob('*.jpg')) + \
#                  list(Path(input_folder).glob('*.jpeg')) + \
#                  list(Path(input_folder).glob('*.png'))
    
#     print(f"找到 {len(image_files)} 張圖片")
    
#     # 處理每張圖片
#     for img_path in tqdm(image_files, desc="旋轉圖片"):
#         # 讀取圖片
#         img = cv2.imread(str(img_path))
        
#         if img is None:
#             print(f"無法讀取圖片: {img_path}")
#             continue
        
#         # 旋轉圖片180度
#         rotated_img = cv2.rotate(img, cv2.ROTATE_180)
        
#         # 決定儲存路徑
#         if output_folder:
#             save_path = Path(output_folder) / img_path.name
#         else:
#             save_path = img_path
        
#         # 儲存圖片
#         cv2.imwrite(str(save_path), rotated_img)

# if __name__ == "__main__":
#     # 設定資料夾路徑
#     input_dir = 'D:/Git/robotlearning/Anomalib_train/datasets/con1/train/good/'  # 請改為您的輸入資料夾路徑
#     output_dir = "data/s" # 可選，如果要保留原圖片
    
#     # 執行旋轉
#     rotate_images(input_dir, output_dir)  # 如果要覆蓋原圖片，移除 output_dir 參數

