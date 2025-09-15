import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import ndimage

class AnomalyDetector:
    def __init__(self, ref_folder, threshold=30):
        self.ref_folder = ref_folder
        self.threshold = threshold
        self.ref_template = None
        self.ref_std = None
        
    def load_reference_images(self):
        """讀取正常圖像資料夾並生成參考模板"""
        ref_images = []
        ref_paths = []
        
        for file in os.listdir(self.ref_folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(self.ref_folder, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    ref_images.append(img)
                    ref_paths.append(img_path)
        
        if not ref_images:
            raise ValueError("正常圖像資料夾中沒有可用的圖像！")
        
        print(f"載入了 {len(ref_images)} 張參考圖像")
        
        # 統一圖像尺寸（使用第一張圖像的尺寸作為標準）
        target_size = ref_images[0].shape
        normalized_images = []
        
        for i, img in enumerate(ref_images):
            if img.shape != target_size:
                img = cv2.resize(img, (target_size[1], target_size[0]))
                print(f"調整圖像尺寸: {ref_paths[i]}")
            normalized_images.append(img.astype(np.float32))
        
        # 計算平均值和標準差
        ref_stack = np.stack(normalized_images, axis=0)
        self.ref_template = np.mean(ref_stack, axis=0).astype(np.uint8)
        self.ref_std = np.std(ref_stack, axis=0).astype(np.float32)
        
        return self.ref_template, self.ref_std
    
    def preprocess_image(self, img):
        """圖像預處理：降噪、直方圖均衡化"""
        # 高斯濾波降噪
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # 直方圖均衡化增強對比度
        img = cv2.equalizeHist(img)
        
        return img
    
    def generate_anomaly_mask_advanced(self, test_img):
        """進階異常mask生成方法"""
        # 確保圖像尺寸一致
        if test_img.shape != self.ref_template.shape:
            test_img = cv2.resize(test_img, (self.ref_template.shape[1], self.ref_template.shape[0]))
        
        # 預處理
        test_img = self.preprocess_image(test_img)
        ref_processed = self.preprocess_image(self.ref_template)
        
        # 方法1: 基本差異檢測
        diff_basic = cv2.absdiff(ref_processed, test_img)
        _, mask_basic = cv2.threshold(diff_basic, self.threshold, 255, cv2.THRESH_BINARY)
        
        # 方法2: 統計異常檢測（基於標準差）
        diff_normalized = np.abs(test_img.astype(np.float32) - ref_processed.astype(np.float32))
        # 考慮標準差，對變化較大的區域更敏感
        std_threshold = self.ref_std + self.threshold
        mask_statistical = (diff_normalized > std_threshold).astype(np.uint8) * 255
        
        # 方法3: 結構相似性檢測
        # 使用Sobel邊緣檢測
        ref_edges = cv2.Sobel(ref_processed, cv2.CV_64F, 1, 1, ksize=3)
        test_edges = cv2.Sobel(test_img, cv2.CV_64F, 1, 1, ksize=3)
        edge_diff = np.abs(ref_edges - test_edges)
        edge_diff = ((edge_diff / edge_diff.max()) * 255).astype(np.uint8)
        _, mask_edges = cv2.threshold(edge_diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # 組合多種方法
        mask_combined = cv2.bitwise_or(mask_basic, mask_statistical)
        mask_combined = cv2.bitwise_or(mask_combined, mask_edges)
        
        return {
            'basic': mask_basic,
            'statistical': mask_statistical,
            'edges': mask_edges,
            'combined': mask_combined
        }
    
    def post_process_mask(self, mask, min_area=100):
        """後處理：去除小噪點、填充孔洞"""
        # 形態學處理
        kernel = np.ones((5, 5), np.uint8)
        
        # 開運算去除小噪點
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 閉運算填充小孔洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 移除小面積區域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # 創建新的mask，只保留面積大於閾值的區域
        filtered_mask = np.zeros_like(mask)
        for i in range(1, num_labels):  # 跳過背景（標籤0）
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered_mask[labels == i] = 255
        
        return filtered_mask
    
    def visualize_results(self, test_img, masks, output_path):
        """視覺化結果"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始測試圖像
        axes[0, 0].imshow(test_img, cmap='gray')
        axes[0, 0].set_title('測試圖像')
        axes[0, 0].axis('off')
        
        # 參考模板
        axes[0, 1].imshow(self.ref_template, cmap='gray')
        axes[0, 1].set_title('參考模板')
        axes[0, 1].axis('off')
        
        # 基本差異
        axes[0, 2].imshow(masks['basic'], cmap='gray')
        axes[0, 2].set_title('基本差異檢測')
        axes[0, 2].axis('off')
        
        # 統計異常
        axes[1, 0].imshow(masks['statistical'], cmap='gray')
        axes[1, 0].set_title('統計異常檢測')
        axes[1, 0].axis('off')
        
        # 邊緣差異
        axes[1, 1].imshow(masks['edges'], cmap='gray')
        axes[1, 1].set_title('邊緣差異檢測')
        axes[1, 1].axis('off')
        
        # 組合結果
        axes[1, 2].imshow(masks['combined'], cmap='gray')
        axes[1, 2].set_title('組合檢測結果')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def calculate_anomaly_score(self, mask):
        """計算異常分數"""
        total_pixels = mask.shape[0] * mask.shape[1]
        anomaly_pixels = np.sum(mask > 0)
        anomaly_ratio = anomaly_pixels / total_pixels
        return anomaly_ratio, anomaly_pixels
    
    def process_test_images(self, test_folder, output_folder, save_visualization=True):
        """處理測試圖像並生成異常mask"""
        # 確保輸出資料夾存在
        os.makedirs(output_folder, exist_ok=True)
        if save_visualization:
            os.makedirs(os.path.join(output_folder, 'visualizations'), exist_ok=True)
        
        # 載入參考模板
        if self.ref_template is None:
            self.load_reference_images()
        
        # 獲取測試圖像
        test_images = [os.path.join(test_folder, file) for file in os.listdir(test_folder)
                      if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not test_images:
            raise ValueError("測試圖像資料夾中沒有可用的圖像！")
        
        results = []
        
        # 處理每張測試圖像
        for test_path in test_images:
            test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
            if test_img is None:
                print(f"無法讀取測試圖像：{test_path}")
                continue
            
            filename = Path(test_path).stem
            print(f"處理圖像：{filename}")
            
            # 生成異常mask
            masks = self.generate_anomaly_mask_advanced(test_img)
            
            # 後處理
            final_mask = self.post_process_mask(masks['combined'])
            
            # 計算異常分數
            anomaly_ratio, anomaly_pixels = self.calculate_anomaly_score(final_mask)
            
            # 保存結果
            cv2.imwrite(os.path.join(output_folder, f"{filename}_mask.png"), final_mask)
            
            # 保存視覺化結果
            if save_visualization:
                viz_path = os.path.join(output_folder, 'visualizations', f"{filename}_analysis.png")
                self.visualize_results(test_img, masks, viz_path)
            
            # 記錄結果
            result = {
                'filename': filename,
                'anomaly_ratio': anomaly_ratio,
                'anomaly_pixels': anomaly_pixels,
                'status': '異常' if anomaly_ratio > 0.05 else '正常'  # 5%閾值
            }
            results.append(result)
            
            print(f"  - 異常比例: {anomaly_ratio:.4f} ({anomaly_pixels} pixels)")
            print(f"  - 判定結果: {result['status']}")
        
        # 保存統計報告
        self.save_report(results, output_folder)
        
        return results
    
    def save_report(self, results, output_folder):
        """保存檢測報告"""
        report_path = os.path.join(output_folder, 'detection_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("異常檢測報告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"檢測閾值: {self.threshold}\n")
            f.write(f"總共檢測圖像數: {len(results)}\n\n")
            
            normal_count = sum(1 for r in results if r['status'] == '正常')
            abnormal_count = len(results) - normal_count
            
            f.write(f"正常圖像: {normal_count}\n")
            f.write(f"異常圖像: {abnormal_count}\n\n")
            
            f.write("詳細結果:\n")
            f.write("-" * 30 + "\n")
            
            for result in results:
                f.write(f"檔案: {result['filename']}\n")
                f.write(f"異常比例: {result['anomaly_ratio']:.4f}\n")
                f.write(f"異常像素數: {result['anomaly_pixels']}\n")
                f.write(f"判定: {result['status']}\n")
                f.write("-" * 30 + "\n")
        
        print(f"檢測報告已保存至: {report_path}")

def main():
    # 配置路徑
    ref_folder = "./pass_image"    # 正常圖像資料夾
    test_folder = "./target"       # 測試圖像資料夾
    output_folder = "./outputv2"     # 輸出資料夾
    
    # 創建異常檢測器
    detector = AnomalyDetector(ref_folder, threshold=30)
    
    try:
        # 執行檢測
        results = detector.process_test_images(
            test_folder, 
            output_folder, 
            save_visualization=True
        )
        
        print("\n檢測完成！")
        print(f"結果保存在: {output_folder}")
        
    except Exception as e:
        print(f"錯誤: {e}")

if __name__ == "__main__":
    main()