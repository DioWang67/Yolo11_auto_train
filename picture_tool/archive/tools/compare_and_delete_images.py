import logging
from pathlib import Path

def setup_logging(log_file="compare_images.log"):
    """設置日誌系統"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def compare_and_delete_images(folder_a, folder_b, logger):
    """比對兩個資料夾的圖片檔案名稱，刪除 folder_b 中多餘的檔案"""
    folder_a = Path(folder_a)
    folder_b = Path(folder_b)
    
    # 檢查資料夾是否存在
    if not folder_a.exists():
        logger.error(f"參考資料夾 {folder_a} 不存在！")
        raise FileNotFoundError(f"參考資料夾 {folder_a} 不存在！")
    if not folder_b.exists():
        logger.error(f"目標資料夾 {folder_b} 不存在！")
        raise FileNotFoundError(f"目標資料夾 {folder_b} 不存在！")
    
    # 支援的圖片格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    
    # 獲取資料夾中的圖片檔案（僅檔案名稱，不含副檔名）
    files_a = {f.stem for f in folder_a.iterdir() if f.is_file() and f.suffix.lower() in image_extensions}
    files_b = {f.stem for f in folder_b.iterdir() if f.is_file() and f.suffix.lower() in image_extensions}
    
    # 檢查資料夾是否為空
    if not files_a:
        logger.error(f"參考資料夾 {folder_a} 中沒有可用的圖片！")
        raise ValueError(f"參考資料夾 {folder_a} 中沒有可用的圖片！")
    if not files_b:
        logger.warning(f"目標資料夾 {folder_b} 中沒有可用的圖片，無需刪除。")
        return
    
    # 找出 folder_b 中多餘的檔案
    extra_files = files_b - files_a
    logger.info(f"參考資料夾 {folder_a} 包含 {len(files_a)} 張圖片")
    logger.info(f"目標資料夾 {folder_b} 包含 {len(files_b)} 張圖片")
    logger.info(f"找到 {len(extra_files)} 張多餘圖片待刪除")
    
    # 刪除多餘的檔案
    for file_stem in extra_files:
        for file_path in folder_b.glob(f"{file_stem}.*"):
            if file_path.suffix.lower() in image_extensions:
                try:
                    file_path.unlink()
                    logger.info(f"已刪除檔案: {file_path}")
                except Exception as e:
                    logger.error(f"刪除檔案 {file_path} 失敗: {e}")
    
    # 檢查刪除後的檔案數量
    files_b_after = {f.stem for f in folder_b.iterdir() if f.is_file() and f.suffix.lower() in image_extensions}
    logger.info(f"刪除完成，目標資料夾 {folder_b} 現包含 {len(files_b_after)} 張圖片")

def main():
    # 直接在程式碼中設定資料夾路徑
    FOLDER_A = r"D:\Git\robotlearning\picture_tool\datasets\for_anomalib\ground_truth\Reversed_Connection"  # 參考資料夾
    FOLDER_B = r"D:\Git\robotlearning\picture_tool\datasets\for_anomalib\test\Reversed_Connection"     # 目標資料夾
    LOG_FILE = "compare_images.log"                   # 日誌檔案
    
    logger = setup_logging(LOG_FILE)
    
    try:
        compare_and_delete_images(FOLDER_A, FOLDER_B, logger)
        logger.info("圖片比對和刪除操作完成")
    except Exception as e:
        logger.error(f"操作失敗: {e}")
        raise

if __name__ == "__main__":
    main()