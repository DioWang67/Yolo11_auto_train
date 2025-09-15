import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# 設置路徑
image_dir = Path("D:/Git/robotlearning/yolo_train/output/images")
label_dir = Path("D:/Git/robotlearning/yolo_train/output/labels")

# 獲取圖片和標籤的完整列表
images = sorted([str(p) for p in image_dir.glob("*.jpg")])  # 假設圖片格式為 jpg
labels = sorted([str(p) for p in label_dir.glob("*.txt")])  # 假設標籤為 txt 文件

# 確保圖片和標籤數量一致
print(len(images), len(labels))
assert len(images) == len(labels), "圖片數量和標籤數量不一致"

# 先分成訓練集和暫時集合（驗證集 + 測試集）
train_images, temp_images, train_labels, temp_labels = train_test_split(
    images, labels, test_size=0.3, random_state=42
)

# 再分成驗證集和測試集
val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, random_state=42
)

# 定義輸出目錄
output_dir = Path("D:/Git/robotlearning/yolo_train/split_dataset")
output_dir.mkdir(parents=True, exist_ok=True)

# 建立子目錄
for split in ["train", "val", "test"]:
    (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

# 移動圖片和標籤到相應的目錄
def copy_files(images, labels, split):
    for img, lbl in zip(images, labels):
        shutil.copy(img, output_dir / split / "images" / Path(img).name)
        shutil.copy(lbl, output_dir / split / "labels" / Path(lbl).name)

copy_files(train_images, train_labels, "train")
copy_files(val_images, val_labels, "val")
copy_files(test_images, test_labels, "test")

print("文件已成功分配並複製到訓練、驗證和測試目錄中")
