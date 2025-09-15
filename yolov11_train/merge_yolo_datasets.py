import os
import shutil
from tqdm import tqdm

def merge_pcba_datasets(dataset_a_path, dataset_b_path, output_path):
    subsets = ['train', 'val', 'test']
    image_folder = 'images'
    label_folder = 'labels'
    allowed_ext = ['.jpg', '.png', '.jpeg', '.bmp']
    
    for subset in subsets:
        os.makedirs(os.path.join(output_path, subset, image_folder), exist_ok=True)
        os.makedirs(os.path.join(output_path, subset, label_folder), exist_ok=True)

    copied_files = 0

    def copy_with_rename(img_src_dir, lbl_src_dir, img_dst_dir, lbl_dst_dir):
        nonlocal copied_files
        if not (os.path.exists(img_src_dir) and os.path.exists(lbl_src_dir)):
            print(f"警告：{img_src_dir} 或 {lbl_src_dir} 不存在，跳過")
            return
        with os.scandir(img_src_dir) as entries:
            for entry in tqdm(entries, desc=f"處理 {img_src_dir}"):
                if not entry.name.lower().endswith(tuple(allowed_ext)):
                    continue
                filename = entry.name
                base, ext = os.path.splitext(filename)
                img_src = os.path.join(img_src_dir, filename)
                lbl_src = os.path.join(lbl_src_dir, f"{base}.txt")

                img_dst = os.path.join(img_dst_dir, filename)
                lbl_dst = os.path.join(lbl_dst_dir, f"{base}.txt")

                counter = 1
                new_base = base
                while os.path.exists(img_dst) or os.path.exists(lbl_dst):
                    new_base = f"{base}_{counter}"
                    img_dst = os.path.join(img_dst_dir, f"{new_base}{ext}")
                    lbl_dst = os.path.join(lbl_dst_dir, f"{new_base}.txt")
                    counter += 1

                try:
                    shutil.copy2(img_src, img_dst)
                    if os.path.exists(lbl_src) and os.path.getsize(lbl_src) > 0:
                        shutil.copy2(lbl_src, lbl_dst)
                    else:
                        print(f"警告：{lbl_src} 不存在或為空，僅複製圖片")
                    copied_files += 1
                except Exception as e:
                    print(f"複製檔案 {img_src} 時出錯：{e}")

    for dataset_path in [dataset_a_path, dataset_b_path]:
        for subset in subsets:
            img_src_dir = os.path.join(dataset_path, subset, image_folder)
            lbl_src_dir = os.path.join(dataset_path, subset, label_folder)
            img_dst_dir = os.path.join(output_path, subset, image_folder)
            lbl_dst_dir = os.path.join(output_path, subset, label_folder)
            copy_with_rename(img_src_dir, lbl_src_dir, img_dst_dir, lbl_dst_dir)

    return f"✅ 合併完成，儲存至：{output_path}，共複製 {copied_files} 組檔案"

# 示範路徑（請替換成你的實際路徑）
dataset_a_path = r"D:\Git\robotlearning\yolov11_train\datasets\PCBA_A"
dataset_b_path = r"D:\Git\robotlearning\yolov11_train\datasets\PCBA_back_0721"
output_path = "./datasets/PCBA"

merge_pcba_datasets(dataset_a_path, dataset_b_path, output_path)
