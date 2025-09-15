import os

def convert_class_ids(input_path: str, output_path: str):
    mapping = {0: 5, 1: 6, 2: 7, 3: 8, 4: 9}

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    converted_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts or len(parts) < 5:
            continue  # 跳過空行或錯誤行
        class_id = int(parts[0])
        if class_id not in mapping:
            new_class_id = class_id
        else:
            new_class_id = mapping[class_id]
        converted_line = ' '.join([str(new_class_id)] + parts[1:])
        converted_lines.append(converted_line)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(converted_lines))

def batch_convert(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            convert_class_ids(input_path, output_path)
            print(f"✅ 轉換完成：{filename}")

if __name__ == "__main__":
    # 請根據你的實際資料夾路徑修改這兩個路徑！
    input_dir = "./datasets/PCBA_back_0721/val/labels"
    output_dir = "./datasets/PCBA_back_0721/val/labels"
    batch_convert(input_dir, output_dir)
