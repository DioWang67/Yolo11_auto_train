import argparse
import yaml
import logging
from pathlib import Path
from picture_tool.format import convert_format
from picture_tool.anomaly import process_anomaly_detection
from picture_tool.augment import YoloDataAugmentor, ImageAugmentor
from picture_tool.split import split_dataset
from picture_tool.train.yolo_trainer import train_yolo
from picture_tool.eval.yolo_evaluator import evaluate_yolo
from picture_tool.report.report_generator import generate_report
from picture_tool.quality.dataset_linter import lint_dataset, preview_dataset

def setup_logging(log_file):
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

def load_config(config_path="config.yaml"):
    """讀取 config.yaml 檔案"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_config_if_updated(config_path, config, logger):
    """檢查配置檔案是否更新，必要時重新載入"""
    config_file = Path(config_path)
    current_mtime = config_file.stat().st_mtime
    last_mtime = getattr(load_config_if_updated, "last_mtime", None)

    if last_mtime is None:
        load_config_if_updated.last_mtime = current_mtime
        return config

    if current_mtime > last_mtime:
        logger.info("檢測到配置檔案更新，重新載入")
        load_config_if_updated.last_mtime = current_mtime
        return load_config(config_path)

    return config

def validate_dependencies(tasks, config, logger):
    """驗證任務依賴關係，並保持原始任務順序"""
    task_dict = {t['name']: t for t in config['pipeline']['tasks']}
    selected_tasks = tasks.copy()  # 複製原始任務列表以保持順序
    required_tasks = []
    
    # 檢查每個任務的依賴
    for task in tasks:
        if task not in task_dict:
            logger.warning(f"未知任務: {task}")
            continue
        if not task_dict[task].get('enabled', True):
            logger.info(f"任務 {task} 在配置文件中被禁用，跳過")
            selected_tasks.remove(task)
            continue
        dependencies = task_dict[task].get('dependencies', [])
        for dep in dependencies:
            if dep not in selected_tasks and dep not in required_tasks:
                logger.info(f"任務 {task} 依賴 {dep}，自動添加")
                required_tasks.append(dep)
    
    # 將依賴任務插入到列表中，確保依賴任務在主任務之前
    final_tasks = []
    for task in selected_tasks:
        dependencies = task_dict.get(task, {}).get('dependencies', [])
        for dep in dependencies:
            if dep in required_tasks and dep not in final_tasks:
                final_tasks.append(dep)
        if task not in final_tasks:
            final_tasks.append(task)
    
    return final_tasks

def get_tasks_from_groups(groups, config):
    """從任務組獲取任務清單"""
    tasks = set()
    for group in groups:
        if group in config['pipeline']['task_groups']:
            tasks.update(config['pipeline']['task_groups'][group])
        else:
            logging.warning(f"未知任務組: {group}")
    return list(tasks)

def interactive_task_selection(config):
    """互動式任務選擇"""
    print("\n可用任務：")
    task_dict = {t['name']: t for t in config['pipeline']['tasks']}
    for i, task in enumerate(task_dict.keys(), 1):
        status = "啟用" if task_dict[task].get('enabled', True) else "禁用"
        print(f"{i}. {task} ({status})")
    
    print("\n輸入要執行的任務編號（多選用空格分隔，輸入 0 選擇全部啟用任務，輸入空行使用配置文件預設）：")
    user_input = input("> ").strip()
    
    if not user_input:
        return [t['name'] for t in config['pipeline']['tasks'] if t.get('enabled', True)]
    if user_input == "0":
        return [t['name'] for t in config['pipeline']['tasks'] if t.get('enabled', True)]
    
    selected_indices = [int(i) - 1 for i in user_input.split()]
    all_tasks = list(task_dict.keys())
    selected_tasks = [all_tasks[i] for i in selected_indices if 0 <= i < len(all_tasks)]
    return selected_tasks

def run_format_conversion(config, args):
    task_config = config['format_conversion'].copy()
    if args.input_format:
        task_config['input_formats'] = [args.input_format]
    if args.output_format:
        task_config['output_format'] = args.output_format
    convert_format(task_config)

def run_anomaly_detection(config, args):
    process_anomaly_detection(config)

def run_yolo_augmentation(config, args):
    augmentor = YoloDataAugmentor()
    augmentor.config = config['yolo_augmentation']
    augmentor._setup_output_dirs()
    augmentor.augmentations = augmentor._create_augmentations()
    augmentor.process_dataset()

def run_image_augmentation(config, args):
    augmentor = ImageAugmentor()
    augmentor.config = config['image_augmentation']
    augmentor._setup_output_dirs()
    augmentor.augmentations = augmentor._create_augmentations()
    augmentor.process_dataset()

def run_dataset_splitter(config, args):
    split_dataset(config)

def run_yolo_train(config, args):
    train_yolo(config)

def run_yolo_evaluation(config, args):
    evaluate_yolo(config)

def run_generate_report(config, args):
    generate_report(config)

def run_dataset_lint(config, args):
    lint_dataset(config)

def run_aug_preview(config, args):
    preview_dataset(config)

TASK_HANDLERS = {
    "format_conversion": run_format_conversion,
    "anomaly_detection": run_anomaly_detection,
    "yolo_augmentation": run_yolo_augmentation,
    "image_augmentation": run_image_augmentation,
    "dataset_splitter": run_dataset_splitter,
    "yolo_train": run_yolo_train,
    "yolo_evaluation": run_yolo_evaluation,
    "generate_report": run_generate_report,
    "dataset_lint": run_dataset_lint,
    "aug_preview": run_aug_preview,
}

def run_pipeline(tasks, config, logger, args, stop_event=None):
    """執行指定的任務流程"""
    tasks = validate_dependencies(tasks, config, logger)

    for task in tasks:
        if stop_event is not None and getattr(stop_event, 'is_set', lambda: False)():
            logger.info("收到停止請求，中止剩餘任務")
            break
        config = load_config_if_updated(args.config, config, logger)
        logger.info(f"開始執行任務: {task}")
        handler = TASK_HANDLERS.get(task)
        if not handler:
            logger.warning(f"未知任務: {task}")
            continue
        try:
            handler(config, args)
            logger.info(f"任務 {task} 完成")
        except Exception as e:
            logger.error(f"任務 {task} 執行失敗: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='圖像處理和數據增強流水線')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路徑')
    parser.add_argument('--tasks', nargs='+', help='要執行的任務列表，選項: format_conversion, anomaly_detection, yolo_augmentation, image_augmentation, dataset_splitter')
    parser.add_argument('--exclude-tasks', nargs='+', help='要排除的任務列表')
    parser.add_argument('--task-groups', nargs='+', help='要執行的任務組，選項: preprocess, augmentation, split')
    parser.add_argument('--interactive', action='store_true', help='啟動互動式任務選擇')
    parser.add_argument('--input-format', type=str, help='覆蓋輸入檔案格式（例如 .bmp）')
    parser.add_argument('--output-format', type=str, help='覆蓋輸出檔案格式（例如 .png）')
    
    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logging(config['pipeline']['log_file'])
    
    # 確定任務清單
    if args.interactive:
        tasks = interactive_task_selection(config)
    elif args.task_groups:
        tasks = get_tasks_from_groups(args.task_groups, config)
    elif args.tasks:
        tasks = args.tasks
    else:
        tasks = [t['name'] for t in config['pipeline']['tasks'] if t.get('enabled', True)]
    
    # 排除指定的任務
    if args.exclude_tasks:
        tasks = [t for t in tasks if t not in args.exclude_tasks]
    
    logger.info(f"最終任務清單: {tasks}")
    run_pipeline(tasks, config, logger, args)
    logger.info("流水線執行完成")

if __name__ == "__main__":
    main()
