import argparse
import yaml
import logging
from pathlib import Path
from typing import Iterable, Optional
from picture_tool.format import convert_format
from picture_tool.anomaly import process_anomaly_detection
from picture_tool.augment import YoloDataAugmentor, ImageAugmentor
from picture_tool.split import split_dataset
from picture_tool.train.yolo_trainer import train_yolo
from picture_tool.eval.yolo_evaluator import evaluate_yolo
from picture_tool.report.report_generator import generate_report
from picture_tool.quality.dataset_linter import lint_dataset, preview_dataset
from picture_tool.infer.batch_infer import run_batch_inference


def setup_logging(log_file):
    """設置日誌系統"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    """讀取 config.yaml 檔案"""
    with open(config_path, "r", encoding="utf-8") as f:
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


def _mtime_latest(paths: Iterable[Path]) -> float:
    mts = []
    for p in paths:
        if p.is_file():
            mts.append(p.stat().st_mtime)
        elif p.is_dir():
            for sub in p.rglob("*"):
                if sub.is_file():
                    mts.append(sub.stat().st_mtime)
    return max(mts) if mts else 0.0


def _exists_and_nonempty(p: Path) -> bool:
    return p.exists() and any(p.iterdir())


def _apply_cli_overrides(config: dict, args, logger: logging.Logger) -> None:
    yt = config.get("yolo_training", {})
    changed = []
    if getattr(args, "device", None):
        yt["device"] = args.device
        changed.append(("device", args.device))
    if getattr(args, "epochs", None):
        yt["epochs"] = int(args.epochs)
        changed.append(("epochs", yt["epochs"]))
    if getattr(args, "imgsz", None):
        yt["imgsz"] = int(args.imgsz)
        changed.append(("imgsz", yt["imgsz"]))
    if getattr(args, "batch", None):
        yt["batch"] = int(args.batch)
        changed.append(("batch", yt["batch"]))
    if getattr(args, "model", None):
        yt["model"] = args.model
        changed.append(("model", yt["model"]))
    if getattr(args, "project", None):
        yt["project"] = args.project
        changed.append(("project", yt["project"]))
    if getattr(args, "name", None):
        yt["name"] = args.name
        changed.append(("name", yt["name"]))
    config["yolo_training"] = yt

    ye = config.get("yolo_evaluation", {})
    if getattr(args, "weights", None):
        ye["weights"] = args.weights
        changed.append(("eval.weights", ye["weights"]))
    config["yolo_evaluation"] = ye

    bi = config.get("batch_inference", {})
    if getattr(args, "infer_input", None):
        bi["input_dir"] = args.infer_input
        changed.append(("infer.input", bi["input_dir"]))
    if getattr(args, "infer_output", None):
        bi["output_dir"] = args.infer_output
        changed.append(("infer.output", bi["output_dir"]))
    config["batch_inference"] = bi

    if changed:
        logger.info("套用 CLI 覆蓋: " + ", ".join([f"{k}={v}" for k, v in changed]))


def _auto_device(config: dict, logger: logging.Logger) -> None:
    yt = config.get("yolo_training", {})
    device = str(yt.get("device", "auto"))
    if device.lower() == "auto":
        try:
            import torch  # type: ignore

            yt["device"] = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            yt["device"] = "cpu"
        logger.info(f"自動選擇裝置: {yt['device']}")
        config["yolo_training"] = yt


def _should_skip(
    task: str, config: dict, args, logger: logging.Logger
) -> Optional[str]:
    force = getattr(args, "force", False) or config.get("pipeline", {}).get(
        "force", False
    )
    if force:
        return None
    if task == "yolo_augmentation":
        ic = config["yolo_augmentation"]["input"]
        oc = config["yolo_augmentation"]["output"]
        in_dirs = [Path(ic["image_dir"]), Path(ic["label_dir"])]
        out_dirs = [Path(oc["image_dir"]), Path(oc["label_dir"])]
        if not all(p.exists() for p in in_dirs):
            raise FileNotFoundError(f"增強輸入不存在: {in_dirs}")
        if all(_exists_and_nonempty(p) for p in out_dirs):
            if _mtime_latest(out_dirs) >= _mtime_latest(in_dirs):
                return "輸出較新，已跳過"
    if task == "dataset_splitter":
        sc = config["train_test_split"]
        in_dirs = [Path(sc["input"]["image_dir"]), Path(sc["input"]["label_dir"])]
        out_root = Path(sc["output"]["output_dir"])
        out_dirs = [
            out_root / "train" / "images",
            out_root / "val" / "images",
            out_root / "test" / "images",
        ]
        if all(p.exists() for p in in_dirs) and all(p.exists() for p in out_dirs):
            if _mtime_latest(out_dirs) >= _mtime_latest(in_dirs):
                return "分割結果較新，已跳過"
    if task == "yolo_train":
        y = config["yolo_training"]
        run_dir = Path(y.get("project", "./runs/detect")) / y.get("name", "train")
        weights = run_dir / "weights" / "best.pt"
        dataset_dir = Path(y.get("dataset_dir", "./data/split"))
        if weights.exists():
            if weights.stat().st_mtime >= _mtime_latest([dataset_dir]):
                return "已有最新 best.pt，已跳過"
    if task == "dataset_lint":
        lint_cfg = config.get("dataset_lint", {})
        img_dir = Path(lint_cfg.get("image_dir", "./data/augmented/images"))
        out_dir = Path(lint_cfg.get("output_dir", "./reports/lint"))
        csv = out_dir / "lint.csv"
        if csv.exists() and csv.stat().st_mtime >= _mtime_latest([img_dir]):
            return "Lint 輸出較新，已跳過"
    if task == "aug_preview":
        p = config.get("aug_preview", {})
        img_dir = Path(p.get("image_dir", "./data/augmented/images"))
        out = Path(p.get("output_dir", "./reports/preview")) / "preview.png"
        if out.exists() and out.stat().st_mtime >= _mtime_latest([img_dir]):
            return "預覽較新，已跳過"
    if task == "batch_inference":
        bi = config.get("batch_inference", {})
        in_dir = Path(bi.get("input_dir", "./data/raw/images"))
        out_dir = Path(bi.get("output_dir", "./reports/infer"))
        csv = out_dir / "predictions.csv"
        if csv.exists() and csv.stat().st_mtime >= _mtime_latest([in_dir]):
            return "推論結果較新，已跳過"
    return None


def validate_dependencies(tasks, config, logger):
    """驗證任務依賴關係，並保持原始任務順序"""
    task_dict = {t["name"]: t for t in config["pipeline"]["tasks"]}
    selected_tasks = tasks.copy()  # 複製原始任務列表以保持順序
    required_tasks = []

    # 檢查每個任務的依賴
    for task in tasks:
        if task not in task_dict:
            logger.warning(f"未知任務: {task}")
            continue
        if not task_dict[task].get("enabled", True):
            logger.info(f"任務 {task} 在配置文件中被禁用，跳過")
            selected_tasks.remove(task)
            continue
        dependencies = task_dict[task].get("dependencies", [])
        for dep in dependencies:
            if dep not in selected_tasks and dep not in required_tasks:
                logger.info(f"任務 {task} 依賴 {dep}，自動添加")
                required_tasks.append(dep)

    # 將依賴任務插入到列表中，確保依賴任務在主任務之前
    final_tasks = []
    for task in selected_tasks:
        dependencies = task_dict.get(task, {}).get("dependencies", [])
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
        if group in config["pipeline"]["task_groups"]:
            tasks.update(config["pipeline"]["task_groups"][group])
        else:
            logging.warning(f"未知任務組: {group}")
    return list(tasks)


def interactive_task_selection(config):
    """互動式任務選擇"""
    print("\n可用任務：")
    task_dict = {t["name"]: t for t in config["pipeline"]["tasks"]}
    for i, task in enumerate(task_dict.keys(), 1):
        status = "啟用" if task_dict[task].get("enabled", True) else "禁用"
        print(f"{i}. {task} ({status})")

    print(
        "\n輸入要執行的任務編號（多選用空格分隔，輸入 0 選擇全部啟用任務，輸入空行使用配置文件預設）："
    )
    user_input = input("> ").strip()

    if not user_input:
        return [
            t["name"] for t in config["pipeline"]["tasks"] if t.get("enabled", True)
        ]
    if user_input == "0":
        return [
            t["name"] for t in config["pipeline"]["tasks"] if t.get("enabled", True)
        ]

    selected_indices = [int(i) - 1 for i in user_input.split()]
    all_tasks = list(task_dict.keys())
    selected_tasks = [all_tasks[i] for i in selected_indices if 0 <= i < len(all_tasks)]
    return selected_tasks


def run_format_conversion(config, args):
    task_config = config["format_conversion"].copy()
    if args.input_format:
        task_config["input_formats"] = [args.input_format]
    if args.output_format:
        task_config["output_format"] = args.output_format
    convert_format(task_config)


def run_anomaly_detection(config, args):
    process_anomaly_detection(config)


def run_yolo_augmentation(config, args):
    augmentor = YoloDataAugmentor()
    augmentor.config = config["yolo_augmentation"]
    augmentor._setup_output_dirs()
    augmentor.augmentations = augmentor._create_augmentations()
    augmentor.process_dataset()


def run_image_augmentation(config, args):
    augmentor = ImageAugmentor()
    augmentor.config = config["image_augmentation"]
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


def run_batch_infer(config, args):
    run_batch_inference(config)


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
    "batch_inference": run_batch_infer,
}


def run_pipeline(tasks, config, logger, args, stop_event=None):
    """執行指定的任務流程"""
    tasks = validate_dependencies(tasks, config, logger)

    for task in tasks:
        if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
            logger.info("收到停止請求，中止剩餘任務")
            break
        config = load_config_if_updated(args.config, config, logger)
        logger.info(f"開始執行任務: {task}")
        _apply_cli_overrides(config, args, logger)
        _auto_device(config, logger)
        skip_reason = _should_skip(task, config, args, logger)
        if skip_reason:
            logger.info(f"跳過 {task}: {skip_reason}")
            continue
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
    parser = argparse.ArgumentParser(description="圖像處理和數據增強流水線")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="配置文件路徑"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="要執行的任務列表，選項: format_conversion, anomaly_detection, yolo_augmentation, image_augmentation, dataset_splitter",
    )
    parser.add_argument("--exclude-tasks", nargs="+", help="要排除的任務列表")
    parser.add_argument(
        "--task-groups",
        nargs="+",
        help="要執行的任務組，選項: preprocess, augmentation, split",
    )
    parser.add_argument("--interactive", action="store_true", help="啟動互動式任務選擇")
    parser.add_argument(
        "--input-format", type=str, help="覆蓋輸入檔案格式（例如 .bmp）"
    )
    parser.add_argument(
        "--output-format", type=str, help="覆蓋輸出檔案格式（例如 .png）"
    )

    # Cache/force and overrides
    parser.add_argument("--force", action="store_true", help="忽略快取並強制重跑任務")
    # Training overrides
    parser.add_argument("--device", type=str, help="覆寫訓練/驗證裝置，例如 0 或 cpu")
    parser.add_argument("--epochs", type=int, help="覆寫訓練 epochs")
    parser.add_argument("--imgsz", type=int, help="覆寫訓練影像尺寸")
    parser.add_argument("--batch", type=int, help="覆寫訓練 batch 大小")
    parser.add_argument("--model", type=str, help="覆寫初始權重路徑/名稱")
    parser.add_argument("--project", type=str, help="覆寫 Ultralytics 專案輸出資料夾")
    parser.add_argument("--name", type=str, help="覆寫實驗名稱")
    parser.add_argument("--weights", type=str, help="覆寫驗證/推論時的權重路徑")
    # Inference overrides
    parser.add_argument("--infer-input", type=str, help="批次推論輸入資料夾")
    parser.add_argument("--infer-output", type=str, help="批次推論輸出資料夾")

    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logging(config["pipeline"]["log_file"])

    # 確定任務清單
    if args.interactive:
        tasks = interactive_task_selection(config)
    elif args.task_groups:
        tasks = get_tasks_from_groups(args.task_groups, config)
    elif args.tasks:
        tasks = args.tasks
    else:
        tasks = [
            t["name"] for t in config["pipeline"]["tasks"] if t.get("enabled", True)
        ]

    # 排除指定的任務
    if args.exclude_tasks:
        tasks = [t for t in tasks if t not in args.exclude_tasks]

    logger.info(f"最終任務清單: {tasks}")
    run_pipeline(tasks, config, logger, args)
    logger.info("流水線執行完成")


if __name__ == "__main__":
    main()
