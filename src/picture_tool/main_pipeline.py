import argparse
import logging
import subprocess
import sys
from importlib import resources
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, Optional

import yaml
from picture_tool.anomaly import process_anomaly_detection
from picture_tool.augment import ImageAugmentor, YoloDataAugmentor
from picture_tool.color import color_verifier
from picture_tool.eval.yolo_evaluator import evaluate_yolo
from picture_tool.format import convert_format
from picture_tool.infer.batch_infer import run_batch_inference
from picture_tool.quality.dataset_linter import lint_dataset, preview_dataset
from picture_tool.report.qc_summary import generate_qc_summary
from picture_tool.report.report_generator import generate_report
from picture_tool.pipeline.core import Pipeline, Task
from picture_tool.split import split_dataset
from picture_tool.train.yolo_trainer import train_yolo
from picture_tool.position import run_position_validation

TASK_DESCRIPTIONS = {
    "format_conversion": "Convert image formats in bulk.",
    "anomaly_detection": "Run anomaly detection on reference/test folders.",
    "yolo_augmentation": "YOLO label-aware augmentation.",
    "image_augmentation": "Image-only augmentation.",
    "dataset_splitter": "Split dataset into train/val/test.",
    "dataset_lint": "Dataset quality linting.",
    "aug_preview": "Preview augmented samples.",
    "yolo_train": "Train YOLO model.",
    "yolo_evaluation": "Evaluate YOLO model.",
    "generate_report": "Aggregate training/eval report.",
    "batch_inference": "Batch inference over a folder.",
    "color_inspection": "Collect SAM color templates.",
    "color_verification": "Verify colors against templates.",
    "position_validation": "Offline position validation.",
    "qc_summary": "Summarise QC outputs into one report.",
}


_DEFAULT_CONFIG_RESOURCE = "default_pipeline.yaml"


def setup_logging(log_file):
    """Initialise logging targets for the pipeline run."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def _load_packaged_default() -> dict:
    """Load the bundled sample config shipped with the package."""
    package_resources = resources.files("picture_tool.resources")
    default_file = package_resources / _DEFAULT_CONFIG_RESOURCE
    with default_file.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(config_path: str | Path = "config.yaml"):
    """Load a pipeline configuration file, falling back to the packaged template."""
    path = Path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    try:
        return _load_packaged_default()
    except FileNotFoundError as exc:  # pragma: no cover - packaging error guard
        raise FileNotFoundError(
            f"Config file '{config_path}' not found and packaged default is missing."
        ) from exc


def load_config_if_updated(config_path, config, logger):
    """Reload the configuration if the on-disk file changed."""
    config_file = Path(config_path)
    if not config_file.exists():
        return config
    current_mtime = config_file.stat().st_mtime
    last_mtime = getattr(load_config_if_updated, "last_mtime", None)

    if last_mtime is None:
        load_config_if_updated.last_mtime = current_mtime
        return config

    if current_mtime > last_mtime:
        logger.info("Detected configuration change; reloading.")
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


def _detect_existing_weights(
    config: dict, prefer: str | None = None
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Locate an existing trained weight file and its run directory (if available).

    Order of precedence:
    1. Explicit override on preferred section (position_validation or evaluation).
    2. Explicit override on the other section.
    3. Latest run under project/name* containing best/last.pt.
    """
    ycfg = config.get("yolo_training", {}) or {}
    pv_cfg = ycfg.get("position_validation", {}) or {}
    ecfg = config.get("yolo_evaluation", {}) or {}
    project = Path(str(ycfg.get("project", "./runs/detect")))
    name_prefix = str(ycfg.get("name", "train"))

    def _candidate_path(path_val: Optional[str | Path]) -> Optional[tuple[Path, Path]]:
        if not path_val:
            return None
        p = Path(str(path_val)).expanduser().resolve()
        if not p.exists():
            return None
        run_dir = p.parent.parent if p.parent.name == "weights" else p.parent
        return p, run_dir.resolve()

    preferred_first = (
        [pv_cfg.get("weights"), ecfg.get("weights")]
        if prefer == "position"
        else [ecfg.get("weights"), pv_cfg.get("weights")]
    )
    for candidate in preferred_first:
        resolved = _candidate_path(candidate)
        if resolved:
            return resolved

    if project.exists():
        runs = [
            p
            for p in project.iterdir()
            if p.is_dir() and p.name.startswith(name_prefix)
        ]
        candidates: list[tuple[float, Path, Path]] = []
        for run in runs:
            for fname in ("best.pt", "last.pt"):
                w = run / "weights" / fname
                if w.exists():
                    candidates.append((w.stat().st_mtime, w.resolve(), run.resolve()))
        if candidates:
            _, w_path, run_dir = max(candidates, key=lambda entry: entry[0])
            return w_path, run_dir

    return None, None


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
        logger.info(
            "Applied CLI overrides: %s",
            ", ".join([f"{key}={value}" for key, value in changed]),
        )


def _auto_device(config: dict, logger: logging.Logger) -> None:
    yt = config.get("yolo_training", {})
    device = str(yt.get("device", "auto"))
    if device.lower() == "auto":
        try:
            import torch  # type: ignore

            yt["device"] = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            yt["device"] = "cpu"
        logger.info(f"Auto-selected device: {yt['device']}")
        config["yolo_training"] = yt


def _should_skip(task: str, config: dict, args, logger: Optional[logging.Logger] = None) -> Optional[str]:
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
            raise FileNotFoundError(f"Augmentation inputs missing: {in_dirs}")
        if all(_exists_and_nonempty(p) for p in out_dirs):
            if _mtime_latest(out_dirs) >= _mtime_latest(in_dirs):
                return "Outputs are newer than inputs; skipping."
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
                return "Split dataset is up-to-date; skipping."
    if task == "yolo_train":
        y = config["yolo_training"]
        run_dir = Path(y.get("project", "./runs/detect")) / y.get("name", "train")
        weights = run_dir / "weights" / "best.pt"
        dataset_dir = Path(y.get("dataset_dir", "./data/split"))
        if weights.exists():
            if weights.stat().st_mtime >= _mtime_latest([dataset_dir]):
                return "Found latest best.pt; skipping training."
    if task == "dataset_lint":
        lint_cfg = config.get("dataset_lint", {})
        img_dir = Path(lint_cfg.get("image_dir", "./data/augmented/images"))
        out_dir = Path(lint_cfg.get("output_dir", "./reports/lint"))
        csv = out_dir / "lint.csv"
        if csv.exists() and csv.stat().st_mtime >= _mtime_latest([img_dir]):
            return "Lint outputs are newer; skipping."
    if task == "aug_preview":
        p = config.get("aug_preview", {})
        img_dir = Path(p.get("image_dir", "./data/augmented/images"))
        out = Path(p.get("output_dir", "./reports/preview")) / "preview.png"
        if out.exists() and out.stat().st_mtime >= _mtime_latest([img_dir]):
            return "Preview output is newer; skipping."
    if task == "batch_inference":
        bi = config.get("batch_inference", {})
        in_dir = Path(bi.get("input_dir", "./data/raw/images"))
        out_dir = Path(bi.get("output_dir", "./reports/infer"))
        csv = out_dir / "predictions.csv"
        if csv.exists() and csv.stat().st_mtime >= _mtime_latest([in_dir]):
            return "Batch inference output is newer; skipping."
    return None


def validate_dependencies(tasks, config, logger):
    """Ensure each selected task honours dependency ordering and availability."""
    pipeline_section = config.setdefault("pipeline", {})
    task_entries = pipeline_section.setdefault("tasks", [])
    task_dict = {t["name"]: t for t in task_entries}
    known_sections = set(config.keys())
    missing_weights: set[str] = set()

    def _ensure_task(task_name: str) -> bool:
        if task_name in task_dict:
            return True
        section = config.get(task_name)
        if section is None and "TASK_HANDLERS" in globals():
            handler_known = task_name in TASK_HANDLERS
        else:
            handler_known = False
        if section is None and handler_known:
            section = {"enabled": True}
            config[task_name] = section
            known_sections.add(task_name)
        if task_name in known_sections and isinstance(section, dict):
            entry = {
                "name": task_name,
                "dependencies": section.get("dependencies", []),
                "enabled": section.get("enabled", True),
            }
            task_entries.append(entry)
            task_dict[task_name] = entry
            logger.info(f"Auto-registered task {task_name} from config section.")
            return True
        return False

    selected_tasks = tasks.copy()
    required_tasks: list[str] = []

    for task in tasks:
        if task not in task_dict and not _ensure_task(task):
            logger.warning(f"Unknown task: {task}")
            continue
        if not task_dict[task].get("enabled", True):
            logger.info(f"Task {task} disabled in config; skipping.")
            selected_tasks.remove(task)
            continue
        dependencies = task_dict[task].get("dependencies", [])
        for dep in dependencies:
            if (
                dep == "yolo_train"
                and task in {"yolo_evaluation", "position_validation"}
                and dep not in selected_tasks
                and dep not in required_tasks
            ):
                prefer = "position" if task == "position_validation" else None
                weights_path, run_dir = _detect_existing_weights(config, prefer=prefer)
                if weights_path:
                    logger.info(
                        "Task %s has existing weights at %s (run_dir=%s); skipping dependency %s.",
                        task,
                        weights_path,
                        run_dir,
                        dep,
                    )
                    continue
                missing_weights.add(task)
                logger.warning(
                    "Task %s requires trained weights but none were found under %s (name prefix '%s'). "
                    "Provide explicit weights or run training manually; dependency %s will not be auto-added.",
                    task,
                    Path(str((config.get("yolo_training", {}) or {}).get("project", "./runs/detect"))),
                    str((config.get("yolo_training", {}) or {}).get("name", "train")),
                    dep,
                )
                continue
            if dep not in selected_tasks and dep not in required_tasks:
                logger.info(f"Task {task} depends on {dep}; adding dependency.")
                required_tasks.append(dep)

    final_tasks: list[str] = []
    for task in selected_tasks:
        dependencies = task_dict.get(task, {}).get("dependencies", [])
        for dep in dependencies:
            if dep in required_tasks and dep not in final_tasks:
                final_tasks.append(dep)
        if task not in final_tasks:
            final_tasks.append(task)

    if missing_weights:
        task_list = ", ".join(sorted(missing_weights))
        raise FileNotFoundError(
            f"No existing YOLO weights detected for: {task_list}. "
            "Set yolo_evaluation.weights or yolo_training.position_validation.weights, "
            "or run yolo_train manually."
        )

    return final_tasks


def get_tasks_from_groups(groups, config):
    """Expand named task groups into a deduplicated task list."""
    tasks = set()
    for group in groups:
        if group in config["pipeline"]["task_groups"]:
            tasks.update(config["pipeline"]["task_groups"][group])
        else:
            logging.warning("Unknown task group: %s", group)
    return list(tasks)


def interactive_task_selection(config):
    """Prompt the user to pick tasks interactively."""
    print("\nAvailable tasks:")
    task_dict = {t["name"]: t for t in config["pipeline"]["tasks"]}
    for i, task in enumerate(task_dict.keys(), 1):
        status = "enabled" if task_dict[task].get("enabled", True) else "disabled"
        print(f"{i}. {task} ({status})")

    print("\nEnter task numbers separated by spaces. Enter 0 to run all enabled tasks, or press Enter to accept defaults.")
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
    weights_path, run_dir = _detect_existing_weights(config, prefer=None)
    if weights_path:
        ecfg = config.setdefault("yolo_evaluation", {})
        ecfg["weights"] = str(weights_path)
        logging.getLogger(__name__).info(
            "Using existing weights for evaluation: %s (run_dir=%s)",
            weights_path,
            run_dir,
        )
    else:
        logging.getLogger(__name__).warning(
            "No existing weights detected before evaluation; will rely on default resolution."
        )
    evaluate_yolo(config)


def run_generate_report(config, args):
    generate_report(config)


def run_dataset_lint(config, args):
    lint_dataset(config)


def run_aug_preview(config, args):
    preview_dataset(config)


def run_batch_infer(config, args):
    run_batch_inference(config)


def run_position_validation_task(config, args):
    """Run offline position validation using trained weights and sample images."""
    ycfg = config.get("yolo_training", {}) if isinstance(config, dict) else {}
    run_root = Path(str(ycfg.get("project", "./runs/detect")))
    run_name = str(ycfg.get("name", "train"))
    default_run_dir = run_root / run_name
    weights_path, detected_run_dir = _detect_existing_weights(config, prefer="position")
    run_dir = detected_run_dir or default_run_dir

    if weights_path:
        pv_cfg = ycfg.get("position_validation", {}) or {}
        pv_cfg["weights"] = str(weights_path)
        ycfg["position_validation"] = pv_cfg
        config["yolo_training"] = ycfg
        logging.getLogger(__name__).info(
            "Using existing weights for position validation: %s (run_dir=%s)",
            weights_path,
            run_dir,
        )

    if not run_dir.exists():
        raise FileNotFoundError(
            "No trained run found for position_validation. "
            f"Checked {run_dir} (project={run_root}, name prefix={run_name}). "
            "Provide yolo_training.position_validation.weights or run yolo_train manually."
        )

    return run_position_validation(config, run_dir, logger=logging.getLogger(__name__))

def _section_enabled(section: Optional[dict]) -> bool:
    if section is None:
        return False
    return section.get("enabled", True)


def _namespace(section: Optional[dict], **base):
    data = dict(base)
    if section:
        data.update(section)
    return SimpleNamespace(**data)


def run_color_inspection(config, args):
    color_cfg = config.get("color_inspection")
    if not _section_enabled(color_cfg):
        logging.getLogger(__name__).info("color_inspection disabled or missing; skipping.")
        return
    input_dir = Path(color_cfg.get("input_dir", "./data/led_qc/samples"))
    output_json = Path(color_cfg.get("output_json", "./reports/led_qc/color_stats.json"))
    colors = color_cfg.get("colors") or []
    sam_cfg = color_cfg.get("sam", {}) or {}
    checkpoint = sam_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError("color_inspection.sam.checkpoint is required.")
    model_type = sam_cfg.get("model_type", "vit_b")
    device = sam_cfg.get("device", "auto")
    max_side = int(sam_cfg.get("max_side", color_cfg.get("max_side", 2048)))

    cmd = [
        sys.executable,
        "-m",
        "picture_tool.color.color_inspection",
        "collect",
        "--input-dir",
        str(input_dir),
        "--output-json",
        str(output_json),
        "--sam-checkpoint",
        str(checkpoint),
        "--sam-model",
        model_type,
        "--device",
        device,
        "--max-side",
        str(max_side),
    ]
    if colors:
        cmd += ["--colors", *colors]
    logging.getLogger(__name__).info(
        "Launching SAM color selection GUI for %s (output -> %s)", input_dir, output_json
    )
    subprocess.run(cmd, check=True)


def run_color_verification(config, args):
    color_cfg = config.get("color_verification")
    if not _section_enabled(color_cfg):
        logging.getLogger(__name__).info("color_verification disabled or missing; skipping.")
        return
    input_dir = Path(color_cfg.get("input_dir", "./data/led_qc/infer"))
    stats_path = Path(color_cfg.get("color_stats", "./reports/led_qc/color_stats.json"))
    output_json = color_cfg.get("output_json")
    output_csv = color_cfg.get("output_csv")
    recursive = bool(color_cfg.get("recursive", False))
    expected_map = color_cfg.get("expected_map")
    infer_from_name = bool(color_cfg.get("expected_from_name", True))
    min_area = float(color_cfg.get("min_area_ratio", 0.01))
    max_area = float(color_cfg.get("max_area_ratio", 0.8))
    hsv_margin = color_cfg.get("hsv_margin", (8.0, 35.0, 40.0))
    lab_margin = color_cfg.get("lab_margin", (12.0, 8.0, 12.0))
    debug_plot = bool(color_cfg.get("debug_plot", False))
    debug_dir = color_cfg.get("debug_dir")
    mask_strategy = str(color_cfg.get("mask_strategy", "auto"))
    strip_cfg = color_cfg.get("strip_sampling", {}) or {}
    strip_opts = color_verifier.StripOptions(
        enabled=bool(strip_cfg.get("enabled", False)),
        segments=int(strip_cfg.get("segments", 10)),
        ratio_threshold=float(strip_cfg.get("threshold", 0.25)),
        orientation=str(strip_cfg.get("orientation", "vertical")),
        min_strip_ratio=float(strip_cfg.get("min_width_ratio", 0.05)),
        edge_margin=float(strip_cfg.get("edge_margin", color_verifier.DEFAULT_EDGE_MARGIN)),
        sat_threshold=float(strip_cfg.get("sat_threshold", color_verifier.DEFAULT_SAT_THRESHOLD)),
        val_threshold=float(strip_cfg.get("val_threshold", color_verifier.DEFAULT_VAL_THRESHOLD)),
        center_bias=strip_cfg.get("center_bias", True),
        center_sigma=float(strip_cfg.get("center_sigma", color_verifier.DEFAULT_CENTER_SIGMA)),
        min_valid_pixels=int(strip_cfg.get("min_valid_pixels", color_verifier.DEFAULT_MIN_VALID_PIXELS)),
        top_k=int(strip_cfg.get("top_k", color_verifier.DEFAULT_TOPK)),
        min_sat_ratio=float(strip_cfg.get("min_sat_ratio", color_verifier.DEFAULT_MIN_SAT_RATIO)),
        max_edge_ratio=float(strip_cfg.get("max_edge_ratio", color_verifier.DEFAULT_MAX_EDGE_RATIO)),
        black_s_threshold=float(strip_cfg.get("black_s_threshold", color_verifier.BLACK_S_THRESHOLD)),
        black_v_threshold=float(strip_cfg.get("black_v_threshold", color_verifier.BLACK_V_THRESHOLD)),
    )

    logger = logging.getLogger(__name__)
    logger.info(
        "Running color verification on %s using %s (recursive=%s)",
        input_dir,
        stats_path,
        recursive,
    )
    color_verifier.verify_directory(
        input_dir=input_dir,
        color_stats=stats_path,
        output_json=Path(output_json) if output_json else None,
        output_csv=Path(output_csv) if output_csv else None,
        recursive=recursive,
        expected_map=Path(expected_map) if expected_map else None,
        infer_expected_from_name=infer_from_name,
        min_area_ratio=min_area,
        max_area_ratio=max_area,
        hsv_margin=hsv_margin,
        lab_margin=lab_margin,
        segments=int(color_cfg.get("segments", strip_opts.segments)),
        orientation=str(color_cfg.get("orientation", strip_opts.orientation)),
        min_strip_ratio=float(color_cfg.get("min_strip_ratio", strip_opts.min_strip_ratio)),
        ratio_threshold=float(color_cfg.get("ratio_threshold", strip_opts.ratio_threshold)),
        edge_margin=float(color_cfg.get("edge_margin", strip_opts.edge_margin)),
        sat_threshold=float(color_cfg.get("sat_threshold", strip_opts.sat_threshold)),
        val_threshold=float(color_cfg.get("val_threshold", strip_opts.val_threshold)),
        min_sat_ratio=float(color_cfg.get("min_sat_ratio", strip_opts.min_sat_ratio)),
        max_edge_ratio=float(color_cfg.get("max_edge_ratio", strip_opts.max_edge_ratio)),
        black_s_threshold=float(color_cfg.get("black_s_threshold", strip_opts.black_s_threshold)),
        black_v_threshold=float(color_cfg.get("black_v_threshold", strip_opts.black_v_threshold)),
        debug_plot=debug_plot,
        debug_dir=Path(debug_dir) if debug_dir else None,
        strip_options=strip_opts,
        mask_strategy=mask_strategy,
        logger=logger,
    )



def run_qc_summary(config, args):
    """Aggregate QC outputs into one concise summary report."""
    generate_qc_summary(config, logger=logging.getLogger(__name__))


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
    "color_inspection": run_color_inspection,
    "color_verification": run_color_verification,
    "batch_inference": run_batch_infer,
    "position_validation": run_position_validation_task,
    "qc_summary": run_qc_summary,
}


def _config_dependencies(config: dict) -> dict[str, list[str]]:
    deps: dict[str, list[str]] = {}
    pipeline_cfg = config.get("pipeline", {})
    for entry in pipeline_cfg.get("tasks", []):
        if isinstance(entry, dict) and entry.get("name"):
            deps[str(entry["name"])] = list(entry.get("dependencies", []))
    return deps


def _make_skipper(name: str) -> Callable[[dict, object], Optional[str]]:
    def _skip(cfg: dict, args: object) -> Optional[str]:
        return _should_skip(name, cfg, args)

    return _skip


def build_task_registry(config: dict) -> dict[str, Task]:
    deps_map = _config_dependencies(config)
    registry: dict[str, Task] = {}
    for name, handler in TASK_HANDLERS.items():
        registry[name] = Task(
            name=name,
            run=handler,
            dependencies=deps_map.get(name, []),
            skip_fn=_make_skipper(name),
            description=TASK_DESCRIPTIONS.get(name, ""),
        )
    return registry


def run_pipeline(tasks, config, logger, args, stop_event=None):
    """Execute each task handler with dependency checks and skipping logic."""
    tasks = validate_dependencies(tasks, config, logger)
    setattr(args, "stop_event", stop_event)

    def _before_task(task_obj: Task, cfg: dict) -> dict:
        fresh_cfg = load_config_if_updated(args.config, cfg, logger)
        _apply_cli_overrides(fresh_cfg, args, logger)
        _auto_device(fresh_cfg, logger)
        return fresh_cfg

    registry = build_task_registry(config)
    pipeline = Pipeline(registry, logger=logger)
    pipeline.run(tasks, config, args, before_task=_before_task)


def main():
    parser = argparse.ArgumentParser(
        description="YOLO auto-train pipeline orchestration tools."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the pipeline config."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help=(
            "Specific tasks to run (e.g. format_conversion, anomaly_detection, "
            "yolo_augmentation, image_augmentation, dataset_splitter)."
        ),
    )
    parser.add_argument(
        "--exclude-tasks", nargs="+", help="Tasks to exclude from execution."
    )
    parser.add_argument(
        "--task-groups",
        nargs="+",
        help="Named task groups to run (e.g. preprocess, augmentation, split).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively select tasks before running.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks and exit.",
    )
    parser.add_argument(
        "--describe-task",
        type=str,
        help="Show details for a specific task and exit.",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        help="Override the expected input format (e.g. .bmp).",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        help="Override the output format (e.g. .png).",
    )

    # Cache/force and overrides
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache heuristics and force all tasks to run.",
    )
    # Training overrides
    parser.add_argument(
        "--device", type=str, help="Override train/eval device, e.g. 0 or cpu."
    )
    parser.add_argument("--epochs", type=int, help="Override training epochs.")
    parser.add_argument("--imgsz", type=int, help="Override training image size.")
    parser.add_argument("--batch", type=int, help="Override training batch size.")
    parser.add_argument("--model", type=str, help="Override model weight path/name.")
    parser.add_argument(
        "--project", type=str, help="Override Ultralytics project output directory."
    )
    parser.add_argument("--name", type=str, help="Override Ultralytics run name.")
    parser.add_argument("--weights", type=str, help="Override eval/infer weights path.")
    # Inference overrides
    parser.add_argument(
        "--infer-input", type=str, help="Override batch inference input directory."
    )
    parser.add_argument(
        "--infer-output", type=str, help="Override batch inference output directory."
    )

    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logging(config["pipeline"]["log_file"])
    registry = build_task_registry(config)

    if args.list_tasks:
        for task_name, task in registry.items():
            deps = ",".join(task.dependencies) if task.dependencies else "-"
            enabled = next((t.get("enabled", True) for t in config["pipeline"]["tasks"] if t["name"] == task_name), True)
            desc = task.description or ""
            print(f"{task_name:20} deps=[{deps}] enabled={enabled} {desc}")
        return
    if args.describe_task:
        task = registry.get(args.describe_task)
        if not task:
            print(f"Unknown task: {args.describe_task}")
            return
        deps = ", ".join(task.dependencies) if task.dependencies else "None"
        enabled = next((t.get("enabled", True) for t in config["pipeline"]["tasks"] if t["name"] == task.name), True)
        print(f"Task: {task.name}\nEnabled: {enabled}\nDependencies: {deps}\nDescription: {task.description or 'N/A'}")
        return

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

    if args.exclude_tasks:
        tasks = [t for t in tasks if t not in args.exclude_tasks]

    logger.info(f"Final task sequence: {tasks}")
    run_pipeline(tasks, config, logger, args)
    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
