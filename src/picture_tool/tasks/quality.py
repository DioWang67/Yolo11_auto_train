import logging
import sys
import subprocess
from pathlib import Path
from picture_tool.anomaly import process_anomaly_detection
from picture_tool.split import split_dataset
from picture_tool.quality.dataset_linter import lint_dataset
from picture_tool.report.report_generator import generate_report
from picture_tool.infer.batch_infer import run_batch_inference
from picture_tool.color import color_verifier
from picture_tool.report.qc_summary import generate_qc_summary
from picture_tool.pipeline.utils import mtime_latest
from picture_tool.pipeline.core import Task
from picture_tool.tasks.quality_schemas import ColorInspectionConfig, ColorVerificationConfig
from pydantic import ValidationError


def run_anomaly_detection(config, args):
    process_anomaly_detection(config)


def run_dataset_splitter(config, args):
    split_dataset(config)


def skip_dataset_splitter(config, args):
    sc = config.get("train_test_split")
    sc = config.get("train_test_split")
    if not sc:
        return None
    in_dirs = [Path(sc["input"]["image_dir"]), Path(sc["input"]["label_dir"])]
    out_root = Path(sc["output"]["output_dir"])
    out_dirs = [
        out_root / "train" / "images",
        out_root / "val" / "images",
        out_root / "test" / "images",
    ]
    if all(p.exists() for p in in_dirs) and all(p.exists() for p in out_dirs):
        if mtime_latest(out_dirs) >= mtime_latest(in_dirs):
            return "Split dataset is up-to-date; skipping."
    return None


def run_dataset_lint(config, args):
    lint_dataset(config)


def skip_dataset_lint(config, args):
    lint_cfg = config.get("dataset_lint", {})
    img_dir = Path(lint_cfg.get("image_dir", "./data/project/processed/images"))
    out_dir = Path(lint_cfg.get("output_dir", "./runs/project/quality/lint"))
    csv = out_dir / "lint.csv"
    if csv.exists() and csv.stat().st_mtime >= mtime_latest([img_dir]):
        return "Lint outputs are newer; skipping."
    return None


def run_generate_report(config, args):
    generate_report(config)


def run_batch_infer(config, args):
    run_batch_inference(config)


def skip_batch_infer(config, args):
    bi = config.get("batch_inference", {})
    in_dir = Path(bi.get("input_dir", "./data/project/split/test/images"))
    out_dir = Path(bi.get("output_dir", "./runs/project/infer"))
    csv = out_dir / "predictions.csv"
    if csv.exists() and csv.stat().st_mtime >= mtime_latest([in_dir]):
        return "Batch inference output is newer; skipping."
    return None


def run_qc_summary(config, args):
    """Aggregate QC outputs into one concise summary report."""
    generate_qc_summary(config, logger=logging.getLogger(__name__))


def _section_enabled(section) -> bool:
    if section is None:
        return False
    return section.get("enabled", True)


def run_color_inspection(config, args):
    color_cfg_dict = config.get("color_inspection")
    if not color_cfg_dict:
        logging.getLogger(__name__).info("color_inspection missing; skipping.")
        return
        
    try:
        cfg = ColorInspectionConfig.model_validate(color_cfg_dict)
    except ValidationError as e:
        logging.getLogger(__name__).error(f"color_inspection configuration error:\n{e}")
        raise ValueError(f"Invalid color_inspection configuration: {e}") from e

    if not cfg.enabled:
        logging.getLogger(__name__).info("color_inspection disabled; skipping.")
        return

    cmd = [
        sys.executable,
        "-m",
        "picture_tool.color.color_inspection",
        "collect",
        "--input-dir",
        str(cfg.input_dir),
        "--output-json",
        str(cfg.output_json),
        "--sam-checkpoint",
        str(cfg.sam.checkpoint),
        "--sam-model",
        cfg.sam.model_type,
        "--device",
        cfg.sam.device,
        "--max-side",
        str(cfg.max_side),
    ]
    if cfg.colors:
        cmd += ["--colors", *cfg.colors]
    logging.getLogger(__name__).info(
        "Launching SAM color selection GUI for %s (output -> %s)",
        cfg.input_dir,
        cfg.output_json,
    )
    subprocess.run(cmd, check=True)


def run_color_verification(config, args):
    color_cfg_dict = config.get("color_verification")
    if not color_cfg_dict:
        logging.getLogger(__name__).info("color_verification missing; skipping.")
        return
        
    try:
        cfg = ColorVerificationConfig.model_validate(color_cfg_dict)
    except ValidationError as e:
        logging.getLogger(__name__).error(f"color_verification configuration error:\n{e}")
        raise ValueError(f"Invalid color_verification configuration: {e}") from e

    if not cfg.enabled:
        logging.getLogger(__name__).info("color_verification disabled; skipping.")
        return

    strip_opts = color_verifier.StripOptions(
        enabled=cfg.strip_sampling.enabled,
        segments=cfg.strip_sampling.segments,
        ratio_threshold=cfg.strip_sampling.threshold,
        orientation=cfg.strip_sampling.orientation,
        min_strip_ratio=cfg.strip_sampling.min_width_ratio,
        edge_margin=cfg.strip_sampling.edge_margin,
        sat_threshold=cfg.strip_sampling.sat_threshold,
        val_threshold=cfg.strip_sampling.val_threshold,
        center_bias=cfg.strip_sampling.center_bias,
        center_sigma=cfg.strip_sampling.center_sigma,
        min_valid_pixels=cfg.strip_sampling.min_valid_pixels,
        top_k=cfg.strip_sampling.top_k,
        min_sat_ratio=cfg.strip_sampling.min_sat_ratio,
        max_edge_ratio=cfg.strip_sampling.max_edge_ratio,
        black_s_threshold=cfg.strip_sampling.black_s_threshold,
        black_v_threshold=cfg.strip_sampling.black_v_threshold,
    )

    logger = logging.getLogger(__name__)
    logger.info(
        "Running color verification on %s using %s (recursive=%s)",
        cfg.input_dir,
        cfg.color_stats,
        cfg.recursive,
    )
    
    def get_override(opt_val, strip_val):
        return opt_val if opt_val is not None else strip_val

    color_verifier.verify_directory(
        input_dir=cfg.input_dir,
        color_stats=cfg.color_stats,
        output_json=cfg.output_json,
        output_csv=cfg.output_csv,
        recursive=cfg.recursive,
        expected_map=cfg.expected_map,
        infer_expected_from_name=cfg.expected_from_name,
        min_area_ratio=cfg.min_area_ratio,
        max_area_ratio=cfg.max_area_ratio,
        hsv_margin=cfg.hsv_margin,
        lab_margin=cfg.lab_margin,
        segments=get_override(cfg.segments, strip_opts.segments),
        orientation=get_override(cfg.orientation, strip_opts.orientation),
        min_strip_ratio=get_override(cfg.min_strip_ratio, strip_opts.min_strip_ratio),
        ratio_threshold=get_override(cfg.ratio_threshold, strip_opts.ratio_threshold),
        edge_margin=get_override(cfg.edge_margin, strip_opts.edge_margin),
        sat_threshold=get_override(cfg.sat_threshold, strip_opts.sat_threshold),
        val_threshold=get_override(cfg.val_threshold, strip_opts.val_threshold),
        min_sat_ratio=get_override(cfg.min_sat_ratio, strip_opts.min_sat_ratio),
        max_edge_ratio=get_override(cfg.max_edge_ratio, strip_opts.max_edge_ratio),
        black_s_threshold=get_override(cfg.black_s_threshold, strip_opts.black_s_threshold),
        black_v_threshold=get_override(cfg.black_v_threshold, strip_opts.black_v_threshold),
        debug_plot=cfg.debug_plot,
        debug_dir=cfg.debug_dir,
        strip_options=strip_opts,
        mask_strategy=cfg.mask_strategy,
        logger=logger,
    )


TASKS = [
    Task(
        name="anomaly_detection",
        run=run_anomaly_detection,
        description="Run anomaly detection on reference/test folders.",
    ),
    Task(
        name="dataset_splitter",
        run=run_dataset_splitter,
        skip_fn=skip_dataset_splitter,
        description="Split dataset into train/val/test.",
    ),
    Task(
        name="dataset_lint",
        run=run_dataset_lint,
        skip_fn=skip_dataset_lint,
        description="Dataset quality linting.",
    ),
    Task(
        name="generate_report",
        run=run_generate_report,
        description="Aggregate training/eval report.",
    ),
    Task(
        name="batch_inference",
        run=run_batch_infer,
        skip_fn=skip_batch_infer,
        description="Batch inference over a folder.",
    ),
    Task(
        name="qc_summary",
        run=run_qc_summary,
        description="Summarise QC outputs into one report.",
    ),
    Task(
        name="color_inspection",
        run=run_color_inspection,
        description="Collect SAM color templates.",
    ),
    Task(
        name="color_verification",
        run=run_color_verification,
        description="Verify colors against templates.",
    ),
]
