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


def run_anomaly_detection(config, args):
    process_anomaly_detection(config)


def run_dataset_splitter(config, args):
    split_dataset(config)


def skip_dataset_splitter(config, args):
    sc = config["train_test_split"]
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
    img_dir = Path(lint_cfg.get("image_dir", "./data/augmented/images"))
    out_dir = Path(lint_cfg.get("output_dir", "./reports/lint"))
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
    in_dir = Path(bi.get("input_dir", "./data/raw/images"))
    out_dir = Path(bi.get("output_dir", "./reports/infer"))
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
