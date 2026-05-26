from __future__ import annotations

import json
import logging
import os
import platform
import shutil
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml  # type: ignore[import]

IMAGE_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")


@dataclass(frozen=True)
class AnomalibModelOption:
    """Human-facing summary for a supported Anomalib model option."""

    name: str
    class_path: str
    summary: str
    best_for: str
    trade_off: str
    default_backbone: str
    default_layers: tuple[str, ...]
    default_n_features: int | None = None


MODEL_OPTIONS: dict[str, AnomalibModelOption] = {
    "padim": AnomalibModelOption(
        name="padim",
        class_path="anomalib.models.Padim",
        summary="Fast feature-statistics baseline with predictable CPU runtime.",
        best_for="Quick PCBA baselines, small datasets, and integration checks.",
        trade_off="Usually less flexible than PatchCore on subtle or varied defects.",
        default_backbone="resnet18",
        default_layers=("layer1",),
        default_n_features=16,
    ),
    "patchcore": AnomalibModelOption(
        name="patchcore",
        class_path="anomalib.models.Patchcore",
        summary="Memory-bank nearest-neighbor anomaly detector.",
        best_for="Higher-quality anomaly localization when you can spend more memory/time.",
        trade_off="Heavier than PaDiM and more sensitive to feature/memory settings.",
        default_backbone="resnet18",
        default_layers=("layer1",),
        default_n_features=None,
    ),
}


@dataclass(frozen=True)
class AnomalibTrainingConfig:
    """Validated configuration for a local Anomalib training run.

    Args:
        name: Dataset/run name passed to Anomalib.
        root: Dataset root directory.
        normal_dir: Directory containing normal training images. Relative paths
            are resolved from ``root``.
        project: Output root directory for Anomalib artifacts.
        model: Supported Anomalib model name.
        task: Anomalib task type, normally ``classification`` or ``segmentation``.
        image_size: Square resize size used by the Anomalib datamodule.
        train_batch_size: Training dataloader batch size.
        eval_batch_size: Validation/test dataloader batch size.
        num_workers: Dataloader worker count.
        accelerator: Lightning accelerator.
        devices: Lightning devices argument.
        max_epochs: Maximum training epochs.
        seed: Optional split seed.
        normal_split_ratio: Ratio used when Anomalib needs to split normal data.
        val_split_mode: Validation split mode.
        val_split_ratio: Validation split ratio when splitting from train.
        test_split_mode: Test split mode.
        test_split_ratio: Test split ratio for synthetic/from-train modes.
        abnormal_dir: Optional abnormal test image directory.
        normal_test_dir: Optional normal test image directory.
        mask_dir: Optional ground-truth mask directory.
        pre_trained: Whether to use pretrained backbone weights.
        backbone: Backbone name for supported models.
        layers: Backbone layers for supported models.
        n_features: Optional PaDiM embedding feature count.
        coreset_sampling_ratio: Patchcore coreset ratio.
        num_neighbors: Patchcore nearest neighbor count.
        patch_anomalib_symlink: Avoid Anomalib's versioned ``latest`` symlink on
            Windows workstations that do not have symlink privileges.
        require_anomalous_validation: Fail fast if no abnormal validation/test
            data is configured.
        limit_train_batches: Optional Lightning smoke-test limit.
        limit_val_batches: Optional Lightning smoke-test limit.
    """

    name: str
    root: Path
    normal_dir: Path
    project: Path
    model: str = "padim"
    task: str = "segmentation"
    image_size: int = 256
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 0
    accelerator: str = "auto"
    devices: str | int = "auto"
    max_epochs: int = 1
    seed: int | None = 42
    normal_split_ratio: float = 0.2
    val_split_mode: str = "from_train"
    val_split_ratio: float = 0.2
    test_split_mode: str = "none"
    test_split_ratio: float = 0.2
    abnormal_dir: Path | None = None
    normal_test_dir: Path | None = None
    mask_dir: Path | None = None
    pre_trained: bool = False
    backbone: str = "resnet18"
    layers: tuple[str, ...] = ("layer1",)
    n_features: int | None = 16
    coreset_sampling_ratio: float = 0.1
    num_neighbors: int = 9
    patch_anomalib_symlink: bool = True
    require_anomalous_validation: bool = False
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None


@dataclass(frozen=True)
class AnomalibFolderTrainingResult:
    """Result summary for the folder-first Anomalib training entrypoint.

    Args:
        run_dir: Anomalib output directory.
        checkpoint_path: Lightning checkpoint path if one was produced.
        report_path: JSON report path.
        baseline_only: Whether the run lacks abnormal validation/test images.
        normal_image_count: Number of normal training images.
        abnormal_image_count: Number of abnormal images configured.
    """

    run_dir: Path
    checkpoint_path: Path | None
    report_path: Path
    baseline_only: bool
    normal_image_count: int
    abnormal_image_count: int


@dataclass(frozen=True)
class AnomalibDeploymentResult:
    """Result summary for deploying an Anomalib run into yolo11_inference."""

    deploy_dir: Path
    config_path: Path
    checkpoint_path: Path
    report_path: Path | None
    baseline_only: bool
    usable_for_deployment: bool
    warnings: list[str] = field(default_factory=list)


def parse_anomalib_training_config(config: dict[str, Any]) -> AnomalibTrainingConfig:
    """Parse and validate the ``anomalib_training`` config block.

    Args:
        config: Full pipeline configuration.

    Returns:
        Validated Anomalib training config.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    raw = config.get("anomalib_training")
    if not isinstance(raw, dict):
        raise ValueError("Missing required config section: anomalib_training")

    root = Path(str(raw.get("root", "./data/anomaly"))).expanduser()
    normal_dir = _resolve_folder_path(root, raw.get("normal_dir", "train/good"))
    cfg = AnomalibTrainingConfig(
        name=str(raw.get("name", "anomaly")),
        root=root,
        normal_dir=normal_dir,
        project=Path(str(raw.get("project", "./runs/anomalib"))).expanduser(),
        model=_normalize_model_name(str(raw.get("model", "padim"))),
        task=str(raw.get("task", "segmentation")),
        image_size=int(raw.get("image_size", 256)),
        train_batch_size=int(raw.get("train_batch_size", 8)),
        eval_batch_size=int(raw.get("eval_batch_size", 8)),
        num_workers=int(raw.get("num_workers", 0)),
        accelerator=str(raw.get("accelerator", "auto")),
        devices=raw.get("devices", "auto"),
        max_epochs=int(raw.get("max_epochs", 1)),
        seed=_optional_int(raw.get("seed", 42)),
        normal_split_ratio=float(raw.get("normal_split_ratio", 0.2)),
        val_split_mode=str(raw.get("val_split_mode", "from_train")),
        val_split_ratio=float(raw.get("val_split_ratio", 0.2)),
        test_split_mode=str(raw.get("test_split_mode", "none")),
        test_split_ratio=float(raw.get("test_split_ratio", 0.2)),
        abnormal_dir=_optional_folder_path(root, raw.get("abnormal_dir")),
        normal_test_dir=_optional_folder_path(root, raw.get("normal_test_dir")),
        mask_dir=_optional_folder_path(root, raw.get("mask_dir")),
        pre_trained=bool(raw.get("pre_trained", False)),
        backbone=str(raw.get("backbone", "resnet18")),
        layers=tuple(str(layer) for layer in raw.get("layers", ["layer1"])),
        n_features=_optional_int(raw.get("n_features", 16)),
        coreset_sampling_ratio=float(raw.get("coreset_sampling_ratio", 0.1)),
        num_neighbors=int(raw.get("num_neighbors", 9)),
        patch_anomalib_symlink=bool(
            raw.get("patch_anomalib_symlink", platform.system() == "Windows")
        ),
        require_anomalous_validation=bool(
            raw.get("require_anomalous_validation", False)
        ),
        limit_train_batches=raw.get("limit_train_batches"),
        limit_val_batches=raw.get("limit_val_batches"),
    )

    _validate_training_config(cfg)
    _validate_folder_dataset(cfg)
    return cfg


def train_anomalib(
    config: dict[str, Any],
    *,
    logger: logging.Logger | None = None,
) -> Path:
    """Train an Anomalib model from the pipeline configuration.

    Args:
        config: Full pipeline configuration containing ``anomalib_training``.
        logger: Optional logger.

    Returns:
        The expected Anomalib run directory.

    Raises:
        ImportError: If Anomalib is not installed.
        RuntimeError: If Anomalib training fails.
        ValueError: If configuration or dataset layout is invalid.
    """
    log = logger or logging.getLogger(__name__)
    cfg = parse_anomalib_training_config(config)

    try:
        from anomalib.engine import Engine  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "anomalib is required for anomalib_train. Install the yolo_anomalib "
            "environment dependencies or add anomalib==1.2.0."
        ) from exc

    datamodule = _build_datamodule(cfg)
    model = _build_model(cfg)
    trainer_kwargs = _trainer_kwargs(cfg)
    expected_run_dir = _expected_workspace_dir(cfg, model_name=getattr(model, "name", cfg.model))

    log.info(
        "Starting Anomalib training | model=%s dataset=%s root=%s normal=%s output=%s",
        cfg.model,
        cfg.name,
        cfg.root,
        cfg.normal_dir,
        expected_run_dir,
    )

    try:
        with _anomalib_symlink_patch(cfg.patch_anomalib_symlink):
            engine = Engine(default_root_dir=cfg.project, **trainer_kwargs)
            engine.fit(model=model, datamodule=datamodule)
    except Exception as exc:
        raise RuntimeError(f"Anomalib training failed: {exc}") from exc

    _write_training_metadata(expected_run_dir, cfg)
    log.info("Anomalib training complete: %s", expected_run_dir)
    return expected_run_dir


def train_anomalib_folder(
    input_dir: Path,
    *,
    product: str,
    area: str,
    project: Path | None = None,
    model: str = "padim",
    image_size: int = 256,
    batch_size: int = 8,
    max_epochs: int = 1,
    accelerator: str = "cpu",
    devices: str | int = 1,
    pre_trained: bool = False,
    require_anomalous_validation: bool = False,
    force: bool = False,
    tmp_dir: Path | None = None,
    logger: logging.Logger | None = None,
) -> AnomalibFolderTrainingResult:
    """Train Anomalib directly from a product/area folder.

    The function recognizes the common project layouts used by this repository:
    ``split/train/images``, Anomalib-style ``train/good`` plus optional
    ``test/good`` and ``test/bad``, or a plain image folder.

    Args:
        input_dir: Product/area folder or direct normal image folder.
        product: Product name used in output metadata.
        area: Area name used in output metadata.
        project: Optional output root. Defaults to ``runs/anomalib/<product>/<area>``.
        model: Supported Anomalib model name.
        image_size: Square resize size.
        batch_size: Train and eval batch size.
        max_epochs: Maximum training epochs.
        accelerator: Lightning accelerator.
        devices: Lightning devices argument.
        pre_trained: Whether to use pretrained backbone weights.
        require_anomalous_validation: Fail when no abnormal folder is found.
        force: Retrain even if a checkpoint already exists.
        tmp_dir: Optional temp directory used to avoid full system temp drives.
        logger: Optional logger.

    Returns:
        Folder training result summary.

    Raises:
        ValueError: If no usable normal image folder is found.
        RuntimeError: If training fails.
    """
    log = logger or logging.getLogger(__name__)
    layout = infer_anomalib_folder_layout(input_dir)
    output_project = project or Path("runs") / "anomalib" / product / area
    run_name = f"{product}_{area}"
    cfg = {
        "anomalib_training": {
            "root": str(layout["root"]),
            "normal_dir": str(layout["normal_dir"]),
            "abnormal_dir": _string_or_none(layout.get("abnormal_dir")),
            "normal_test_dir": _string_or_none(layout.get("normal_test_dir")),
            "mask_dir": _string_or_none(layout.get("mask_dir")),
            "project": str(output_project),
            "name": run_name,
            "model": model,
            "task": "segmentation",
            "image_size": image_size,
            "train_batch_size": batch_size,
            "eval_batch_size": batch_size,
            "num_workers": 0,
            "accelerator": accelerator,
            "devices": devices,
            "max_epochs": max_epochs,
            "val_split_mode": "from_train"
            if layout.get("abnormal_dir") is None
            else "from_test",
            "val_split_ratio": 0.2,
            "test_split_mode": "none"
            if layout.get("abnormal_dir") is None
            else "from_dir",
            "pre_trained": pre_trained,
            "backbone": "resnet18",
            "layers": ["layer1"],
            "n_features": 16,
            "patch_anomalib_symlink": True,
            "require_anomalous_validation": require_anomalous_validation,
        }
    }

    parsed_cfg = parse_anomalib_training_config(cfg)
    existing_run_dir = find_existing_anomalib_run(cfg)
    if existing_run_dir is not None and not force:
        log.info("Using existing Anomalib checkpoint: %s", existing_run_dir)
        return _write_folder_training_report(
            input_dir=input_dir,
            product=product,
            area=area,
            run_dir=existing_run_dir,
            cfg=cfg,
            parsed_cfg=parsed_cfg,
        )

    tmp_root = tmp_dir or Path("runs") / "tmp"
    with _temporary_training_env(tmp_root):
        run_dir = train_anomalib(cfg, logger=log)

    return _write_folder_training_report(
        input_dir=input_dir,
        product=product,
        area=area,
        run_dir=run_dir,
        cfg=cfg,
        parsed_cfg=parsed_cfg,
    )


def _write_folder_training_report(
    *,
    input_dir: Path,
    product: str,
    area: str,
    run_dir: Path,
    cfg: dict[str, Any],
    parsed_cfg: AnomalibTrainingConfig,
) -> AnomalibFolderTrainingResult:
    checkpoint_path = _find_checkpoint(run_dir)
    normal_count = _count_images(parsed_cfg.normal_dir)
    abnormal_count = (
        _count_images(parsed_cfg.abnormal_dir)
        if parsed_cfg.abnormal_dir is not None
        else 0
    )
    baseline_only = abnormal_count == 0
    report_path = run_dir / "training_report.json"
    report = {
        "product": product,
        "area": area,
        "input_dir": str(input_dir),
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "baseline_only": baseline_only,
        "usable_for_deployment": not baseline_only,
        "normal_image_count": normal_count,
        "abnormal_image_count": abnormal_count,
        "warnings": _folder_training_warnings(baseline_only),
        "model_option": asdict(MODEL_OPTIONS[parsed_cfg.model]),
        "config": cfg["anomalib_training"],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    return AnomalibFolderTrainingResult(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        baseline_only=baseline_only,
        normal_image_count=normal_count,
        abnormal_image_count=abnormal_count,
    )


def infer_anomalib_folder_layout(input_dir: Path) -> dict[str, Path | None]:
    """Infer Anomalib folder settings from a user-provided directory.

    Args:
        input_dir: Product/area folder or image folder.

    Returns:
        Mapping containing ``root``, ``normal_dir`` and optional test folders.

    Raises:
        ValueError: If no normal images can be found.
    """
    base = input_dir.expanduser()
    if not base.exists():
        raise ValueError(f"Input folder does not exist: {base}")

    candidates = [
        (base / "anomalib", Path("train/good")),
        (base, Path("train/good")),
        (base / "split", Path("train/images")),
        (base, Path("split/train/images")),
        (base, Path("good")),
    ]
    for root, normal_rel in candidates:
        normal_abs = root / normal_rel
        if normal_abs.exists() and _count_images(normal_abs) > 0:
            return _layout_with_optional_test_dirs(root, normal_rel)

    if _count_images(base) > 0:
        return {
            "root": base.parent,
            "normal_dir": Path(base.name),
            "abnormal_dir": None,
            "normal_test_dir": None,
            "mask_dir": None,
        }

    raise ValueError(
        "No normal training images found. Expected one of: split/train/images, "
        "train/good, anomalib/train/good, good, or a direct image folder."
    )


def supported_anomalib_models() -> list[AnomalibModelOption]:
    """Return supported model options in display order."""
    return [MODEL_OPTIONS["padim"], MODEL_OPTIONS["patchcore"]]


def deploy_anomalib_run(
    run_dir: Path,
    *,
    inference_root: Path,
    product: str,
    area: str,
    threshold: float = 0.5,
    force: bool = False,
) -> AnomalibDeploymentResult:
    """Deploy a trained Anomalib run into yolo11_inference model layout.

    Args:
        run_dir: Anomalib run directory containing ``weights/lightning/*.ckpt``.
        inference_root: Root of the yolo11_inference repository.
        product: Product name.
        area: Area name.
        threshold: Runtime anomaly score threshold.
        force: Overwrite existing deployed files.

    Returns:
        Deployment result summary.

    Raises:
        FileNotFoundError: If required source files are missing.
        FileExistsError: If destination files exist and ``force`` is false.
    """
    resolved_run = run_dir.expanduser()
    checkpoint = _find_checkpoint(resolved_run)
    if checkpoint is None:
        raise FileNotFoundError(f"No Anomalib checkpoint found under {resolved_run}")

    report = _read_training_report(resolved_run)
    model_name = str((report.get("config") or {}).get("model", "padim"))
    model_option = MODEL_OPTIONS.get(_normalize_model_name(model_name), MODEL_OPTIONS["padim"])
    baseline_only = bool(report.get("baseline_only", True))
    usable_for_deployment = bool(report.get("usable_for_deployment", False))
    warnings = list(report.get("warnings") or [])
    if baseline_only and not warnings:
        warnings = _folder_training_warnings(True)

    deploy_dir = inference_root / "models" / product / area / "anomalib"
    weights_dir = deploy_dir / "weights"
    dest_checkpoint = weights_dir / "model.ckpt"
    dest_report = deploy_dir / "training_report.json"
    config_path = deploy_dir / "config.yaml"

    for path in (dest_checkpoint, dest_report, config_path):
        if path.exists() and not force:
            raise FileExistsError(f"Destination already exists: {path}. Use --force to overwrite.")

    weights_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, dest_checkpoint)

    source_report = resolved_run / "training_report.json"
    report_path: Path | None = None
    if source_report.exists():
        shutil.copy2(source_report, dest_report)
        report_path = dest_report

    config = _build_inference_anomalib_config(
        product=product,
        area=area,
        model_option=model_option,
        threshold=threshold,
        baseline_only=baseline_only,
        usable_for_deployment=usable_for_deployment,
        warnings=warnings,
    )
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

    return AnomalibDeploymentResult(
        deploy_dir=deploy_dir,
        config_path=config_path,
        checkpoint_path=dest_checkpoint,
        report_path=report_path,
        baseline_only=baseline_only,
        usable_for_deployment=usable_for_deployment,
        warnings=warnings,
    )


def find_existing_anomalib_run(config: dict[str, Any]) -> Path | None:
    """Return the expected run directory if it already contains a checkpoint."""
    cfg = parse_anomalib_training_config(config)
    model_name = "Patchcore" if cfg.model == "patchcore" else "Padim"
    run_dir = _expected_workspace_dir(cfg, model_name=model_name)
    checkpoint_dir = run_dir / "weights" / "lightning"
    if any(checkpoint_dir.glob("*.ckpt")):
        return run_dir
    return None


def _layout_with_optional_test_dirs(root: Path, normal_rel: Path) -> dict[str, Path | None]:
    normal_test_rel = _first_existing_relative(
        root,
        [Path("test/good"), Path("test/normal")],
    )
    abnormal_rel = _first_existing_relative(
        root,
        [Path("test/bad"), Path("test/abnormal"), Path("test/defect")],
    )
    mask_rel = _first_existing_relative(
        root,
        [Path("ground_truth/bad"), Path("ground_truth/abnormal"), Path("ground_truth/defect")],
    )
    return {
        "root": root,
        "normal_dir": normal_rel,
        "abnormal_dir": abnormal_rel,
        "normal_test_dir": normal_test_rel,
        "mask_dir": mask_rel,
    }


def _first_existing_relative(root: Path, candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if (root / candidate).exists() and _count_images(root / candidate) > 0:
            return candidate
    return None


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


@contextmanager
def _temporary_training_env(tmp_dir: Path) -> Iterator[None]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    previous = {key: os.environ.get(key) for key in ("TEMP", "TMP", "YOLO_CONFIG_DIR")}
    os.environ["TEMP"] = str(tmp_dir.resolve())
    os.environ["TMP"] = str(tmp_dir.resolve())
    os.environ.setdefault("YOLO_CONFIG_DIR", str((Path("runs") / "ultralytics_config").resolve()))
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _find_checkpoint(run_dir: Path) -> Path | None:
    checkpoint_dir = run_dir / "weights" / "lightning"
    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"))
    return checkpoints[-1] if checkpoints else None


def _read_training_report(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "training_report.json"
    if not report_path.exists():
        return {
            "baseline_only": True,
            "usable_for_deployment": False,
            "warnings": ["training_report.json not found; deployment marked as baseline-only."],
            "config": {"model": "padim"},
        }
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Training report must be a JSON object: {report_path}")
    return raw


def _build_inference_anomalib_config(
    *,
    product: str,
    area: str,
    model_option: AnomalibModelOption,
    threshold: float,
    baseline_only: bool,
    usable_for_deployment: bool,
    warnings: list[str],
) -> dict[str, Any]:
    ckpt_path = f"models/{product}/{area}/anomalib/weights/model.ckpt"
    return {
        "enable_yolo": False,
        "enable_anomalib": True,
        "output_dir": "Result",
        "anomalib_config": {
            "threshold": threshold,
            "output": f"Result/anomalib/{product}/{area}/YYYYMMDD",
            "baseline_only": baseline_only,
            "usable_for_deployment": usable_for_deployment,
            "warnings": warnings,
            "model": {"class_path": model_option.class_path},
            "data": {},
            "models": {
                product: {
                    area: {
                        "ckpt_path": ckpt_path,
                        "threshold": threshold,
                        "baseline_only": baseline_only,
                        "usable_for_deployment": usable_for_deployment,
                    }
                }
            },
        },
    }


def _folder_training_warnings(baseline_only: bool) -> list[str]:
    if not baseline_only:
        return []
    return [
        "No abnormal validation/test images were configured.",
        "This run is a baseline-only model; adaptive thresholds are not deployment-grade.",
    ]


def _resolve_folder_path(root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return root / path


def _optional_folder_path(root: Path, value: object) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_folder_path(root, value)


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _normalize_model_name(value: str) -> str:
    normalized = value.strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "padim": "padim",
        "paDiM".lower(): "padim",
        "patchcore": "patchcore",
    }
    if normalized not in aliases:
        raise ValueError(
            "Unsupported anomalib_training.model. Supported values: padim, patchcore"
        )
    return aliases[normalized]


def _validate_training_config(cfg: AnomalibTrainingConfig) -> None:
    if cfg.image_size <= 0:
        raise ValueError("anomalib_training.image_size must be greater than 0")
    if cfg.train_batch_size <= 0 or cfg.eval_batch_size <= 0:
        raise ValueError("Anomalib batch sizes must be greater than 0")
    if cfg.num_workers < 0:
        raise ValueError("anomalib_training.num_workers must be >= 0")
    if cfg.max_epochs <= 0:
        raise ValueError("anomalib_training.max_epochs must be greater than 0")
    if not 0 <= cfg.normal_split_ratio <= 1:
        raise ValueError("anomalib_training.normal_split_ratio must be between 0 and 1")
    if not 0 <= cfg.val_split_ratio <= 1:
        raise ValueError("anomalib_training.val_split_ratio must be between 0 and 1")
    if not 0 <= cfg.test_split_ratio <= 1:
        raise ValueError("anomalib_training.test_split_ratio must be between 0 and 1")
    if cfg.model == "patchcore" and cfg.num_neighbors <= 0:
        raise ValueError("anomalib_training.num_neighbors must be greater than 0")
    if cfg.n_features is not None and cfg.n_features <= 0:
        raise ValueError("anomalib_training.n_features must be greater than 0")


def _validate_folder_dataset(cfg: AnomalibTrainingConfig) -> None:
    if not cfg.root.exists():
        raise ValueError(f"anomalib_training.root does not exist: {cfg.root}")
    if not cfg.normal_dir.exists():
        raise ValueError(
            "anomalib_training.normal_dir does not exist: "
            f"{cfg.normal_dir}. Expected normal images for training."
        )
    normal_count = _count_images(cfg.normal_dir)
    if normal_count == 0:
        raise ValueError(f"No normal training images found in {cfg.normal_dir}")
    if cfg.require_anomalous_validation and cfg.abnormal_dir is None:
        raise ValueError(
            "require_anomalous_validation is true but abnormal_dir is not configured"
        )
    for label, folder in (
        ("abnormal_dir", cfg.abnormal_dir),
        ("normal_test_dir", cfg.normal_test_dir),
        ("mask_dir", cfg.mask_dir),
    ):
        if folder is not None and not folder.exists():
            raise ValueError(f"anomalib_training.{label} does not exist: {folder}")


def _count_images(folder: Path) -> int:
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _build_datamodule(cfg: AnomalibTrainingConfig) -> Any:
    from anomalib.data import Folder  # type: ignore

    return Folder(
        name=cfg.name,
        root=cfg.root,
        normal_dir=_path_for_anomalib(cfg.root, cfg.normal_dir),
        abnormal_dir=_optional_path_for_anomalib(cfg.root, cfg.abnormal_dir),
        normal_test_dir=_optional_path_for_anomalib(cfg.root, cfg.normal_test_dir),
        mask_dir=_optional_path_for_anomalib(cfg.root, cfg.mask_dir),
        normal_split_ratio=cfg.normal_split_ratio,
        train_batch_size=cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        task=cfg.task,
        image_size=(cfg.image_size, cfg.image_size),
        test_split_mode=cfg.test_split_mode,
        test_split_ratio=cfg.test_split_ratio,
        val_split_mode=cfg.val_split_mode,
        val_split_ratio=cfg.val_split_ratio,
        seed=cfg.seed,
    )


def _path_for_anomalib(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path


def _optional_path_for_anomalib(root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return _path_for_anomalib(root, path)


def _build_model(cfg: AnomalibTrainingConfig) -> Any:
    if cfg.model == "padim":
        from anomalib.models import Padim  # type: ignore

        return Padim(
            backbone=cfg.backbone,
            layers=list(cfg.layers),
            pre_trained=cfg.pre_trained,
            n_features=cfg.n_features,
        )

    from anomalib.models import Patchcore  # type: ignore

    return Patchcore(
        backbone=cfg.backbone,
        layers=cfg.layers,
        pre_trained=cfg.pre_trained,
        coreset_sampling_ratio=cfg.coreset_sampling_ratio,
        num_neighbors=cfg.num_neighbors,
    )


def _trainer_kwargs(cfg: AnomalibTrainingConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "accelerator": cfg.accelerator,
        "devices": cfg.devices,
        "max_epochs": cfg.max_epochs,
    }
    if cfg.limit_train_batches is not None:
        kwargs["limit_train_batches"] = cfg.limit_train_batches
    if cfg.limit_val_batches is not None:
        kwargs["limit_val_batches"] = cfg.limit_val_batches
    return kwargs


def _expected_workspace_dir(cfg: AnomalibTrainingConfig, *, model_name: str) -> Path:
    return cfg.project / model_name / cfg.name / "latest"


@contextmanager
def _anomalib_symlink_patch(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    import anomalib.engine.engine as engine_module  # type: ignore

    original = engine_module.create_versioned_dir

    def _create_latest_dir(root_dir: Path) -> Path:
        latest_dir = Path(root_dir) / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        return latest_dir

    engine_module.create_versioned_dir = _create_latest_dir
    try:
        yield
    finally:
        engine_module.create_versioned_dir = original


def _write_training_metadata(run_dir: Path, cfg: AnomalibTrainingConfig) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": _json_safe_config(cfg),
        "normal_image_count": _count_images(cfg.normal_dir),
        "abnormal_image_count": _count_images(cfg.abnormal_dir)
        if cfg.abnormal_dir is not None
        else 0,
    }
    (run_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _json_safe_config(cfg: AnomalibTrainingConfig) -> dict[str, Any]:
    data = asdict(cfg)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
        elif isinstance(value, tuple):
            data[key] = list(value)
    return data
