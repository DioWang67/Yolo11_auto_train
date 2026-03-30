"""Pre-execution validation for the picture_tool pipeline.

``PreflightChecker.run(tasks, config)`` scans the selected tasks and active
configuration for likely failures *before* the pipeline starts.  Results are
``PreflightIssue`` objects with two severity levels:

* ``Severity.ERROR``   — blocks execution; must be fixed first.
* ``Severity.WARNING`` — shown to the user, who may choose to proceed anyway.

Adding a new check
------------------
Implement a ``_check_*`` method on ``PreflightChecker`` and call it from
``run()``.  Keep checks fast (no network, no heavy I/O); sample files rather
than reading entire datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List


class Severity(Enum):
    ERROR = "error"      # Blocks pipeline execution
    WARNING = "warning"  # User may continue after seeing this


@dataclass
class PreflightIssue:
    severity: Severity
    task: str       # Task name this issue relates to, or "config" for global
    message: str

    @property
    def is_blocking(self) -> bool:
        return self.severity is Severity.ERROR


class PreflightChecker:
    """Run a suite of cheap validations before pipeline execution."""

    # Maximum label files to scan for class-ID check (avoid scanning huge datasets)
    _MAX_LABEL_SAMPLE = 300

    def run(self, tasks: list[str], config: dict) -> list[PreflightIssue]:
        """Return all issues found for the given *tasks* and *config*."""
        issues: list[PreflightIssue] = []
        ycfg: dict = config.get("yolo_training", {}) or {}
        task_set = set(tasks)

        # ── Global checks (always run) ────────────────────────────────
        self._check_class_names_vs_labels(ycfg, issues)

        # ── Task-specific checks ──────────────────────────────────────
        if task_set & {"dataset_splitter", "yolo_train"}:
            self._check_split_input_paths(config, issues)

        if "yolo_train" in task_set:
            self._check_class_names_defined(ycfg, issues)
            self._check_yolo_base_weights(ycfg, issues)

        if "color_inspection" in task_set:
            self._check_sam_checkpoint(config, issues)

        if "color_verification" in task_set:
            self._check_color_stats(config, issues)

        if "deploy" in task_set:
            self._check_deploy_config(ycfg, issues)
            self._check_deploy_has_weights(ycfg, tasks, issues)

        return issues

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_class_names_vs_labels(
        self, ycfg: dict, issues: list[PreflightIssue]
    ) -> None:
        """ERROR if label class IDs exceed the length of class_names."""
        class_names: list = ycfg.get("class_names") or []
        if not class_names:
            return  # Handled by _check_class_names_defined

        dataset_dir = ycfg.get("dataset_dir", "")
        if not dataset_dir:
            return

        label_dir = Path(dataset_dir) / "train" / "labels"
        if not label_dir.exists():
            return  # Path check will surface this

        label_files = list(label_dir.glob("*.txt"))[: self._MAX_LABEL_SAMPLE]
        if not label_files:
            return

        max_id = -1
        try:
            for lf in label_files:
                for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = line.strip().split()
                    if parts:
                        try:
                            max_id = max(max_id, int(parts[0]))
                        except ValueError:
                            pass
        except OSError:
            return

        if max_id < 0:
            return

        n = len(class_names)
        if max_id >= n:
            issues.append(
                PreflightIssue(
                    severity=Severity.ERROR,
                    task="yolo_train",
                    message=(
                        f"Label 檔案中最大 class ID 為 {max_id}，"
                        f"但 class_names 只有 {n} 個（ID 0~{n - 1}）。\n"
                        f"請確認 class_names 順序與 labels 一致，否則訓練結果將錯誤。"
                    ),
                )
            )

    def _check_split_input_paths(
        self, config: dict, issues: list[PreflightIssue]
    ) -> None:
        """ERROR if dataset_splitter input directories are missing."""
        split_cfg: dict = config.get("train_test_split", {}) or {}
        inp: dict = split_cfg.get("input", {}) or {}
        for key, label in [("image_dir", "圖片目錄"), ("label_dir", "標籤目錄")]:
            path_str = inp.get(key, "")
            if path_str and not Path(path_str).exists():
                issues.append(
                    PreflightIssue(
                        severity=Severity.ERROR,
                        task="dataset_splitter",
                        message=f"找不到 {label}：{path_str}",
                    )
                )

    def _check_class_names_defined(
        self, ycfg: dict, issues: list[PreflightIssue]
    ) -> None:
        """ERROR if class_names is empty or missing."""
        if not ycfg.get("class_names"):
            issues.append(
                PreflightIssue(
                    severity=Severity.ERROR,
                    task="yolo_train",
                    message=(
                        "yolo_training.class_names 未設定。"
                        "YOLO 訓練無法在沒有類別名稱的情況下執行。"
                    ),
                )
            )

    def _check_yolo_base_weights(
        self, ycfg: dict, issues: list[PreflightIssue]
    ) -> None:
        """WARNING if the base model weights file cannot be found locally.

        Ultralytics may auto-download it, so this is a warning, not an error.
        """
        model = ycfg.get("model", "")
        if not model:
            return
        candidates = [Path(model), Path("models") / Path(model).name]
        if not any(p.exists() for p in candidates):
            issues.append(
                PreflightIssue(
                    severity=Severity.WARNING,
                    task="yolo_train",
                    message=(
                        f"本地找不到 YOLO 底模：{model}。"
                        f"Ultralytics 將嘗試自動下載（需要網路連線）。"
                    ),
                )
            )

    def _check_sam_checkpoint(
        self, config: dict, issues: list[PreflightIssue]
    ) -> None:
        """ERROR if the SAM2 checkpoint required by color_inspection is missing."""
        ci_cfg: dict = config.get("color_inspection", {}) or {}
        sam_path = ci_cfg.get("sam_checkpoint", "")
        if sam_path and not Path(sam_path).exists():
            issues.append(
                PreflightIssue(
                    severity=Severity.ERROR,
                    task="color_inspection",
                    message=(
                        f"找不到 SAM2 checkpoint：{sam_path}。\n"
                        "請至 Meta AI 官方頁面下載 sam2_b.pt 並放置於 models/ 目錄。"
                    ),
                )
            )

    def _check_color_stats(
        self, config: dict, issues: list[PreflightIssue]
    ) -> None:
        """ERROR if color_verification is selected but color_stats.json doesn't exist."""
        cv_cfg: dict = config.get("color_verification", {}) or {}
        stats_path = cv_cfg.get("color_stats", "")
        if stats_path and not Path(stats_path).exists():
            issues.append(
                PreflightIssue(
                    severity=Severity.ERROR,
                    task="color_verification",
                    message=(
                        f"找不到顏色統計檔：{stats_path}。\n"
                        "請先執行 color_inspection task 建立 stats.json，再進行 color_verification。"
                    ),
                )
            )

    def _check_deploy_has_weights(
        self, ycfg: dict, tasks: list[str], issues: list[PreflightIssue]
    ) -> None:
        """WARNING if deploy is selected but no trained weights exist and yolo_train is not selected."""
        if "yolo_train" in tasks:
            return  # yolo_train will produce weights before deploy runs

        project = ycfg.get("project", "")
        name = ycfg.get("name", "train")
        if project and name:
            weights = Path(project) / name / "weights" / "best.pt"
            if not weights.exists():
                issues.append(
                    PreflightIssue(
                        severity=Severity.WARNING,
                        task="deploy",
                        message=(
                            f"找不到已訓練的 weights：{weights}，"
                            f"且本次未選取 yolo_train。\n"
                            "若先前從未訓練過，deploy 將無任何檔案可部署。"
                        ),
                    )
                )

    def _check_deploy_config(
        self, ycfg: dict, issues: list[PreflightIssue]
    ) -> None:
        """Check deploy task prerequisites."""
        dcfg: dict = ycfg.get("deploy", {}) or {}
        if not dcfg.get("enabled", False):
            issues.append(
                PreflightIssue(
                    severity=Severity.WARNING,
                    task="deploy",
                    message=(
                        "deploy task 已選取，但 yolo_training.deploy.enabled 為 false。"
                        "任務將跳過，不會部署任何檔案。"
                    ),
                )
            )
            return

        inf_dir = dcfg.get("inference_models_dir", "")
        if not inf_dir:
            issues.append(
                PreflightIssue(
                    severity=Severity.ERROR,
                    task="deploy",
                    message="deploy.inference_models_dir 未設定，無法執行部署。",
                )
            )
            return

        inf_path = Path(inf_dir)
        if not inf_path.is_absolute():
            inf_path = (Path.cwd() / inf_path).resolve()
        if not inf_path.exists():
            issues.append(
                PreflightIssue(
                    severity=Severity.WARNING,
                    task="deploy",
                    message=f"部署目標目錄不存在，執行時將自動建立：{inf_path}",
                )
            )
