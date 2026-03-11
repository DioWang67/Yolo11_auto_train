import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

try:
    import mlflow  # type: ignore
    from mlflow.exceptions import MlflowException  # type: ignore
except ImportError:
    mlflow = None  # type: ignore

    # Create dummy exception classes if mlflow is not installed
    class MlflowException(Exception):
        pass


class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking."""

    @abstractmethod
    def start_run(self, run_name: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        pass

    @abstractmethod
    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        pass

    @abstractmethod
    def end_run(self) -> None:
        pass


class MLflowTracker(ExperimentTracker):
    """MLflow implementation of ExperimentTracker."""

    def __init__(
        self, experiment_name: str = "yolo_training", tracking_uri: Optional[str] = None
    ):
        self._enabled: bool = True
        if mlflow is None:
            logging.warning("mlflow not installed. Tracking will be disabled.")
            self._enabled = False
            return

        self._enabled = True
        try:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            else:
                # Default to absolute path ./mlruns to avoid CWD issues with YOLO. Use POSIX for URI compatibility on Windows.
                from pathlib import Path
                mlruns = Path("mlruns").resolve().as_posix()
                mlflow.set_tracking_uri(f"file:///{mlruns}")

            mlflow.set_experiment(experiment_name)
        except MlflowException as e:
            logging.warning(f"Failed to setup MLflow tracking/experiment: {e}")
            self._enabled = False
        except OSError as e:
            logging.warning(f"OS/Network error during MLflow setup: {e}")
            self._enabled = False
            self._enabled = False

    def start_run(self, run_name: Optional[str] = None) -> None:
        if not self._enabled:
            return
        mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        # Flatten dict if necessary, MLflow handles basic types
        try:
            mlflow.log_params(params)
        except MlflowException as e:
            logging.warning(f"MLflow log_params failed: {e}")
        except OSError as e:
            logging.warning(f"OS/Network error during MLflow log_params: {e}")

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        if not self._enabled:
            return
        try:
            mlflow.log_metrics(metrics, step=step)
        except MlflowException as e:
            logging.warning(f"MLflow log_metrics failed: {e}")
        except OSError as e:
            logging.warning(f"OS/Network error during MLflow log_metrics: {e}")

    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        if not self._enabled:
            return
        if not os.path.exists(local_path):
            logging.warning(f"Artifact not found: {local_path}")
            return
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except MlflowException as e:
            logging.warning(f"MLflow log_artifact failed: {e}")
        except OSError as e:
            logging.warning(f"OS/Network error during MLflow log_artifact: {e}")

    def end_run(self) -> None:
        if not self._enabled:
            return
        mlflow.end_run()


class NullTracker(ExperimentTracker):
    """Dummy tracker when tracking is disabled."""

    def start_run(self, run_name: Optional[str] = None) -> None:
        pass

    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        pass

    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        pass

    def end_run(self) -> None:
        pass


def get_tracker(config: Dict[str, Any]) -> ExperimentTracker:
    """Factory to get configured tracker."""
    track_cfg = config.get("tracking", {})
    if not track_cfg.get("enabled", False):
        return NullTracker()

    backend = track_cfg.get("backend", "mlflow")
    if backend == "mlflow":
        return MLflowTracker(
            experiment_name=track_cfg.get("experiment_name", "Yolo11_Experiment"),
            tracking_uri=track_cfg.get("uri"),
        )

    return NullTracker()
