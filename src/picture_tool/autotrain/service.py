"""The service interface a future agent will drive.

Every method is a plain Python function returning JSON-serialisable data.
There is no LLM call anywhere in this package, and there should never be one
below this line: an agent belongs *above* this interface, choosing which
operations to invoke, not inside the training core deciding what a metric
means.

The read methods work whether or not the feature is enabled --- reporting on
a disabled subsystem is useful and harmless. Every method that writes calls
``require_enabled`` first.

Nothing exposed here can deploy a model. That is not an oversight to be
fixed later by adding a ``promote`` method; it is the boundary. Adopting a
challenger stays a named human action in the inspection release flow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from picture_tool.autotrain import AutoTrainError
from picture_tool.autotrain import golden as golden_module
from picture_tool.autotrain.candidate_pool import CandidatePool
from picture_tool.autotrain.collector import collect_production_records
from picture_tool.autotrain.config import AutoTrainConfig
from picture_tool.autotrain.dataset_versions import (
    DatasetVersionStore,
    LabelledSample,
)
from picture_tool.autotrain.evaluator import evaluate_candidate as _evaluate_candidate
from picture_tool.autotrain.labeling import verified_samples
from picture_tool.autotrain.metrics import ModelMetrics, compare
from picture_tool.autotrain.orchestrator import TrainingCycle, run_training_cycle
from picture_tool.autotrain.paths import AutoTrainPaths
from picture_tool.autotrain.class_schema import (
    resolve_class_schema,
    schema_from_station_config,
)
from picture_tool.autotrain.registry import (
    CandidateRegistry,
    read_champion,
    station_class_schema,
)

LOGGER = logging.getLogger(__name__)

#: Statuses that mean production judged the board bad.
_FAILURE_STATUSES = frozenset({"DETECTION_FAIL", "FAIL", "NG", "ERROR"})


class AutoTrainService:
    """Stable facade over the autonomous training subsystem."""

    def __init__(
        self,
        config: AutoTrainConfig | None = None,
        paths: AutoTrainPaths | None = None,
    ) -> None:
        self.config = config or AutoTrainConfig.load()
        self.paths = paths or AutoTrainPaths.discover()
        self.product = self.config.station.product
        self.area = self.config.station.area

    # -- reads ------------------------------------------------------------------

    def get_model_health(self) -> dict[str, Any]:
        """Champion, candidates and whether the subsystem is even switched on."""
        champion = read_champion(
            self.paths.production_model_dir(self.product, self.area)
        )
        registry = self._registry()
        by_status: dict[str, int] = {}
        for model in registry.list_models():
            by_status[model.status] = by_status.get(model.status, 0) + 1
        golden_status = golden_module.resolve(
            self.config.golden.dataset_path, self.config.golden.manifest_sha256
        )
        return {
            "enabled": self.config.enabled,
            "product": self.product,
            "area": self.area,
            "champion": champion.to_dict() if champion else None,
            "candidates_by_status": by_status,
            "pool": self._pool().statistics(),
            "dataset_versions": len(self._datasets().list_versions()),
            "golden": golden_status.to_dict(),
        }

    def get_recent_failures(self, limit: int = 50) -> list[dict[str, Any]]:
        """Recent production inspections that did not pass.

        Reads production records directly rather than the candidate pool, so
        it reports what the line actually saw, including inspections no
        selector picked.
        """
        self.config.require_enabled()
        collected = collect_production_records(self.paths, self.config)
        failures = [
            record
            for record in collected.records
            if str(record.status).upper() in _FAILURE_STATUSES
        ]
        # Sorted on the ISO string, not the datetime: production records may
        # carry naive or timezone-aware timestamps depending on when they were
        # written, and comparing the two raises.
        failures.sort(key=_timestamp_key, reverse=True)
        return [record.to_dict() for record in failures[:limit]]

    def get_candidate_samples(
        self, label_state: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Pooled candidates, optionally filtered by label state."""
        pool = self._pool()
        samples = (
            pool.by_label_state(label_state) if label_state else pool.load()
        )
        return [sample.to_dict() for sample in samples[:limit]]

    def get_dataset_statistics(self) -> dict[str, Any]:
        """Pool composition and the dataset versions built from it."""
        store = self._datasets()
        versions = store.history()
        return {
            "pool": self._pool().statistics(),
            "versions": [
                {
                    "version": version.version,
                    "parent_version": version.parent_version,
                    "created_at": version.created_at,
                    "sample_count": len(version.sample_ids),
                    "added": len(version.added_samples),
                    "removed": len(version.removed_samples),
                    "content_id": version.content_id,
                }
                for version in versions
            ],
            "latest": versions[-1].version if versions else "",
        }

    def get_training_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Past cycles, newest first, as recorded in their reports."""
        import json

        root = self.paths.cycles_root
        if not root.is_dir():
            return []
        history: list[dict[str, Any]] = []
        for directory in sorted(root.iterdir(), reverse=True):
            report = directory / "report.json"
            if not report.is_file():
                continue
            try:
                payload = json.loads(report.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                LOGGER.warning("Skipping unreadable cycle report %s: %s", report, exc)
                continue
            history.append(
                {
                    "cycle_id": payload.get("cycle_id", directory.name),
                    "dataset_version": payload.get("dataset_version", ""),
                    "challenger": payload.get("challenger", ""),
                    "decision": (payload.get("decision") or {}).get("decision", ""),
                    "deployed": payload.get("deployed", False),
                    "report_path": str(directory / "report.md"),
                }
            )
            if len(history) >= limit:
                break
        return history

    def get_model_registry(self) -> dict[str, Any]:
        """Champion plus every candidate, in one view.

        The champion entry is read from production and is not editable
        through this interface --- there is deliberately no method that can
        change which model the line runs.
        """
        champion = read_champion(
            self.paths.production_model_dir(self.product, self.area)
        )
        return {
            "production": champion.to_dict() if champion else None,
            "candidates": [
                model.to_dict() for model in self._registry().list_models()
            ],
        }

    # -- writes -----------------------------------------------------------------

    def create_dataset_version(self, description: str = "") -> dict[str, Any]:
        """Freeze the currently verified candidates into a new version."""
        self.config.require_enabled()
        usable = verified_samples(self._pool())
        if not usable:
            raise AutoTrainError(
                "No candidates carry a human-verified label. Export a labelling "
                "request and import the annotated result first."
            )
        version = self._datasets().create(
            [
                LabelledSample(
                    sample_id=sample.sample_id,
                    image_path=Path(sample.image_path),
                    label_path=Path(sample.label_path),
                    origin=sample.selector,
                )
                for sample in usable
            ],
            source="autotrain-pool",
            label_source="human-verified",
            class_schema=resolve_class_schema(
                [
                    station_class_schema(
                        self.paths.production_model_dir(self.product, self.area)
                    ),
                    schema_from_station_config(
                        self.paths.production_model_dir(self.product, self.area)
                    ),
                ],
                context=f"{self.product}/{self.area}",
            ),
            description=description,
        )
        return version.to_dict()

    def train_candidate(
        self,
        *,
        cycle_id: str | None = None,
        runner: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Train one challenger from the latest dataset version."""
        self.config.require_enabled()
        cycle = TrainingCycle(
            self.config, self.paths, cycle_id=cycle_id, runner=runner
        )
        latest = self._datasets().latest()
        if latest is None:
            raise AutoTrainError("No dataset version exists yet.")
        cycle.state.data["dataset_version"] = latest.version
        cycle.state.data["dataset_content_id"] = latest.content_id
        cycle.state.step("dataset")["status"] = "COMPLETED"
        model_version = cycle.train()
        return {"cycle_id": cycle.cycle_id, "model_version": model_version}

    def evaluate_candidate(
        self,
        *,
        champion_weights: str,
        challenger_weights: str,
        data_yaml: str,
        validator: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Measure a champion/challenger pair on one split."""
        self.config.require_enabled()
        golden_status = golden_module.resolve(
            self.config.golden.dataset_path, self.config.golden.manifest_sha256
        )
        report = _evaluate_candidate(
            champion_weights=champion_weights,
            challenger_weights=challenger_weights,
            data_yaml=data_yaml,
            golden_status=golden_status,
            imgsz=self.config.training.imgsz,
            device=self.config.training.device,
            batch=self.config.training.batch,
            validator=validator,
        )
        return report.to_dict()

    def compare_models(
        self,
        champion_metrics: Mapping[str, Any],
        challenger_metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Compare two already-measured models without re-running inference."""
        return compare(
            ModelMetrics.from_dict(champion_metrics),
            ModelMetrics.from_dict(challenger_metrics),
        ).to_dict()

    def run_training_cycle(
        self,
        *,
        cycle_id: str | None = None,
        runner: Callable[..., Any] | None = None,
        validator: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Run a full cycle and return its report payload."""
        self.config.require_enabled()
        result = run_training_cycle(
            self.config,
            self.paths,
            cycle_id=cycle_id,
            runner=runner,
            validator=validator,
        )
        return {
            "cycle_id": result.cycle_id,
            "status": result.status,
            "decision": result.decision.to_dict() if result.decision else None,
            "deployed": result.deployed,
            "blocked_reason": result.blocked_reason,
            "report_path": str(result.report_path) if result.report_path else "",
            "report_text": result.report_text,
        }

    # -- internals ---------------------------------------------------------------

    def _pool(self) -> CandidatePool:
        return CandidatePool(self.paths.pool_dir(self.product, self.area))

    def _datasets(self) -> DatasetVersionStore:
        return DatasetVersionStore(
            self.paths.dataset_station_root(self.product, self.area),
            product=self.product,
            area=self.area,
        )

    def _registry(self) -> CandidateRegistry:
        return CandidateRegistry(
            self.paths.candidates_root / self.product / self.area,
            product=self.product,
            area=self.area,
        )


def _timestamp_key(record: Any) -> str:
    timestamp = getattr(record, "timestamp", None)
    return timestamp.isoformat() if timestamp is not None else ""


#: Names an agent may call. Anything not listed is internal and may change.
TOOL_SURFACE: Sequence[str] = (
    "get_model_health",
    "get_recent_failures",
    "get_candidate_samples",
    "get_dataset_statistics",
    "get_training_history",
    "get_model_registry",
    "create_dataset_version",
    "train_candidate",
    "evaluate_candidate",
    "compare_models",
    "run_training_cycle",
)
