"""The training cycle.

``run_training_cycle`` chains the steps: collect production records, select
candidates, build a dataset version from *verified* labels, train a
challenger, evaluate it against the champion, apply the promotion policy,
and write a report.

Three properties shape the design:

* **Every step is separately runnable, retryable and resumable.** Each writes
  its outcome into ``cycle.json`` as it goes, so a cycle that died at
  training can be resumed at training rather than re-collecting and
  re-selecting.
* **Failure is contained.** A step that raises is recorded as ``FAILED`` and
  the cycle stops there with a report. Nothing in production is touched at
  any point, so a failed cycle cannot affect inference --- production does
  not read anything this writes.
* **Nothing is deployed.** The strongest outcome is a recommendation.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from picture_tool.autotrain import AutoTrainError
from picture_tool.autotrain import golden as golden_module
from picture_tool.autotrain import promotion as promotion_module
from picture_tool.autotrain import reports
from picture_tool.autotrain.candidate_pool import CandidatePool
from picture_tool.autotrain.collector import collect_production_records
from picture_tool.autotrain.config import AutoTrainConfig
from picture_tool.autotrain.dataset_versions import (
    DatasetVersionStore,
    LabelledSample,
)
from picture_tool.autotrain.evaluator import EvaluationReport, evaluate_candidate
from picture_tool.autotrain.labeling import verified_samples
from picture_tool.autotrain.paths import AutoTrainPaths
from picture_tool.autotrain.registry import (
    EVALUATING,
    FAILED,
    PROMOTION_CANDIDATE,
    REJECTED,
    TRAINED,
    CandidateRegistry,
    read_champion,
)
from picture_tool.autotrain.selectors import run_selectors
from picture_tool.autotrain.trainer import train_candidate

LOGGER = logging.getLogger(__name__)

CYCLE_STATE_FILENAME = "cycle.json"
CYCLE_SCHEMA_VERSION = 1

PENDING = "PENDING"
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
STEP_FAILED = "FAILED"
BLOCKED = "BLOCKED"

STEP_COLLECT = "collect"
STEP_SELECT = "select"
STEP_DATASET = "dataset"
STEP_TRAIN = "train"
STEP_EVALUATE = "evaluate"
STEP_DECIDE = "decide"
STEP_REPORT = "report"

CYCLE_STEPS = (
    STEP_COLLECT,
    STEP_SELECT,
    STEP_DATASET,
    STEP_TRAIN,
    STEP_EVALUATE,
    STEP_DECIDE,
    STEP_REPORT,
)


class CycleError(AutoTrainError):
    """Raised when a cycle cannot be started or resumed."""


@dataclass
class CycleState:
    """Per-step progress, persisted after every step."""

    cycle_id: str
    product: str
    area: str
    created_at: str
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    status: str = RUNNING
    schema_version: int = CYCLE_SCHEMA_VERSION

    def step(self, name: str) -> dict[str, Any]:
        return self.steps.setdefault(name, {"status": PENDING})

    def is_done(self, name: str) -> bool:
        return self.step(name).get("status") == COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cycle_id": self.cycle_id,
            "product": self.product,
            "area": self.area,
            "created_at": self.created_at,
            "status": self.status,
            "steps": self.steps,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CycleState":
        return cls(
            cycle_id=str(payload.get("cycle_id", "")),
            product=str(payload.get("product", "")),
            area=str(payload.get("area", "")),
            created_at=str(payload.get("created_at", "")),
            steps=dict(payload.get("steps") or {}),
            data=dict(payload.get("data") or {}),
            status=str(payload.get("status", RUNNING)),
            schema_version=int(payload.get("schema_version", 0)),
        )


@dataclass(frozen=True)
class CycleResult:
    """What a cycle produced, whether or not it got all the way through."""

    cycle_id: str
    status: str
    state: CycleState
    report_text: str
    report_path: Path | None = None
    decision: promotion_module.PromotionDecision | None = None
    evaluation: EvaluationReport | None = None
    blocked_reason: str = ""

    @property
    def deployed(self) -> bool:
        """Always false. Kept explicit so callers can assert on it."""
        return False


class TrainingCycle:
    """One cycle, with each step callable on its own."""

    def __init__(
        self,
        config: AutoTrainConfig,
        paths: AutoTrainPaths,
        *,
        cycle_id: str | None = None,
        logger: logging.Logger | None = None,
        runner: Callable[..., Any] | None = None,
        validator: Callable[..., Any] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        config.require_enabled()
        self.config = config
        self.paths = paths
        self.logger = logger or LOGGER
        self.runner = runner
        self.validator = validator
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.product = config.station.product
        self.area = config.station.area
        self.cycle_id = cycle_id or self._new_cycle_id()
        self.directory = paths.cycle_dir(self.cycle_id)
        self.state = self._load_state()

    # -- state ------------------------------------------------------------------

    def _new_cycle_id(self) -> str:
        return self._now().strftime("cycle_%Y%m%dT%H%M%SZ")

    def _load_state(self) -> CycleState:
        path = self.directory / CYCLE_STATE_FILENAME
        if not path.is_file():
            return CycleState(
                cycle_id=self.cycle_id,
                product=self.product,
                area=self.area,
                created_at=self._now().isoformat(),
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CycleError(f"Unable to resume {self.cycle_id}: {exc}") from exc
        return CycleState.from_dict(payload)

    def save(self) -> None:
        self.paths.ensure_dir(self.directory)
        path = self.directory / CYCLE_STATE_FILENAME
        handle, temporary = tempfile.mkstemp(dir=str(self.directory), prefix=".cycle-")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(
                    self.state.to_dict(),
                    stream,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                try:
                    os.remove(temporary)
                except OSError:  # pragma: no cover - best effort cleanup
                    self.logger.debug("Could not remove %s", temporary)

    def _begin(self, name: str) -> None:
        entry = self.state.step(name)
        entry["status"] = RUNNING
        entry["started_at"] = self._now().isoformat()
        self.save()

    def _finish(self, name: str, **result: Any) -> None:
        entry = self.state.step(name)
        entry["status"] = COMPLETED
        entry["finished_at"] = self._now().isoformat()
        entry["result"] = result
        self.save()

    def _fail(self, name: str, detail: str) -> None:
        entry = self.state.step(name)
        entry["status"] = STEP_FAILED
        entry["finished_at"] = self._now().isoformat()
        entry["detail"] = detail
        self.state.status = STEP_FAILED
        self.save()

    def _block(self, name: str, detail: str) -> None:
        """A step that cannot proceed for a legitimate reason, not a failure."""
        entry = self.state.step(name)
        entry["status"] = BLOCKED
        entry["finished_at"] = self._now().isoformat()
        entry["detail"] = detail
        self.state.status = BLOCKED
        self.save()

    # -- steps ------------------------------------------------------------------

    @property
    def pool(self) -> CandidatePool:
        return CandidatePool(self.paths.pool_dir(self.product, self.area))

    @property
    def dataset_store(self) -> DatasetVersionStore:
        return DatasetVersionStore(
            self.paths.dataset_station_root(self.product, self.area),
            product=self.product,
            area=self.area,
        )

    @property
    def registry(self) -> CandidateRegistry:
        return CandidateRegistry(
            self.paths.candidates_root / self.product / self.area,
            product=self.product,
            area=self.area,
        )

    def collect(self) -> dict[str, Any]:
        """Read recent production inspections. Read-only."""
        self._begin(STEP_COLLECT)
        result = collect_production_records(self.paths, self.config)
        self.state.data["records"] = [
            record.to_dict() for record in result.records
        ]
        summary = result.summary()
        self._finish(STEP_COLLECT, **summary)
        self.logger.info("Collected %s production record(s)", summary["records"])
        return summary

    def select(self) -> dict[str, Any]:
        """Run the enabled selectors and pool what they pick."""
        self._begin(STEP_SELECT)
        records = _rehydrate_records(self.state.data.get("records", []))
        candidates = run_selectors(records, self.config.selectors)
        added = self.pool.add(candidates)
        summary = {**added.summary(), "selected": len(candidates)}
        self._finish(STEP_SELECT, **summary)
        self.logger.info(
            "Selected %s candidate(s); %s new in the pool",
            summary["selected"],
            summary["added"],
        )
        return summary

    def build_dataset(self) -> str:
        """Create an immutable dataset version from verified labels only.

        Blocks rather than fails when nothing is labelled yet: waiting for
        annotation is the expected state of a healthy cycle, not an error.
        """
        self._begin(STEP_DATASET)
        usable = verified_samples(self.pool)
        if not usable:
            self._block(
                STEP_DATASET,
                "No candidates carry a human-verified label yet. Export a "
                "labelling request, annotate it, and import the result before "
                "training.",
            )
            return ""
        samples = [
            LabelledSample(
                sample_id=sample.sample_id,
                image_path=Path(sample.image_path),
                label_path=Path(sample.label_path),
                origin=sample.selector,
            )
            for sample in usable
        ]
        version = self.dataset_store.create(
            samples,
            source="autotrain-pool",
            label_source="human-verified",
            description=f"cycle {self.cycle_id}",
            split_policy={"splitter": "picture_tool.split.dataset_splitter"},
        )
        self.state.data["dataset_version"] = version.version
        self.state.data["dataset_content_id"] = version.content_id
        self._finish(
            STEP_DATASET,
            version=version.version,
            content_id=version.content_id,
            samples=len(version.sample_ids),
        )
        return version.version

    def train(self) -> str:
        """Train the challenger. Never runs a publishing task."""
        self._begin(STEP_TRAIN)
        version_name = self.state.data.get("dataset_version", "")
        if not version_name:
            self._block(STEP_TRAIN, "No dataset version was built.")
            return ""
        version = self.dataset_store.load(version_name)

        champion = read_champion(self.paths.production_model_dir(self.product, self.area))
        if champion is None:
            self._block(
                STEP_TRAIN,
                "No champion is deployed for this station, so there is no base "
                "model to continue from and nothing to compare against.",
            )
            return ""
        self.state.data["champion"] = champion.to_dict()

        base_model = champion.training_weight_path or champion.weights_path
        model_version = f"{self.product}_{self.area}_{self.cycle_id}_candidate"
        registry = self.registry
        registry.register(
            model_version,
            dataset_version=version.version,
            dataset_content_id=version.content_id,
            parent_model=champion.model_version,
            cycle_id=self.cycle_id,
            training_config={
                "epochs": self.config.training.epochs,
                "imgsz": self.config.training.imgsz,
                "batch": self.config.training.batch,
                "device": self.config.training.device,
                "base_model": base_model,
            },
        )
        try:
            result = train_candidate(
                dataset_version=version,
                base_config=_load_base_pipeline_config(self.paths, self.product),
                candidate_dir=registry.directory_for(model_version),
                work_dir=self.directory / "work",
                model_version=model_version,
                base_model=base_model,
                class_names=self._class_names(),
                epochs=self.config.training.epochs,
                imgsz=self.config.training.imgsz,
                batch=self.config.training.batch,
                device=self.config.training.device,
                runner=self.runner,
                logger=self.logger,
            )
        except Exception as exc:  # noqa: BLE001 - recorded, never propagated blindly
            registry.update_status(model_version, FAILED, notes=str(exc))
            self._fail(STEP_TRAIN, str(exc))
            raise
        registry.update_status(
            model_version,
            TRAINED,
            training_metrics=dict(result.metrics),
            weight_sha256=result.weight_sha256,
        )
        self.state.data["challenger"] = result.to_dict()
        self._finish(
            STEP_TRAIN, model_version=model_version, weights=str(result.weights_path)
        )
        return model_version

    def evaluate(self) -> EvaluationReport | None:
        """Measure champion and challenger on the same split."""
        self._begin(STEP_EVALUATE)
        challenger = self.state.data.get("challenger") or {}
        champion = self.state.data.get("champion") or {}
        if not challenger or not champion:
            self._block(STEP_EVALUATE, "No challenger was trained.")
            return None

        model_version = str(challenger.get("model_version", ""))
        registry = self.registry
        registry.update_status(model_version, EVALUATING)

        golden_status = golden_module.resolve(
            self.config.golden.dataset_path,
            self.config.golden.manifest_sha256,
            training_sample_ids=self._training_sample_ids(),
        )
        data_yaml = Path(self.directory) / "work" / "split" / "data.yaml"
        # The training weight is preferred: a station running ONNX still keeps
        # its paired .pt, and comparing a .pt challenger against an .onnx
        # champion would measure the export as much as the model.
        champion_weights = str(
            champion.get("training_weight_path") or champion.get("weights_path") or ""
        )
        try:
            report = evaluate_candidate(
                champion_weights=champion_weights,
                challenger_weights=str(challenger.get("weights_path", "")),
                data_yaml=data_yaml,
                golden_status=golden_status,
                confidence=0.4,
                imgsz=self.config.training.imgsz,
                device=self.config.training.device,
                batch=self.config.training.batch,
                validator=self.validator,
            )
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
            registry.update_status(model_version, FAILED, notes=str(exc))
            self._fail(STEP_EVALUATE, str(exc))
            raise
        self.state.data["evaluation"] = report.to_dict()
        self._finish(STEP_EVALUATE, status=report.status)
        return report

    def decide(self, report: EvaluationReport) -> promotion_module.PromotionDecision:
        """Apply the deterministic promotion policy."""
        self._begin(STEP_DECIDE)
        if not report.is_valid:
            decision = promotion_module.PromotionDecision(
                decision=promotion_module.REJECTED,
                reasons=(f"evaluation was {report.status}: {report.detail}",),
                passed_checks=(),
                golden_status=report.golden.status,
            )
        else:
            decision = promotion_module.decide(
                report.comparison, report.golden, self.config.promotion
            )
        model_version = str(
            (self.state.data.get("challenger") or {}).get("model_version", "")
        )
        if model_version:
            self.registry.update_status(
                model_version,
                PROMOTION_CANDIDATE if decision.is_candidate else REJECTED,
                evaluation_metrics=report.comparison.challenger.to_dict(),
                promotion_decision=decision.to_dict(),
            )
        self.state.data["decision"] = decision.to_dict()
        self._finish(STEP_DECIDE, decision=decision.decision)
        return decision

    def write_report(
        self,
        *,
        evaluation: EvaluationReport | None,
        decision: promotion_module.PromotionDecision | None,
        notes: Sequence[str] = (),
    ) -> tuple[str, Path]:
        """Render and persist both report forms."""
        self._begin(STEP_REPORT)
        champion = self.state.data.get("champion") or {}
        challenger = self.state.data.get("challenger") or {}
        text = reports.render_text(
            cycle_id=self.cycle_id,
            champion=str(champion.get("model_version", "")),
            challenger=str(challenger.get("model_version", "")),
            dataset_version=str(self.state.data.get("dataset_version", "")),
            evaluation=evaluation,
            decision=decision,
            notes=notes,
        )
        payload = reports.build_payload(
            cycle_id=self.cycle_id,
            product=self.product,
            area=self.area,
            champion=champion or None,
            challenger=str(challenger.get("model_version", "")),
            dataset_version=str(self.state.data.get("dataset_version", "")),
            evaluation=evaluation,
            decision=decision,
            steps=self.state.steps,
            notes=notes,
        )
        text_path, _ = reports.write_reports(
            self.directory, text=text, payload=payload
        )
        self._finish(STEP_REPORT, report=str(text_path))
        return text, text_path

    # -- helpers ----------------------------------------------------------------

    def _class_names(self) -> tuple[str, ...]:
        """Class list from the deployed model, so the contract cannot drift."""
        records = self.state.data.get("records") or []
        for record in records:
            names = record.get("class_names") or []
            if names:
                return tuple(str(name) for name in names)
        model_config = (
            self.paths.production_model_dir(self.product, self.area) / "config.yaml"
        )
        if model_config.is_file():
            try:
                payload = yaml.safe_load(model_config.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, yaml.YAMLError):
                payload = None
            if isinstance(payload, dict):
                names = payload.get("class_names") or payload.get("names")
                if isinstance(names, (list, tuple)) and names:
                    return tuple(str(name) for name in names)
        raise CycleError(
            "Could not determine the class list for this station. A challenger "
            "trained on a different class order would be incomparable to the "
            "champion."
        )

    def _training_sample_ids(self) -> tuple[str, ...]:
        version_name = self.state.data.get("dataset_version", "")
        if not version_name:
            return ()
        try:
            return self.dataset_store.load(version_name).sample_ids
        except AutoTrainError:
            return ()


# ---------------------------------------------------------------------------


def run_training_cycle(
    config: AutoTrainConfig,
    paths: AutoTrainPaths | None = None,
    *,
    cycle_id: str | None = None,
    logger: logging.Logger | None = None,
    runner: Callable[..., Any] | None = None,
    validator: Callable[..., Any] | None = None,
) -> CycleResult:
    """Run one full cycle, stopping cleanly at the first blocking condition.

    Never raises for an ordinary outcome --- no labelled data, no champion, a
    worse challenger --- because each of those is a result a person should
    read in the report. Genuine failures are recorded in ``cycle.json`` and
    re-raised after the report is written.
    """
    resolved_paths = paths or AutoTrainPaths.discover()
    cycle = TrainingCycle(
        config,
        resolved_paths,
        cycle_id=cycle_id,
        logger=logger,
        runner=runner,
        validator=validator,
    )
    log = cycle.logger
    notes: list[str] = []

    try:
        if not cycle.state.is_done(STEP_COLLECT):
            cycle.collect()
        if not cycle.state.is_done(STEP_SELECT):
            cycle.select()
        if not cycle.state.is_done(STEP_DATASET):
            if not cycle.build_dataset():
                return _blocked(cycle, STEP_DATASET, notes)
        if not cycle.state.is_done(STEP_TRAIN):
            if not cycle.train():
                return _blocked(cycle, STEP_TRAIN, notes)
        report = cycle.evaluate()
        if report is None:
            return _blocked(cycle, STEP_EVALUATE, notes)
        decision = cycle.decide(report)
    except Exception as exc:  # noqa: BLE001 - report first, then propagate
        log.exception("Training cycle %s failed", cycle.cycle_id)
        notes.append(f"cycle failed: {exc}")
        text, path = cycle.write_report(evaluation=None, decision=None, notes=notes)
        cycle.state.status = STEP_FAILED
        cycle.save()
        raise CycleError(
            f"Training cycle {cycle.cycle_id} failed: {exc}. Report: {path}"
        ) from exc

    text, path = cycle.write_report(
        evaluation=report, decision=decision, notes=notes
    )
    cycle.state.status = COMPLETED
    cycle.save()
    return CycleResult(
        cycle_id=cycle.cycle_id,
        status=COMPLETED,
        state=cycle.state,
        report_text=text,
        report_path=path,
        decision=decision,
        evaluation=report,
    )


def _blocked(cycle: TrainingCycle, step: str, notes: list[str]) -> CycleResult:
    detail = str(cycle.state.step(step).get("detail", ""))
    notes.append(detail)
    text, path = cycle.write_report(evaluation=None, decision=None, notes=notes)
    cycle.state.status = BLOCKED
    cycle.save()
    return CycleResult(
        cycle_id=cycle.cycle_id,
        status=BLOCKED,
        state=cycle.state,
        report_text=text,
        report_path=path,
        blocked_reason=detail,
    )


def _rehydrate_records(payloads: Sequence[Mapping[str, Any]]):
    """Rebuild collector records from persisted cycle state.

    Selection reads only a handful of fields, so a light object is enough and
    keeps a resumed cycle from having to re-read the production tree.
    """
    from picture_tool.autotrain.collector import DetectionRecord, ProductionRecord
    from picture_tool.autotrain.image_quality import ImageQuality

    records = []
    for payload in payloads:
        quality_payload = payload.get("image_quality")
        quality = (
            ImageQuality(**quality_payload)
            if isinstance(quality_payload, dict)
            else None
        )
        timestamp = payload.get("timestamp") or ""
        records.append(
            ProductionRecord(
                inspection_id=str(payload.get("inspection_id", "")),
                timestamp=_parse_iso(timestamp),
                status=str(payload.get("status", "")),
                detector=str(payload.get("detector", "")),
                product=str(payload.get("product", "")),
                area=str(payload.get("area", "")),
                model_version=str(payload.get("model_version", "")),
                model_weights=str(payload.get("model_weights", "")),
                conf_threshold=payload.get("conf_threshold"),
                class_names=tuple(payload.get("class_names") or ()),
                detections=tuple(
                    DetectionRecord(
                        class_name=str(item.get("class_name", "")),
                        confidence=item.get("confidence"),
                        bbox=tuple(item["bbox"]) if item.get("bbox") else None,
                    )
                    for item in payload.get("detections") or []
                ),
                original_path=str(payload.get("original_path", "")),
                preprocessed_path=str(payload.get("preprocessed_path", "")),
                annotated_path=str(payload.get("annotated_path", "")),
                camera_parameters=dict(payload.get("camera_parameters") or {}),
                equipment=dict(payload.get("equipment") or {}),
                fail_reasons=tuple(payload.get("fail_reasons") or ()),
                config_hash=str(payload.get("config_hash", "")),
                snapshot_path=str(payload.get("snapshot_path", "")),
                schema_version=int(payload.get("schema_version", 0)),
                image_quality=quality,
                review_outcome=str(payload.get("review_outcome", "")),
                review_label=str(payload.get("review_label", "")),
                failure_category=str(payload.get("failure_category", "")),
                action_route=str(payload.get("action_route", "")),
            )
        )
    return records


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _load_base_pipeline_config(paths: AutoTrainPaths, product: str) -> dict[str, Any]:
    """Load the station's pipeline config, falling back to the default.

    Reuses the existing loader so packaged defaults and validation behave
    exactly as they do for the operator flow.
    """
    from picture_tool.config_loader import load_config

    configs_dir = paths.workspace.training_project / "configs"
    for name in (f"{product.lower()}_pipeline.yaml", "default_pipeline.yaml"):
        candidate = configs_dir / name
        if candidate.is_file():
            return load_config(str(candidate))
    return load_config("config.yaml")
