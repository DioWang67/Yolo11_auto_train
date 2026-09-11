"""Registry of challenger models.

This registry owns *candidates* only. The deployed champion is not recorded
here --- it is read from the production project's own
``deployment_manifest.yaml`` and station ``config.yaml``, which are already
the authority on what the line is running. A second copy would be a second
truth, and the two would disagree the first time someone activated a model
through the existing release flow without telling this subsystem.

So ``PRODUCTION`` is a status this registry can *report* but never *assign*.
Nothing here can change which model production loads.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from picture_tool.autotrain import AutoTrainError
from picture_tool.autotrain.class_schema import (
    ClassSchema,
    read_checkpoint_class_schema,
)

LOGGER = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1

#: Where ``deploy`` puts a station's weight files, and the segment its
#: deployment manifest leaves out when naming them.
STATION_WEIGHTS_DIRNAME = "weights"
#: The directory every station lives under in the inference project. Used to
#: find the root that a station config's relative paths are written against.
MODELS_DIRNAME = "models"

#: Read from production, never assigned here.
PRODUCTION = "PRODUCTION"
#: A training run is in flight.
TRAINING = "TRAINING"
#: Training finished and produced weights.
TRAINED = "TRAINED"
#: Evaluation against the champion is in flight.
EVALUATING = "EVALUATING"
#: Evaluated and found not better, per the promotion policy.
REJECTED = "REJECTED"
#: Evaluated and worth a human's consideration. Still not deployed.
PROMOTION_CANDIDATE = "PROMOTION_CANDIDATE"
#: Superseded or deliberately retired.
ARCHIVED = "ARCHIVED"
#: Training or evaluation could not complete. Distinct from REJECTED, which
#: is a measured verdict --- conflating "worse" with "crashed" would hide
#: infrastructure failures inside what looks like a quality result.
FAILED = "FAILED"

CANDIDATE_STATUSES = (
    TRAINING,
    TRAINED,
    EVALUATING,
    REJECTED,
    PROMOTION_CANDIDATE,
    ARCHIVED,
    FAILED,
)

_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    TRAINING: frozenset({TRAINED, FAILED, ARCHIVED}),
    TRAINED: frozenset({EVALUATING, FAILED, ARCHIVED}),
    EVALUATING: frozenset({PROMOTION_CANDIDATE, REJECTED, FAILED, ARCHIVED}),
    PROMOTION_CANDIDATE: frozenset({ARCHIVED, REJECTED}),
    REJECTED: frozenset({ARCHIVED}),
    FAILED: frozenset({ARCHIVED}),
    ARCHIVED: frozenset(),
}


class ModelRegistryError(AutoTrainError):
    """Raised on an unknown model or an illegal status transition."""


@dataclass(frozen=True)
class CandidateModel:
    """One challenger and everything known about where it came from."""

    model_version: str
    product: str
    area: str
    status: str
    parent_model: str = ""
    dataset_version: str = ""
    dataset_content_id: str = ""
    cycle_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    artifact_path: str = ""
    weight_sha256: str = ""
    training_config: Mapping[str, Any] = field(default_factory=dict)
    training_metrics: Mapping[str, Any] = field(default_factory=dict)
    evaluation_metrics: Mapping[str, Any] = field(default_factory=dict)
    promotion_decision: Mapping[str, Any] = field(default_factory=dict)
    #: The class contract this candidate was trained against. Recorded so a
    #: later comparison can refuse two models whose class ids mean different
    #: things before it reads a single metric off them.
    class_schema: ClassSchema | None = None
    notes: str = ""
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "product": self.product,
            "area": self.area,
            "status": self.status,
            "parent_model": self.parent_model,
            "dataset_version": self.dataset_version,
            "dataset_content_id": self.dataset_content_id,
            "cycle_id": self.cycle_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "artifact_path": self.artifact_path,
            "weight_sha256": self.weight_sha256,
            "training_config": dict(self.training_config),
            "training_metrics": dict(self.training_metrics),
            "evaluation_metrics": dict(self.evaluation_metrics),
            "promotion_decision": dict(self.promotion_decision),
            "class_schema": (
                self.class_schema.to_dict() if self.class_schema else None
            ),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateModel":
        try:
            return cls(
                model_version=str(payload["model_version"]),
                product=str(payload.get("product", "")),
                area=str(payload.get("area", "")),
                status=str(payload.get("status", TRAINING)),
                parent_model=str(payload.get("parent_model", "")),
                dataset_version=str(payload.get("dataset_version", "")),
                dataset_content_id=str(payload.get("dataset_content_id", "")),
                cycle_id=str(payload.get("cycle_id", "")),
                created_at=str(payload.get("created_at", "")),
                updated_at=str(payload.get("updated_at", "")),
                artifact_path=str(payload.get("artifact_path", "")),
                weight_sha256=str(payload.get("weight_sha256", "")),
                training_config=dict(payload.get("training_config") or {}),
                training_metrics=dict(payload.get("training_metrics") or {}),
                evaluation_metrics=dict(payload.get("evaluation_metrics") or {}),
                promotion_decision=dict(payload.get("promotion_decision") or {}),
                class_schema=(
                    ClassSchema.from_dict(payload["class_schema"])
                    if payload.get("class_schema")
                    else None
                ),
                notes=str(payload.get("notes", "")),
                schema_version=int(payload.get("schema_version", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelRegistryError(f"Malformed candidate manifest: {exc}") from exc


@dataclass(frozen=True)
class ChampionModel:
    """The model production is actually running, as read from production."""

    model_version: str
    weights_path: str
    weight_sha256: str
    training_weight_path: str
    dataset_id: str
    evaluation_metrics: Mapping[str, Any]
    source: str
    #: Filled in by :func:`read_champion_class_schema`, which has to open the
    #: checkpoint; left unset by the cheap metadata read so that reporting a
    #: champion does not pay for loading torch.
    class_schema: ClassSchema | None = None
    status: str = PRODUCTION

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "weights_path": self.weights_path,
            "weight_sha256": self.weight_sha256,
            "training_weight_path": self.training_weight_path,
            "dataset_id": self.dataset_id,
            "evaluation_metrics": dict(self.evaluation_metrics),
            "class_schema": (
                self.class_schema.to_dict() if self.class_schema else None
            ),
            "status": self.status,
            "source": self.source,
        }


class CandidateRegistry:
    """Stores candidate manifests under ``models/candidates/<product>/<area>``."""

    def __init__(self, root: str | Path, *, product: str = "", area: str = "") -> None:
        self.root = Path(root).expanduser().resolve()
        self.product = product
        self.area = area

    def directory_for(self, model_version: str) -> Path:
        return self.root / model_version

    # -- reading ----------------------------------------------------------------

    def list_models(self) -> tuple[CandidateModel, ...]:
        """Every readable candidate, newest first."""
        if not self.root.is_dir():
            return ()
        models: list[CandidateModel] = []
        for entry in sorted(self.root.iterdir()):
            manifest = entry / MANIFEST_FILENAME
            if not manifest.is_file():
                continue
            try:
                models.append(self._read(manifest))
            except ModelRegistryError as exc:
                LOGGER.warning("Skipping unreadable candidate %s: %s", entry.name, exc)
        models.sort(key=lambda model: (model.created_at, model.model_version), reverse=True)
        return tuple(models)

    def get(self, model_version: str) -> CandidateModel:
        manifest = self.directory_for(model_version) / MANIFEST_FILENAME
        if not manifest.is_file():
            raise ModelRegistryError(f"Unknown candidate model: {model_version}")
        return self._read(manifest)

    def by_status(self, status: str) -> tuple[CandidateModel, ...]:
        return tuple(model for model in self.list_models() if model.status == status)

    # -- writing ----------------------------------------------------------------

    def register(
        self,
        model_version: str,
        *,
        dataset_version: str = "",
        dataset_content_id: str = "",
        parent_model: str = "",
        cycle_id: str = "",
        training_config: Mapping[str, Any] | None = None,
        class_schema: ClassSchema | None = None,
        notes: str = "",
    ) -> CandidateModel:
        """Create a candidate in ``TRAINING``.

        Registered before training starts so a crashed run still leaves a
        record explaining what was attempted.
        """
        directory = self.directory_for(model_version)
        if (directory / MANIFEST_FILENAME).is_file():
            raise ModelRegistryError(
                f"Candidate {model_version} is already registered."
            )
        now = _utc_now()
        model = CandidateModel(
            model_version=model_version,
            product=self.product,
            area=self.area,
            status=TRAINING,
            parent_model=parent_model,
            dataset_version=dataset_version,
            dataset_content_id=dataset_content_id,
            cycle_id=cycle_id,
            created_at=now,
            updated_at=now,
            artifact_path=str(directory),
            training_config=dict(training_config or {}),
            class_schema=class_schema,
            notes=notes,
        )
        directory.mkdir(parents=True, exist_ok=True)
        self._write(model)
        return model

    def update_status(
        self,
        model_version: str,
        status: str,
        *,
        notes: str = "",
        **updates: Any,
    ) -> CandidateModel:
        """Move a candidate to a new status, refusing illegal transitions.

        The state machine is enforced rather than advisory: a candidate that
        jumped from TRAINING straight to PROMOTION_CANDIDATE would be one
        nobody evaluated, and that must be impossible, not merely unlikely.
        """
        if status not in CANDIDATE_STATUSES:
            raise ModelRegistryError(
                f"{status!r} is not a candidate status. Valid: "
                + ", ".join(CANDIDATE_STATUSES)
            )
        if status == PRODUCTION:  # pragma: no cover - excluded by the check above
            raise ModelRegistryError(
                "PRODUCTION is read from the inference project and cannot be "
                "assigned here."
            )
        current = self.get(model_version)
        allowed = _ALLOWED_TRANSITIONS.get(current.status, frozenset())
        if status != current.status and status not in allowed:
            raise ModelRegistryError(
                f"Cannot move {model_version} from {current.status} to {status}. "
                f"Allowed from {current.status}: "
                + (", ".join(sorted(allowed)) or "nothing")
            )
        model = replace(
            current,
            status=status,
            updated_at=_utc_now(),
            notes=notes or current.notes,
            **updates,
        )
        self._write(model)
        return model

    # -- internals ---------------------------------------------------------------

    def _read(self, manifest: Path) -> CandidateModel:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelRegistryError(f"Unable to read {manifest}: {exc}") from exc
        return CandidateModel.from_dict(payload)

    def _write(self, model: CandidateModel) -> None:
        directory = self.directory_for(model.model_version)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MANIFEST_FILENAME
        handle, temporary = tempfile.mkstemp(dir=str(directory), prefix=".manifest-")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(
                    model.to_dict(), stream, ensure_ascii=False, indent=2, sort_keys=True
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except OSError as exc:
            raise ModelRegistryError(f"Unable to write {path}: {exc}") from exc
        finally:
            if os.path.exists(temporary):
                try:
                    os.remove(temporary)
                except OSError:  # pragma: no cover - best effort cleanup
                    LOGGER.debug("Could not remove %s", temporary)


# ---------------------------------------------------------------------------
# The champion, read from production


def read_champion(model_dir: str | Path) -> ChampionModel | None:
    """Read the deployed model from the production station directory.

    Read-only. Prefers ``deployment_manifest.yaml`` because it carries the
    checksums and the metrics the model was deployed on; falls back to the
    station ``config.yaml``'s ``weights`` key so a station deployed before
    manifests existed still reports a champion instead of nothing.
    """
    directory = Path(model_dir).expanduser()
    manifest_path = directory / "deployment_manifest.yaml"
    payload = _read_yaml(manifest_path)
    if payload:
        weights = str(
            payload.get("deployed_file")
            or payload.get("deployed_weight_file")
            or payload.get("weights")
            or ""
        )
        return ChampionModel(
            model_version=str(payload.get("deployed_version") or ""),
            weights_path=_resolve_station_artifact(directory, weights),
            weight_sha256=str(payload.get("weight_sha256") or ""),
            training_weight_path=_resolve_station_artifact(
                directory, str(payload.get("training_weight_file") or "")
            ),
            dataset_id=str(payload.get("dataset_id") or ""),
            evaluation_metrics=dict(payload.get("evaluation_metrics") or {}),
            source=str(manifest_path),
        )

    config = _read_yaml(directory / "config.yaml")
    if not config:
        return None
    weights = str(config.get("weights") or "")
    if not weights:
        return None
    return ChampionModel(
        model_version=str(config.get("model_version") or ""),
        weights_path=_resolve_station_config_weight(directory, weights),
        weight_sha256="",
        training_weight_path="",
        dataset_id="",
        evaluation_metrics={},
        source=str(directory / "config.yaml"),
    )


def station_class_schema(model_dir: str | Path) -> ClassSchema | None:
    """The deployed champion's class contract for a station, if there is one.

    The one-call form of "read the champion, then read its checkpoint", so
    callers that only need the contract do not each repeat the two steps and
    risk one of them drifting.
    """
    champion = read_champion(model_dir)
    if champion is None:
        return None
    return read_champion_class_schema(champion)


def read_champion_class_schema(champion: ChampionModel) -> ClassSchema | None:
    """Read the champion's own class contract from its training checkpoint.

    The checkpoint is the strongest statement of what the deployed model's
    class ids mean --- it is what inference itself reads at load time, and
    what every production record's ``class_names`` is derived from. Nothing
    else in production states it: the deployment manifest records checksums,
    metrics and paths but carries no class list at all, and the station
    ``config.yaml`` has only ``expected_items``, which is not one.

    Separate from :func:`read_champion` because this opens the weight file.
    Returns ``None`` when there is no readable ``.pt`` to open, leaving the
    caller to fall through to another source rather than fail here.
    """
    path = champion.training_weight_path or champion.weights_path
    if not path:
        return None
    return read_checkpoint_class_schema(path)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        LOGGER.warning("Unable to read %s: %s", path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_station_artifact(directory: Path, value: str) -> str:
    """Resolve a file named by the station's deployment manifest.

    ``deploy`` writes artifacts into the station's ``weights`` directory but
    records them in the manifest by bare filename, so that segment has to be
    put back here. A value that already carries a directory is taken as
    written, which is what a hand-built or older manifest looks like.
    """
    if not value:
        return ""
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    if len(candidate.parts) == 1:
        candidate = Path(STATION_WEIGHTS_DIRNAME) / candidate
    return str((directory / candidate).resolve())


def _resolve_station_config_weight(directory: Path, value: str) -> str:
    """Resolve the ``weights`` value from a station ``config.yaml``.

    This one does not follow the manifest's convention: it is written
    relative to the inference *project root*
    (``models/<product>/<area>/yolo/weights/...``), which is how the
    inference project itself resolves it. Joining it to the station
    directory instead would repeat the whole ``models/...`` prefix.
    """
    if not value:
        return ""
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    root = _inference_project_root(directory)
    return str(((root or directory) / candidate).resolve())


def _inference_project_root(directory: Path) -> Path | None:
    """The project root a station config's paths are written against.

    Station directories are ``<root>/models/<product>/<area>/yolo``, so the
    root is the parent of the ``models`` directory above this one. Found by
    name rather than by counting levels, so a deeper or shallower station
    layout does not silently resolve to the wrong place.
    """
    for parent in directory.resolve().parents:
        if parent.name == MODELS_DIRNAME:
            return parent.parent
    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
