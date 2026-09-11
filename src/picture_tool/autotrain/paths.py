"""Filesystem layout for the autonomous training subsystem.

Every writable path lives inside the training project. The production
inference project is reachable read-only, and :func:`assert_not_production`
is the guard that keeps it that way.

These paths are derived from the existing :class:`WorkspacePaths` contract
rather than added to ``workspace.yaml``: that manifest is validated by
``scripts/validate_workspace.py`` against *both* repositories' independent
implementations, so a new key there would have to be implemented twice to
keep the contract test green.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from picture_tool.autotrain import AutoTrainError
from picture_tool.workspace_paths import WorkspacePaths

#: Container for everything this subsystem owns, under the training data root.
AUTOTRAIN_DIRNAME = ".autotrain"

#: Candidate weights live beside the training project's other models, never
#: under the inference project's ``models/`` tree.
CANDIDATES_DIRNAME = "candidates"


class ProductionWriteAttemptError(AutoTrainError):
    """Raised when a path that would be written resolves into production.

    Treated as a programming error rather than a runtime condition: no
    configuration should be able to produce one, so it fails loudly.
    """


@dataclass(frozen=True)
class AutoTrainPaths:
    """Resolved locations for pools, datasets, cycles and candidate models."""

    workspace: WorkspacePaths
    root: Path
    pool_root: Path
    labeling_root: Path
    datasets_root: Path
    cycles_root: Path
    candidates_root: Path

    @classmethod
    def from_workspace(cls, workspace: WorkspacePaths) -> "AutoTrainPaths":
        """Derive the layout from an already-resolved workspace contract."""
        root = (workspace.training_data / AUTOTRAIN_DIRNAME).resolve()
        return cls(
            workspace=workspace,
            root=root,
            pool_root=root / "pool",
            labeling_root=root / "labeling",
            datasets_root=root / "datasets",
            cycles_root=root / "cycles",
            candidates_root=(
                workspace.training_project / "models" / CANDIDATES_DIRNAME
            ).resolve(),
        )

    @classmethod
    def discover(cls, start: Path | None = None) -> "AutoTrainPaths":
        """Discover the workspace, then derive the layout from it."""
        return cls.from_workspace(WorkspacePaths.discover(start))

    # -- station-scoped locations -------------------------------------------------

    def pool_dir(self, product: str, area: str) -> Path:
        """Candidate pool for one station."""
        return self.pool_root / product / area

    def dataset_dir(self, product: str, area: str, version: str) -> Path:
        """One immutable dataset version for one station."""
        return self.datasets_root / product / area / version

    def dataset_station_root(self, product: str, area: str) -> Path:
        """Container holding every dataset version for one station."""
        return self.datasets_root / product / area

    def cycle_dir(self, cycle_id: str) -> Path:
        """Working directory and report location for one training cycle."""
        return self.cycles_root / cycle_id

    def candidate_dir(self, product: str, area: str, version: str) -> Path:
        """Artifact directory for one challenger model."""
        return self.candidates_root / product / area / version

    # -- production, read-only ----------------------------------------------------

    def production_model_dir(
        self, product: str, area: str, model_type: str = "yolo"
    ) -> Path:
        """Deployed model directory for one station. Read-only by contract."""
        return self.workspace.inference_models / product / area / model_type

    def production_results_root(self) -> Path:
        """Root of the production inspection results tree. Read-only."""
        return self.workspace.inference_results

    def production_database(self) -> Path:
        """Production inspection history database. Opened read-only."""
        return self.workspace.inference_results / "inspection_records.sqlite3"

    # -- guard --------------------------------------------------------------------

    def assert_not_production(self, path: Path) -> Path:
        """Reject any write target that resolves inside the inference project.

        The check is on the resolved path, so a candidate directory assembled
        from configured strings cannot reach production through ``..`` or a
        symlink.
        """
        resolved = Path(path).expanduser().resolve()
        for guarded in (
            self.workspace.inference_project,
            self.workspace.inference_models,
            self.workspace.inference_results,
            self.workspace.station_data,
        ):
            if resolved == guarded or resolved.is_relative_to(guarded):
                raise ProductionWriteAttemptError(
                    f"Refusing to write inside the production inference project: "
                    f"{resolved} is under {guarded}"
                )
        return resolved

    def ensure_dir(self, path: Path) -> Path:
        """Create a directory after proving it is not a production path."""
        safe = self.assert_not_production(path)
        safe.mkdir(parents=True, exist_ok=True)
        return safe
