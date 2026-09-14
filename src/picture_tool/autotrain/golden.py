"""The locked evaluation dataset.

This module provides the *structure* for a golden set: registration, content
locking, and contamination checks. It deliberately does not choose the data.
Deciding which images become the evaluation ground truth for a station is a
human judgement with long consequences --- a golden set assembled by a script
out of whatever was lying around is worse than none, because it looks like
evidence.

So the flow is: a person assembles a directory, runs ``register``, and puts
the resulting path and manifest hash in the configuration. Until then
evaluation reports ``NOT_CONFIGURED`` and promotion is refused. That is
fail-closed on purpose: "we could not check" must never read as "it passed".

Three properties are enforced once configured:

* **Locked** --- the manifest is hashed, and a changed manifest or a changed
  image is a failure, not a silent update. A golden set that drifts stops
  being a fixed yardstick.
* **Uncontaminated** --- no golden sample may appear in a training dataset
  version. Evaluating on data the model trained on measures memorisation.
* **Never written** --- nothing in this subsystem modifies the golden
  directory after registration.

  One caveat, confirmed against ultralytics 8.3.156 rather than assumed:
  validating the set makes *ultralytics* write its own label cache
  (``labels/<split>.cache``) beside the labels. It is derived data, it does
  not touch any registered image, and :func:`verify_content` hashes only the
  images, so the set still resolves ``OK``. But "the directory is byte-for-byte
  untouched" is not true of the real evaluation path, and a caller comparing
  whole-tree snapshots will see it. Placing a golden set on read-only storage
  is therefore safe but will make ultralytics warn on every run.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from picture_tool.autotrain import AutoTrainError
from picture_tool.autotrain.class_schema import ClassSchema

LOGGER = logging.getLogger(__name__)

MANIFEST_FILENAME = "golden_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

#: No golden set has been configured. Evaluation still runs; promotion cannot.
NOT_CONFIGURED = "NOT_CONFIGURED"
#: Configured, present, locked and uncontaminated.
OK = "OK"
#: Configured but the directory or manifest is not there.
MISSING = "MISSING"
#: Present but its content no longer matches what was registered.
MISMATCH = "MISMATCH"
#: Present and intact, but overlaps a training dataset version.
CONTAMINATED = "CONTAMINATED"
#: Present but unreadable or structurally invalid.
INVALID = "INVALID"

#: The only status that may support a promotion recommendation.
PASSING_STATUSES = (OK,)


class GoldenDatasetError(AutoTrainError):
    """Raised when a golden dataset cannot be registered."""


@dataclass(frozen=True)
class GoldenDataset:
    """A registered, content-locked evaluation set."""

    root: Path
    manifest_path: Path
    manifest_sha256: str
    sample_ids: tuple[str, ...]
    images: Mapping[str, str]
    registered_by: str
    registered_at: str
    description: str
    #: The class contract these labels were drawn against. A golden set whose
    #: class order differs from the model under test measures nothing: every
    #: per-class number would compare two different classes.
    class_schema: ClassSchema | None = None
    #: Per-sample ``representative`` / ``hard_case`` label, where the person
    #: who assembled the set said which is which. Absent means unsplit.
    groups: Mapping[str, str] = field(default_factory=dict)
    schema_version: int = MANIFEST_SCHEMA_VERSION

    @property
    def image_count(self) -> int:
        return len(self.sample_ids)

    @property
    def ungrouped_sample_ids(self) -> tuple[str, ...]:
        """Registered samples carrying no group label.

        Reported rather than hidden: per-group metrics cover only what is
        labelled, so a caller that does not know how many samples fall
        outside both groups cannot tell whether the two subsets describe the
        set or a corner of it.
        """
        return tuple(
            sample_id for sample_id in self.sample_ids if sample_id not in self.groups
        )

    def group_counts(self) -> dict[str, int]:
        """How many registered samples carry each group label."""
        counts: dict[str, int] = {}
        for sample_id in self.sample_ids:
            label = self.groups.get(sample_id)
            if label:
                counts[label] = counts.get(label, 0) + 1
        return counts

    def sample_ids_in_group(self, group: str) -> tuple[str, ...]:
        """Registered sample ids carrying ``group``, in manifest order."""
        return tuple(
            sample_id
            for sample_id in self.sample_ids
            if self.groups.get(sample_id) == group
        )


@dataclass(frozen=True)
class GoldenStatus:
    """Outcome of resolving and checking the configured golden set."""

    status: str
    dataset: GoldenDataset | None = None
    problems: tuple[str, ...] = ()
    detail: str = ""
    contaminated_samples: tuple[str, ...] = ()

    @property
    def is_passing(self) -> bool:
        return self.status in PASSING_STATUSES

    @property
    def is_configured(self) -> bool:
        return self.status != NOT_CONFIGURED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "detail": self.detail,
            "problems": list(self.problems),
            "contaminated_samples": list(self.contaminated_samples),
            "image_count": self.dataset.image_count if self.dataset else 0,
            "root": str(self.dataset.root) if self.dataset else "",
            "groups": self.dataset.group_counts() if self.dataset else {},
            "ungrouped": (
                len(self.dataset.ungrouped_sample_ids) if self.dataset else 0
            ),
        }


# ---------------------------------------------------------------------------
# Registration (run once, by a person, on data they chose)


def register(
    root: str | Path,
    *,
    registered_by: str,
    class_schema: ClassSchema,
    description: str = "",
    groups: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> GoldenDataset:
    """Lock an existing directory as a golden evaluation set.

    Does not select, copy or move any data --- it records what is already
    there and hashes it. ``registered_by`` is required so the record says who
    made this call: this is the approval marker, and a set with nobody's name
    on it is not approved.

    ``class_schema`` is required for the same reason it is required of a
    dataset version. These labels store class ids, and a yardstick whose ids
    mean something different from the model's is worse than no yardstick --- it
    reports confident numbers about the wrong classes.

    ``groups`` maps a sample id --- the sha256 of the image bytes --- to
    ``representative`` or ``hard_case``. Only assignments describing an image
    actually present are recorded; see :func:`_intersect_groups`.
    """
    directory = Path(root).expanduser().resolve()
    if not directory.is_dir():
        raise GoldenDatasetError(f"Golden dataset directory not found: {directory}")
    if not registered_by.strip():
        raise GoldenDatasetError(
            "registered_by is required: a golden set is evidence, and evidence "
            "needs an owner."
        )

    manifest_path = directory / MANIFEST_FILENAME
    if manifest_path.exists() and not overwrite:
        raise GoldenDatasetError(
            f"{manifest_path} already exists. A golden set is locked once "
            "registered; pass overwrite=True only if you intend to replace it, "
            "which invalidates every evaluation made against the old one."
        )

    images = _hash_images(directory)
    if not images:
        raise GoldenDatasetError(
            f"No images found under {directory}; nothing to register."
        )

    resolved_groups = _intersect_groups(groups, images)

    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "registered_by": registered_by.strip(),
        "description": description,
        "images": {sample_id: relative for sample_id, relative in sorted(images.items())},
        "image_count": len(images),
        "class_schema": class_schema.to_dict(),
        "groups": dict(sorted(resolved_groups.items())),
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    manifest_path.write_text(serialized, encoding="utf-8")

    return GoldenDataset(
        root=directory,
        manifest_path=manifest_path,
        manifest_sha256=_sha256_bytes(serialized.encode("utf-8")),
        sample_ids=tuple(sorted(images)),
        images=dict(images),
        registered_by=registered_by.strip(),
        registered_at=str(payload["registered_at"]),
        description=description,
        class_schema=class_schema,
        groups=resolved_groups,
    )


# ---------------------------------------------------------------------------
# Resolution and checking


def resolve(
    dataset_path: str,
    expected_manifest_sha256: str = "",
    *,
    training_sample_ids: Iterable[str] | None = None,
) -> GoldenStatus:
    """Resolve the configured golden set and check it end to end.

    Every failure mode returns a status rather than raising: the caller's job
    is to refuse promotion and say why, not to crash a training cycle that
    otherwise produced a usable challenger.
    """
    if not dataset_path or not dataset_path.strip():
        return GoldenStatus(
            status=NOT_CONFIGURED,
            detail=(
                "No golden evaluation dataset is configured, so a challenger "
                "cannot be recommended for promotion. Register one and set "
                "golden.dataset_path."
            ),
        )

    directory = Path(dataset_path).expanduser()
    manifest_path = directory / MANIFEST_FILENAME
    if not directory.is_dir():
        return GoldenStatus(
            status=MISSING, detail=f"Golden dataset directory not found: {directory}"
        )
    if not manifest_path.is_file():
        return GoldenStatus(
            status=MISSING,
            detail=(
                f"{manifest_path} not found. Register the directory before "
                "using it for evaluation."
            ),
        )

    try:
        raw = manifest_path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return GoldenStatus(status=INVALID, detail=f"Unreadable golden manifest: {exc}")
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), dict):
        return GoldenStatus(
            status=INVALID, detail=f"{manifest_path} is not a golden manifest."
        )

    manifest_sha256 = _sha256_bytes(raw.encode("utf-8"))
    dataset = GoldenDataset(
        root=directory.resolve(),
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        sample_ids=tuple(sorted(str(k) for k in payload["images"])),
        images={str(k): str(v) for k, v in payload["images"].items()},
        registered_by=str(payload.get("registered_by", "")),
        registered_at=str(payload.get("registered_at", "")),
        description=str(payload.get("description", "")),
        class_schema=(
            ClassSchema.from_dict(payload["class_schema"])
            if payload.get("class_schema")
            else None
        ),
        groups={str(k): str(v) for k, v in (payload.get("groups") or {}).items()},
        schema_version=int(payload.get("schema_version", 0)),
    )

    expected = expected_manifest_sha256.strip()
    if expected and expected != manifest_sha256:
        return GoldenStatus(
            status=MISMATCH,
            dataset=dataset,
            detail=(
                "The golden manifest has changed since it was pinned in the "
                "configuration. Every evaluation pinned to the old hash is no "
                "longer comparable."
            ),
            problems=(
                f"expected manifest sha256 {expected}, found {manifest_sha256}",
            ),
        )

    problems = verify_content(dataset)
    if problems:
        return GoldenStatus(
            status=MISMATCH,
            dataset=dataset,
            detail="Golden dataset content no longer matches its manifest.",
            problems=problems,
        )

    if training_sample_ids is not None:
        overlap = contamination(dataset, training_sample_ids)
        if overlap:
            return GoldenStatus(
                status=CONTAMINATED,
                dataset=dataset,
                detail=(
                    "Golden samples also appear in the training data. Scores "
                    "measured on them would report memorisation, not accuracy."
                ),
                contaminated_samples=overlap,
            )

    return GoldenStatus(status=OK, dataset=dataset, detail="")


def verify_content(dataset: GoldenDataset) -> tuple[str, ...]:
    """Re-hash the registered images and report any drift."""
    problems: list[str] = []
    for sample_id, relative in sorted(dataset.images.items()):
        path = dataset.root / relative
        if not path.is_file():
            problems.append(f"registered image is missing: {relative}")
            continue
        actual = _sha256_file(path)
        if actual is None:
            problems.append(f"registered image could not be read: {relative}")
        elif actual != sample_id:
            problems.append(f"registered image changed on disk: {relative}")
    return tuple(problems)


def contamination(
    dataset: GoldenDataset, training_sample_ids: Iterable[str]
) -> tuple[str, ...]:
    """Golden sample ids that also appear in training data.

    Both sides identify a sample by the sha256 of its image bytes, so this
    catches the same photo arriving under a different filename --- which is
    how contamination actually happens.
    """
    training = {str(sample_id) for sample_id in training_sample_ids}
    return tuple(sorted(training.intersection(dataset.sample_ids)))


# ---------------------------------------------------------------------------


def _intersect_groups(
    groups: Mapping[str, str] | None, images: Mapping[str, str]
) -> dict[str, str]:
    """Keep only the group labels that describe an image actually present.

    A candidate report covers far more images than a reviewer keeps, so most
    of its assignments legitimately fall away here. Recording them anyway
    would make the manifest claim the set contains samples it does not.

    Matching nothing at all is different in kind: it means the wrong file was
    passed, or the join key was wrong, and the set would then register with no
    split while looking like it had one. That is refused.
    """
    if not groups:
        return {}
    matched = {
        sample_id: str(label).strip()
        for sample_id, label in groups.items()
        if sample_id in images and str(label).strip()
    }
    if not matched:
        raise GoldenDatasetError(
            f"None of the {len(groups)} supplied group assignments match an "
            "image in this directory. Group keys are the sha256 of the image "
            "bytes; check that the assignments come from a candidate report "
            "covering these images."
        )
    return matched


def _hash_images(root: Path) -> dict[str, str]:
    images: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        digest = _sha256_file(path)
        if digest is None:
            LOGGER.warning("Skipping unreadable golden image %s", path)
            continue
        images[digest] = path.relative_to(root).as_posix()
    return images


def _sha256_file(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
