"""The human labelling queue.

This is the deliberate seam in the autonomous path. Selection is automatic;
annotation is not. Low-confidence and randomly-sampled images carry no ground
truth, and the model's own predictions are not evidence --- training on them
teaches the model its own mistakes, which is exactly why the operator
workflow already refuses to do it.

So a cycle exports a request, a person labels it with whatever tool they
already use, and the cycle imports the result. Labels are validated with the
same :func:`picture_tool.pending_annotations.validate_yolo_label_text` the
operator flow uses, so a label that would be rejected there is rejected here.

Import is all-or-nothing per request: a batch containing one invalid label
changes nothing, so the person fixes it and retries against an unchanged
pool rather than reconciling a half-applied import.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from picture_tool.autotrain import AutoTrainError
from picture_tool.autotrain.candidate_pool import (
    DISCARDED,
    NEEDS_LABEL,
    VERIFIED,
    CandidatePool,
    CandidatePoolError,
    CandidateSample,
    sample_id_for_image,
    utc_now_iso,
)
from picture_tool.autotrain.class_schema import (
    ClassSchema,
    ClassSchemaError,
    normalize_class_names,
)
from picture_tool.autotrain.golden import IMAGE_SUFFIXES
from picture_tool.pending_annotations import validate_yolo_label_text

LOGGER = logging.getLogger(__name__)

REQUEST_MANIFEST_NAME = "request.json"
CLASSES_FILENAME = "classes.txt"
IMAGES_DIRNAME = "images"
LABELS_DIRNAME = "labels"
REQUEST_SCHEMA_VERSION = 1
#: Selector recorded for candidates that arrived already labelled, so the pool
#: and every dataset version built from it can tell them from production picks.
IMPORTED_SELECTOR = "imported"


class LabelingError(AutoTrainError):
    """Raised when a labelling request cannot be exported or imported."""


@dataclass(frozen=True)
class LabelingRequest:
    """One exported batch of images awaiting annotation."""

    request_id: str
    root: Path
    product: str
    area: str
    class_names: tuple[str, ...]
    sample_ids: tuple[str, ...]
    created_at: str
    schema_version: int = REQUEST_SCHEMA_VERSION

    @property
    def images_dir(self) -> Path:
        return self.root / IMAGES_DIRNAME

    @property
    def labels_dir(self) -> Path:
        return self.root / LABELS_DIRNAME

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "product": self.product,
            "area": self.area,
            "class_names": list(self.class_names),
            "sample_ids": list(self.sample_ids),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ImportResult:
    """What one import applied, and what it refused."""

    verified: tuple[str, ...] = ()
    skipped_unlabelled: tuple[str, ...] = ()
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def applied(self) -> bool:
        return bool(self.verified)

    def summary(self) -> dict[str, Any]:
        return {
            "verified": len(self.verified),
            "skipped_unlabelled": len(self.skipped_unlabelled),
            "errors": list(self.errors),
        }


def export_request(
    pool: CandidatePool,
    destination: Path,
    *,
    product: str,
    area: str,
    class_names: Sequence[str],
    request_id: str | None = None,
    limit: int | None = None,
    samples: Sequence[CandidateSample] | None = None,
) -> LabelingRequest:
    """Write the pending candidates out for a human to annotate.

    Filenames are prefixed with the selector that picked each image, so an
    annotator can work through one kind of case at a time --- the same reason
    the operator flow prefixes its filenames by repair type.
    """
    if not class_names:
        raise LabelingError(
            "class_names is required: an annotation tool with the wrong class "
            "list produces labels that cannot be trained on."
        )
    pending = list(samples if samples is not None else pool.by_label_state(NEEDS_LABEL))
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        raise LabelingError("There are no candidates waiting for labels.")

    identifier = request_id or datetime.now(timezone.utc).strftime("req_%Y%m%dT%H%M%SZ")
    root = Path(destination) / identifier
    images = root / IMAGES_DIRNAME
    labels = root / LABELS_DIRNAME
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    exported: list[str] = []
    for sample in pending:
        source = Path(sample.image_path)
        if not source.is_file():
            LOGGER.warning(
                "Candidate %s has no pooled image at %s; not exported.",
                sample.sample_id,
                source,
            )
            continue
        name = f"{sample.selector}-{sample.sample_id}{source.suffix.lower()}"
        shutil.copy2(source, images / name)
        exported.append(sample.sample_id)

    if not exported:
        raise LabelingError(
            "No pooled images could be exported; the pool may be damaged."
        )

    (root / CLASSES_FILENAME).write_text(
        "\n".join(class_names) + "\n", encoding="utf-8"
    )
    request = LabelingRequest(
        request_id=identifier,
        root=root,
        product=product,
        area=area,
        class_names=tuple(class_names),
        sample_ids=tuple(exported),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    (root / REQUEST_MANIFEST_NAME).write_text(
        json.dumps(request.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return request


def load_request(root: str | Path) -> LabelingRequest:
    """Read an exported request back from disk."""
    directory = Path(root).expanduser().resolve()
    manifest = directory / REQUEST_MANIFEST_NAME
    if not manifest.is_file():
        raise LabelingError(f"Not a labelling request: {manifest} is missing.")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LabelingError(f"Unable to read {manifest}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LabelingError(f"{manifest} is not a request manifest.")
    return LabelingRequest(
        request_id=str(payload.get("request_id", directory.name)),
        root=directory,
        product=str(payload.get("product", "")),
        area=str(payload.get("area", "")),
        class_names=tuple(str(n) for n in payload.get("class_names", [])),
        sample_ids=tuple(str(s) for s in payload.get("sample_ids", [])),
        created_at=str(payload.get("created_at", "")),
        schema_version=int(payload.get("schema_version", 0)),
    )


def import_request(pool: CandidatePool, root: str | Path) -> ImportResult:
    """Validate annotated labels and mark their candidates verified.

    All-or-nothing: every label is validated before any is stored, so a batch
    with one bad file leaves the pool exactly as it was.
    """
    request = load_request(root)
    if not request.class_names:
        raise LabelingError(
            f"Request {request.request_id} declares no classes; cannot validate "
            "labels against an unknown class list."
        )

    by_id = {sample.sample_id: sample for sample in pool.load()}
    accepted: list[tuple[str, str]] = []
    unlabelled: list[str] = []
    errors: list[str] = []

    for sample_id in request.sample_ids:
        if sample_id not in by_id:
            errors.append(f"{sample_id}: no longer in the candidate pool")
            continue
        label_path = _find_label(request, sample_id)
        if label_path is None:
            unlabelled.append(sample_id)
            continue
        try:
            text = label_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"{sample_id}: unreadable label ({exc})")
            continue
        problems = validate_yolo_label_text(text, len(request.class_names))
        if problems:
            errors.append(f"{sample_id}: " + "; ".join(problems))
            continue
        accepted.append((sample_id, text))

    if errors:
        return ImportResult(
            verified=(),
            skipped_unlabelled=tuple(unlabelled),
            errors=tuple(errors),
        )

    pool.labels_dir.mkdir(parents=True, exist_ok=True)
    verified: list[str] = []
    for sample_id, text in accepted:
        stored = pool.labels_dir / f"{sample_id}.txt"
        stored.write_text(text, encoding="utf-8")
        pool.update_label_state(sample_id, VERIFIED, label_path=str(stored))
        verified.append(sample_id)

    return ImportResult(
        verified=tuple(verified),
        skipped_unlabelled=tuple(unlabelled),
        errors=(),
    )


@dataclass(frozen=True)
class FolderImportResult:
    """What importing an already-labelled folder applied, and what it refused."""

    verified: tuple[str, ...] = ()
    already_verified: tuple[str, ...] = ()
    duplicates: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def applied(self) -> bool:
        return not self.errors and bool(self.verified)

    def summary(self) -> dict[str, Any]:
        return {
            "verified": len(self.verified),
            "already_verified": len(self.already_verified),
            "duplicates_in_folder": len(self.duplicates),
            "errors": list(self.errors),
        }


def import_labelled_folder(
    pool: CandidatePool,
    root: str | Path,
    *,
    class_schema: ClassSchema,
    product: str,
    area: str,
    golden_sample_ids: Iterable[str] = (),
) -> FolderImportResult:
    """Add a folder that is already labelled to the pool, as verified.

    The folder is ``images/`` and ``labels/`` with matching relative paths
    (nested splits such as ``images/train`` are fine) and a ``classes.txt``
    or YAML ``names`` stating what the class ids mean. Everything is checked
    before anything is written, so a folder with one problem changes nothing:

    * its class list must equal the station's contract, in order --- the
      same names in another order would train every box as the wrong class;
    * every image needs a label file. An empty file is a verified negative;
      a missing one is not, and the two must stay distinguishable;
    * labels pass the operator flow's own validator;
    * one image stored twice with different labels is a conflict, not a
      choice to make silently --- public datasets do contain these;
    * an image that is in the golden set is refused here, where it can
      still be removed, rather than surfacing later as a contaminated
      evaluation;
    * an image already in the pool may be verified by this import, but a
      different label for one a person already verified, or a label for one
      a person discarded, is a conflict.
    """
    folder = Path(root).expanduser()
    images_dir, labels_dir = folder / IMAGES_DIRNAME, folder / LABELS_DIRNAME
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return FolderImportResult(
            errors=(f"{folder} must contain {IMAGES_DIRNAME}/ and {LABELS_DIRNAME}/",)
        )
    try:
        declared = read_folder_class_schema(folder)
    except ClassSchemaError as exc:
        return FolderImportResult(errors=(str(exc),))
    if declared is None:
        return FolderImportResult(
            errors=(
                f"{folder} declares no classes; add {CLASSES_FILENAME} or a YAML "
                "file with 'names' so the class ids can be checked.",
            )
        )
    if not declared.agrees_with(class_schema):
        return FolderImportResult(
            errors=(
                f"class list {list(declared.names)} does not match this station's "
                f"{list(class_schema.names)} (order matters).",
            )
        )

    golden = set(golden_sample_ids)
    existing = {sample.sample_id: sample for sample in pool.load()}
    accepted: dict[str, tuple[Path, str, str]] = {}
    already: list[str] = []
    duplicates: list[str] = []
    errors: list[str] = []

    for image in sorted(images_dir.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = image.relative_to(images_dir).as_posix()
        label = labels_dir / Path(relative).with_suffix(".txt")
        if not label.is_file():
            errors.append(
                f"{relative}: no label file (write an empty one for an image "
                "with no objects)"
            )
            continue
        try:
            text = label.read_text(encoding="utf-8")
            sample_id = sample_id_for_image(image)
        except (OSError, UnicodeDecodeError, CandidatePoolError) as exc:
            errors.append(f"{relative}: unreadable ({exc})")
            continue
        problems = validate_yolo_label_text(text, len(class_schema.names))
        if problems:
            errors.append(f"{relative}: " + "; ".join(problems))
            continue
        if sample_id in golden:
            errors.append(
                f"{relative}: this image is in the golden set; training on it "
                "would contaminate the evaluation"
            )
            continue
        earlier = accepted.get(sample_id)
        if earlier is not None:
            if _label_lines(earlier[1]) == _label_lines(text):
                duplicates.append(relative)
            else:
                errors.append(
                    f"{relative}: same image as {earlier[2]} but with a different label"
                )
            continue
        current = existing.get(sample_id)
        if current is not None and current.label_state == DISCARDED:
            errors.append(f"{relative}: a person discarded this image from the pool")
            continue
        if current is not None and current.label_state == VERIFIED:
            stored = _read_label(current.label_path)
            if stored is not None and _label_lines(stored) == _label_lines(text):
                already.append(sample_id)
                continue
            if stored is not None:
                errors.append(
                    f"{relative}: already verified in the pool with a different label"
                )
                continue
        accepted[sample_id] = (image, text, relative)

    if errors:
        return FolderImportResult(
            already_verified=tuple(already),
            duplicates=tuple(duplicates),
            errors=tuple(errors),
        )

    selected_at = utc_now_iso()
    added = pool.add(
        CandidateSample(
            sample_id=sample_id,
            inspection_id="",
            product=product,
            area=area,
            selector=IMPORTED_SELECTOR,
            selected_reason=f"imported labelled image {relative} from {folder}",
            score=0.0,
            source_model="",
            source_model_version="",
            selected_at=selected_at,
            production_timestamp="",
            image_path="",
            source_image_path=str(image),
        )
        for sample_id, (image, _text, relative) in accepted.items()
    )
    pool.labels_dir.mkdir(parents=True, exist_ok=True)
    verified: list[str] = []
    for sample_id, (_image, text, _relative) in accepted.items():
        if sample_id in added.skipped_missing_image:
            continue
        stored_label = pool.labels_dir / f"{sample_id}.txt"
        stored_label.write_text(text, encoding="utf-8")
        pool.update_label_state(sample_id, VERIFIED, label_path=str(stored_label))
        verified.append(sample_id)
    return FolderImportResult(
        verified=tuple(verified),
        already_verified=tuple(already),
        duplicates=tuple(duplicates),
        errors=tuple(
            f"{accepted[sample_id][2]}: could not be copied into the pool"
            for sample_id in added.skipped_missing_image
        ),
    )


def read_folder_class_schema(root: str | Path) -> ClassSchema | None:
    """The class list a labelled folder declares, or None if it declares none.

    ``classes.txt`` (one name per line, the labelImg convention) wins over a
    YAML ``names`` entry because it is what annotation tools write.
    """
    folder = Path(root)
    classes_file = folder / CLASSES_FILENAME
    if classes_file.is_file():
        names = [
            line.strip()
            for line in classes_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return normalize_class_names(names, source=str(classes_file)) if names else None
    for descriptor in sorted(folder.glob("*.yaml")) + sorted(folder.glob("*.yml")):
        try:
            payload = yaml.safe_load(descriptor.read_text(encoding="utf-8")) or {}
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            continue
        if isinstance(payload, dict) and payload.get("names"):
            return normalize_class_names(payload["names"], source=str(descriptor))
    return None


def _label_lines(text: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in text.splitlines() if line.strip())


def _read_label(path: str) -> str | None:
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def verified_samples(pool: CandidatePool) -> tuple[CandidateSample, ...]:
    """Pool entries with a human-verified label that still exists on disk."""
    usable: list[CandidateSample] = []
    for sample in pool.by_label_state(VERIFIED):
        if not sample.label_path or not Path(sample.label_path).is_file():
            LOGGER.warning(
                "Candidate %s is marked verified but its label is missing at %s",
                sample.sample_id,
                sample.label_path,
            )
            continue
        if not sample.image_path or not Path(sample.image_path).is_file():
            LOGGER.warning(
                "Candidate %s is marked verified but its image is missing at %s",
                sample.sample_id,
                sample.image_path,
            )
            continue
        usable.append(sample)
    return tuple(usable)


def _find_label(request: LabelingRequest, sample_id: str) -> Path | None:
    """Locate the annotator's output for one sample.

    Accepts either the exported ``<selector>-<sample_id>.txt`` name or a bare
    ``<sample_id>.txt``, because annotation tools differ in whether they keep
    the image's stem.
    """
    exact = request.labels_dir / f"{sample_id}.txt"
    if exact.is_file():
        return exact
    if not request.labels_dir.is_dir():
        return None
    for path in sorted(request.labels_dir.glob(f"*{sample_id}.txt")):
        if path.is_file():
            return path
    return None


def request_statistics(request: LabelingRequest) -> Mapping[str, int]:
    """How much of a request has been annotated so far."""
    labelled = sum(
        1 for sample_id in request.sample_ids if _find_label(request, sample_id)
    )
    return {
        "requested": len(request.sample_ids),
        "labelled": labelled,
        "outstanding": len(request.sample_ids) - labelled,
    }
