"""The ``class_id -> class_name`` contract, and where it may come from.

A YOLO label file stores a class *index*. The index means nothing on its own:
``0`` is whatever the ordered name list says position zero is. Two models
trained on the same images with the names in a different order are not
comparable, and a challenger trained against the wrong order produces a model
that is confidently, silently wrong --- it would call a red wire black and
report excellent metrics for doing so.

So the ordered list is treated here as a contract with an identity, not as a
convenience list of strings:

* It is **ordered**. Comparison is by sequence, never by set membership.
* It carries a **hash**, computed exactly the way the existing operator
  handoff computes it, so a schema recorded by this path and one recorded by
  the operator flow are directly comparable.
* Names must be **unique**. A duplicate makes ``class_id -> class_name``
  non-invertible, and is the signature of the one input this module must
  never accept --- see below.

``expected_items`` is not a class schema
----------------------------------------
A station config carries ``expected_items``, e.g. for Cable1/A
``['Red', 'Green', 'Orange', 'Yellow', 'Black', 'Black']``. It is tempting
because it is a list of the right-looking words, and it is wrong: it is the
multiset of physical items the station expects to see, compared with
``set(expected_items) - detected`` by the inference detector. Its order is
unrelated to the model's class indices and it contains a duplicate.

Nothing here derives a schema from it, and the duplicate rule means it cannot
be smuggled in: constructing a schema from that list is refused by
:func:`normalize_class_names` rather than silently de-duplicated or sorted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from picture_tool.autotrain import AutoTrainError

# The operator handoff already defines this contract's checksum. Importing it
# rather than restating the algorithm is what keeps a schema hash written here
# comparable with one written there; a second implementation would be free to
# drift. picture_tool.portable_training_package imports it the same way.
from picture_tool.pending_annotations import _class_schema_hash

LOGGER = logging.getLogger(__name__)

#: Bumped only if the recorded shape changes, not when class names change.
CLASS_SCHEMA_VERSION = 1

#: Where a schema came from, most authoritative first. Used for reporting and
#: to make the precedence in :func:`resolve_class_schema` explicit.
SOURCE_DATASET = "dataset_version"
SOURCE_CHAMPION = "champion_checkpoint"
SOURCE_RECORDED = "recorded_class_names"
SOURCE_STATION_CONFIG = "station_config"


class ClassSchemaError(AutoTrainError):
    """Raised when the class contract is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class ClassSchema:
    """An ordered YOLO class list, with the identity to compare it by."""

    names: tuple[str, ...]
    source: str = ""

    @property
    def schema_hash(self) -> str:
        """Checksum over the ordered names, shared with the operator handoff."""
        return _class_schema_hash(list(self.names))

    @property
    def mapping(self) -> dict[int, str]:
        return {index: name for index, name in enumerate(self.names)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CLASS_SCHEMA_VERSION,
            "names": list(self.names),
            "schema_hash": self.schema_hash,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClassSchema":
        schema = normalize_class_names(
            payload.get("names"), source=str(payload.get("source") or "")
        )
        recorded = str(payload.get("schema_hash") or "")
        if recorded and recorded != schema.schema_hash:
            raise ClassSchemaError(
                "Recorded class schema does not match its own hash: names "
                f"{list(schema.names)} hash to {schema.schema_hash}, but "
                f"{recorded} was stored. The record has been edited or is "
                "corrupt; it cannot be used to decide what a class id means."
            )
        return schema

    def describe(self) -> str:
        """A one-line ``0=Name, 1=Name`` rendering for error messages."""
        pairs = ", ".join(f"{i}={name}" for i, name in enumerate(self.names))
        return f"[{pairs}] (hash {self.schema_hash[:12]}, from {self.source or '?'})"

    def agrees_with(self, other: "ClassSchema") -> bool:
        return self.names == other.names


def normalize_class_names(raw: Any, *, source: str = "") -> ClassSchema:
    """Build a schema from any of the shapes this contract is stored in.

    Accepts an ordered sequence, or a mapping of class id to name with either
    integer or string keys --- ultralytics hands back ``{0: 'Black', ...}``
    while anything that has been through JSON has ``{'0': 'Black', ...}``.

    Refuses anything it cannot read as an exact, gapless ``0..n-1`` mapping.
    The refusals are the point: each one is a way a wrong schema could
    otherwise reach a trainer.
    """
    if raw is None:
        raise ClassSchemaError(f"No class names supplied (source: {source or '?'}).")

    if isinstance(raw, Mapping):
        names = _names_from_mapping(raw, source=source)
    elif isinstance(raw, (str, bytes)):
        raise ClassSchemaError(
            f"Class names must be a sequence or mapping, not a string "
            f"(source: {source or '?'}, value: {raw!r})."
        )
    elif isinstance(raw, Iterable):
        names = [str(name).strip() for name in raw]
    else:
        raise ClassSchemaError(
            f"Class names must be a sequence or mapping, got {type(raw).__name__} "
            f"(source: {source or '?'})."
        )

    if not names:
        raise ClassSchemaError(
            f"Class names are empty (source: {source or '?'}). Training without "
            "a class contract would silently assign meanings to class ids."
        )
    if any(not name for name in names):
        raise ClassSchemaError(
            f"Class names contain a blank entry (source: {source or '?'}): {names}."
        )

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ClassSchemaError(
            f"Class names contain duplicates {duplicates} (source: "
            f"{source or '?'}): {names}. A class id must map to exactly one "
            "name. Note that a station's expected_items list legitimately "
            "repeats entries and is not a class schema --- it is the items the "
            "station expects to see, not the model's classes."
        )
    return ClassSchema(names=tuple(names), source=source)


def _names_from_mapping(raw: Mapping[Any, Any], *, source: str) -> list[str]:
    """Order a ``{class_id: name}`` mapping, insisting the ids are ``0..n-1``."""
    indices: dict[int, str] = {}
    for key, value in raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise ClassSchemaError(
                f"Class id {key!r} is not an integer (source: {source or '?'})."
            ) from exc
        if index in indices:
            raise ClassSchemaError(
                f"Class id {index} appears twice (source: {source or '?'})."
            )
        indices[index] = str(value).strip()

    expected = set(range(len(indices)))
    if set(indices) != expected:
        missing = sorted(expected - set(indices))
        extra = sorted(set(indices) - expected)
        raise ClassSchemaError(
            f"Class ids must be a gapless 0..{len(indices) - 1} range "
            f"(source: {source or '?'}); missing {missing}, unexpected {extra}. "
            "A gap means the stored mapping is partial, and a partial mapping "
            "cannot say what the other ids mean."
        )
    return [indices[index] for index in sorted(indices)]


def read_checkpoint_class_schema(weights_path: str | Path) -> ClassSchema | None:
    """Read ``model.names`` from a YOLO checkpoint.

    Returns ``None`` whenever the checkpoint cannot supply names --- missing
    file, not a ``.pt``, ultralytics absent, or the file unreadable. This is a
    *source*, and a source that cannot answer should let the next one try;
    :func:`resolve_class_schema` is what refuses when none of them can, so
    failing hard here would only remove the chance to fall through without
    making anything safer.

    ultralytics is imported lazily: it pulls in torch, and resolving a schema
    from an already-recorded source must not pay for that.
    """
    path = Path(weights_path)
    if not path.is_file() or path.suffix.lower() != ".pt":
        # An exported runtime (.onnx) carries no usable names; the paired
        # training checkpoint is the one to read.
        return None
    try:
        from ultralytics import YOLO
    except ImportError:  # pragma: no cover - environment without ultralytics
        LOGGER.warning("ultralytics is unavailable; cannot read %s", path)
        return None
    try:
        raw = getattr(YOLO(str(path)).model, "names", None)
        if not raw:
            return None
        # Parsed inside the guard too: whatever this file yields is untrusted
        # input, and a checkpoint that answers with something unreadable is
        # still just a source that could not answer.
        return normalize_class_names(raw, source=SOURCE_CHAMPION)
    except Exception as exc:  # noqa: BLE001 - ultralytics raises many types
        LOGGER.warning("Could not read class names from checkpoint %s: %s", path, exc)
        return None


def resolve_class_schema(
    candidates: Sequence[ClassSchema | None],
    *,
    context: str = "",
) -> ClassSchema:
    """Pick the schema to train against, refusing to guess.

    ``candidates`` is in precedence order, most authoritative first. Every
    present candidate must agree: precedence decides which one is *reported*,
    never which one wins an argument. Disagreement and total absence are both
    hard failures --- "no schema" must not read as "any schema", and
    "two schemas" must not read as "pick one".
    """
    present = [schema for schema in candidates if schema is not None]
    if not present:
        raise ClassSchemaError(
            "No class schema is available from any accepted source"
            + (f" for {context}" if context else "")
            + ". Accepted sources are the dataset version's recorded schema, "
            "the champion checkpoint's model.names, and class names recorded "
            "in production records or the handoff. A station's expected_items "
            "is deliberately not one of them. Refusing to train: a guessed "
            "class order produces a model that is wrong without looking wrong."
        )

    chosen = present[0]
    conflicts = [other for other in present[1:] if not chosen.agrees_with(other)]
    if conflicts:
        lines = [
            "Class schemas disagree"
            + (f" for {context}" if context else "")
            + "; refusing to train.",
            f"  {chosen.source or 'candidate 1'}: {chosen.describe()}",
        ]
        lines.extend(
            f"  {other.source or 'candidate'}: {other.describe()}"
            for other in conflicts
        )
        lines.append(
            "  Same names in a different order is still a disagreement: the "
            "class id is what the labels store."
        )
        raise ClassSchemaError("\n".join(lines))
    return chosen


def validate_label_class_ids(labels_dir: str | Path, schema: ClassSchema) -> None:
    """Check every label's class id against the schema.

    An id outside the schema either crashes the trainer or, worse, is accepted
    against a longer class list and trains a class that means nothing.
    """
    directory = Path(labels_dir)
    if not directory.is_dir():
        raise ClassSchemaError(f"Label directory does not exist: {directory}")

    limit = len(schema.names)
    offenders: list[str] = []
    for label in sorted(directory.glob("*.txt")):
        if label.name == "classes.txt":
            continue
        for number, line in enumerate(
            label.read_text(encoding="utf-8").splitlines(), start=1
        ):
            text = line.strip()
            if not text:
                continue
            token = text.split()[0]
            try:
                class_id = int(float(token))
            except ValueError:
                offenders.append(f"{label.name}:{number} class id {token!r} is not a number")
                continue
            if not 0 <= class_id < limit:
                offenders.append(
                    f"{label.name}:{number} class id {class_id} outside 0..{limit - 1}"
                )
        if len(offenders) >= 20:
            break

    if offenders:
        raise ClassSchemaError(
            f"Labels do not fit the class schema {schema.describe()}; refusing "
            "to train.\n  " + "\n  ".join(offenders[:20])
        )


def schema_from_station_config(model_dir: str | Path) -> ClassSchema | None:
    """An explicit class list declared in a station's ``config.yaml``.

    Only ``class_names`` and ``names`` are read, and only when present. Both
    are explicit declarations of the contract, which is what makes them
    usable.

    ``expected_items`` sits in the same file, looks like a list of the same
    words, and is **not** read: it is the multiset of items the station
    expects to see, in its own order and with repeats. Cable1/A's is
    ``['Red', 'Green', 'Orange', 'Yellow', 'Black', 'Black']`` against a real
    contract of ``['Black', 'Green', 'Orange', 'Red', 'Yellow']`` --- every
    single class id would be wrong.
    """
    path = Path(model_dir) / "config.yaml"
    if not path.is_file():
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        LOGGER.warning("Unable to read %s: %s", path, exc)
        return None
    if not isinstance(payload, Mapping):
        return None
    declared = payload.get("class_names") or payload.get("names")
    if not declared:
        return None
    return normalize_class_names(declared, source=SOURCE_STATION_CONFIG)


def schema_from_json_field(value: Any, *, source: str) -> ClassSchema | None:
    """Read a schema from a manifest field that may be a JSON string.

    Handoff manifests store the contract as ``class_names_json`` and
    ``class_map_json`` text. Real rows carry any of: a populated list, the
    empty list, a full id map, a partial id map, or nothing --- so an unusable
    value returns ``None`` to fall through rather than failing the cycle.
    """
    if value in (None, "", b""):
        return None
    payload: Any = value
    if isinstance(value, (str, bytes)):
        try:
            payload = json.loads(value)
        except (ValueError, TypeError):
            return None
    if not payload:
        return None
    try:
        return normalize_class_names(payload, source=source)
    except ClassSchemaError:
        # A partial class map is normal in production rows; it describes the
        # classes one image happened to contain, not the station's schema.
        return None
