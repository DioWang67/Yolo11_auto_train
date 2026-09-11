"""The class_id -> class_name contract.

Every test here is about one failure mode: a model trained against the wrong
class order. It does not crash and it does not look wrong --- the metrics come
out fine, and the model calls a red wire black. So the assertions are mostly
about refusing, and the hardest case to get right is the one where two sources
list the same names in a different order.
"""

from __future__ import annotations

import json

import pytest
import yaml

from picture_tool.autotrain.class_schema import (
    SOURCE_CHAMPION,
    SOURCE_DATASET,
    ClassSchema,
    ClassSchemaError,
    normalize_class_names,
    read_checkpoint_class_schema,
    resolve_class_schema,
    schema_from_json_field,
    schema_from_station_config,
    validate_label_class_ids,
)
from picture_tool.autotrain.dataset_versions import DatasetVersionStore, LabelledSample

#: The real Cable1/A contract, as recorded by the station's champion, by its
#: handoff manifests, and by the data.yaml of every job trained from them.
CANONICAL = ["Black", "Green", "Orange", "Red", "Yellow"]

#: What the station config actually carries instead. Different order, and
#: "Black" twice --- it is the items the station expects to see, not classes.
EXPECTED_ITEMS = ["Red", "Green", "Orange", "Yellow", "Black", "Black"]

#: The hash the operator handoff writes for CANONICAL, taken from the real
#: manifests under data/.operator_handoff. Pinned so a change to the hashing
#: shows up here rather than as two subsystems that quietly stop agreeing.
CANONICAL_HASH = "05f915927011ba63db6d16d535e714c014b53f3f6602391688536aa5b3119df9"


def _schema(names=None, source="test") -> ClassSchema:
    return normalize_class_names(list(names or CANONICAL), source=source)


# ---------------------------------------------------------------------------
# Normalizing the shapes this contract is stored in


def test_a_list_of_names_keeps_its_order():
    schema = _schema()

    assert schema.names == tuple(CANONICAL)
    assert schema.mapping == {0: "Black", 1: "Green", 2: "Orange", 3: "Red", 4: "Yellow"}


def test_an_int_keyed_mapping_is_ordered_by_class_id():
    """The shape ultralytics hands back from model.names."""
    schema = normalize_class_names(
        {0: "Black", 1: "Green", 2: "Orange", 3: "Red", 4: "Yellow"}, source="model"
    )

    assert schema.names == tuple(CANONICAL)


def test_a_string_keyed_mapping_is_ordered_numerically_not_lexically():
    """The shape anything that has been through JSON has.

    Lexical ordering of "0".."10" puts 10 second, so the keys have to be read
    as numbers. With five classes this test would pass either way, so it uses
    enough to tell the two apart.
    """
    raw = {str(index): f"c{index}" for index in range(12)}

    schema = normalize_class_names(raw, source="json")

    assert schema.names == tuple(f"c{index}" for index in range(12))


def test_a_mapping_in_arbitrary_key_order_still_orders_by_id():
    schema = normalize_class_names(
        {"3": "Red", "0": "Black", "4": "Yellow", "1": "Green", "2": "Orange"},
        source="json",
    )

    assert schema.names == tuple(CANONICAL)


def test_the_hash_matches_the_operator_handoffs_own_checksum():
    """Both sides must compute the same identity or comparison is meaningless."""
    assert _schema().schema_hash == CANONICAL_HASH


def test_order_changes_the_hash():
    reordered = _schema(["Green", "Black", "Orange", "Red", "Yellow"])

    assert reordered.schema_hash != CANONICAL_HASH


# ---------------------------------------------------------------------------
# expected_items is not a class schema


def test_expected_items_cannot_become_a_class_schema():
    """The station's list is refused structurally, not by being named.

    It repeats an entry, and a class id that maps to two names is not a
    mapping. Nothing has to remember that this particular list is dangerous.
    """
    with pytest.raises(ClassSchemaError, match="duplicates"):
        normalize_class_names(EXPECTED_ITEMS, source="station config")


def test_expected_items_is_not_read_from_a_station_config(tmp_path):
    model_dir = tmp_path / "models" / "Cable1" / "A" / "yolo"
    model_dir.mkdir(parents=True)
    (model_dir / "config.yaml").write_text(
        yaml.safe_dump({"expected_items": {"Cable1": {"A": EXPECTED_ITEMS}}}),
        encoding="utf-8",
    )

    assert schema_from_station_config(model_dir) is None


def test_a_station_config_that_declares_class_names_is_read(tmp_path):
    """An explicit declaration is a contract; expected_items is not."""
    model_dir = tmp_path / "models" / "Cable1" / "A" / "yolo"
    model_dir.mkdir(parents=True)
    (model_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {"class_names": CANONICAL, "expected_items": {"Cable1": {"A": EXPECTED_ITEMS}}}
        ),
        encoding="utf-8",
    )

    schema = schema_from_station_config(model_dir)

    assert schema is not None
    assert schema.names == tuple(CANONICAL)


# ---------------------------------------------------------------------------
# Malformed contracts


def test_an_empty_class_list_is_refused():
    with pytest.raises(ClassSchemaError, match="empty"):
        normalize_class_names([], source="test")


def test_a_gapped_class_map_is_refused():
    """A partial map is what a production row carries; it is not a schema."""
    with pytest.raises(ClassSchemaError, match="gapless"):
        normalize_class_names({"0": "Black", "4": "Yellow"}, source="row")


def test_a_blank_name_is_refused():
    with pytest.raises(ClassSchemaError, match="blank"):
        normalize_class_names(["Black", "", "Orange"], source="test")


def test_a_string_is_not_a_class_list():
    with pytest.raises(ClassSchemaError, match="not a string"):
        normalize_class_names("Black,Green", source="test")


def test_a_recorded_schema_whose_hash_does_not_match_is_refused():
    """Catches an edited record, which would silently redefine every id."""
    payload = _schema().to_dict()
    payload["names"] = ["Green", "Black", "Orange", "Red", "Yellow"]

    with pytest.raises(ClassSchemaError, match="does not match its own hash"):
        ClassSchema.from_dict(payload)


# ---------------------------------------------------------------------------
# Resolution: agreement, disagreement, and absence


def test_matching_sources_resolve_to_the_shared_contract():
    champion = _schema(source=SOURCE_CHAMPION)
    dataset = _schema(source=SOURCE_DATASET)

    resolved = resolve_class_schema([dataset, champion], context="Cable1/A")

    assert resolved.names == tuple(CANONICAL)
    assert resolved.source == SOURCE_DATASET


def test_the_same_names_in_a_different_order_is_a_disagreement():
    """The case that motivates all of this.

    Set comparison would call these identical. The labels store ids, so under
    this pair every single one means something different.
    """
    champion = _schema(source=SOURCE_CHAMPION)
    dataset = _schema(
        ["Green", "Black", "Orange", "Red", "Yellow"], source=SOURCE_DATASET
    )

    with pytest.raises(ClassSchemaError) as excinfo:
        resolve_class_schema([dataset, champion], context="Cable1/A")

    message = str(excinfo.value)
    assert "disagree" in message
    assert SOURCE_CHAMPION in message and SOURCE_DATASET in message
    assert "0=Black" in message and "0=Green" in message


def test_an_extra_class_in_the_dataset_is_a_disagreement():
    champion = _schema(source=SOURCE_CHAMPION)
    dataset = _schema(CANONICAL + ["Blue"], source=SOURCE_DATASET)

    with pytest.raises(ClassSchemaError, match="disagree"):
        resolve_class_schema([dataset, champion], context="Cable1/A")


def test_a_missing_class_in_the_dataset_is_a_disagreement():
    champion = _schema(source=SOURCE_CHAMPION)
    dataset = _schema(CANONICAL[:-1], source=SOURCE_DATASET)

    with pytest.raises(ClassSchemaError, match="disagree"):
        resolve_class_schema([dataset, champion], context="Cable1/A")


def test_no_source_at_all_fails_closed():
    """"Cannot tell" must never resolve to "carry on"."""
    with pytest.raises(ClassSchemaError, match="No class schema is available"):
        resolve_class_schema([None, None, None], context="Cable1/A")


def test_the_failure_message_says_expected_items_is_not_a_source():
    """Whoever hits this will be looking straight at a list that would 'work'."""
    with pytest.raises(ClassSchemaError, match="expected_items"):
        resolve_class_schema([None], context="Cable1/A")


# ---------------------------------------------------------------------------
# Reading model.names off a checkpoint


def test_model_names_are_read_from_a_checkpoint(monkeypatch):
    """The real champion's shape: a dict with integer keys."""

    class _Model:
        names = {0: "Black", 1: "Green", 2: "Orange", 3: "Red", 4: "Yellow"}

    class _YOLO:
        def __init__(self, path):
            self.model = _Model()

    module = type("ultralytics", (), {"YOLO": _YOLO})
    monkeypatch.setitem(__import__("sys").modules, "ultralytics", module)
    weights = _touch_pt(monkeypatch)

    schema = read_checkpoint_class_schema(weights)

    assert schema is not None
    assert schema.names == tuple(CANONICAL)
    assert schema.schema_hash == CANONICAL_HASH


def test_an_unreadable_checkpoint_yields_no_schema_rather_than_raising(tmp_path):
    """A source that cannot answer lets the next one try; the resolver refuses."""
    weights = tmp_path / "not-really.pt"
    weights.write_bytes(b"not a checkpoint")

    assert read_checkpoint_class_schema(weights) is None


def test_a_runtime_export_is_not_asked_for_class_names(tmp_path):
    runtime = tmp_path / "model.onnx"
    runtime.write_bytes(b"onnx")

    assert read_checkpoint_class_schema(runtime) is None


def _touch_pt(monkeypatch):
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "best.training.pt"
    path.write_bytes(b"weights")
    return path


# ---------------------------------------------------------------------------
# Manifest fields, as real rows actually look


def test_a_populated_class_names_json_is_read():
    schema = schema_from_json_field(json.dumps(CANONICAL), source="manifest")

    assert schema is not None
    assert schema.names == tuple(CANONICAL)


def test_an_empty_or_partial_manifest_field_yields_nothing():
    """Both shapes occur in the real handoff manifests under data/."""
    assert schema_from_json_field("[]", source="manifest") is None
    assert schema_from_json_field("{}", source="manifest") is None
    assert schema_from_json_field(None, source="manifest") is None
    # A per-image class map naming only what that image contained.
    assert schema_from_json_field('{"0":"Black","4":"Yellow"}', source="row") is None


def test_a_full_class_map_field_is_read():
    payload = json.dumps({str(i): name for i, name in enumerate(CANONICAL)})

    schema = schema_from_json_field(payload, source="manifest")

    assert schema is not None
    assert schema.names == tuple(CANONICAL)


# ---------------------------------------------------------------------------
# Labels must fit the contract


def test_labels_within_the_schema_pass(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n4 0.1 0.1 0.1 0.1\n", "utf-8")

    validate_label_class_ids(labels, _schema())


def test_a_class_id_past_the_end_of_the_schema_is_refused(tmp_path):
    """Five classes means ids 0..4; a 5 would train a class that means nothing."""
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("5 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(ClassSchemaError, match="outside 0..4"):
        validate_label_class_ids(labels, _schema())


def test_a_negative_class_id_is_refused(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("-1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(ClassSchemaError, match="outside"):
        validate_label_class_ids(labels, _schema())


def test_the_offending_file_and_line_are_named(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "good.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (labels / "bad.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n9 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )

    with pytest.raises(ClassSchemaError) as excinfo:
        validate_label_class_ids(labels, _schema())

    assert "bad.txt:2" in str(excinfo.value)


def test_an_empty_label_file_is_fine(tmp_path):
    """A verified negative: the image legitimately contains nothing."""
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "empty.txt").write_text("", encoding="utf-8")

    validate_label_class_ids(labels, _schema())


# ---------------------------------------------------------------------------
# A dataset version carries its contract


def test_a_dataset_version_records_and_reloads_its_class_schema(tmp_path):
    store = DatasetVersionStore(tmp_path / "datasets", product="Cable1", area="A")
    image = tmp_path / "src" / "a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"pixels")
    label = tmp_path / "src" / "a.txt"
    label.write_text("3 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    version = store.create(
        [LabelledSample(sample_id="a", image_path=image, label_path=label)],
        source="pool",
        label_source="human",
        class_schema=_schema(),
    )
    reloaded = store.load(version.version)

    assert reloaded.class_schema is not None
    assert reloaded.class_schema.names == tuple(CANONICAL)
    assert reloaded.class_schema.schema_hash == CANONICAL_HASH


def test_a_version_whose_labels_exceed_its_schema_is_refused(tmp_path):
    """The version is immutable, so a bad id would be recorded permanently."""
    store = DatasetVersionStore(tmp_path / "datasets", product="Cable1", area="A")
    image = tmp_path / "src" / "a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"pixels")
    label = tmp_path / "src" / "a.txt"
    label.write_text("7 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(ClassSchemaError, match="outside 0..4"):
        store.create(
            [LabelledSample(sample_id="a", image_path=image, label_path=label)],
            source="pool",
            label_source="human",
            class_schema=_schema(),
        )
