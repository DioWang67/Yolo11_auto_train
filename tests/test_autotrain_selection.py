"""Candidate selection and the pool that stores it.

Two invariants carry most of the weight here: a candidate always enters the
pool as NEEDS_LABEL (predictions are never treated as ground truth), and the
image is copied in at selection time so production retention cannot empty the
pool later.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from picture_tool.autotrain.candidate_pool import (
    DISCARDED,
    NEEDS_LABEL,
    VERIFIED,
    CandidatePool,
    CandidatePoolError,
    build_candidate,
    sample_id_for_image,
)
from picture_tool.autotrain.collector import ProductionRecord
from picture_tool.autotrain.config import SelectorSettings
from picture_tool.autotrain.image_quality import ImageQuality
from picture_tool.autotrain.selectors import (
    SelectorError,
    available_selectors,
    create_selector,
    register_selector,
    run_selectors,
    unregister_selector,
)

REFERENCE = datetime(2026, 9, 10, 12, 0, 0)


def _image(tmp_path: Path, name: str, payload: bytes = b"") -> Path:
    path = tmp_path / "images" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload or name.encode("utf-8"))
    return path


def _record(tmp_path, inspection_id, *, confidence=None, outcome="", image=None):
    source = image if image is not None else _image(tmp_path, f"{inspection_id}.jpg")
    detections = ()
    if confidence is not None:
        from picture_tool.autotrain.collector import DetectionRecord

        detections = (
            DetectionRecord(class_name="Red", confidence=confidence, bbox=(0, 0, 1, 1)),
        )
    return ProductionRecord(
        inspection_id=inspection_id,
        timestamp=REFERENCE,
        status="PASS",
        detector="yolo",
        product="Cable1",
        area="A",
        model_version="1.2.0",
        model_weights="models/Cable1/A/yolo/weights/best.onnx",
        conf_threshold=0.4,
        class_names=("Red",),
        detections=detections,
        original_path=str(source),
        preprocessed_path="",
        annotated_path="",
        camera_parameters={"gain": "23.0"},
        equipment={},
        fail_reasons=(),
        config_hash="hash",
        snapshot_path="",
        schema_version=2,
        review_outcome=outcome,
    )


# ---------------------------------------------------------------------------
# Registry


def test_default_selectors_are_registered():
    assert available_selectors() == [
        "low_confidence",
        "random_sample",
        "review_correction",
    ]


def test_unknown_selector_is_an_error():
    with pytest.raises(SelectorError, match="Unknown selector"):
        create_selector("does_not_exist")


def test_selectors_can_be_registered_and_removed():
    class Dummy:
        name = "dummy"

        def select(self, records, *, options, selected_at=None):
            return []

    register_selector("dummy", Dummy)
    try:
        assert "dummy" in available_selectors()
        assert create_selector("dummy").name == "dummy"
    finally:
        unregister_selector("dummy")
    assert "dummy" not in available_selectors()


# ---------------------------------------------------------------------------
# Low confidence


def test_low_confidence_picks_only_below_the_threshold(tmp_path):
    records = [
        _record(tmp_path, "sure", confidence=0.95),
        _record(tmp_path, "unsure", confidence=0.31),
    ]

    picked = create_selector("low_confidence").select(
        records, options={"conf_below": 0.55}
    )

    assert [c.inspection_id for c in picked] == ["unsure"]
    assert "0.310" in picked[0].selected_reason
    assert picked[0].metrics["min_confidence"] == pytest.approx(0.31)


def test_low_confidence_prefers_the_least_confident_when_capped(tmp_path):
    records = [
        _record(tmp_path, "a", confidence=0.50),
        _record(tmp_path, "b", confidence=0.10),
        _record(tmp_path, "c", confidence=0.30),
    ]

    picked = create_selector("low_confidence").select(
        records, options={"conf_below": 0.55, "max_samples": 2}
    )

    assert [c.inspection_id for c in picked] == ["b", "c"]


def test_low_confidence_ignores_records_without_predictions(tmp_path):
    picked = create_selector("low_confidence").select(
        [_record(tmp_path, "empty")], options={"conf_below": 0.55}
    )

    assert picked == []


# ---------------------------------------------------------------------------
# Review corrections


def test_review_correction_picks_operator_disagreements(tmp_path):
    records = [
        _record(tmp_path, "agreed", outcome="confirmed_ng"),
        _record(tmp_path, "overkill", outcome="false_reject"),
        _record(tmp_path, "escape", outcome="false_accept"),
        _record(tmp_path, "unreviewed", outcome=""),
    ]

    picked = create_selector("review_correction").select(records, options={})

    assert [c.inspection_id for c in picked] == ["escape", "overkill"]


def test_review_correction_ranks_escapes_above_overkills(tmp_path):
    records = [
        _record(tmp_path, "overkill", outcome="false_reject"),
        _record(tmp_path, "escape", outcome="false_accept"),
    ]

    picked = create_selector("review_correction").select(records, options={})

    assert picked[0].inspection_id == "escape"
    assert picked[0].score > picked[1].score


def test_review_correction_skips_undecidable_images(tmp_path):
    """The operator flow already refuses to train on these; so do we."""
    records = [_record(tmp_path, "unclear", outcome="undecidable")]

    assert create_selector("review_correction").select(records, options={}) == []


# ---------------------------------------------------------------------------
# Random sampling


def test_random_sample_is_reproducible_for_a_seed(tmp_path):
    records = [_record(tmp_path, f"insp-{i}", confidence=0.9) for i in range(50)]
    options = {"fraction": 0.2, "seed": 7}

    first = create_selector("random_sample").select(records, options=options)
    second = create_selector("random_sample").select(records, options=options)

    assert [c.inspection_id for c in first] == [c.inspection_id for c in second]
    assert len(first) == 10


def test_random_sample_respects_the_cap(tmp_path):
    records = [_record(tmp_path, f"insp-{i}", confidence=0.9) for i in range(50)]

    picked = create_selector("random_sample").select(
        records, options={"fraction": 0.9, "max_samples": 3, "seed": 1}
    )

    assert len(picked) == 3


def test_random_sample_scores_below_every_real_signal(tmp_path):
    """So a random pick never outranks a genuine reason when pools merge."""
    records = [_record(tmp_path, "insp-0", confidence=0.9)]

    picked = create_selector("random_sample").select(records, options={"seed": 1})

    assert picked[0].score == 0.0


# ---------------------------------------------------------------------------
# Running several selectors


def _settings(**enabled):
    return [
        SelectorSettings(name=name, enabled=True, options=options)
        for name, options in enabled.items()
    ]


def test_run_selectors_merges_a_sample_picked_twice(tmp_path):
    shared = _image(tmp_path, "shared.jpg")
    records = [
        _record(tmp_path, "shared", confidence=0.2, image=shared),
    ]

    picked = run_selectors(
        records,
        _settings(
            low_confidence={"conf_below": 0.55},
            random_sample={"fraction": 1.0, "seed": 1},
        ),
    )

    assert len(picked) == 1
    assert picked[0].selector == "low_confidence"  # the stronger signal wins
    assert "random_sample" in picked[0].also_selected_by


def test_a_failing_selector_does_not_lose_the_others(tmp_path):
    class Exploding:
        name = "exploding"

        def select(self, records, *, options, selected_at=None):
            raise ValueError("boom")

    register_selector("exploding", Exploding)
    try:
        picked = run_selectors(
            [_record(tmp_path, "unsure", confidence=0.2)],
            [
                SelectorSettings(name="exploding", enabled=True, options={}),
                SelectorSettings(
                    name="low_confidence", enabled=True, options={"conf_below": 0.55}
                ),
            ],
        )
    finally:
        unregister_selector("exploding")

    assert [c.inspection_id for c in picked] == ["unsure"]


def test_disabled_selectors_are_not_run(tmp_path):
    picked = run_selectors(
        [_record(tmp_path, "unsure", confidence=0.2)],
        [SelectorSettings(name="low_confidence", enabled=False, options={})],
    )

    assert picked == []


# ---------------------------------------------------------------------------
# The pool


def test_candidates_always_enter_needing_a_label(tmp_path):
    """Predictions are never promoted to ground truth."""
    pool = CandidatePool(tmp_path / "pool")
    candidate = build_candidate(
        _record(tmp_path, "insp-1", confidence=0.2),
        selector="low_confidence",
        reason="uncertain",
        score=0.3,
    )

    pool.add([candidate])

    stored = pool.load()
    assert len(stored) == 1
    assert stored[0].label_state == NEEDS_LABEL
    assert stored[0].label_path == ""


def test_the_image_is_copied_into_the_pool(tmp_path):
    """Production retention must not be able to empty the pool later."""
    pool = CandidatePool(tmp_path / "pool")
    source = _image(tmp_path, "insp-1.jpg", b"original-bytes")
    candidate = build_candidate(
        _record(tmp_path, "insp-1", confidence=0.2, image=source),
        selector="low_confidence",
        reason="uncertain",
        score=0.3,
    )
    pool.add([candidate])

    source.unlink()

    stored = pool.load()[0]
    assert Path(stored.image_path).is_file()
    assert Path(stored.image_path).read_bytes() == b"original-bytes"


def test_sample_identity_is_the_image_content_hash(tmp_path):
    same_a = _image(tmp_path, "a.jpg", b"identical")
    same_b = _image(tmp_path, "b.jpg", b"identical")

    assert sample_id_for_image(same_a) == sample_id_for_image(same_b)


def test_the_same_photo_selected_twice_is_one_candidate(tmp_path):
    pool = CandidatePool(tmp_path / "pool")
    source = _image(tmp_path, "insp-1.jpg", b"same-photo")
    record = _record(tmp_path, "insp-1", confidence=0.2, image=source)
    first = build_candidate(record, selector="low_confidence", reason="a", score=0.3)
    second = build_candidate(record, selector="random_sample", reason="b", score=0.0)

    pool.add([first])
    result = pool.add([second])

    assert len(pool.load()) == 1
    assert result.merged == (pool.load()[0].sample_id,)
    assert "random_sample" in pool.load()[0].also_selected_by


def test_reselecting_does_not_reset_human_label_state(tmp_path):
    pool = CandidatePool(tmp_path / "pool")
    source = _image(tmp_path, "insp-1.jpg", b"photo")
    record = _record(tmp_path, "insp-1", confidence=0.2, image=source)
    candidate = build_candidate(record, selector="low_confidence", reason="a", score=0.3)
    pool.add([candidate])
    sample_id = pool.load()[0].sample_id
    pool.update_label_state(sample_id, VERIFIED, label_path="labels/insp-1.txt")

    pool.add([build_candidate(record, selector="random_sample", reason="b", score=0.0)])

    assert pool.load()[0].label_state == VERIFIED
    assert pool.load()[0].label_path == "labels/insp-1.txt"


def test_a_candidate_without_an_image_is_not_pooled(tmp_path):
    """Nothing to label means nothing to pool."""
    pool = CandidatePool(tmp_path / "pool")
    source = _image(tmp_path, "gone.jpg")
    candidate = build_candidate(
        _record(tmp_path, "gone", confidence=0.2, image=source),
        selector="low_confidence",
        reason="uncertain",
        score=0.3,
    )
    source.unlink()

    result = pool.add([candidate])

    assert result.added == ()
    assert result.skipped_missing_image != ()
    assert pool.load() == ()


def test_build_candidate_returns_none_when_the_image_is_missing(tmp_path):
    source = _image(tmp_path, "gone.jpg")
    record = _record(tmp_path, "gone", confidence=0.2, image=source)
    source.unlink()

    assert build_candidate(record, selector="s", reason="r", score=1.0) is None


def test_image_quality_is_carried_into_the_pool(tmp_path):
    pool = CandidatePool(tmp_path / "pool")
    record = replace(
        _record(tmp_path, "insp-1", confidence=0.2),
        image_quality=ImageQuality(brightness=120.0, saturation=44.0, blur_score=88.0),
    )
    pool.add([build_candidate(record, selector="s", reason="r", score=1.0)])

    assert pool.load()[0].image_quality == {
        "brightness": 120.0,
        "saturation": 44.0,
        "blur_score": 88.0,
    }


def test_label_state_transitions_are_validated(tmp_path):
    pool = CandidatePool(tmp_path / "pool")
    pool.add(
        [
            build_candidate(
                _record(tmp_path, "insp-1", confidence=0.2),
                selector="s",
                reason="r",
                score=1.0,
            )
        ]
    )
    sample_id = pool.load()[0].sample_id

    with pytest.raises(CandidatePoolError, match="Unknown label state"):
        pool.update_label_state(sample_id, "SOMETHING_ELSE")
    with pytest.raises(CandidatePoolError, match="Unknown candidate"):
        pool.update_label_state("no-such-sample", VERIFIED)


def test_statistics_report_states_and_selectors(tmp_path):
    pool = CandidatePool(tmp_path / "pool")
    pool.add(
        [
            build_candidate(
                _record(tmp_path, f"insp-{i}", confidence=0.2),
                selector="low_confidence",
                reason="r",
                score=1.0,
            )
            for i in range(3)
        ]
    )
    pool.update_label_state(pool.load()[0].sample_id, VERIFIED)
    pool.update_label_state(pool.load()[1].sample_id, DISCARDED)

    stats = pool.statistics()

    assert stats["total"] == 3
    assert stats["by_label_state"][VERIFIED] == 1
    assert stats["by_label_state"][DISCARDED] == 1
    assert stats["by_label_state"][NEEDS_LABEL] == 1
    assert stats["by_selector"]["low_confidence"] == 3


def test_a_corrupt_pool_line_is_skipped_not_fatal(tmp_path):
    pool = CandidatePool(tmp_path / "pool")
    pool.add(
        [
            build_candidate(
                _record(tmp_path, "insp-1", confidence=0.2),
                selector="s",
                reason="r",
                score=1.0,
            )
        ]
    )
    with pool.manifest_path.open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")

    assert len(pool.load()) == 1


def test_the_manifest_is_valid_jsonl(tmp_path):
    pool = CandidatePool(tmp_path / "pool")
    pool.add(
        [
            build_candidate(
                _record(tmp_path, f"insp-{i}", confidence=0.2),
                selector="s",
                reason="r",
                score=1.0,
            )
            for i in range(3)
        ]
    )

    lines = pool.manifest_path.read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 3
    for line in lines:
        assert set(json.loads(line)) >= {
            "sample_id",
            "selector",
            "selected_reason",
            "score",
            "source_model",
            "selected_at",
            "label_state",
        }


def test_an_empty_pool_reads_as_empty(tmp_path):
    assert CandidatePool(tmp_path / "never-written").load() == ()
