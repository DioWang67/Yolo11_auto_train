"""Golden *candidates*, and the line between a candidate and a golden set.

The single most important property here is the one in the first test: nothing
this module produces is a golden set. Everything else --- the grouping, the
thinning, the statistics --- is in service of a person making that call.
"""

from __future__ import annotations

import csv
import json

import pytest

from picture_tool.autotrain import golden
from picture_tool.autotrain.class_schema import normalize_class_names
from picture_tool.autotrain.golden_candidates import (
    HARD_CASE,
    NEEDS_ANNOTATION,
    READY_TO_REVIEW,
    REASON_CORRECTION,
    REASON_EXTRA_DETECTION,
    REASON_LOW_CONFIDENCE,
    REASON_MISSED_DETECTION,
    REPRESENTATIVE,
    Candidate,
    DetectionEvidence,
    GoldenCandidateError,
    build_candidates,
    classify,
    group_duplicates,
    read_group_assignments,
    read_review_manifests,
    summarise,
    thin_by_group,
    write_report,
)

SCHEMA = normalize_class_names(
    ["Black", "Green", "Orange", "Red", "Yellow"], source="test"
)

#: Cable1/A photographs six objects across five classes: Black appears twice.
EXPECTED_BOXES = 6


def _detection(name, verified=None, confidence=0.95):
    return {
        "class": name,
        "class_id": SCHEMA.names.index(name),
        "verified_class": verified or name,
        "confidence": confidence,
        "bbox": [10, 10, 30, 40],
    }


def _good_row():
    return [_detection(n) for n in ("Black", "Black", "Green", "Orange", "Red", "Yellow")]


def _evidence(detections, sample_id="s0", tmp_path=None):
    image = (tmp_path / f"{sample_id}.jpg") if tmp_path else None
    if image is not None:
        image.write_bytes(sample_id.encode())
    return DetectionEvidence(
        sample_id=sample_id,
        image_path=image or f"{sample_id}.jpg",
        timestamp="2026-09-01T00:00:00",
        model_version="1.0.6",
        camera_id="cam0",
        status="PASS",
        detections=tuple(detections),
    )


def _write_image(path, payload=b"pixels"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


# ---------------------------------------------------------------------------
# A candidate is not a golden set


def test_a_candidate_pass_registers_nothing(tmp_path):
    """The load-bearing test. Candidates are a reading list, not evidence."""
    out = tmp_path / "report"
    candidates = [
        Candidate(
            sample_id="s0",
            image_path=str(_write_image(tmp_path / "img" / "s0.jpg")),
            status=NEEDS_ANNOTATION,
            group=HARD_CASE,
            source="production:PASS",
        )
    ]

    write_report(candidates, summarise(candidates, SCHEMA), out)

    assert not list(out.rglob(golden.MANIFEST_FILENAME))
    # And the configured golden set is still simply absent.
    assert golden.resolve("").status == golden.NOT_CONFIGURED


def test_the_report_says_in_writing_that_nothing_is_golden(tmp_path):
    out = tmp_path / "report"
    candidates = [
        Candidate(
            sample_id="s0",
            image_path=str(_write_image(tmp_path / "img" / "s0.jpg")),
            status=NEEDS_ANNOTATION,
            group=HARD_CASE,
            source="production:PASS",
        )
    ]

    written = write_report(candidates, summarise(candidates, SCHEMA), out)

    text = written["readme"].read_text(encoding="utf-8")
    assert "None of this is a golden set yet" in text
    assert "registered_by" in text or "--registered-by" in text


def test_an_unapproved_candidate_directory_is_not_an_evaluation_set(tmp_path):
    """Pointing the config at a candidate directory must not pass.

    The directory has images in it and looks the part; what it does not have
    is a manifest, and therefore anybody's name against it.
    """
    candidates_dir = tmp_path / "candidates"
    _write_image(candidates_dir / "a.jpg")

    status = golden.resolve(str(candidates_dir))

    assert not status.is_passing
    assert status.status == golden.MISSING


# ---------------------------------------------------------------------------
# Classification


def test_a_clean_row_is_representative(tmp_path):
    group, reasons, _ = classify(
        _evidence(_good_row(), tmp_path=tmp_path),
        expected_boxes=EXPECTED_BOXES,
        low_confidence_below=0.55,
    )

    assert group == REPRESENTATIVE
    assert reasons == ()


def test_a_human_correction_makes_it_a_hard_case(tmp_path):
    detections = _good_row()
    detections[3] = _detection("Red", verified="Orange")

    group, reasons, detail = classify(
        _evidence(detections, tmp_path=tmp_path),
        expected_boxes=EXPECTED_BOXES,
        low_confidence_below=0.55,
    )

    assert group == HARD_CASE
    assert REASON_CORRECTION in reasons
    assert "Red->Orange" in detail


def test_a_low_confidence_detection_makes_it_a_hard_case(tmp_path):
    detections = _good_row()
    detections[2] = _detection("Green", confidence=0.31)

    group, reasons, detail = classify(
        _evidence(detections, tmp_path=tmp_path),
        expected_boxes=EXPECTED_BOXES,
        low_confidence_below=0.55,
    )

    assert group == HARD_CASE
    assert REASON_LOW_CONFIDENCE in reasons
    assert "0.310" in detail


def test_a_missed_object_is_a_hard_case(tmp_path):
    """The kind production records worst, and the kind golden most needs."""
    group, reasons, detail = classify(
        _evidence(_good_row()[:4], tmp_path=tmp_path),
        expected_boxes=EXPECTED_BOXES,
        low_confidence_below=0.55,
    )

    assert group == HARD_CASE
    assert REASON_MISSED_DETECTION in reasons
    assert "4 of 6" in detail


def test_an_extra_box_is_a_hard_case(tmp_path):
    group, reasons, _ = classify(
        _evidence(_good_row() + [_detection("Red")], tmp_path=tmp_path),
        expected_boxes=EXPECTED_BOXES,
        low_confidence_below=0.55,
    )

    assert group == HARD_CASE
    assert REASON_EXTRA_DETECTION in reasons


def test_six_boxes_over_five_classes_is_not_treated_as_an_extra(tmp_path):
    """Black twice is the station, not a duplicate detection."""
    group, reasons, _ = classify(
        _evidence(_good_row(), tmp_path=tmp_path),
        expected_boxes=EXPECTED_BOXES,
        low_confidence_below=0.55,
    )

    assert reasons == ()
    assert group == REPRESENTATIVE


# ---------------------------------------------------------------------------
# Duplicates


def test_byte_identical_images_group_together(tmp_path):
    a = _write_image(tmp_path / "a.jpg", b"same")
    b = _write_image(tmp_path / "b.jpg", b"same")
    c = _write_image(tmp_path / "c.jpg", b"different")

    groups = group_duplicates([a, b, c])

    assert groups[a] == groups[b]
    assert groups[a].startswith("sha:")
    assert groups[c] != groups[a]


def test_exact_duplicates_keep_exactly_one_however_wide_the_group_limit(tmp_path):
    """Two copies of one file are one sample, not two."""
    candidates = [
        Candidate(
            sample_id=f"s{i}",
            image_path=str(tmp_path / f"s{i}.jpg"),
            status=NEEDS_ANNOTATION,
            group=REPRESENTATIVE,
            source="production:PASS",
            duplicate_group="sha:abc123",
        )
        for i in range(4)
    ]

    kept, dropped = thin_by_group(candidates, per_group=3)

    assert len(kept) == 1
    assert dropped == 3


def test_perceptual_groups_may_keep_several(tmp_path):
    """Similar-looking photographs are still different photographs."""
    candidates = [
        Candidate(
            sample_id=f"s{i}",
            image_path=str(tmp_path / f"s{i}.jpg"),
            status=NEEDS_ANNOTATION,
            group=REPRESENTATIVE,
            source="production:PASS",
            duplicate_group="phash:ff00",
        )
        for i in range(5)
    ]

    kept, dropped = thin_by_group(candidates, per_group=2)

    assert len(kept) == 2
    assert dropped == 3


def test_thinning_keeps_ground_truth_over_production_evidence(tmp_path):
    """A handoff job copies its production image, so both arrive.

    Keeping the "more interesting" production row would discard the only copy
    with human-drawn boxes, which is the thing that makes it usable at all.
    """
    labelled = Candidate(
        sample_id="s0",
        image_path=str(tmp_path / "labelled.jpg"),
        status=READY_TO_REVIEW,
        group=REPRESENTATIVE,
        source="handoff:job1",
        duplicate_group="sha:abc123",
        label_path=str(tmp_path / "s0.txt"),
    )
    production = Candidate(
        sample_id="s0",
        image_path=str(tmp_path / "production.jpg"),
        status=NEEDS_ANNOTATION,
        group=HARD_CASE,
        source="production:FAIL",
        min_confidence=0.2,
        duplicate_group="sha:abc123",
    )

    kept, _ = thin_by_group([production, labelled], per_group=1)

    assert [c.status for c in kept] == [READY_TO_REVIEW]
    assert kept[0].label_path


# ---------------------------------------------------------------------------
# Statistics a reviewer needs


def test_red_orange_confusions_are_counted(tmp_path):
    """The station's dominant confusion; the reviewer has to be able to see it."""
    candidates = [
        Candidate(
            sample_id=f"s{i}",
            image_path=str(tmp_path / f"s{i}.jpg"),
            status=NEEDS_ANNOTATION,
            group=HARD_CASE,
            source="production:FAIL",
            reasons=(REASON_CORRECTION,),
            corrections=("Red->Orange",),
        )
        for i in range(3)
    ] + [
        Candidate(
            sample_id="s9",
            image_path=str(tmp_path / "s9.jpg"),
            status=NEEDS_ANNOTATION,
            group=HARD_CASE,
            source="production:FAIL",
            reasons=(REASON_CORRECTION,),
            corrections=("Orange->Red",),
        )
    ]

    summary = summarise(candidates, SCHEMA)

    assert summary["confusions"]["Red->Orange"] == 3
    assert summary["confusions"]["Orange->Red"] == 1


def test_a_hard_case_kind_with_no_examples_is_reported_as_a_gap(tmp_path):
    """An absent failure mode is a finding, not something to invent."""
    candidates = [
        Candidate(
            sample_id="s0",
            image_path=str(tmp_path / "s0.jpg"),
            status=NEEDS_ANNOTATION,
            group=HARD_CASE,
            source="production:FAIL",
            reasons=(REASON_CORRECTION,),
        )
    ]

    summary = summarise(candidates, SCHEMA)

    assert REASON_CORRECTION not in summary["coverage_gaps"]
    assert REASON_MISSED_DETECTION in summary["coverage_gaps"]
    assert summary["hard_case_reasons"][REASON_MISSED_DETECTION] == 0


def test_the_summary_carries_the_class_contract(tmp_path):
    summary = summarise([], SCHEMA)

    assert summary["class_schema"]["names"] == list(SCHEMA.names)
    assert summary["class_schema"]["schema_hash"] == SCHEMA.schema_hash


# ---------------------------------------------------------------------------
# Reading production manifests


def test_rows_whose_image_is_gone_are_skipped(tmp_path):
    """Production retention deletes images; a row pointing at nothing is not a
    candidate, because nobody can review it."""
    manifest = tmp_path / "manifest.csv"
    live = _write_image(tmp_path / "live.jpg")
    with open(manifest, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["product", "area", "original_path", "detections_json"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "product": "Cable1",
                "area": "A",
                "original_path": str(live),
                "detections_json": json.dumps(_good_row()),
            }
        )
        writer.writerow(
            {
                "product": "Cable1",
                "area": "A",
                "original_path": str(tmp_path / "deleted.jpg"),
                "detections_json": json.dumps(_good_row()),
            }
        )

    evidence = read_review_manifests([manifest], product="Cable1", area="A")

    assert [e.sample_id for e in evidence] == ["live"]


def test_another_station_is_not_read(tmp_path):
    manifest = tmp_path / "manifest.csv"
    other = _write_image(tmp_path / "other.jpg")
    with open(manifest, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["product", "area", "original_path"])
        writer.writeheader()
        writer.writerow({"product": "PCBA1", "area": "C", "original_path": str(other)})

    assert read_review_manifests([manifest], product="Cable1", area="A") == []


def test_a_pass_reads_production_and_labelled_data_into_separate_statuses(tmp_path):
    manifest = tmp_path / "manifest.csv"
    production = _write_image(tmp_path / "prod" / "p0.jpg", b"production")
    with open(manifest, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["product", "area", "original_path", "detections_json"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "product": "Cable1",
                "area": "A",
                "original_path": str(production),
                "detections_json": json.dumps(_good_row()[:3]),
            }
        )

    raw = tmp_path / "job" / "raw"
    _write_image(raw / "images" / "L0.jpg", b"labelled")
    (raw / "labels").mkdir(parents=True, exist_ok=True)
    (raw / "labels" / "L0.txt").write_text("0 .5 .5 .2 .2\n3 .4 .4 .1 .1\n", "utf-8")

    candidates, summary = build_candidates(
        schema=SCHEMA,
        review_manifests=[manifest],
        labelled_roots=[raw],
        expected_boxes=EXPECTED_BOXES,
        measure_quality=False,
    )

    by_status = {c.status: c for c in candidates}
    assert set(by_status) == {NEEDS_ANNOTATION, READY_TO_REVIEW}
    assert by_status[READY_TO_REVIEW].class_counts == {"Black": 1, "Red": 1}
    assert by_status[READY_TO_REVIEW].label_path
    # The production row had three of six objects, so it is a hard case and
    # carries no ground truth.
    assert REASON_MISSED_DETECTION in by_status[NEEDS_ANNOTATION].reasons
    assert not by_status[NEEDS_ANNOTATION].label_path
    assert summary["by_status"][READY_TO_REVIEW] == 1


# ---------------------------------------------------------------------------
# The report itself


def test_the_csv_lists_hard_cases_first(tmp_path):
    candidates = [
        Candidate(
            sample_id="easy",
            image_path=str(_write_image(tmp_path / "img" / "easy.jpg", b"1")),
            status=NEEDS_ANNOTATION,
            group=REPRESENTATIVE,
            source="production:PASS",
        ),
        Candidate(
            sample_id="tricky",
            image_path=str(_write_image(tmp_path / "img" / "tricky.jpg", b"2")),
            status=NEEDS_ANNOTATION,
            group=HARD_CASE,
            source="production:FAIL",
            reasons=(REASON_CORRECTION,),
        ),
    ]

    written = write_report(candidates, summarise(candidates, SCHEMA), tmp_path / "out")

    with open(written["csv"], encoding="utf-8-sig", newline="") as handle:
        order = [row["sample_id"] for row in csv.DictReader(handle)]
    assert order == ["tricky", "easy"]


def test_the_report_writes_only_into_its_own_directory(tmp_path):
    source = _write_image(tmp_path / "img" / "s0.jpg")
    before = source.read_bytes()
    candidates = [
        Candidate(
            sample_id="s0",
            image_path=str(source),
            status=NEEDS_ANNOTATION,
            group=HARD_CASE,
            source="production:FAIL",
        )
    ]

    out = tmp_path / "out"
    write_report(candidates, summarise(candidates, SCHEMA), out)

    assert source.read_bytes() == before
    assert {p.name for p in out.iterdir()} == {
        "candidates.csv",
        "summary.json",
        "REVIEW.md",
    }


# ---------------------------------------------------------------------------
# Protection: an approved golden set stays out of training


def test_a_registered_golden_set_records_its_class_contract(tmp_path):
    root = tmp_path / "golden"
    _write_image(root / "g0.jpg")

    dataset = golden.register(root, registered_by="engineer", class_schema=SCHEMA)

    assert dataset.class_schema is not None
    assert dataset.class_schema.schema_hash == SCHEMA.schema_hash
    manifest = json.loads(dataset.manifest_path.read_text(encoding="utf-8"))
    assert manifest["class_schema"]["names"] == list(SCHEMA.names)


def test_registration_without_a_name_is_refused(tmp_path):
    """The approval marker. A set nobody signed is not approved."""
    root = tmp_path / "golden"
    _write_image(root / "g0.jpg")

    with pytest.raises(golden.GoldenDatasetError, match="registered_by"):
        golden.register(root, registered_by="   ", class_schema=SCHEMA)


def test_a_golden_sample_appearing_in_training_is_contamination(tmp_path):
    """Evaluating on trained data measures memory, not generalisation.

    Identity is the sha256 of the image bytes on both sides, so renaming the
    file on its way into training changes nothing --- which is how
    contamination actually happens.
    """
    root = tmp_path / "golden"
    _write_image(root / "g0.jpg", b"a golden picture")
    dataset = golden.register(root, registered_by="engineer", class_schema=SCHEMA)
    (sample_id,) = dataset.sample_ids

    status = golden.resolve(
        str(root),
        dataset.manifest_sha256,
        training_sample_ids=[sample_id],
    )

    assert status.status == golden.CONTAMINATED
    assert not status.is_passing
    assert sample_id in status.contaminated_samples


def test_a_clean_training_set_leaves_the_golden_set_passing(tmp_path):
    root = tmp_path / "golden"
    _write_image(root / "g0.jpg", b"a golden picture")
    dataset = golden.register(root, registered_by="engineer", class_schema=SCHEMA)

    status = golden.resolve(
        str(root), dataset.manifest_sha256, training_sample_ids=["something-else"]
    )

    assert status.status == golden.OK
    assert status.is_passing


def test_an_edited_golden_image_stops_the_set_passing(tmp_path):
    """Immutability that is checkable, not merely intended."""
    root = tmp_path / "golden"
    image = _write_image(root / "g0.jpg")
    dataset = golden.register(root, registered_by="engineer", class_schema=SCHEMA)

    image.write_bytes(b"augmented")

    status = golden.resolve(str(root), dataset.manifest_sha256)
    assert not status.is_passing


# ---------------------------------------------------------------------------
# Source lineage and eligibility


def test_an_augmented_image_traces_back_to_its_original():
    """Reuses the splitter's own convention rather than inventing a second."""
    from picture_tool.autotrain.golden_candidates import source_image_id

    assert source_image_id("yolo_Cable1_A_142300_aug_3.png") == "yolo_Cable1_A_142300"
    assert source_image_id("yolo_Cable1_A_142300.jpg") == "yolo_Cable1_A_142300"


def test_an_unnameable_image_reports_unknown_lineage():
    from picture_tool.autotrain.golden_candidates import UNKNOWN_SOURCE, source_image_id

    assert source_image_id("") == UNKNOWN_SOURCE


def _eligible_candidate(tmp_path, **overrides):
    from picture_tool.autotrain.golden_candidates import ELIGIBLE  # noqa: F401

    base = dict(
        sample_id="g0",
        image_path=str(tmp_path / "g0.jpg"),
        label_path=str(tmp_path / "g0.txt"),
        status=READY_TO_REVIEW,
        group=REPRESENTATIVE,
        source="handoff:job1",
        class_counts={"Black": 2, "Green": 1, "Orange": 1, "Red": 1, "Yellow": 1},
        source_image_id="yolo_Cable1_A_1",
        image_sha256="deadbeef",
    )
    base.update(overrides)
    return Candidate(**base)


def _assess(candidate, **kwargs):
    from picture_tool.autotrain.golden_candidates import assess_eligibility

    return assess_eligibility(
        candidate, schema=SCHEMA, expected_boxes=EXPECTED_BOXES, **kwargs
    )


def test_a_complete_traceable_uncontaminated_label_is_eligible(tmp_path):
    from picture_tool.autotrain.golden_candidates import ELIGIBLE

    assert _assess(_eligible_candidate(tmp_path)) == (True, ELIGIBLE)


def test_the_exact_image_having_been_trained_on_is_contamination(tmp_path):
    from picture_tool.autotrain.golden_candidates import TRAINING_CONTAMINATION

    eligible, reason = _assess(
        _eligible_candidate(tmp_path), trained_sha256=frozenset({"deadbeef"})
    )

    assert (eligible, reason) == (False, TRAINING_CONTAMINATION)


def test_an_augmentation_of_the_image_having_been_trained_on_is_contamination(tmp_path):
    """The case a byte hash cannot see, and the one that actually happens.

    Training augments before it splits, so what reaches the model is a flipped,
    brightened derivative whose bytes match nothing. The source id is what
    connects them.
    """
    from picture_tool.autotrain.golden_candidates import TRAINING_CONTAMINATION

    eligible, reason = _assess(
        _eligible_candidate(tmp_path),
        trained_source_ids=frozenset({"yolo_Cable1_A_1"}),
        trained_sha256=frozenset({"a-completely-different-hash"}),
    )

    assert (eligible, reason) == (False, TRAINING_CONTAMINATION)


def test_a_production_image_without_boxes_is_not_eligible(tmp_path):
    from picture_tool.autotrain.golden_candidates import NEEDS_LABEL

    candidate = _eligible_candidate(
        tmp_path, status=NEEDS_ANNOTATION, label_path="", class_counts={}
    )

    assert _assess(candidate) == (False, NEEDS_LABEL)


def test_a_human_correction_alone_does_not_make_ground_truth(tmp_path):
    """A corrected prediction says one box was wrong, not that all are right.

    The row carries corrections and a verified class per detected box, and
    still has no statement about what the model failed to detect.
    """
    from picture_tool.autotrain.golden_candidates import NEEDS_LABEL

    candidate = _eligible_candidate(
        tmp_path,
        status=NEEDS_ANNOTATION,
        label_path="",
        reasons=(REASON_CORRECTION,),
        corrections=("Red->Orange",),
        class_counts={"Black": 2, "Green": 1, "Orange": 1, "Red": 1, "Yellow": 1},
    )

    assert _assess(candidate) == (False, NEEDS_LABEL)


def test_a_box_count_that_disagrees_with_the_station_is_incomplete(tmp_path):
    from picture_tool.autotrain.golden_candidates import LABEL_INCOMPLETE

    candidate = _eligible_candidate(
        tmp_path,
        class_counts={"Black": 1, "Green": 1, "Orange": 1, "Red": 1, "Yellow": 1},
    )

    assert _assess(candidate) == (False, LABEL_INCOMPLETE)


def test_copies_that_disagree_about_their_labels_are_incomplete(tmp_path):
    """This station's own data contains exactly one such image."""
    from picture_tool.autotrain.golden_candidates import LABEL_INCOMPLETE

    eligible, reason = _assess(
        _eligible_candidate(tmp_path),
        conflicting_source_ids=frozenset({"yolo_Cable1_A_1"}),
    )

    assert (eligible, reason) == (False, LABEL_INCOMPLETE)


def test_a_class_the_schema_cannot_explain_is_a_mismatch(tmp_path):
    from picture_tool.autotrain.golden_candidates import SCHEMA_MISMATCH

    candidate = _eligible_candidate(
        tmp_path, class_counts={"Black": 2, "Green": 1, "Orange": 1, "Red": 1, "Blue": 1}
    )

    assert _assess(candidate) == (False, SCHEMA_MISMATCH)


def test_an_untraceable_image_is_not_quietly_accepted(tmp_path):
    """Unknown lineage cannot be cleared of leakage, so it is not eligible."""
    from picture_tool.autotrain.golden_candidates import (
        SOURCE_UNKNOWN,
        UNKNOWN_SOURCE,
    )

    candidate = _eligible_candidate(tmp_path, source_image_id=UNKNOWN_SOURCE)

    assert _assess(candidate) == (False, SOURCE_UNKNOWN)


def test_unreadable_provenance_is_refused_rather_than_treated_as_empty(tmp_path):
    """An empty answer here would clear every candidate of contamination."""
    from picture_tool.autotrain.golden_candidates import (
        GoldenCandidateError,
        read_training_provenance,
    )

    with pytest.raises(GoldenCandidateError, match="Refusing to assume"):
        read_training_provenance(tmp_path / "missing.json")


def test_provenance_yields_both_source_ids_and_hashes(tmp_path):
    from picture_tool.autotrain.golden_candidates import read_training_provenance

    path = tmp_path / "training_provenance.json"
    path.write_text(
        json.dumps(
            {
                "images": [
                    {"file_name": "cap_1_aug_0.png", "sha256": "aaa"},
                    {"file_name": "cap_1_aug_1.png", "sha256": "bbb"},
                    {"file_name": "cap_2.jpg", "sha256": "ccc"},
                ]
            }
        ),
        encoding="utf-8",
    )

    sources, hashes = read_training_provenance(path)

    assert sources == {"cap_1", "cap_2"}
    assert hashes == {"aaa", "bbb", "ccc"}


# ---------------------------------------------------------------------------
# Reading the split back out, for `golden register --groups`


def _report_with(tmp_path, candidates):
    out = tmp_path / "report"
    write_report(candidates, summarise(candidates, SCHEMA), out)
    return out


def test_the_split_is_read_back_keyed_by_image_content_hash(tmp_path):
    """The load-bearing detail: a row's sample_id is its *filename stem*,
    while a golden set identifies a sample by the sha256 of its bytes.
    Joining on sample_id would match nothing, and would do it silently."""
    candidates = [
        Candidate(
            sample_id="s0",
            image_path=str(_write_image(tmp_path / "img" / "s0.jpg")),
            status=NEEDS_ANNOTATION,
            group=HARD_CASE,
            source="production:PASS",
            image_sha256="a" * 64,
        ),
        Candidate(
            sample_id="s1",
            image_path=str(_write_image(tmp_path / "img" / "s1.jpg")),
            status=READY_TO_REVIEW,
            group=REPRESENTATIVE,
            source="handoff:job",
            image_sha256="b" * 64,
        ),
    ]

    assignments = read_group_assignments(_report_with(tmp_path, candidates))

    assert assignments == {"a" * 64: HARD_CASE, "b" * 64: REPRESENTATIVE}
    assert "s0" not in assignments


def test_either_the_report_directory_or_the_csv_is_accepted(tmp_path):
    candidates = [
        Candidate(
            sample_id="s0",
            image_path=str(_write_image(tmp_path / "img" / "s0.jpg")),
            status=NEEDS_ANNOTATION,
            group=HARD_CASE,
            source="production:PASS",
            image_sha256="a" * 64,
        )
    ]
    out = _report_with(tmp_path, candidates)

    assert read_group_assignments(out) == read_group_assignments(
        out / "candidates.csv"
    )


def test_rows_with_no_recorded_hash_are_dropped_not_guessed(tmp_path):
    candidates = [
        Candidate(
            sample_id="traceable",
            image_path=str(_write_image(tmp_path / "img" / "a.jpg")),
            status=NEEDS_ANNOTATION,
            group=HARD_CASE,
            source="production:PASS",
            image_sha256="a" * 64,
        ),
        Candidate(
            sample_id="untraceable",
            image_path=str(_write_image(tmp_path / "img" / "b.jpg")),
            status=NEEDS_ANNOTATION,
            group=REPRESENTATIVE,
            source="production:PASS",
            image_sha256="",
        ),
    ]

    assignments = read_group_assignments(_report_with(tmp_path, candidates))

    assert assignments == {"a" * 64: HARD_CASE}


def test_one_image_in_two_groups_is_refused(tmp_path):
    """Concatenated reports would make the split arbitrary."""
    out = tmp_path / "report"
    out.mkdir()
    with open(out / "candidates.csv", "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "group", "image_sha256"])
        writer.writerow(["s0", HARD_CASE, "a" * 64])
        writer.writerow(["s0-again", REPRESENTATIVE, "a" * 64])

    with pytest.raises(GoldenCandidateError, match="more than one group"):
        read_group_assignments(out)


def test_a_csv_that_is_not_a_candidate_report_is_refused(tmp_path):
    out = tmp_path / "report"
    out.mkdir()
    (out / "candidates.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(GoldenCandidateError, match="no image_sha256 column"):
        read_group_assignments(out)


def test_a_missing_report_is_refused(tmp_path):
    with pytest.raises(GoldenCandidateError, match="not found"):
        read_group_assignments(tmp_path / "nope")


def test_a_report_with_no_usable_rows_is_refused(tmp_path):
    out = tmp_path / "report"
    out.mkdir()
    with open(out / "candidates.csv", "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "group", "image_sha256"])
        writer.writerow(["s0", "", ""])

    with pytest.raises(GoldenCandidateError, match="no usable group assignments"):
        read_group_assignments(out)


def test_the_split_read_back_registers_a_real_set(tmp_path):
    """End to end: the group a reviewer saw is the group in the manifest."""
    image = _write_image(tmp_path / "img" / "s0.jpg")
    import hashlib

    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    candidates = [
        Candidate(
            sample_id="s0",
            image_path=str(image),
            status=READY_TO_REVIEW,
            group=HARD_CASE,
            source="handoff:job",
            image_sha256=digest,
        )
    ]
    out = _report_with(tmp_path, candidates)

    chosen = tmp_path / "golden"
    chosen.mkdir()
    (chosen / "s0.jpg").write_bytes(image.read_bytes())
    dataset = golden.register(
        chosen,
        class_schema=SCHEMA,
        registered_by="engineer",
        groups=read_group_assignments(out),
    )

    assert dataset.group_counts() == {HARD_CASE: 1}
