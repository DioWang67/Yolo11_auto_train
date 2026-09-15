"""Reading paid-for verdicts back out of an interrupted run's console log.

Every line quoted here is copied verbatim from ``runs/vision_full.log``, the
log of the run that was interrupted at 81 of 213 on 2026-09-14. That matters:
the reason a recovery parser is worth testing at all is that its input is a
log format nobody designed as a data format, and a fixture invented to match
the parser would prove nothing about the file it has to read.

The load-bearing test is the one for a reason truncated mid-bracket. The
logger clipped reasons to 80 characters, which routinely severs a ``(`` from
its ``)``, and a parser that matched brackets would silently drop or mangle
those records.
"""

from __future__ import annotations

from scripts.vision_review_recover import recover

# Verbatim from runs/vision_full.log.
HEADER = (
    "INFO vision_review_dryrun: 1/213 "
    "yolo_Cable1_A_091307_708368_c632b447c5bd: 2 crop(s) + whole frame"
)
TRUNCATED_MID_BRACKET = (
    "INFO vision_review_dryrun:    -> REJECT Orange (Crop 1 is clearly an "
    "orange wire (matching the orange wire at the P+ pad in the )"
)
CROPS_ONLY_HEADER = (
    "INFO vision_review_dryrun: 2/213 "
    "yolo_Cable1_A_092204_064038_676d56e550d4: 1 crop(s)"
)
RETRY_LINE = (
    "INFO vision_review_dryrun:    -> RETRY Red (Crop 1 is unambiguously a "
    "red wire (detector 0.97 and colour score 0.68 agree, a)"
)
PARSE_FAILURE = (
    "ERROR vision_review_dryrun: yolo_Cable1_A_133309: yolo_Cable1_A_133309: "
    "no usable answer after 3 attempt(s). Reply JSON did not parse: "
    "Expecting ',' delimiter: line 1 column 319 (char 318)"
)


def test_a_reason_truncated_mid_bracket_is_recovered_whole() -> None:
    (record,) = recover([HEADER, TRUNCATED_MID_BRACKET])

    assert record["verdict"] == "REJECT"
    assert record["class_name"] == "Orange"
    assert record["reason"] == (
        "Crop 1 is clearly an orange wire (matching the orange wire at the "
        "P+ pad in the "
    )
    assert record["reason_truncated"] is True


def test_the_header_supplies_what_the_request_carried() -> None:
    (record,) = recover([HEADER, TRUNCATED_MID_BRACKET])

    assert record["sample_id"] == "yolo_Cable1_A_091307_708368_c632b447c5bd"
    assert record["crops_sent"] == 2
    assert record["whole_frame_sent"] is True


def test_a_crops_only_sample_is_not_read_as_carrying_the_frame() -> None:
    (record,) = recover([CROPS_ONLY_HEADER, RETRY_LINE])

    assert record["whole_frame_sent"] is False
    assert record["verdict"] == "RETRY"


def test_what_the_log_never_held_is_null_rather_than_guessed() -> None:
    (record,) = recover([HEADER, TRUNCATED_MID_BRACKET])

    assert record["confidence"] is None
    assert record["next_action"] is None
    assert record["attempts"] is None
    assert record["usage"] == {}
    assert record["fidelity"] == "recovered_from_log"


def test_a_failure_is_recovered_as_unfinished_work() -> None:
    """No verdict, so the resume asks this sample again."""
    (record,) = recover([HEADER, PARSE_FAILURE])

    assert record["verdict"] is None
    assert "Reply JSON did not parse" in record["error"]
    assert record["sample_id"] == "yolo_Cable1_A_091307_708368_c632b447c5bd"


def test_a_sample_cut_off_mid_request_yields_no_record() -> None:
    """Sample 81 was sent and never answered; it must not look answered."""
    records = recover([HEADER, TRUNCATED_MID_BRACKET, CROPS_ONLY_HEADER])

    assert len(records) == 1
    assert records[0]["sample_id"] == "yolo_Cable1_A_091307_708368_c632b447c5bd"


def test_the_whole_exchange_recovers_in_order() -> None:
    records = recover(
        [HEADER, TRUNCATED_MID_BRACKET, CROPS_ONLY_HEADER, RETRY_LINE]
    )

    assert [r["verdict"] for r in records] == ["REJECT", "RETRY"]
    assert [r["sample_id"] for r in records] == [
        "yolo_Cable1_A_091307_708368_c632b447c5bd",
        "yolo_Cable1_A_092204_064038_676d56e550d4",
    ]


def test_preamble_lines_are_ignored() -> None:
    assert (
        recover(
            [
                "INFO vision_review_dryrun: 213 NEEDS_REVIEW samples available",
                "INFO vision_review_dryrun: Endpoint configured from A / B",
            ]
        )
        == []
    )
