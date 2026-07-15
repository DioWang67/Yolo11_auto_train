from picture_tool.gui.operator_workflow_panel import (
    build_operator_workflow_presentation,
)


def test_waiting_annotation_presentation_includes_pending_count() -> None:
    presentation = build_operator_workflow_presentation(
        "waiting_annotation", pending_count=3
    )

    assert presentation.step_index == 1
    assert presentation.title == "需要補齊 3 張標註"
    assert presentation.progress == 10


def test_operator_workflow_advances_through_quality_gate_and_deploy() -> None:
    training = build_operator_workflow_presentation("training")
    evaluating = build_operator_workflow_presentation("evaluating")
    deployed = build_operator_workflow_presentation("deployed")

    assert training.step_index == 2
    assert evaluating.step_index == 3
    assert deployed.step_index == 4
    assert deployed.progress == 100
    assert deployed.is_terminal is True
    assert deployed.is_success is True


def test_operator_progress_is_clamped_to_valid_range() -> None:
    assert build_operator_workflow_presentation("training", progress=-5).progress == 0
    assert build_operator_workflow_presentation("training", progress=120).progress == 100
