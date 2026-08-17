import ast
from pathlib import Path

import picture_tool.gui.operator_workflow_panel as panel_module
from picture_tool.gui.operator_workflow_panel import (
    PUBLISHED_JOB_STATES,
    build_operator_workflow_presentation,
)


MAIN_WINDOW_SOURCE = (
    Path(panel_module.__file__).with_name("main_window.py")
)


def _string_constants(node: ast.AST) -> set[str]:
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


def _states_published_by_main_window() -> set[str]:
    """Collect every lifecycle state literal the training GUI can publish.

    The panel is the only operator-visible rendering of these states, so a new
    state that never reaches ``_STATE_PRESENTATIONS`` silently renders as
    ``queued``. Reading the publisher's source keeps that contract enforced
    without importing PyQt widgets.
    """
    tree = ast.parse(MAIN_WINDOW_SOURCE.read_text(encoding="utf-8"))
    published: set[str] = set()
    for node in ast.walk(tree):
        # ``_publish_operator_status(state=...)`` including conditional values.
        if isinstance(node, ast.keyword) and node.arg == "state":
            published |= _string_constants(node.value)
        # ``operator_states = {task: (state, message, progress), ...}``
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "operator_states"
            for target in node.targets
        ):
            for value in getattr(node.value, "values", []):
                if isinstance(value, ast.Tuple) and value.elts:
                    published |= _string_constants(value.elts[0])
        # ``_operator_error_state`` maps exceptions onto lifecycle states.
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_operator_error_state"
        ):
            for statement in ast.walk(node):
                if isinstance(statement, ast.Return) and statement.value is not None:
                    published |= _string_constants(statement.value)
    return published


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


def test_every_published_state_has_its_own_presentation() -> None:
    """A missing presentation renders a stopped job as one that is starting."""
    missing = _states_published_by_main_window() - PUBLISHED_JOB_STATES
    assert not missing, (
        "These states are published by the training GUI but fall back to the "
        f"'queued' presentation: {sorted(missing)}"
    )


def test_recoverable_shortage_is_terminal_and_keeps_progress() -> None:
    presentation = build_operator_workflow_presentation("waiting_feedback")

    assert presentation.title != build_operator_workflow_presentation("queued").title
    assert presentation.is_terminal is True
    assert presentation.is_success is False
    assert presentation.keeps_reached_progress is True


def test_cancelling_is_not_terminal_and_keeps_progress() -> None:
    presentation = build_operator_workflow_presentation("cancelling")

    assert presentation.title == "正在安全停止"
    assert presentation.is_terminal is False
    assert presentation.keeps_reached_progress is True
