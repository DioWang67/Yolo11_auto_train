"""
Extended tests for pipeline/core.py module.
Coverage target: 13% → 90%+
"""

import logging
from types import SimpleNamespace
import pytest

from picture_tool.pipeline.core import Task, Pipeline


class TestTask:
    """Test Task dataclass."""

    def test_task_creation_with_minimal_params(self):
        """Should create task with just name and run function."""

        def dummy_run(config, args):
            pass

        task = Task(name="test_task", run=dummy_run)

        assert task.name == "test_task"
        assert task.run == dummy_run
        assert task.dependencies == []
        assert task.skip_fn is None
        assert task.description == ""

    def test_task_creation_with_all_params(self):
        """Should create task with all parameters."""

        def dummy_run(config, args):
            pass

        def skip_check(config, args):
            return "reason"

        task = Task(
            name="full_task",
            run=dummy_run,
            dependencies=["dep1", "dep2"],
            skip_fn=skip_check,
            description="Test task",
        )

        assert task.name == "full_task"
        assert task.dependencies == ["dep1", "dep2"]
        assert task.skip_fn == skip_check
        assert task.description == "Test task"


class TestPipelineCollect:
    """Test Pipeline._collect method."""

    def test_collect_single_task_no_dependencies(self):
        """Should collect single task without dependencies."""

        def run_fn(c, a):
            pass

        tasks = {"task1": Task("task1", run_fn)}
        pipeline = Pipeline(tasks)

        collected = pipeline._collect(["task1"])

        assert "task1" in collected
        assert len(collected) == 1

    def test_collect_with_dependencies(self):
        """Should collect task and its dependencies."""

        def run_fn(c, a):
            pass

        tasks = {
            "task1": Task("task1", run_fn),
            "task2": Task("task2", run_fn, dependencies=["task1"]),
            "task3": Task("task3", run_fn, dependencies=["task2"]),
        }
        pipeline = Pipeline(tasks)

        collected = pipeline._collect(["task3"])

        assert len(collected) == 3
        assert "task1" in collected
        assert "task2" in collected
        assert "task3" in collected

    def test_collect_detects_circular_dependency(self):
        """Should raise ValueError on circular dependencies."""

        def run_fn(c, a):
            pass

        tasks = {
            "task1": Task("task1", run_fn, dependencies=["task2"]),
            "task2": Task("task2", run_fn, dependencies=["task1"]),
        }
        pipeline = Pipeline(tasks)

        with pytest.raises(ValueError, match="Cycle detected"):
            pipeline._collect(["task1"])

    def test_collect_handles_unknown_task_throws_value_error(self):
        """Should raise ValueError on unknown task dependency."""

        def run_fn(c, a):
            pass

        tasks = {"task1": Task("task1", run_fn)}
        pipeline = Pipeline(tasks)

        with pytest.raises(ValueError, match="Unknown task requested or missing dependency"):
            pipeline._collect(["unknown_task"])

    def test_collect_deduplicates_tasks(self):
        """Should not duplicate tasks in collection."""

        def run_fn(c, a):
            pass

        tasks = {
            "base": Task("base", run_fn),
            "task1": Task("task1", run_fn, dependencies=["base"]),
            "task2": Task("task2", run_fn, dependencies=["base"]),
        }
        pipeline = Pipeline(tasks)

        # Request both task1 and task2, both depend on base
        collected = pipeline._collect(["task1", "task2"])

        # base should only appear once
        assert len(collected) == 3


class TestPipelineToposort:
    """Test Pipeline._toposort method."""

    def test_toposort_orders_by_dependencies(self):
        """Should order tasks by dependency chain."""

        def run_fn(c, a):
            pass

        tasks = {
            "task3": Task("task3", run_fn, dependencies=["task2"]),
            "task1": Task("task1", run_fn),
            "task2": Task("task2", run_fn, dependencies=["task1"]),
        }
        pipeline = Pipeline(tasks)

        ordered = pipeline._toposort(tasks)

        names = [t.name for t in ordered]
        # task1 should come before task2, task2 before task3
        assert names.index("task1") < names.index("task2")
        assert names.index("task2") < names.index("task3")

    def test_toposort_handles_diamond_dependency(self):
        """Should handle diamond-shaped dependency graph."""

        def run_fn(c, a):
            pass

        # Diamond: top -> left,right -> bottom
        tasks = {
            "top": Task("top", run_fn),
            "left": Task("left", run_fn, dependencies=["top"]),
            "right": Task("right", run_fn, dependencies=["top"]),
            "bottom": Task("bottom", run_fn, dependencies=["left", "right"]),
        }
        pipeline = Pipeline(tasks)

        ordered = pipeline._toposort(tasks)

        names = [t.name for t in ordered]
        # top must come first
        assert names[0] == "top"
        # bottom must come last
        assert names[-1] == "bottom"

    def test_toposort_deduplicates_results(self):
        """Should not include duplicate tasks in result."""

        def run_fn(c, a):
            pass

        tasks = {"task1": Task("task1", run_fn), "task2": Task("task2", run_fn)}
        pipeline = Pipeline(tasks)

        ordered = pipeline._toposort(tasks)

        assert len(ordered) == 2
        names = [t.name for t in ordered]
        assert len(names) == len(set(names))  # No duplicates


class TestPipelineRun:
    """Test Pipeline.run method."""

    def test_run_executes_tasks_in_order(self):
        """Should execute tasks in dependency order."""
        execution_order = []

        def make_run(name):
            def run_fn(c, a):
                execution_order.append(name)

            return run_fn

        tasks = {
            "task1": Task("task1", make_run("task1")),
            "task2": Task("task2", make_run("task2"), dependencies=["task1"]),
            "task3": Task("task3", make_run("task3"), dependencies=["task2"]),
        }
        pipeline = Pipeline(tasks)

        pipeline.run(["task3"], {}, SimpleNamespace())

        assert execution_order == ["task1", "task2", "task3"]

    def test_run_executes_skip_function(self):
        """Should evaluate skip_fn and skip task if reason returned."""
        execution_log = []

        def run_fn(c, a):
            execution_log.append("executed")

        def skip_fn(c, a):
            return "skipped for test"

        tasks = {"task1": Task("task1", run_fn, skip_fn=skip_fn)}
        pipeline = Pipeline(tasks)

        pipeline.run(["task1"], {}, SimpleNamespace(force=False))

        assert execution_log == []  # Should not execute

    def test_run_ignores_skip_when_force_enabled(self):
        """Should ignore skip_fn when force=True."""
        execution_log = []

        def run_fn(c, a):
            execution_log.append("executed")

        def skip_fn(c, a):
            return "should be ignored"

        tasks = {"task1": Task("task1", run_fn, skip_fn=skip_fn)}
        pipeline = Pipeline(tasks)

        pipeline.run(["task1"], {}, SimpleNamespace(force=True))

        assert "executed" in execution_log

    def test_run_honors_stop_event(self):
        """Should stop execution when stop_event is set."""
        execution_log = []

        class MockStopEvent:
            def __init__(self):
                self.count = 0

            def is_set(self):
                self.count += 1
                return self.count > 1  # Stop after first task

        def make_run(name):
            def run_fn(c, a):
                execution_log.append(name)

            return run_fn

        tasks = {
            "task1": Task("task1", make_run("task1")),
            "task2": Task("task2", make_run("task2"), dependencies=["task1"]),
        }
        pipeline = Pipeline(tasks)

        stop_event = MockStopEvent()
        args = SimpleNamespace(stop_event=stop_event)
        pipeline.run(["task2"], {}, args)

        assert "task1" in execution_log
        assert "task2" not in execution_log

    def test_run_executes_before_task_hook(self):
        """Should execute before_task hook before each task."""
        hook_calls = []

        def before_hook(task, config):
            hook_calls.append(task.name)
            return config

        def run_fn(c, a):
            pass

        tasks = {"task1": Task("task1", run_fn), "task2": Task("task2", run_fn)}
        pipeline = Pipeline(tasks)

        pipeline.run(["task1", "task2"], {}, SimpleNamespace(), before_task=before_hook)

        assert hook_calls == ["task1", "task2"]

    def test_run_handles_before_task_hook_exception(self, caplog):
        """Should log warning if before_task hook fails."""

        def failing_hook(task, config):
            raise RuntimeError("Hook failed")

        def run_fn(c, a):
            pass

        tasks = {"task1": Task("task1", run_fn)}
        pipeline = Pipeline(tasks)

        with caplog.at_level(logging.WARNING):
            pipeline.run(["task1"], {}, SimpleNamespace(), before_task=failing_hook)

        assert "Pre-task hook" in caplog.text
        assert "failed" in caplog.text

    def test_run_handles_skip_fn_exception(self):
        """Should raise RuntimeError if skip_fn fails."""

        def run_fn(c, a):
            pass

        def failing_skip(c, a):
            raise RuntimeError("Skip check failed")

        tasks = {"task1": Task("task1", run_fn, skip_fn=failing_skip)}
        pipeline = Pipeline(tasks)

        with pytest.raises(RuntimeError, match="Task skip evaluation failed"):
            pipeline.run(["task1"], {}, SimpleNamespace(force=False))

    def test_run_logs_task_execution(self, caplog):
        """Should log task start and completion."""

        def run_fn(c, a):
            pass

        tasks = {"task1": Task("task1", run_fn)}
        pipeline = Pipeline(tasks)

        with caplog.at_level(logging.INFO):
            pipeline.run(["task1"], {}, SimpleNamespace())

        assert "Running task: task1" in caplog.text
        assert "Task task1 completed" in caplog.text
