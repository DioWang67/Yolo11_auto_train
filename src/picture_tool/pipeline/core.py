from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set


@dataclass
class Task:
    """A unit of work in the picture-tool pipeline.

    Attributes:
        name: Unique identifier for this task.
        run: Callable ``(config, args) -> Any`` that executes the task.
        dependencies: Names of tasks that must run before this one.
        skip_fn: Optional callable ``(config, args) -> Optional[str]``.
            **Contract**: return ``None`` to proceed with execution, or a
            non-empty ``str`` describing the reason to skip.  When the
            pipeline's ``force`` flag is set, ``skip_fn`` is bypassed
            entirely.
        description: Human-readable summary shown in CLI/GUI listings.
    """

    name: str
    run: Callable[[dict, Any], object]
    dependencies: List[str] = field(default_factory=list)
    skip_fn: Optional[Callable[[dict, Any], Optional[str]]] = None
    description: str = ""


class Pipeline:
    """DAG-aware pipeline executor with optional cache skipping."""

    def __init__(self, tasks: Dict[str, Task], logger: Optional[logging.Logger] = None):
        self.tasks = tasks
        self.logger = logger or logging.getLogger(__name__)

    def _collect(self, requested: Iterable[str]) -> Dict[str, Task]:
        collected: Dict[str, Task] = {}
        visiting: Set[str] = set()

        def dfs(name: str) -> None:
            if name in collected:
                return
            if name in visiting:
                raise ValueError(f"Cycle detected at task '{name}'")
            visiting.add(name)
            task = self.tasks.get(name)
            if not task:
                raise ValueError(f"Unknown task requested or missing dependency: {name}")
            for dep in task.dependencies:
                dfs(dep)
            collected[name] = task
            visiting.remove(name)

        for t in requested:
            dfs(t)
        return collected

    def _toposort(self, tasks: Dict[str, Task]) -> List[Task]:
        ordered: List[Task] = []
        temporary: Set[str] = set()
        permanent: Set[str] = set()

        def visit(name: str) -> None:
            if name in permanent:
                return
            if name in temporary:
                raise ValueError(f"Cycle detected at task '{name}'")
            temporary.add(name)
            task = tasks.get(name)
            if not task:
                temporary.remove(name)
                return
            for dep in task.dependencies:
                if dep in tasks:
                    visit(dep)
            permanent.add(name)
            temporary.remove(name)
            ordered.append(task)

        for name in tasks:
            visit(name)
        # Preserve requested order for tasks sharing the same dep tree
        dedup: List[Task] = []
        seen: Set[str] = set()
        for task in ordered:
            if task.name not in seen:
                dedup.append(task)
                seen.add(task.name)
        return dedup

    def run(
        self,
        requested: List[str],
        config: dict,
        args: Any,
        before_task: Optional[Callable[["Task", dict], dict]] = None,
        after_task: Optional[Callable[["Task", int, int], None]] = None,
    ) -> None:
        """Resolve dependencies, topo-sort, and execute tasks with skip logic.

        Args:
            requested: List of task names to execute.
            config: Pipeline configuration dictionary.
            args: Parsed CLI arguments namespace.
            before_task: Optional ``(task, config) -> config`` hook called
                before each task.  May return a refreshed config dict.
            after_task: Optional ``(task, index, total) -> None`` hook called
                after each task completes (including skipped tasks).
        """
        collected = self._collect(requested)
        ordered = self._toposort(collected)
        self.logger.info(f"Resolved task order: {[t.name for t in ordered]}")
        force = bool(
            getattr(args, "force", False) or config.get("pipeline", {}).get("force")
        )
        total = len(ordered)

        for index, task in enumerate(ordered):
            if before_task:
                try:
                    config = before_task(task, config)
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.warning(f"Pre-task hook for {task.name} failed: {exc}")
            if (
                hasattr(args, "stop_event")
                and getattr(args.stop_event, "is_set", lambda: False)()
            ):
                self.logger.info("Stop requested; aborting remaining tasks.")
                break
            skip_reason: Optional[str] = None
            if not force and task.skip_fn:
                try:
                    skip_reason = task.skip_fn(config, args)
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.error("skip_fn for %s failed: %s", task.name, exc)
                    raise RuntimeError(f"Task skip evaluation failed for {task.name}: {exc}") from exc
            if skip_reason:
                self.logger.info(f"Skipping task {task.name}: {skip_reason}")
                if after_task:
                    after_task(task, index, total)
                continue
            self.logger.info(f"Running task: {task.name}")
            task.run(config, args)
            self.logger.info(f"Task {task.name} completed.")
            if after_task:
                after_task(task, index, total)
