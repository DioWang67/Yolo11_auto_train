# Pipeline API

## `Task`

```python
from picture_tool.pipeline.core import Task

task = Task(
    name="my_task",
    run=my_run_fn,
    dependencies=["other_task"],
    skip_fn=my_skip_fn,
    description="Does something useful.",
)
```

A unit of work in the pipeline DAG.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Unique identifier. Used as the key in `Pipeline.tasks`. |
| `run` | `Callable[[dict, Any], Any]` | Executes the task. Signature: `(config, args) -> Any`. |
| `dependencies` | `list[str]` | Names of tasks that must run before this one. |
| `skip_fn` | `Callable[[dict, Any], str \| None] \| None` | Return `None` to run, or a non-empty string describing why to skip. Bypassed when `force=True`. |
| `description` | `str` | Human-readable summary for `--list-tasks` and GUI. |

---

## `Pipeline`

```python
from picture_tool.pipeline.core import Pipeline

pipeline = Pipeline(tasks={"task_a": task_a, "task_b": task_b}, logger=logger)
pipeline.run(requested=["task_b"], config=cfg, args=args)
```

DAG-aware executor. Resolves dependencies, topologically sorts, then runs each task in order.

### Constructor

```python
Pipeline(tasks: dict[str, Task], logger: logging.Logger | None = None)
```

### `Pipeline.run`

```python
pipeline.run(
    requested: list[str],
    config: dict,
    args: Any,
    before_task: Callable[[Task, dict], dict] | None = None,
    after_task: Callable[[Task, int, int], None] | None = None,
)
```

| Parameter | Description |
|-----------|-------------|
| `requested` | Task names to execute (dependencies are pulled in automatically). |
| `config` | Pipeline configuration dictionary. |
| `args` | Parsed CLI/GUI args namespace. Must have `force: bool` attribute to bypass skip logic. |
| `before_task` | Optional hook `(task, config) -> config` called before each task. May return a refreshed config. |
| `after_task` | Optional hook `(task, index, total) -> None` called after each task completes (including skipped). |

### Stop / Cancellation

If `args.stop_event` is set and `stop_event.is_set()` returns `True`, the pipeline stops after the current task completes.

### Force Mode

Set `args.force = True` or `config["pipeline"]["force"] = True` to bypass all `skip_fn` checks and run every task unconditionally.

---

## `skip_fn` Contract

```python
def my_skip_fn(config: dict, args: Any) -> str | None:
    if outputs_are_fresh(config):
        return "Outputs up to date; skipping."
    return None   # proceed
```

- Return `None` → task runs.
- Return a non-empty string → task is skipped; the string appears in the log.
- If `skip_fn` raises an exception, `Pipeline` re-raises it as `RuntimeError`.

---

## Built-in Skip Functions

| Task | Skip function | Skip condition |
|------|--------------|----------------|
| `yolo_train` | `skip_yolo_train` | dataset hash + config hash unchanged AND `best.pt` exists in latest versioned run dir |
| `yolo_augmentation` | `skip_yolo_augmentation` | output files newer than input files (mtime) |
| `dataset_splitter` | `skip_dataset_splitter` | split output directories already populated |
| `dataset_lint` | `skip_dataset_lint` | lint CSV already exists |
| `aug_preview` | `skip_aug_preview` | preview images already exist |
| `batch_inference` | `skip_batch_infer` | predictions CSV already exists |
