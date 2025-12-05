# picture-tool Improvement Tracker

This file keeps a concise record of planned and completed improvements.

## Completed
- doctor: when `create_demo=True`, now writes `data/demo_doctor/images` and `labels`, fixing the smoke test.
- typing: mypy runs clean on `src` and `tests`.
- Performance: cached letterbox params in position validator to avoid repeated scale/pad math.
- UX: batch inference now shows tqdm progress.
- Config: added schema validation hook (pydantic if available, otherwise manual) and config reload caching to avoid repeated YAML reads.
- SAM UX: added cancel-aware SAM inference tasks with progress status updates in the GUI.
- Augmentor: optional process pool path for image augmentation to bypass GIL for CPU-heavy runs.
- SAM GUI: progress/cancel now routed through signals to keep UI thread updates safe.

## Planned (high priority)
- Color verifier refactor: split the large `color_verifier` logic into strategy/rule components (black/yellow fast paths, orange/red split, green correction, generic matcher).

## Planned (medium)
- (none currently)

## To evaluate (longer term)
- Task registry/plugin: replace hardcoded `TASK_HANDLERS` with a registration mechanism if tasks will grow.
- Event/trace logging: record task start/end and config changes for easier debugging and replay.
