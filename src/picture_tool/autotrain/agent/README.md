# `agent/` — reserved scaffolding, not implemented

This directory holds the shape of a future AI vision engineer, and nothing
else. There is no agent here, no LLM call anywhere in `picture_tool.autotrain`,
and no code in this directory runs as part of a training cycle.

It exists so the next phase extends a declared boundary rather than inventing
one, and so the constraints below are written down before there is any
pressure to bend them.

## The boundary

An agent belongs **above** `picture_tool.autotrain.service.AutoTrainService`,
choosing which operations to call. It does not belong inside the training
core deciding what a metric means.

Specifically, and permanently:

- **The promotion decision stays in `promotion.py`.** It is a pure function of
  measured metrics and configured thresholds. "The new model looks better to
  me" is exactly the reasoning that gate exists to exclude, and an agent must
  not be able to supply it.
- **Nothing in this package deploys.** There is no method on the service that
  changes which model production runs, and adding one is not a future
  enhancement — adopting a challenger is a named human action in the
  inspection release flow.
- **The golden dataset is not agent-writable.** An evaluation set an agent can
  edit is not evidence.
- **Labels come from humans.** An agent may decide *what* to send for
  annotation; it may not decide what the annotation is.

## Directories

| Directory    | Intended use |
| ------------ | ------------ |
| `tools/`     | Small, reviewed, single-purpose functions an agent may call. Each should be as testable as the rest of the package — an analysis script an agent wrote is code, and gets reviewed like code. |
| `sandbox/`   | Where agent-written analysis scripts execute, isolated from the pool, dataset versions and production. Nothing here should be able to write outside itself. |
| `incidents/` | One directory per investigated failure: the symptom, the hypothesis, what was measured, and the conclusion. Append-only, so a later investigation can see what was already ruled out. |
| `memory/`    | Durable notes across investigations — what has been tried on this station, which explanations turned out to be wrong. Distinct from `incidents/`, which is per-event. |

## What the next phase needs to decide first

1. **When a cycle should run.** Today a human schedules it. The signals are
   already available through the service (`get_model_health`,
   `get_recent_failures`, `get_dataset_statistics`); what is missing is an
   agreed trigger policy, and that policy should be deterministic code with
   an agent proposing changes to it, not an agent evaluating it each time.
2. **What the sandbox is allowed to touch.** Read-only access to a *copy* of a
   dataset version is probably right; access to the pool is probably not.
3. **How an incident ends.** An investigation that concludes "no action" is a
   result worth keeping, and needs somewhere to be kept.
