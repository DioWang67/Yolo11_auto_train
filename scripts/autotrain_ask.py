#!/usr/bin/env python3
"""Say what you want in a sentence; the model picks the workflow.

The point of this file is the narrowness of what it can do. An operator's
sentence is turned into exactly one of the workflows listed below --- each
of which is a command this repository already runs and a person has already
reviewed --- or into a refusal. It never assembles a command, never fills in
a path it was not given, and has no workflow that publishes anything.

The model is used here for the one thing it is good at on this station:
reading a request and naming which of a short list it matches. It is not
used to decide *whether* the request is a good idea, and it cannot act on
its own answer --- this script runs the command, after printing what it
understood, in the words of what is about to happen.

``NONE`` is on the list on purpose. A chooser with no way to say "that is
not one of mine" produces a confident answer to every sentence, and the
confident answer here would be a job nobody asked for.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picture_tool.bootstrap import dispatch  # noqa: E402
from picture_tool.bootstrap import vision_client as vc  # noqa: E402

LOGGER = logging.getLogger("autotrain_ask")

FLOWS: tuple[dispatch.Flow, ...] = (
    dispatch.Flow(
        name="FULL",
        summary="the whole thing: train on the prepared labels, score the "
                "result on the acceptance boards, and decide what to do next",
        steps=(
            "檢查人工標註（空標註、缺標註會被剔除）",
            "增強 → lint → 切分（按來源分組）",
            "訓練",
            "拿驗收集算獨立分數",
            "由模型判斷下一步，寫進 decisions.jsonl",
        ),
    ),
    dispatch.Flow(
        name="TRAIN_ONLY",
        summary="train on the prepared labels and stop; no scoring, no "
                "decision",
        steps=("檢查人工標註", "增強 → lint → 切分", "訓練後停下"),
    ),
    dispatch.Flow(
        name="CHECK_DATA",
        summary="only check that the prepared labels are usable; trains "
                "nothing and touches no model",
        steps=("檢查人工標註是否合格，然後停下",),
    ),
)

#: The flags each flow adds. Kept as a table beside the list rather than
#: assembled from the model's reply: the reply chooses a name, and only a
#: name; what that name runs is written here by a person.
FLAGS: dict[str, list[str]] = {
    "FULL": [],
    "TRAIN_ONLY": ["--no-decide"],
    "CHECK_DATA": ["--check-data-only"],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("instruction", help="What you want, in a sentence.")
    p.add_argument("--raw", type=Path, required=True,
                   help="The prepared labels: images/ and labels/.")
    p.add_argument("--acceptance", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Say which workflow it picked and stop.")
    p.add_argument("--url-env", default="QWEN_URL")
    p.add_argument("--key-env", default="")
    p.add_argument("--model", default="Qwen3.8-27B-GGUF")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Arguments passed through to the workflow verbatim.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = vc.openai_compatible_profile(
        model=args.model, url_env=args.url_env, key_env=args.key_env,
        requires_credential=bool(args.key_env), max_tokens=1024,
        max_retries=1, timeout_seconds=120.0,
    )
    try:
        decision = dispatch.choose_flow(
            vc.HttpVisionLLMClient(cfg), args.instruction, flows=FLOWS)
    except dispatch.DispatchError as exc:
        LOGGER.error("沒有執行任何東西：%s", exc)
        LOGGER.error("可用的流程：%s", ", ".join(f.name for f in FLOWS))
        return 1

    flow = next(f for f in FLOWS if f.name == decision.action)
    print(f"\n你說：{args.instruction}")
    print(f"我理解成：{flow.name}")
    print(f"理由：{decision.reason}\n")
    print("將要執行：")
    for index, step in enumerate(flow.steps, 1):
        print(f"  {index}. {step}")

    workflow = PROJECT_ROOT / "scripts" / "autotrain_agentic.py"
    command = [sys.executable, str(workflow), "--raw", str(args.raw)]
    command += FLAGS[flow.name]
    if flow.name == "FULL":
        if not args.acceptance:
            LOGGER.error(
                "\nFULL 需要 --acceptance：沒有獨立的驗收集，唯一的分數就是"
                "模型自己給自己打的，那個分數在這個站別高了約 30 分。")
            return 1
        command += ["--acceptance", str(args.acceptance)]
    command += list(args.extra)

    print(f"\n指令：{' '.join(command[1:])}\n")
    if args.dry_run:
        return 0
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
