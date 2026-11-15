"""Experiment AG: change one fact in standalone rule-application pairs."""
import argparse
import json
import random
import re
from pathlib import Path

import torch

from scripts.philosophy_pilot import CONFIGS, configuration, generate, messages
from wi.steer import Steerable


@torch.no_grad()
def run(cases):
    tasks = []
    for index, case in enumerate(cases):
        allow_letter = "AB"[index % 2]
        deny_letter = "AB"[1 - index % 2]
        choices = sorted([(allow_letter, "The rule permits it."), (deny_letter, "The rule does not permit it.")])
        options = "\n".join(f"{letter}. {meaning}" for letter, meaning in choices)
        for variant, (fact, allowed) in enumerate(zip(case["facts"], case["allow"], strict=True)):
            prompt = (f"Rule: {case['rule']}\nCase: {fact}\n"
                      f"Apply the stated rule as written. Does the case qualify?\n{options}\n"
                      "Start with A or B and then state the deciding fact in one sentence.")
            tasks.append({"id": case["id"], "kind": case["kind"], "variant": variant,
                          "expected": allow_letter if allowed else deny_letter, "prompt": prompt})
    random.Random(42).shuffle(tasks)
    model = Steerable()
    for name in CONFIGS:
        with configuration(model, name) as system:
            replies = generate(model, [messages(task["prompt"], system) for task in tasks], 256)
        for task, reply in zip(tasks, replies, strict=True):
            match = re.match(r"^\s*\**([AB])\b", reply["answer"])
            yield {**task, **reply, "config": name, "choice": match[1] if match else None}
        print(f"{name}: {len(replies)} replies", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/premise_sensitivity.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/premise_sensitivity/outputs.jsonl"))
    args = parser.parse_args()
    cases = json.loads(args.data.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    with temporary.open("w") as output:
        for row in run(cases):
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
    temporary.replace(args.out)


if __name__ == "__main__":
    main()
