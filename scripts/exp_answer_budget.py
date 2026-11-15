"""Experiment AE: paired Qwen3.5-4B completion sweep.

HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.exp_answer_budget
"""
import json

import torch

from scripts.philosophy_pilot import CONFIGS, ROOT, RUNS, configuration, generate, messages
from wi.steer import Steerable

CONDITIONS = ((120, None), (256, None), (512, None), (512, 90))
OUT = RUNS / "answer_budget" / "outputs.jsonl"


@torch.no_grad()
def main():
    cases = json.loads((ROOT / "data/answer_budget.json").read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = Steerable()
    temporary = OUT.with_suffix(".tmp")
    with temporary.open("w") as output:
        for name in CONFIGS:
            with configuration(model, name) as system:
                for budget, word_limit in CONDITIONS:
                    prompts = [case["prompt"] for case in cases]
                    if word_limit:
                        prompts = [f"{prompt}\nUse at most {word_limit} words." for prompt in prompts]
                    histories = [messages(prompt, system) for prompt in prompts]
                    replies = generate(model, histories, budget, record_tokens=True)
                    for case, history, reply in zip(cases, histories, replies, strict=True):
                        row = {"id": case["id"], "config": name, "budget": budget,
                               "word_limit": word_limit, "messages": history, **reply}
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    print(f"{name}: {budget} tokens, word limit {word_limit}, {len(replies)} replies", flush=True)
    temporary.replace(OUT)


if __name__ == "__main__":
    main()
