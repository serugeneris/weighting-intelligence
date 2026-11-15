"""Experiment AH: warm-start the rank-8 student on paired instruction conditions."""
import argparse
import ast
import json
from pathlib import Path

import torch

import scripts.exp_distill as distill
from scripts.exp_teacher import load_lora
from scripts.philosophy_pilot import ADAPTER, ROOT, RUNS, generate, messages
from wi.steer import Steerable

OUT = RUNS / "conditional_reflection"
TRAIN_PROMPTS = {
    "requested": "Give the practical answer first, then add one sentence reflecting on why the distinction matters.",
    "optional": "Answer the practical question directly. Add reflection only if it is useful.",
    "suppressed": "Output just the answer in the requested form. Do not include a reflection, heading, or explanation.",
}
TEST_PROMPTS = {
    "requested": "On the first line, provide the result. On the next line, consider a broader implication in one sentence.",
    "optional": "Provide the requested result. Use your judgment about adding any commentary.",
    "suppressed": "Return the result alone. Omit every word or formatting element beyond the requested result.",
}
VARIANT = distill.VARIANTS["mlp-r8"]
ADAPTERS = {"base": None, "student": ADAPTER, "mixed": OUT / "mixed.pt"}
SWITCH_MODES = ("requested", "suppressed", "requested")


def training_rows(cases):
    return [{"system": system, "q": question,
             "target": f"{answer}\n{reflection}" if mode == "requested" else answer}
            for question, answer, reflection in cases for mode, system in TRAIN_PROMPTS.items()]


def matches(answer, expected, kind):
    if kind == "json":
        def reject_duplicates(pairs):
            if len({key for key, _ in pairs}) != len(pairs):
                raise ValueError("duplicate JSON keys")
            return dict(pairs)

        try:
            observed = json.loads(answer, object_pairs_hook=reject_duplicates)
            wanted = json.loads(expected)
        except ValueError:
            return False
        return (type(observed) is dict and observed.keys() == wanted.keys()
                and all(type(observed[key]) is type(value) and observed[key] == value
                        for key, value in wanted.items()))
    if kind == "code":
        try:
            return ast.dump(ast.parse(answer)) == ast.dump(ast.parse(expected))
        except SyntaxError:
            return False
    return answer.strip() == expected


def train(model, cases):
    torch.manual_seed(99)
    added = load_lora(model, VARIANT, ADAPTER)
    distill.OUT = OUT
    try:
        distill.train(model, training_rows(cases), "mixed", {**VARIANT, "lr": 1e-4}, added=added)
    finally:
        distill.detach(added)


@torch.no_grad()
def evaluate(model, cases):
    with (OUT / "eval.jsonl").open("w") as output:
        for name, path in ADAPTERS.items():
            added = load_lora(model, VARIANT, path) if path else []
            try:
                for mode, system in TEST_PROMPTS.items():
                    replies = generate(model, [messages(case[0], system) for case in cases], 128)
                    for (question, expected, kind), reply in zip(cases, replies, strict=True):
                        first_line = reply["answer"].strip().partition("\n")[0]
                        row = {"config": name, "mode": mode, "question": question, "expected": expected,
                               "kind": kind, "exact": matches(reply["answer"], expected, kind),
                               "first_line_correct": matches(first_line, expected, kind), **reply}
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    print(f"{name} / {mode}: {len(replies)} replies", flush=True)
            finally:
                distill.detach(added)


@torch.no_grad()
def evaluate_switches(model, cases):
    # One starting task from each format. After the suppression turn, change format.
    starting = cases[::2]
    with (OUT / "switches.jsonl").open("w") as output:
        for name, path in ADAPTERS.items():
            added = load_lora(model, VARIANT, path) if path else []
            try:
                histories = [[] for _ in starting]
                for turn, mode in enumerate(SWITCH_MODES):
                    tasks = starting if turn < 2 else starting[1:] + starting[:1]
                    for history, (question, _, _) in zip(histories, tasks, strict=True):
                        history.append({"role": "user", "content": question + "\n" + TEST_PROMPTS[mode]})
                    replies = generate(model, histories, 128)
                    for index, (history, (_, expected, kind), reply) in enumerate(
                            zip(histories, tasks, replies, strict=True)):
                        answer = reply["answer"]
                        first, _, remaining = answer.strip().partition("\n")
                        exact = matches(answer, expected, kind)
                        follow = exact if mode == "suppressed" else (
                            matches(first, expected, kind) and bool(remaining.strip()))
                        record = {"config": name, "sequence": index, "turn": turn + 1,
                                  "mode": mode, "messages": history, "expected": expected,
                                  "followed": follow, **reply}
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        history.append({"role": "assistant", "content": answer})
                    output.flush()
                print(f"{name}: four instruction switches", flush=True)
            finally:
                distill.detach(added)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "eval", "switch", "all"), default="all")
    parser.add_argument("--data", type=Path, default=ROOT / "data/conditional_reflection.json")
    args = parser.parse_args()
    cases = json.loads(args.data.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    model = Steerable()
    if args.stage in ("train", "all"):
        train(model, cases["train"])
    if args.stage in ("eval", "all"):
        evaluate(model, cases["test"])
    if args.stage in ("switch", "all"):
        evaluate_switches(model, cases["test"])


if __name__ == "__main__":
    main()
