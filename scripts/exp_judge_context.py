"""Experiment AF: test which distinctions survive when the judge loses task context."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.claude_judge import call_claude, parse

RUBRIC = """Judge each pair of candidate replies using only the supplied conversation.
Prefer the reply that follows explicit instructions, answers the question accurately,
respects the earlier exchange, and gives relevant, valid reasoning when requested.
Use tie if neither reply has an established advantage from the supplied context.
Return only a JSON array of objects with keys id and winner (A, B, or tie).

"""
MODES = ("full", "last-user")


def judge(cases, model, mode, reversed_order):
    items = []
    for case in cases:
        answers = case["answers"][::-1] if reversed_order else case["answers"]
        conversation = case["messages"] if mode == "full" else case["messages"][-1:]
        items.append({"id": case["id"], "conversation": conversation,
                      "A": answers[0], "B": answers[1]})
    request = RUBRIC + json.dumps(items, ensure_ascii=False)
    decisions = parse(call_claude(request, model))
    by_id = {row["id"]: row["winner"] for row in decisions}
    if len(decisions) != len(items) or set(by_id) != {item["id"] for item in items}:
        raise ValueError("Judge returned missing, repeated, or unknown case IDs")
    if set(by_id.values()) - {"A", "B", "tie"}:
        raise ValueError("Judge returned an invalid winner")
    return {"mode": mode, "reversed": reversed_order, "model": model,
            "request_sha256": hashlib.sha256(request.encode()).hexdigest(), "decisions": by_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/judge_context.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/judge_context/results.json"))
    parser.add_argument("--model", default="sonnet")
    args = parser.parse_args()
    cases = json.loads(args.data.read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(judge, cases, args.model, mode, reverse)
                for mode in MODES for reverse in (False, True)]
        runs = [job.result() for job in jobs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(runs, indent=2) + "\n")


if __name__ == "__main__":
    main()
