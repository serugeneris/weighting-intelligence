"""Experiment U: a non-Claude judge. Does Qwen3.5-4B, scoring the same answers, agree with Sonnet and Opus?

Every result since experiment O rests on a Claude judge scoring answers whose style was taught by Claude (the
Claude-written targets). A judge can prefer text that reads like its own (self-preference bias), so the headline
"the Claude-taught student matches the prompt's philosophy with ~+2 helpfulness" could partly be the judge liking
its own register. Here the six 48-prompt pools that already have Sonnet and Opus scores (confirm, claude_teacher,
more_teacher, deeper_teacher, capacity, dpo_claude; ~2,000 answers) are re-scored by the base Qwen3.5-4B, with no
steering, one criterion per question (philo, helpful, coherence; the claude_judge.py rubric wording), as the expected
value of its next-digit distribution 0-9. Qwen is not neutral either (it may prefer Qwen-like text), so the test is
whether the *conclusions* survive a judge with the opposite bias, not whether Qwen's numbers are "right".

  stage score    -> runs/qwen_judge/scores.jsonl  (one row per pool/cfg/item, all three judges side by side)
  stage analyze  -> runs/qwen_judge/summary.json  (item correlations, config rankings, paired contrasts vs the
                    woven prompt with SE over prompts, per judge)
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "qwen_judge"
POOLS = ["confirm", "claude_teacher", "more_teacher", "deeper_teacher", "capacity", "dpo_claude"]
CRIT = {
    "philo": "Rate how much the response engages in genuine philosophical reflection (meaning, existence, knowledge, "
             "ethics, mind, time, value). 0 = purely practical/factual. 9 = substantive philosophical thinking. Mere "
             "repetition of words like \"being\" or \"truth\", or incoherent mystical-sounding fragments, is NOT philosophy.",
    "helpful": "Rate how well the response actually addresses the user's request. 0 = ignores it; 9 = fully useful answer.",
    "coherence": "Rate how fluent, grammatical and logically connected the response's prose is. 0 = gibberish or "
                 "loops; 9 = fully coherent.",
}


def rows():
    out = []
    for pool in POOLS:
        son = [json.loads(l) for l in (RUNS / pool / "judge.jsonl").read_text().splitlines()]
        opu = {(r["cfg"], r["i"]): r for r in map(json.loads, (RUNS / pool / "judge_opus.jsonl").read_text().splitlines())}
        for r in son:
            o = opu.get((r["cfg"], r["i"]))
            if o is None or o["output"] != r["output"]:
                continue
            out.append({"pool": pool, "cfg": r["cfg"], "i": r["i"], "prompt": r["prompt"], "output": r["output"],
                        **{f"sonnet_{c}": r[c] for c in CRIT}, **{f"opus_{c}": o[c] for c in CRIT},
                        "sonnet_degenerate": r["degenerate"], "opus_degenerate": o["degenerate"]})
    return out


def query(s, prompt, output, crit):
    return s.chat(f"You are grading an assistant's response.\n\nUser request:\n\"\"\"\n{prompt.strip()}\n\"\"\"\n\n"
                  f"Response:\n\"\"\"\n{output.strip()[:1500]}\n\"\"\"\n\n{CRIT[crit]}\n\nAnswer with a single digit from 0 to 9.")


@torch.no_grad()
def score(s, texts, bs=8):
    digit_ids = [s.tok.convert_tokens_to_ids(str(d)) for d in range(10)]
    s.tok.padding_side = "left"
    order = sorted(range(len(texts)), key=lambda k: len(texts[k]))     # similar lengths per batch: little padding
    res = [None] * len(texts)
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        enc = s.tok([texts[k] for k in idx], return_tensors="pt", padding=True).to(s.device)
        logits = s.model(**enc, logits_to_keep=1).logits[:, -1, digit_ids].float()
        ev = (logits.softmax(-1) * torch.arange(10, device=logits.device)).sum(-1)
        for k, v in zip(idx, ev.tolist()):
            res[k] = v
    return res


def stage_score():
    from wi.steer import Steerable
    s = Steerable()
    data = rows()
    print(f"{len(data)} answers in {len(POOLS)} pools", flush=True)
    # padding check: batched vs one-at-a-time on 12 items must agree
    probe = [query(s, d["prompt"], d["output"], "philo") for d in data[:12]]
    a, b = score(s, probe), [score(s, [t], bs=1)[0] for t in probe]
    gap = max(abs(x - y) for x, y in zip(a, b))
    print(f"batched vs single max gap {gap:.3f}", flush=True)
    bs = 8 if gap < 0.15 else 1
    for c in CRIT:
        vals = score(s, [query(s, d["prompt"], d["output"], c) for d in data], bs=bs)
        for d, v in zip(data, vals):
            d[f"qwen_{c}"] = round(v, 3)
        print(f"scored {c}", flush=True)
    with (OUT / "scores.jsonl").open("w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    (OUT / "padding_check.json").write_text(json.dumps({"max_gap": gap, "batch_size_used": bs}))


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


def ranks(v):
    o = sorted(range(len(v)), key=lambda k: v[k])
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def stage_analyze():
    data = [json.loads(l) for l in (OUT / "scores.jsonl").read_text().splitlines()]
    J = ["sonnet", "opus", "qwen"]
    summ = {"n": len(data), "item_corr": {}, "config_rank": {}, "config_means": {}, "contrasts": {}}
    for c in CRIT:
        summ["item_corr"][c] = {f"{a}~{b}": round(pearson([d[f"{a}_{c}"] for d in data], [d[f"{b}_{c}"] for d in data]), 3)
                                for a, b in (("sonnet", "opus"), ("qwen", "sonnet"), ("qwen", "opus"))}
    by = defaultdict(lambda: defaultdict(dict))
    for d in data:
        by[d["pool"]][d["cfg"]][d["i"]] = d
    for pool, cfgs in by.items():
        full = {k: v for k, v in cfgs.items() if len(v) == 48}
        means = {k: {f"{j}_{c}": round(sum(r[f"{j}_{c}"] for r in v.values()) / len(v), 3) for j in J for c in CRIT}
                 for k, v in cfgs.items()}
        summ["config_means"][pool] = means
        names = list(full)
        summ["config_rank"][pool] = {c: {f"{a}~{b}": round(spearman([means[k][f"{a}_{c}"] for k in names],
                                                                     [means[k][f"{b}_{c}"] for k in names]), 3)
                                         for a, b in (("sonnet", "opus"), ("qwen", "sonnet"), ("qwen", "opus"))}
                                     for c in CRIT}
        ref = full.get("prompt woven")
        if not ref:
            continue
        for k, v in full.items():
            if k == "prompt woven":
                continue
            row = {}
            for j in J:
                for c in ("philo", "helpful"):
                    diffs = [v[i][f"{j}_{c}"] - ref[i][f"{j}_{c}"] for i in v]
                    m = sum(diffs) / len(diffs)
                    se = math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1) / len(diffs))
                    row[f"{j}_{c}"] = [round(m, 2), round(se, 2)]
            summ["contrasts"][f"{pool} | {k} − prompt woven"] = row
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps(summ["item_corr"], indent=1))
    for k, v in summ["contrasts"].items():
        print(k, {a: b for a, b in v.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["score", "analyze"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stage_score() if args.stage == "score" else stage_analyze()


if __name__ == "__main__":
    main()
