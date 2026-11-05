"""Experiment B: the prompting baseline, and a direction built from the model's own answers.

1. Prompting baseline: a system prompt asks the model to answer fully and then add a short philosophical
   reflection. Any steering method has to be compared with simply asking.
2. Self-generated direction: for the 32 contrast questions, the model answers twice, greedily, with and without
   that system prompt. Both answers are then fed back *without* the system prompt, and the per-layer diff-means
   (prompted answer - plain answer) gives a direction in the model's own words. Steering with it is a way of
   "distilling" the prompt into a vector, so it works without the prompt in the context.

Round 2 adds prompt variants that bring the reflection earlier (reflect-first, woven), the first prompt with a
200-token answer, and the two variants combined with a light push along the original L8 direction (alpha 0.35).

Outputs under runs/selfgen/: pairs.json (the generated answers), diffmeans.pt, directions.json, gen.jsonl
Then: scripts/claude_judge.py --input selfgen/gen.jsonl --output selfgen/judge.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wi.steer import Steerable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "selfgen"
SYSTEM = ("Answer the user's request fully and helpfully first. Then add one or two sentences of genuine "
          "philosophical reflection connected to the topic (meaning, time, knowledge, ethics, mind or value).")

PROMPTS = {
    "reflect-first": "Open with one sentence of genuine philosophical reflection on the topic (meaning, time, knowledge, "
                     "ethics, mind or value). Then answer the user's request fully and helpfully.",
    "woven": "You are a thoughtful assistant with a philosophical turn of mind. Answer the user's request fully and "
             "helpfully, weaving brief philosophical reflections into the answer itself.",
}


def eval_set(s):
    prompts = json.loads((ROOT / "data/eval_prompts.json").read_text())
    heldout = json.loads((ROOT / "data/heldout_prompts.json").read_text())
    return prompts + heldout, ["eval"] * len(prompts) + ["heldout"] * len(heldout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 12])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    ap.add_argument("--tokens", type=int, default=160, help="max new tokens for the self-generated pairs")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    s = Steerable()
    s.p50 = {int(k): v for k, v in json.loads((RUNS / "p50.json").read_text()).items()}
    old = torch.load(RUNS / "diffmeans.pt")
    qs = [p["q"] for p in json.loads((ROOT / "data/contrast_pairs.json").read_text())]

    pf = OUT / "pairs.json"
    if pf.exists():
        pairs = json.loads(pf.read_text())
    else:
        plain = s.generate([s.chat(q) for q in qs], max_new_tokens=args.tokens)
        philo = s.generate([s.chat(q, system=SYSTEM) for q in qs], max_new_tokens=args.tokens)
        pairs = [{"q": q, "plain": a.strip(), "philo": b.strip()} for q, a, b in zip(qs, plain, philo)]
        pf.write_text(json.dumps(pairs, indent=1, ensure_ascii=False))
    layers = sorted(s.layers)
    R = {L: [] for L in layers}
    for p in pairs:
        prompt = s.chat(p["q"])                                   # no system prompt in either pass
        a = s.resid_mean(prompt + p["philo"], len(prompt), layers)
        b = s.resid_mean(prompt + p["plain"], len(prompt), layers)
        for L in layers:
            R[L].append(a[L] - b[L])
    dm = {L: torch.stack(R[L]).mean(0) for L in layers}
    torch.save(dm, OUT / "diffmeans.pt")
    cos = lambda a, b: round(float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0)), 3)
    rep = {"cos_original": {L: cos(dm[L], old[L]) for L in (4, 8, 12, 16, 20, 24)},
           "norm_over_p50": {L: round(float(dm[L].norm() / s.p50[L]), 3) for L in (4, 8, 12, 16, 20, 24)}}
    ar = RUNS / "answer_reflect"
    for v in ("appended", "woven"):
        if (ar / f"diffmeans_{v}.pt").exists():
            d = torch.load(ar / f"diffmeans_{v}.pt")
            rep[f"cos_{v}"] = {L: cos(dm[L], d[L]) for L in (4, 8, 12, 16, 20, 24)}
    (OUT / "directions.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)

    all_prompts, split = eval_set(s)
    chats = [s.chat(p) for p in all_prompts]
    unit = lambda d: d.float() / d.float().norm()
    cfgs = [("baseline", chats, None), ("prompted (system prompt)", [s.chat(p, system=SYSTEM) for p in all_prompts], None)]
    for L in args.layers:
        for a in args.alphas:
            cfgs.append((f"selfgen L{L}@{a}", chats, {L: a * s.p50[L] * unit(dm[L])}))
    # Round 2: prompt variants that bring the reflection earlier, a longer answer, and prompt + light steering.
    unit8 = unit(old[8])
    for key, sysmsg in PROMPTS.items():
        cfgs.append((f"prompt {key}", [s.chat(p, system=sysmsg) for p in all_prompts], None))
    cfgs.append(("prompt answer-then-reflect, 200 tokens", [s.chat(p, system=SYSTEM) for p in all_prompts], None))
    for key in ("reflect-first", "woven"):
        cfgs.append((f"prompt {key} + original L8@0.35", [s.chat(p, system=PROMPTS[key]) for p in all_prompts],
                     {8: 0.35 * s.p50[8] * unit8}))
    out = OUT / "gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()
    for i, (name, cs, vec) in enumerate(cfgs):
        if name in done:
            continue
        t0 = time.time()
        with s.steer([], vec):
            outs = s.generate(cs, max_new_tokens=200 if "200 tokens" in name else 120)
        kind = "steer" if vec else ("prompt" if "prompt" in name else "baseline")
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": kind, "prompts": all_prompts, "split": split, "outputs": outs}) + "\n")
        print(f"[{i + 1}/{len(cfgs)}] {name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
