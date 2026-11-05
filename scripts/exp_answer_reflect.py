"""Experiment A: "answer + reflect" contrast data.

The original philosophical answers (data/contrast_pairs.json) *replace* the practical answer, so the diff-means
direction encodes "reflect instead of answering" (FINDINGS Results 6). data/contrast_pairs_v2.json keeps the plain
answer and adds two new philosophical versions that keep all the practical content:
  appended - the plain answer verbatim, then one or two reflective sentences
  woven    - the same facts and steps, with reflective framing woven into the answer
For each variant we compute per-layer diff-means (variant - plain, mean residual over answer tokens), compare it
with the original direction, and steer layer 8 at several alphas. A random unit direction at the largest alpha is
the control. The natural v2 texts are judged in the same blind pool (helpfulness ceiling of each target).

Outputs (all under runs/answer_reflect/, nothing earlier is overwritten):
  diffmeans_appended.pt, diffmeans_woven.pt, directions.json, gen.jsonl
Then: scripts/claude_judge.py --input answer_reflect/gen.jsonl --output answer_reflect/judge.jsonl
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wi.steer import Steerable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "answer_reflect"
VARIANTS = ("appended", "woven")
L0 = 8
ALPHAS = (0.5, 0.75, 1.0)


def directions(s, pairs):
    layers = sorted(s.layers)
    R = {v: {L: [] for L in layers} for v in VARIANTS}
    for i, p in enumerate(pairs):
        prompt = s.chat(p["q"])
        base = s.resid_mean(prompt + p["plain"], len(prompt), layers)
        for v in VARIANTS:
            r = s.resid_mean(prompt + p[v], len(prompt), layers)
            for L in layers:
                R[v][L].append(r[L] - base[L])
        print(f"pair {i + 1}/{len(pairs)}", end="\r", flush=True)
    print()
    return {v: {L: torch.stack(R[v][L]).mean(0) for L in layers} for v in VARIANTS}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = Steerable()
    s.p50 = {int(k): v for k, v in json.loads((RUNS / "p50.json").read_text()).items()}
    old = torch.load(RUNS / "diffmeans.pt")
    pairs = json.loads((ROOT / "data/contrast_pairs_v2.json").read_text())

    dms = directions(s, pairs)
    report = {}
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0))
    for v in VARIANTS:
        torch.save(dms[v], OUT / f"diffmeans_{v}.pt")
        report[v] = {L: {"cos_original": round(cos(dms[v][L], old[L]), 3),
                         "norm_over_p50": round(float(dms[v][L].norm() / s.p50[L]), 3)} for L in (4, 8, 12, 16, 20, 24)}
    report["appended_vs_woven"] = {L: round(cos(dms["appended"][L], dms["woven"][L]), 3) for L in (4, 8, 12, 16, 20, 24)}
    report["original_norm_over_p50"] = {L: round(float(old[L].norm() / s.p50[L]), 3) for L in (4, 8, 12, 16, 20, 24)}
    (OUT / "directions.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)

    prompts = json.loads((ROOT / "data/eval_prompts.json").read_text())
    heldout = json.loads((ROOT / "data/heldout_prompts.json").read_text())
    all_prompts = prompts + heldout
    split = ["eval"] * len(prompts) + ["heldout"] * len(heldout)
    chats = [s.chat(p) for p in all_prompts]
    unit = lambda d: d.float() / d.float().norm()
    g = torch.Generator().manual_seed(0)
    rand = unit(torch.randn(old[L0].shape[0], generator=g))
    cfgs = [("baseline", None), (f"original L{L0}@0.5", 0.5 * s.p50[L0] * unit(old[L0]))]
    for v in VARIANTS:
        for a in ALPHAS:
            cfgs.append((f"{v} L{L0}@{a}", a * s.p50[L0] * unit(dms[v][L0])))
    cfgs.append((f"random-direction L{L0}@{ALPHAS[-1]}", ALPHAS[-1] * s.p50[L0] * rand))

    out = OUT / "gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()
    for i, (name, vec) in enumerate(cfgs):
        if name in done:
            continue
        t0 = time.time()
        with s.steer([], {L0: vec} if vec is not None else None):
            outs = s.generate(chats, max_new_tokens=120)
        kind = "baseline" if vec is None else ("control" if name.startswith("random") else "steer")
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": kind, "prompts": all_prompts, "split": split, "outputs": outs}) + "\n")
        print(f"[{i + 1}/{len(cfgs)}] {name} ({time.time() - t0:.0f}s)", flush=True)
    if "natural appended" not in done:
        with out.open("a") as f:
            for v in VARIANTS:
                f.write(json.dumps({"name": f"natural {v}", "kind": "natural", "prompts": [p["q"] for p in pairs],
                                    "split": ["pairs"] * len(pairs), "outputs": [p[v] for p in pairs]}) + "\n")


if __name__ == "__main__":
    main()
