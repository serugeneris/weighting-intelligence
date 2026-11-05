"""Experiment E: spread the push over several layers instead of one big push at layer 8.

The pathway check (Results 4) found the L8 push is 2-5x the natural L8 difference and loses the natural pathway in
the middle layers; the dose sweep (Results 5) found that shrinking a single push switches philosophy off. Here the
same total on-direction push is split across layers, each layer pushed along its *own* diff-means direction:
  sum_L  (alpha / n) * p50_L * unit(diffmeans_L)          (n = number of layers)
so each layer gets a gentler push, closer to its natural size, and the later layers are nudged directly instead of
relying on L8's push to travel. Also tried: recreating the natural trajectory. The residual stream is a running
sum, so pushing each layer L by the *increment* of the natural difference,
  g * (diffmeans_L - diffmeans_{L-1})     for L = 0..K      ("natural path x g, up to L K")
makes the steered offset after layer L equal g * diffmeans_L (before the model's own response), i.e. the same
philosophy-minus-plain difference the model shows naturally, layer by layer.
Layer sweep: single layers 4-24, each at alpha 0.5 and at 2x its own natural difference, to test whether the
choice of layer 8 mattered (hypothesis H5 in LITERATURE.md).
Directions: the original contrast pairs (runs/diffmeans.pt).

Outputs under runs/multilayer/: gen.jsonl. Then:
  scripts/claude_judge.py --input multilayer/gen.jsonl --output multilayer/judge.jsonl
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
OUT = RUNS / "multilayer"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = Steerable()
    s.p50 = {int(k): v for k, v in json.loads((RUNS / "p50.json").read_text()).items()}
    dm = torch.load(RUNS / "diffmeans.pt")
    unit = lambda d: d.float() / d.float().norm()
    prompts = json.loads((ROOT / "data/eval_prompts.json").read_text())
    heldout = json.loads((ROOT / "data/heldout_prompts.json").read_text())
    all_prompts = prompts + heldout
    split = ["eval"] * len(prompts) + ["heldout"] * len(heldout)
    chats = [s.chat(p) for p in all_prompts]

    def spread(layers, alpha):
        return {L: alpha / len(layers) * s.p50[L] * unit(dm[L]) for L in layers}

    def natural_path(K, g):
        return {L: g * (dm[L].float() - (dm[L - 1].float() if L else 0)) for L in range(K + 1)}

    cfgs = [("baseline", None), ("single L8 α0.5", spread([8], 0.5))]
    for layers in ([8, 12, 16], [4, 8, 12, 16, 20], list(range(4, 25, 2))):
        tag = f"L{layers[0]}-{layers[-1]}" + (f" every 2 ({len(layers)} layers)" if len(layers) > 5 else f" ({len(layers)} layers)")
        for a in (0.5, 1.0):
            cfgs.append((f"spread {tag} total α{a}", spread(layers, a)))
    for K in (12, 24):
        for g in (1.0, 2.0, 3.0):
            cfgs.append((f"natural path ×{g} up to L{K}", natural_path(K, g)))
    # Layer sweep (literature hypothesis H5, "abstraction lives in the middle layers"): one layer at a time, at a
    # fixed alpha 0.5 and at twice that layer's own natural difference (the threshold found at L8 in Results 5).
    for L in (4, 8, 12, 16, 20, 24):
        if L != 8:
            cfgs.append((f"single L{L} α0.5", spread([L], 0.5)))
        cfgs.append((f"single L{L} 2×natural", {L: 2 * dm[L].float()}))
    # report push sizes
    for name, vec in cfgs[1:]:
        tot = sum(float(v.norm() / s.p50[L]) for L, v in vec.items())
        print(f"{name}: {len(vec)} layers, sum |push|/p50 = {tot:.2f}", flush=True)

    out = OUT / "gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()
    for i, (name, vec) in enumerate(cfgs):
        if name in done:
            continue
        t0 = time.time()
        with s.steer([], vec):
            outs = s.generate(chats, max_new_tokens=120)
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": "baseline" if vec is None else "steer",
                                "layers": sorted(vec) if vec else [], "prompts": all_prompts, "split": split,
                                "outputs": outs}) + "\n")
        print(f"[{i + 1}/{len(cfgs)}] {name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
