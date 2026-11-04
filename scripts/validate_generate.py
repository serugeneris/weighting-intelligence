"""Generate outputs for the shortlisted neuron groups plus matched random-group controls.

Prompts: the 8 original eval prompts + 8 held-out prompts never used for selection.
Matched control for a group G (layer L, k neurons, doses d_i): k random neurons in the same layer,
given the same |d_i| values with random signs. Same layer, same count, same total push magnitude --
only the choice of neurons differs.
Writes runs/validate.jsonl (resumable); scoring is done separately by scripts/claude_judge.py.
"""
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.neuron_search import omp  # noqa: E402
from wi.steer import Push, Steerable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SHORTLIST = [(8, 16, 0.5), (8, 16, 1.0), (16, 16, 0.5), (16, 64, 1.0)]  # (layer, k, alpha), rescaled OMP
N_CONTROLS = 3


def main():
    s = Steerable()
    s.p50 = {int(k): v for k, v in json.loads((RUNS / "p50.json").read_text()).items()}
    dm = torch.load(RUNS / "diffmeans.pt")
    prompts = json.loads((ROOT / "data/eval_prompts.json").read_text())
    heldout = json.loads((ROOT / "data/heldout_prompts.json").read_text())
    all_prompts = prompts + heldout
    split = ["eval"] * len(prompts) + ["heldout"] * len(heldout)
    chats = [s.chat(p) for p in all_prompts]

    cfgs = [{"name": "baseline", "kind": "baseline", "pushes": []},
            {"name": "diffmeans L8@0.5", "kind": "diffmeans", "layer": 8, "alpha": 0.5, "pushes": []}]
    rng = random.Random(42)
    for L, k, a in SHORTLIST:
        W = s.down[L].weight.detach().float().cpu()
        U = W / W.norm(dim=0, keepdim=True)
        d = dm[L] / dm[L].norm()
        idx, coef, c = omp(U, a * s.p50[L] * d, k)
        doses = (coef / c ** 2 / s.p50[L]).tolist()
        name = f"omp* L{L} k={k} a={a}"
        cfgs.append({"name": name, "kind": "omp", "pushes": [[L, n, dz] for n, dz in zip(idx, doses)]})
        for j in range(N_CONTROLS):
            rn = rng.sample([n for n in range(s.n_neurons) if n not in idx], k)
            cfgs.append({"name": f"ctrl{j} of {name}", "kind": "control", "matches": name,
                         "pushes": [[L, n, abs(dz) * rng.choice([-1, 1])] for n, dz in zip(rn, doses)]})

    out = RUNS / "validate.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()
    for i, cfg in enumerate(cfgs):
        if cfg["name"] in done:
            continue
        t0 = time.time()
        vectors = None
        if cfg["kind"] == "diffmeans":
            v = dm[cfg["layer"]]
            vectors = {cfg["layer"]: cfg["alpha"] * s.p50[cfg["layer"]] * v / v.norm()}
        with s.steer([Push(*p) for p in cfg["pushes"]], vectors):
            outs = s.generate(chats, max_new_tokens=120)
        with out.open("a") as f:
            f.write(json.dumps({**cfg, "prompts": all_prompts, "split": split, "outputs": outs}) + "\n")
        print(f"[{i + 1}/{len(cfgs)}] {cfg['name']} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
