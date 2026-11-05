"""Less overdrive: does a smaller L8 push keep the philosophy and lose less helpfulness?

The pathway check (scripts/pathway_check.py) found that the validated pushes are 2-5x the natural L8
philosophical difference (0.24*p50) and that the 16-neuron group's push (1.11*p50, cos 0.45) sets off an
off-pathway disturbance as large as random groups of the same size. This sweep shrinks the push two ways:
  - smaller alpha (on-direction size): 0.25, 0.35, 0.5
  - larger OMP groups (better aligned, so less off-direction push for the same alpha): k = 16, 64, 256
plus the full diff-means direction at the same alphas, and one matched random group per OMP config
(same layer, same k, same |doses|, random signs).

  --stage pathway   forward passes on the 32 contrast pairs; per-config disturbance and pathway match,
                    averaged over layer bands -> runs/dose_pathway.json
  --stage generate  greedy generations on the 8 practice + 8 held-out prompts -> runs/dose_sweep.jsonl,
                    then score with: scripts/claude_judge.py --input dose_sweep.jsonl --output claude_judge_dose.jsonl
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.neuron_search import omp  # noqa: E402
from scripts.pathway_check import collect, metrics  # noqa: E402
from wi.steer import Push, Steerable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
L0 = 8
KS = (16, 64, 256)
ALPHAS = (0.25, 0.35, 0.5)
BANDS = {"L9-13": (9, 13), "L14-23": (14, 23), "L24-31": (24, 31)}


def configs(s: Steerable, dm: dict) -> list[dict]:
    W = s.down[L0].weight.detach().float().cpu()
    U = W / W.norm(dim=0, keepdim=True)
    d = dm[L0] / dm[L0].norm()
    cfgs = []
    for a in ALPHAS:
        cfgs.append({"name": f"diffmeans L{L0}@{a}", "kind": "diffmeans", "pushes": [],
                     "vector": (a * s.p50[L0] * d).tolist()})
    for k in KS:
        idx, coef0, c = omp(U, s.p50[L0] * d, k)              # OMP is scale-free: coefs scale linearly with alpha
        for a in ALPHAS:
            doses = (a * coef0 / c ** 2 / s.p50[L0]).tolist()
            name = f"omp* L{L0} k={k} a={a}"
            cfgs.append({"name": name, "kind": "omp", "cos": c, "pushes": [[L0, n, dz] for n, dz in zip(idx, doses)]})
            rng = random.Random(f"{k}-{a}")
            rn = rng.sample([n for n in range(s.n_neurons) if n not in idx], k)
            cfgs.append({"name": f"ctrl0 of {name}", "kind": "control", "matches": name,
                         "pushes": [[L0, n, abs(dz) * rng.choice([-1, 1])] for n, dz in zip(rn, doses)]})
    return cfgs


def steer_args(cfg):
    vec = {L0: torch.tensor(cfg["vector"])} if cfg.get("vector") else None
    return [Push(*p) for p in cfg["pushes"]], vec


def stage_pathway(s, cfgs):
    pairs = json.loads((ROOT / "data/contrast_pairs.json").read_text())
    conds = {}
    for c in cfgs:
        pushes, vec = steer_args(c)
        conds[c["name"]] = {"pushes": pushes, "vectors": vec}
    R, A, act_layers = collect(s, pairs, conds)
    dN, dNA = (R["philo"] - R["plain"]).mean(0, keepdim=True), (A["philo"] - A["plain"]).mean(0, keepdim=True)
    layers = list(range(L0 + 1, s.n_layers))
    nat8 = float(dN[0, L0].norm() / s.p50[L0])
    out = {"natural_L8_over_p50": round(nat8, 3), "bands": BANDS, "configs": {}}
    print(f"natural L8 difference = {nat8:.2f} p50\n")
    print(f"{'config':<30}{'push/p50':>9}" + "".join(f"{b + ' ' + m:>17}" for m in ("resp", "off") for b in BANDS))
    for c in cfgs:
        n = c["name"]
        dS, dSA = (R[n] - R["plain"]).mean(0, keepdim=True), (A[n] - A["plain"]).mean(0, keepdim=True)
        m = metrics(dN, dS, dNA, dSA, act_layers)
        row = {"push_over_p50": round(float(dS[0, L0].norm() / s.p50[L0]), 3)}
        for key in ("response", "acts", "recreated", "off"):
            for b, (lo, hi) in BANDS.items():
                sel = [layers.index(L) for L in range(lo, hi + 1)]
                row[f"{key} {b}"] = round(float(m[key][0, sel].mean()), 3)
        out["configs"][n] = row
        print(f"{n:<30}{row['push_over_p50']:>9.2f}"
              + "".join(f"{row[f'{key} {b}']:>17.2f}" for key in ("response", "off") for b in BANDS), flush=True)
    (RUNS / "dose_pathway.json").write_text(json.dumps(out, indent=1))


def stage_generate(s, cfgs):
    prompts = json.loads((ROOT / "data/eval_prompts.json").read_text())
    heldout = json.loads((ROOT / "data/heldout_prompts.json").read_text())
    all_prompts = prompts + heldout
    split = ["eval"] * len(prompts) + ["heldout"] * len(heldout)
    chats = [s.chat(p) for p in all_prompts]
    out = RUNS / "dose_sweep.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [{"name": "baseline", "kind": "baseline", "pushes": []}] + cfgs
    for i, cfg in enumerate(todo):
        if cfg["name"] in done:
            continue
        t0 = time.time()
        pushes, vec = steer_args(cfg)
        with s.steer(pushes, vec):
            outs = s.generate(chats, max_new_tokens=120)
        rec = {k: v for k, v in cfg.items() if k != "vector"}
        with out.open("a") as f:
            f.write(json.dumps({**rec, "prompts": all_prompts, "split": split, "outputs": outs}) + "\n")
        print(f"[{i + 1}/{len(todo)}] {cfg['name']} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pathway", "generate"], required=True)
    args = ap.parse_args()
    s = Steerable()
    s.p50 = {int(k): v for k, v in json.loads((RUNS / "p50.json").read_text()).items()}
    dm = torch.load(RUNS / "diffmeans.pt")
    cfgs = configs(s, dm)
    (stage_pathway if args.stage == "pathway" else stage_generate)(s, cfgs)


if __name__ == "__main__":
    main()
