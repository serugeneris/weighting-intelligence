"""Push candidate neurons on neutral prompts and score philosophy vs coherence.

Configs: baseline, top-K candidate neurons x doses (pushed in their philosophical sign),
random-neuron controls, and the per-layer diff-means residual direction.
Scores (by the *unsteered* model): philo 0-9, coherence 0-9, perplexity of the completion.
Appends to runs/sweep.jsonl (resumable) and prints a ranked table.
"""
import argparse
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wi.steer import COHERENCE_Q, PHILO_Q, Push, Steerable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def build_configs(k: int, doses: list[float], n_rand: int, dm_layers: list[int], dm_alphas: list[float], n_neurons: int):
    cands = json.loads((RUNS / "candidates.json").read_text())[:k]
    cfgs = [{"name": "baseline", "kind": "baseline"}]
    for c in cands:
        for d in doses:
            cfgs.append({"name": f"L{c['layer']}N{c['neuron']}{'+' if c['sign'] > 0 else '-'}@{d}", "kind": "neuron",
                         "pushes": [[c["layer"], c["neuron"], c["sign"] * d]]})
    rng = random.Random(0)
    for i in range(n_rand):
        L, n, sg = rng.randint(6, 25), rng.randrange(n_neurons), rng.choice([-1, 1])
        for d in doses:
            cfgs.append({"name": f"rand{i}:L{L}N{n}{'+' if sg > 0 else '-'}@{d}", "kind": "random",
                         "pushes": [[L, n, sg * d]]})
    for L in dm_layers:
        for a in dm_alphas:
            cfgs.append({"name": f"diffmeans L{L}@{a}", "kind": "diffmeans", "layer": L, "alpha": a})
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--doses", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--n-rand", type=int, default=6)
    ap.add_argument("--dm-layers", type=int, nargs="+", default=[8, 12, 16, 20])
    ap.add_argument("--dm-alphas", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--out", default="sweep.jsonl")
    args = ap.parse_args()

    s = Steerable()
    s.p50 = {int(k): v for k, v in json.loads((RUNS / "p50.json").read_text()).items()}
    dm = torch.load(RUNS / "diffmeans.pt")
    prompts = json.loads((ROOT / "data/eval_prompts.json").read_text())
    chats = [s.chat(p) for p in prompts]

    cfgs = build_configs(args.k, args.doses, args.n_rand, args.dm_layers, args.dm_alphas, s.n_neurons)
    run(s, cfgs, RUNS / args.out, chats, dm, args.max_new)


def run(s, cfgs, out_path, chats, dm, max_new=96):
    done = set()
    if out_path.exists():
        done = {json.loads(l)["name"] for l in out_path.read_text().splitlines() if l.strip()}
    print(f"{len(cfgs)} configs, {len(done)} already done")
    for ci, cfg in enumerate(cfgs):
        if cfg["name"] in done:
            continue
        t0 = time.time()
        pushes = [Push(*p) for p in cfg.get("pushes", [])]
        vectors = None
        if cfg["kind"] == "diffmeans":
            v = dm[cfg["layer"]]
            vectors = {cfg["layer"]: cfg["alpha"] * s.p50[cfg["layer"]] * v / v.norm()}
        with s.steer(pushes, vectors):
            outs = s.generate(chats, max_new_tokens=max_new)
        philo = s.judge(outs, PHILO_Q)
        coh = s.judge(outs, COHERENCE_Q)
        ppl = [s.ppl(c, o) for c, o in zip(chats, outs)]
        rec = {**cfg, "philo": st.mean(philo), "coh": st.mean(coh), "ppl": st.median(ppl),
               "philo_each": philo, "coh_each": coh, "outputs": outs}
        with out_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{ci + 1}/{len(cfgs)}] {cfg['name']:<32} philo={rec['philo']:.2f} coh={rec['coh']:.2f} "
              f"ppl={rec['ppl']:.1f}  ({time.time() - t0:.0f}s)", flush=True)

    recs = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    print("\nRanked by philo (coh >= 5):")
    for r in sorted((r for r in recs if r["coh"] >= 5), key=lambda r: -r["philo"])[:20]:
        print(f"  {r['name']:<28} {r['kind']:<9} philo={r['philo']:.2f} coh={r['coh']:.2f} ppl={r['ppl']:.1f}")


if __name__ == "__main__":
    main()
