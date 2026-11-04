"""Rank Qwen3.5-4B MLP neurons by how differently they fire on philosophical vs plain answers.

For each contrast pair (same question, two answers) we take the mean activation of every MLP neuron
(input to down_proj) over the answer tokens only, then compute per-neuron paired statistics:
    t = mean(d) / (std(d) / sqrt(N)),  d = act_philo - act_plain
    consistency = fraction of pairs where sign(d) == sign(mean d)
Outputs runs/candidates.json, runs/p50.json and runs/diffmeans.pt (per-layer residual diff-means).
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wi.steer import Steerable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)


def main(top_k: int = 60):
    pairs = json.loads((ROOT / "data/contrast_pairs.json").read_text())
    s = Steerable()
    print(f"device={s.device} layers={s.n_layers} neurons/layer={s.n_neurons}")

    s.calibrate([s.chat(p["q"], p["plain"]) for p in pairs])
    (RUNS / "p50.json").write_text(json.dumps(s.p50, indent=1))

    layers = sorted(s.down)
    D = []  # [N, L, n]
    R = {L: [] for L in layers}
    for i, p in enumerate(pairs):
        prompt = s.chat(p["q"])
        a_phi = s.neuron_acts(prompt + p["philo"], start_char=len(prompt))
        a_pln = s.neuron_acts(prompt + p["plain"], start_char=len(prompt))
        D.append(torch.stack([a_phi[L] - a_pln[L] for L in layers]))
        r_phi = s.resid_mean(prompt + p["philo"], len(prompt), layers)
        r_pln = s.resid_mean(prompt + p["plain"], len(prompt), layers)
        for L in layers:
            R[L].append(r_phi[L] - r_pln[L])
        print(f"pair {i + 1}/{len(pairs)}", end="\r", flush=True)
    print()

    D = torch.stack(D)  # N x L x n
    N = D.shape[0]
    mean, std = D.mean(0), D.std(0) + 1e-6
    t = mean / (std / N ** 0.5)
    consistency = ((D.sign() == mean.sign().unsqueeze(0)).float().mean(0))

    # Only consider neurons that are reliably directional; rank by |t|.
    score = t.abs() * (consistency >= 0.8)
    flat = score.flatten().topk(top_k)
    cands = []
    for v, idx in zip(flat.values.tolist(), flat.indices.tolist()):
        li, n = divmod(idx, s.n_neurons)
        L = layers[li]
        cands.append({
            "layer": L, "neuron": n,
            "t": round(t[li, n].item(), 2),
            "mean_diff": round(mean[li, n].item(), 4),
            "consistency": round(consistency[li, n].item(), 3),
            "sign": 1 if mean[li, n] > 0 else -1,
        })
    (RUNS / "candidates.json").write_text(json.dumps(cands, indent=1))

    diffmeans = {L: torch.stack(R[L]).mean(0) for L in layers}
    torch.save(diffmeans, RUNS / "diffmeans.pt")

    # Per-layer summary: how many strong neurons live in each layer
    strong = (score > 6).sum(1)
    print("layer  strong(|t|>6, cons>=.8)  max|t|  diffmean_norm/p50")
    for li, L in enumerate(layers):
        print(f"{L:5d}  {int(strong[li]):8d}  {score[li].max().item():8.1f}  "
              f"{diffmeans[L].norm().item() / s.p50[L]:.3f}")
    print("\nTop 20 candidates:")
    for c in cands[:20]:
        print(c)


if __name__ == "__main__":
    main()
