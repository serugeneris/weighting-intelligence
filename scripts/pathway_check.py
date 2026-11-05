"""Downstream pathway check: does an L8 push recreate the model's own philosophical pathway in later layers?

For each of the 32 contrast pairs we teacher-force the *plain* answer and record, on answer tokens, the mean
residual stream after every layer and the mean MLP neuron activations in layers 8-31, under:
  plain   - no push                                   (reference)
  philo   - the philosophical answer, no push         -> natural change  N = philo - plain
  omp     - plain + L8 16-neuron group, alpha 0.5     -> steered change  S = omp - plain
  dm      - plain + full diff-means L8, alpha 0.5
  ctrl0-2 - plain + the 3 matched random 16-neuron groups from validation (null reference)

The push is added at L8's MLP output, so the residual after L8 already contains it and every later
residual carries it along. To see what later layers do *in response*, we compare, per layer L > 8:
  update   cos(dS_L, dN_L) with d*_L = (*_L - *_{L-1}): change in what layer L itself writes
  response cos(S_L - S_8, N_L - N_8): everything layers 9..L added on top of the L8 state
  acts     cos over the 9,216 neurons of layer L: change in which neurons fire
  recreated (S_L - S_8) . unit(N_L - N_8) / |N_L - N_8|: fraction of the natural response reproduced
  off       |component of (S_L - S_8) orthogonal to the natural response| / |N_L - N_8|
Caveat: N also contains token-identity effects (philo and plain answers are different words), which no push
on plain text can reproduce; the random controls give the floor for these cosines.

Writes runs/pathway.pt (per-pair tensors) and runs/pathway.json (per-layer metrics with bootstrap 90% CIs).
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
L0, K, ALPHA = 8, 16, 0.5
N_BOOT = 1000


def conditions(s: Steerable, dm: dict) -> dict[str, dict]:
    """Same groups as scripts/validate_generate.py (first shortlist entry, rng seed 42)."""
    W = s.down[L0].weight.detach().float().cpu()
    U = W / W.norm(dim=0, keepdim=True)
    d = dm[L0] / dm[L0].norm()
    idx, coef, c = omp(U, ALPHA * s.p50[L0] * d, K)
    doses = (coef / c ** 2 / s.p50[L0]).tolist()
    conds = {"omp": {"pushes": [Push(L0, n, dz) for n, dz in zip(idx, doses)]},
             "dm": {"vectors": {L0: ALPHA * s.p50[L0] * d}}}
    rng = random.Random(42)
    for j in range(3):
        rn = rng.sample([n for n in range(s.n_neurons) if n not in idx], K)
        conds[f"ctrl{j}"] = {"pushes": [Push(L0, n, abs(dz) * rng.choice([-1, 1])) for n, dz in zip(rn, doses)]}
    print(f"omp group cos={c:.3f}, neurons={idx}", flush=True)
    return conds


def collect(s, pairs, conds):
    res_layers = sorted(s.layers)
    act_layers = [L for L in sorted(s.down) if L >= L0]
    names = ["plain", "philo", *conds]
    R = {n: [] for n in names}
    A = {n: [] for n in names}
    for i, p in enumerate(pairs):
        t0 = time.time()
        prompt = s.chat(p["q"])
        for n in names:
            text = prompt + (p["philo"] if n == "philo" else p["plain"])
            cfg = conds.get(n, {})
            with s.steer(cfg.get("pushes", []), cfg.get("vectors")):
                r, a = s.capture(text, len(prompt), res_layers, act_layers)
            R[n].append(torch.stack([r[L] for L in res_layers]))
            A[n].append(torch.stack([a[L] for L in act_layers]))
        print(f"[{i + 1}/{len(pairs)}] ({time.time() - t0:.0f}s)", flush=True)
    R = {n: torch.stack(v) for n, v in R.items()}   # [pairs, n_layers, d_model]
    A = {n: torch.stack(v) for n, v in A.items()}   # [pairs, n_act_layers, n_neurons]
    return R, A, act_layers


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def metrics(N, S, NA, SA, act_layers):
    """N, S: [B, n_layers, d] mean differences (possibly bootstrap means). NA, SA: [B, n_act, n_neurons]."""
    nresp, sresp = N - N[:, L0:L0 + 1], S - S[:, L0:L0 + 1]
    nupd, supd = N[:, 1:] - N[:, :-1], S[:, 1:] - S[:, :-1]       # index L-1 holds layer L's update
    nn2 = (nresp ** 2).sum(-1).clamp_min(1e-12)
    proj = (sresp * nresp).sum(-1) / nn2
    orth = sresp - proj.unsqueeze(-1) * nresp
    a0 = act_layers[0]
    return {
        "update": cos(supd, nupd)[:, L0:],                       # layers L0+1..
        "response": cos(sresp, nresp)[:, L0 + 1:],
        "resid": cos(S, N)[:, L0 + 1:],
        "acts": cos(SA, NA)[:, L0 + 1 - a0:],
        "recreated": proj[:, L0 + 1:],
        "off": (orth.norm(dim=-1) / nn2.sqrt())[:, L0 + 1:],
    }


def summarize(R, A, act_layers, names):
    g = torch.Generator().manual_seed(0)
    n_pairs = R["plain"].shape[0]
    W = torch.zeros(N_BOOT, n_pairs)
    W.scatter_add_(1, torch.randint(0, n_pairs, (N_BOOT, n_pairs), generator=g), torch.ones(N_BOOT, n_pairs))
    W /= n_pairs
    dN = R["philo"] - R["plain"]
    dNA = A["philo"] - A["plain"]
    out = {}
    for n in names:
        dS, dSA = R[n] - R["plain"], A[n] - A["plain"]
        point = metrics(dN.mean(0, keepdim=True), dS.mean(0, keepdim=True),
                        dNA.mean(0, keepdim=True), dSA.mean(0, keepdim=True), act_layers)
        boot = {k: [] for k in point}
        for b in W.split(100):                                   # chunk the bootstrap to bound memory
            m = metrics(torch.einsum("bp,pld->bld", b, dN), torch.einsum("bp,pld->bld", b, dS),
                        torch.einsum("bp,pld->bld", b, dNA), torch.einsum("bp,pld->bld", b, dSA), act_layers)
            for k in m:
                boot[k].append(m[k])
        out[n] = {}
        for k in point:
            bk = torch.cat(boot[k])
            lo, hi = bk.quantile(0.05, dim=0), bk.quantile(0.95, dim=0)
            out[n][k] = [[round(float(v), 4), round(float(l), 4), round(float(h), 4)]
                         for v, l, h in zip(point[k][0], lo, hi)]
    # Noise ceiling: split-half agreement of the natural change itself (16 vs 16 pairs).
    rng = random.Random(0)
    ceil_upd, ceil_acts = [], []
    for _ in range(200):
        perm = rng.sample(range(n_pairs), n_pairs)
        h1, h2 = perm[:n_pairs // 2], perm[n_pairs // 2:]
        u1, u2 = dN[h1].mean(0), dN[h2].mean(0)
        ceil_upd.append(cos(u1[1:] - u1[:-1], u2[1:] - u2[:-1])[L0:])
        ceil_acts.append(cos(dNA[h1].mean(0), dNA[h2].mean(0))[L0 + 1 - act_layers[0]:])
    out["split_half_natural"] = {"update": [round(float(v), 4) for v in torch.stack(ceil_upd).mean(0)],
                                 "acts": [round(float(v), 4) for v in torch.stack(ceil_acts).mean(0)]}
    # Sanity: the steered residual right after L8 is exactly the push; diff-means after L8 matches runs/diffmeans.pt.
    out["sanity"] = {n: {"resid_change_before_L8": round(float((R[n][:, :L0] - R["plain"][:, :L0]).norm(dim=-1).max()), 4),
                         "acts_change_at_L8": round(float((A[n][:, 0] - A["plain"][:, 0]).norm(dim=-1).max()), 4)}
                     for n in names}
    return out


def main():
    s = Steerable()
    s.p50 = {int(k): v for k, v in json.loads((RUNS / "p50.json").read_text()).items()}
    dm = torch.load(RUNS / "diffmeans.pt")
    pairs = json.loads((ROOT / "data/contrast_pairs.json").read_text())
    conds = conditions(s, dm)
    R, A, act_layers = collect(s, pairs, conds)
    torch.save({"R": {k: v.half() for k, v in R.items()}, "A": {k: v.half() for k, v in A.items()},
                "act_layers": act_layers}, RUNS / "pathway.pt")
    dN8 = (R["philo"] - R["plain"]).mean(0)[L0]
    print(f"natural change at L8 vs runs/diffmeans.pt: cos={float(cos(dN8, dm[L0].float())):.4f}", flush=True)
    names = list(conds)
    out = {"layers": list(range(L0 + 1, s.n_layers)), "conditions": names, **summarize(R, A, act_layers, names)}
    (RUNS / "pathway.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["sanity"]), flush=True)
    hdr = "layer " + " ".join(f"{n:>8}" for n in names) + "   ceiling"
    for key in ("update", "acts", "response", "recreated", "off"):
        print(f"\n{key}" + (" (cos with natural)" if key in ("update", "acts", "response") else ""))
        print(hdr)
        for j, L in enumerate(out["layers"]):
            ceil = out["split_half_natural"].get(key)
            print(f"L{L:<5}" + " ".join(f"{out[n][key][j][0]:>8.3f}" for n in names)
                  + (f"   {ceil[j]:.3f}" if ceil else ""))


if __name__ == "__main__":
    main()
