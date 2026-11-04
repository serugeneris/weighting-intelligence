"""Write-side neuron search: find neurons whose W_down columns point along the philosophy direction.

A push on neuron n of layer L adds dose * p50_L * unit(W_down_L[:, n]) to the residual stream, so what
matters for steering is the *output column*, not when the neuron fires. For each target layer we:
  1. rank single neurons by cos(W_down[:, n], diffmeans_L), push them in the aligned sign;
  2. greedily approximate target = alpha * p50 * unit(diffmeans_L) with k columns (orthogonal matching
     pursuit) and convert the least-squares coefficients into per-neuron doses (dose = coef / p50).
Controls: random neurons in the same layers, and the full diff-means vector as reference.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sweep import run  # noqa: E402
from wi.steer import Steerable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def omp(U: torch.Tensor, target: torch.Tensor, k: int):
    """Orthogonal matching pursuit over unit columns U (d x n). Returns (indices, coefs, cos(recon, target))."""
    idx, r = [], target.clone()
    for _ in range(k):
        j = int((r @ U).abs().argmax())
        idx.append(j)
        A = U[:, idx]
        coef = torch.linalg.lstsq(A, target.unsqueeze(1)).solution.squeeze(1)
        r = target - A @ coef
    recon = U[:, idx] @ coef
    return idx, coef, float(torch.nn.functional.cosine_similarity(recon, target, dim=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--doses", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--n-rand", type=int, default=2)
    ap.add_argument("--out", default="neuron_search.jsonl")
    ap.add_argument("--rescale", action="store_true",
                    help="divide OMP doses by cos^2 so the push's projection on the direction equals alpha")
    ap.add_argument("--alphas", type=float, nargs="+", default=None, help="OMP alphas (default: --alpha)")
    ap.add_argument("--omp-only", action="store_true", help="skip single-neuron and random configs")
    args = ap.parse_args()

    s = Steerable()
    s.p50 = {int(k): v for k, v in json.loads((RUNS / "p50.json").read_text()).items()}
    dm = torch.load(RUNS / "diffmeans.pt")

    cfgs = [{"name": "baseline", "kind": "baseline"}]
    report = {}
    rng = random.Random(1)
    for L in args.layers:
        W = s.down[L].weight.detach().float().cpu()
        U = W / W.norm(dim=0, keepdim=True)
        d = dm[L] / dm[L].norm()
        cos = d @ U
        top = cos.abs().topk(args.top).indices.tolist()
        report[L] = {"top_single": [(n, round(float(cos[n]), 3)) for n in top], "omp": {}}
        for n in ([] if args.omp_only else top):
            sg = 1 if cos[n] > 0 else -1
            for dose in args.doses:
                cfgs.append({"name": f"align L{L}N{n}{'+' if sg > 0 else '-'}@{dose}", "kind": "aligned",
                             "pushes": [[L, n, sg * dose]]})
        for alpha in args.alphas or [args.alpha]:
            target = alpha * s.p50[L] * d
            for k in args.ks:
                idx, coef, c = omp(U, target, k)
                if args.rescale:
                    coef = coef / c ** 2
                doses = (coef / s.p50[L]).tolist()
                tag = f"omp{'*' if args.rescale else ''} L{L} k={k} a={alpha}"
                report[L]["omp"][f"{k}@{alpha}"] = {"cos": round(c, 3), "max_abs_dose": round(max(map(abs, doses)), 3)}
                cfgs.append({"name": f"{tag} (cos {c:.2f})", "kind": "omp",
                             "pushes": [[L, n, dz] for n, dz in zip(idx, doses)]})
        for i in range(0 if args.omp_only else args.n_rand):
            n, sg = rng.randrange(s.n_neurons), rng.choice([-1, 1])
            for dose in args.doses:
                cfgs.append({"name": f"rand L{L}N{n}{'+' if sg > 0 else '-'}@{dose}", "kind": "random",
                             "pushes": [[L, n, sg * dose]]})
        if not args.omp_only:
            cfgs.append({"name": f"diffmeans L{L}@{args.alpha}", "kind": "diffmeans", "layer": L, "alpha": args.alpha})

    print(json.dumps(report, indent=1))
    (RUNS / "neuron_search_report.json").write_text(json.dumps(report, indent=1))
    prompts = json.loads((ROOT / "data/eval_prompts.json").read_text())
    run(s, cfgs, RUNS / args.out, [s.chat(p) for p in prompts], dm)


if __name__ == "__main__":
    main()
