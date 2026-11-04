"""Astra-style weight-space edits: bake the philosophy direction into mlp.down_proj, no gradients.

Mirrors "Sparks of Weight-Space Programming" (Takehi et al., MIT, 2026), where GPT-6 Astra edited
Qwen3.5-0.8B's down_proj with a masked rank-1 projection: W' = W - 6 * outer(u, mask * (u @ W)).
Here u = unit philosophy direction (diff-means at layer L) and we try two masked rank-1 forms:

  amp  (Astra's form, opposite sign): W' = W + beta * outer(u, mask * (u @ W))
        -> amplifies whatever the masked neurons already write along u (beta < 0 suppresses it).
  cond (activation-conditioned):      W' = W + alpha * p50 * outer(u, m_mask) / ||m_mask||^2
        m = mean neuron-activation difference (philo - plain) at layer L, restricted to the top-k |t| neurons.
        -> the push along u is alpha*p50 * (m.act)/||m||^2: grows when the layer's activations look
           "philosophical", so the edit amplifies philosophy instead of adding a constant.
A constant push (what activation steering does) is impossible as a pure weight edit: down_proj has no bias.

Protocol (as in the paper): candidates are scored on the *practice* prompts (data/eval_prompts.json) with the
cheap self-judge; the best ones that don't break coherence are then generated on practice + held-out prompts
and written to runs/astra.jsonl for the blind Claude judge (scripts/claude_judge.py --input astra.jsonl).
"""
import contextlib
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wi.steer import COHERENCE_Q, PHILO_Q, Steerable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
LAYERS = [8, 16]


@contextlib.contextmanager
def edited(s: Steerable, L: int, delta: torch.Tensor):
    W = s.down[L].weight
    orig = W.data.clone()
    W.data = (W.data.float() + delta.to(W.device)).to(W.dtype)
    try:
        yield
    finally:
        W.data = orig


def layer_stats(s: Steerable, pairs, layers):
    """Per-neuron mean activation diff and paired t at the given layers."""
    D = {L: [] for L in layers}
    for p in pairs:
        prompt = s.chat(p["q"])
        a = s.neuron_acts(prompt + p["philo"], len(prompt), layers)
        b = s.neuron_acts(prompt + p["plain"], len(prompt), layers)
        for L in layers:
            D[L].append(a[L] - b[L])
    out = {}
    for L in layers:
        X = torch.stack(D[L])
        m = X.mean(0)
        out[L] = (m, m / (X.std(0) / len(X) ** 0.5 + 1e-6))
    return out


def make_delta(s, dm, stats, form, L, k, strength):
    W = s.down[L].weight.detach().float().cpu()
    u = dm[L] / dm[L].norm()
    if form == "amp":
        proj = u @ W                                      # what each column writes along u
        mask = torch.zeros_like(proj)
        mask[proj.abs().topk(k).indices] = 1
        return strength * torch.outer(u, mask * proj)
    m, t = stats[L]
    mm = torch.zeros_like(m)
    idx = t.abs().topk(k).indices
    mm[idx] = m[idx]
    return strength * s.p50[L] * torch.outer(u, mm) / mm.norm() ** 2


def score(s, chats, outs):
    return st.mean(s.judge(outs, PHILO_Q)), st.mean(s.judge(outs, COHERENCE_Q)), s.judge(outs, COHERENCE_Q)


def main():
    s = Steerable()
    s.p50 = {int(k): v for k, v in json.loads((RUNS / "p50.json").read_text()).items()}
    dm = torch.load(RUNS / "diffmeans.pt")
    pairs = json.loads((ROOT / "data/contrast_pairs.json").read_text())
    practice = json.loads((ROOT / "data/eval_prompts.json").read_text())
    heldout = json.loads((ROOT / "data/heldout_prompts.json").read_text())
    pchats = [s.chat(p) for p in practice]

    print("diagnose: neuron stats on contrast pairs ...", flush=True)
    stats = layer_stats(s, pairs, LAYERS)

    base_out = s.generate(pchats, max_new_tokens=96)
    bp, bc, bce = score(s, pchats, base_out)
    print(f"baseline practice philo={bp:.2f} coh={bc:.2f}", flush=True)

    grid = []
    for L in LAYERS:
        for k in (64, 512, 9216):
            for beta in (2.0, 4.0, 8.0):
                grid.append(("amp", L, k, beta))
            for alpha in (0.5, 1.0, 2.0):
                grid.append(("cond", L, k, alpha))
    log = RUNS / "astra_search.jsonl"
    results = []
    for form, L, k, a in grid:
        t0 = time.time()
        delta = make_delta(s, dm, stats, form, L, k, a)
        rel = float(delta.norm() / s.down[L].weight.float().norm())
        with edited(s, L, delta):
            outs = s.generate(pchats, max_new_tokens=96)
        p, c, ce = score(s, pchats, outs)
        breaks = sum(x <= y - 2 for x, y in zip(ce, bce))
        r = {"form": form, "layer": L, "k": k, "strength": a, "rel_change": rel, "philo": p, "coh": c,
             "breaks": breaks, "sample": outs[2][:200]}
        results.append(r)
        with log.open("a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"{form:<4} L{L} k={k:<5} s={a:<4} |dW|/|W|={rel:.3f}  philo={p:.2f} coh={c:.2f} breaks={breaks} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # Keep the best 2 edits per form that don't break coherence on practice prompts.
    ok = [r for r in results if r["breaks"] <= 2]
    best = []
    for form in ("amp", "cond"):
        best += sorted((r for r in ok if r["form"] == form), key=lambda r: -r["philo"])[:2]
    print("\nselected for held-out:", [(b["form"], b["layer"], b["k"], b["strength"]) for b in best], flush=True)

    all_prompts = practice + heldout
    split = ["eval"] * len(practice) + ["heldout"] * len(heldout)
    chats = [s.chat(p) for p in all_prompts]
    out = RUNS / "astra.jsonl"
    out.unlink(missing_ok=True)
    recs = [{"name": "baseline", "kind": "baseline"}] + [
        {"name": f"edit {b['form']} L{b['layer']} k={b['k']} s={b['strength']}", "kind": "edit", **b} for b in best]
    for r in recs:
        if r["kind"] == "baseline":
            outs = s.generate(chats, max_new_tokens=120)
        else:
            with edited(s, r["layer"], make_delta(s, dm, stats, r["form"], r["layer"], r["k"], r["strength"])):
                outs = s.generate(chats, max_new_tokens=120)
        with out.open("a") as f:
            f.write(json.dumps({**r, "prompts": all_prompts, "split": split, "outputs": outs}) + "\n")
        print(f"held-out generated: {r['name']}", flush=True)


if __name__ == "__main__":
    main()
