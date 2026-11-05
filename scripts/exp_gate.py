"""Experiment L: why does the rank-1 LoRA work? Separate its gate (when, how much) from its direction (where).

The rank-1 LoRA of experiment D adds, at every token t of layer 8's down_proj output,
    push_t = g_t * unit(B),      g_t = scale * |B| * (A . a_t)     (a_t = the 9216 MLP activations)
a token-dependent size g_t (the "gate", read side A) along one fixed direction B (the write side). Steering adds a
constant size along diff-means. This experiment swaps the two parts (LITERATURE.md H2, gated/conditional steering):

  LoRA                       gate A, direction B         (the trained update itself)
  constant along B           fixed size c, direction B   (c = 0.12, 0.2, 0.4 p50; 0.12 and 0.2 are the LoRA's mean
                                                          push on answer and on prompt tokens, Results 10)
  gated diff-means           gate A, direction diff-means (x1, x2)
  gated random direction     gate A, a random unit direction (x1, x2): control for "any direction with this gate"
  LoRA gate, sign flipped    -g_t along B: does the sign of the read matter?

Round 2 (--round 2): constant pushes along B looked more efficient than diff-means, so they are mapped further
(0.4-0.8 p50) next to the LoRA amplified x1.5 and x2 and diff-means alpha 0.5, in one pool.

Outputs under runs/gate/: gen.jsonl, gen_round2.jsonl. Then:
  scripts/claude_judge.py --input gate/gen.jsonl --output gate/judge.jsonl
  scripts/claude_judge.py --input gate/gen_round2.jsonl --output gate/judge_round2.jsonl
"""
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wi.steer import Steerable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "gate"
L = 8


@contextlib.contextmanager
def gated(s, A, gain, direction):
    """Add gain * (A . a_t) * direction to layer L's down_proj output at every token."""
    A = A.to(s.device, torch.float32)
    d = direction.to(s.device, torch.float32)

    def hook(_m, inp, out):
        g = inp[0].float() @ A                              # [B, T]
        return out + (gain * g[..., None] * d).to(out.dtype)
    h = s.down[L].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    s = Steerable()
    p50 = json.loads((RUNS / "p50.json").read_text())[str(L)]
    lo = torch.load(RUNS / "distill/down8-r1.pt")[f"layers.{L}.mlp.down_proj"]
    A, B, sc = lo["A"][0].float(), lo["B"][:, 0].float(), lo["scale"]
    uB = B / B.norm()
    dm = torch.load(RUNS / "diffmeans.pt")[L].float()
    udm = dm / dm.norm()
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(B.shape[0], generator=g)
    urnd = rnd / rnd.norm()
    print(f"cos(B, diffmeans) = {float(uB @ udm):.3f}, cos(B, random) = {float(uB @ urnd):.3f}", flush=True)
    G = sc * float(B.norm())                                 # the LoRA = gated(A, G, unit(B))

    prompts = json.loads((ROOT / "data/eval_prompts.json").read_text())
    heldout = json.loads((ROOT / "data/heldout_prompts.json").read_text())
    all_prompts = prompts + heldout
    split = ["eval"] * len(prompts) + ["heldout"] * len(heldout)
    chats = [s.chat(p) for p in all_prompts]
    cfgs = [("baseline", None), ("LoRA (gate A, direction B)", ("gate", G, uB))]
    for c in (0.12, 0.2, 0.4):
        cfgs.append((f"constant {c}·p50 along B", ("const", c * p50 * uB)))
    for k in (1, 2):
        cfgs.append((f"gate A, diff-means direction ×{k}", ("gate", k * G, udm)))
        cfgs.append((f"gate A, random direction ×{k}", ("gate", k * G, urnd)))
    cfgs.append(("gate A sign flipped, direction B", ("gate", -G, uB)))

    if args.round == 2:
        cfgs = [("baseline", None), ("diff-means L8 α0.5", ("const", 0.5 * p50 * udm))]
        cfgs += [(f"LoRA ×{k}", ("gate", k * G, uB)) for k in (1, 1.5, 2)]
        cfgs += [(f"constant {c}·p50 along B", ("const", c * p50 * uB)) for c in (0.4, 0.5, 0.6, 0.8)]
    out = OUT / ("gen.jsonl" if args.round == 1 else "gen_round2.jsonl")
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()
    for i, (name, c) in enumerate(cfgs):
        if name in done:
            continue
        t0 = time.time()
        if c is None:
            ctx = contextlib.nullcontext()
        elif c[0] == "const":
            ctx = s.steer([], {L: c[1]})
        else:
            ctx = gated(s, A, c[1], c[2])
        with ctx:
            outs = s.generate(chats, max_new_tokens=120)
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": "baseline" if c is None else "steer", "prompts": all_prompts,
                                "split": split, "outputs": outs}) + "\n")
        print(f"[{i + 1}/{len(cfgs)}] {name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
