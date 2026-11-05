"""Experiment H: change the push's geometry instead of its size or place (LITERATURE.md H4, plus an orthogonal direction).

Two ideas to lower the helpfulness cost at layer 8.

1. Remove the "drop the answer" part of the direction. The original contrast pairs compared a plain answer with a
   philosophical text that *replaces* the answer; experiment A's "appended" pairs compared it with the same answer plus
   a reflection. So  drop = diffmeans_original - diffmeans_appended  approximates "stop answering", and pushing along
   a direction with that component projected out should keep the reflection and lose less of the answer.
   Measured first (L8): the appended direction has cos 0.92 with the original and 40% of its size, and drop has
   cos 0.97 with the original, so the three are nearly collinear; projecting drop out of the original leaves a small
   remainder (cos 0.23 with the original). We still test it, plus the appended and woven directions with drop
   projected out (60% and ~50% of their length survive).
     dir  = unit(d - (d . unit(drop)) unit(drop))      pushed as  alpha * p50 * dir
2. Norm-preserving pushes (Selective Steering, COAST, the angle/norm paper). Adding a vector also makes the residual
   longer, and a longer residual is out of distribution for the next layers. Here the residual after L8 is
     h' = (h + alpha * p50 * unit(d)) * (gain * |h| / |h + alpha * p50 * unit(d)|)
   i.e. it turns toward d by the same angle as the plain push but keeps its length (gain 1), or grows it by 10%
   (gain 1.1, the "mild norm increase" the geometric paper found best for refusal).

Outputs under runs/geometry/: gen.jsonl. Then:
  scripts/claude_judge.py --input geometry/gen.jsonl --output geometry/judge.jsonl
"""
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
OUT = RUNS / "geometry"
L = 8


def unit(x):
    x = x.float()
    return x / x.norm()


def without(d, drop):
    u = unit(drop)
    return unit(d.float() - (d.float() @ u) * u)


@contextlib.contextmanager
def norm_push(s, layer, v, gain=1.0):
    """Turn the residual after `layer` toward v by the same angle as adding v, at gain x its original length."""
    v = v.to(s.device, torch.float32)

    def hook(_m, _i, o):
        h = o[0] if isinstance(o, tuple) else o
        hf = h.float()
        n = hf.norm(dim=-1, keepdim=True)
        new = hf + v
        new = (new * (gain * n / new.norm(dim=-1, keepdim=True))).to(h.dtype)
        return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
    hk = s.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        hk.remove()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = Steerable()
    p50 = json.loads((RUNS / "p50.json").read_text())[str(L)]
    orig = torch.load(RUNS / "diffmeans.pt")[L]
    app = torch.load(RUNS / "answer_reflect/diffmeans_appended.pt")[L]
    wov = torch.load(RUNS / "answer_reflect/diffmeans_woven.pt")[L]
    drop = orig.float() - app.float()
    dirs = {"original": unit(orig), "original ⊥ drop": without(orig, drop),
            "appended ⊥ drop": without(app, drop), "woven ⊥ drop": without(wov, drop)}
    for k, d in dirs.items():
        print(f"{k}: cos with original {float(d @ dirs['original']):.3f}", flush=True)

    prompts = json.loads((ROOT / "data/eval_prompts.json").read_text())
    heldout = json.loads((ROOT / "data/heldout_prompts.json").read_text())
    all_prompts = prompts + heldout
    split = ["eval"] * len(prompts) + ["heldout"] * len(heldout)
    chats = [s.chat(p) for p in all_prompts]

    cfgs = [("baseline", None)]
    for a in (0.5, 0.75):
        cfgs.append((f"add α{a}", ("add", dirs["original"], a)))
    for a in (0.5, 0.75, 1.0, 1.5):
        cfgs.append((f"norm-preserving α{a}", ("norm", dirs["original"], a, 1.0)))
    for a in (0.75, 1.0):
        cfgs.append((f"norm +10% α{a}", ("norm", dirs["original"], a, 1.1)))
    for k in ("original ⊥ drop", "appended ⊥ drop", "woven ⊥ drop"):
        for a in (0.5, 1.0):
            cfgs.append((f"{k} α{a}", ("add", dirs[k], a)))

    out = OUT / "gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()
    for i, (name, c) in enumerate(cfgs):
        if name in done:
            continue
        t0 = time.time()
        if c is None:
            ctx = contextlib.nullcontext()
        elif c[0] == "add":
            ctx = s.steer([], {L: c[2] * p50 * c[1]})
        else:
            ctx = norm_push(s, L, c[2] * p50 * c[1], gain=c[3])
        with ctx:
            outs = s.generate(chats, max_new_tokens=120)
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": "baseline" if c is None else "steer", "prompts": all_prompts,
                                "split": split, "outputs": outs}) + "\n")
        print(f"[{i + 1}/{len(cfgs)}] {name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
