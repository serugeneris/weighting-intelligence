"""Experiment K: steer at the prompt's hot spots found by experiment G, with prompt-derived vectors.

Experiment G found where the woven system prompt's effect travels: three full-attention heads (L15 head 8, L15
head 2, L11 head 0) carry 38% of it, attention layers 11 and 15 most of all, and DeltaNet layers 16-18 right
after them. The attention-steering literature (LITERATURE.md H3: Izawa et al.; GCAD; ITI) reports that steering
a few heads beats residual steering on the trait-vs-coherence trade-off.

Vectors: for the 32 contrast questions, the prompted model's own woven answer (runs/distill/targets.json) is fed
after the prompted and after the unprompted context. At each site we take the mean (prompted - unprompted) output
over the answer tokens: per head, the head's 256-dim slice of o_proj's input; per layer, the token mixer's output.
These are what the prompt adds at that site, averaged. During generation with no prompt we add g times that mean
at every position (a constant, so it equals a residual push in that site's output subspace; the test is whether
these prompt-derived sites and directions do better than the diff-means direction at L8).

Round 2 (--round 2): the only setting that worked in round 1 was the sum over all 32 token mixers (x2), so it is
mapped more finely (x1.5, x2, x2.5, x3) in one pool with the woven prompt and diff-means L8 alpha 0.5.

Outputs under runs/heads/: vectors.pt, gen.jsonl (round 1), gen_round2.jsonl. Then:
  scripts/claude_judge.py --input heads/gen.jsonl --output heads/judge.jsonl
  scripts/claude_judge.py --input heads/gen_round2.jsonl --output heads/judge_round2.jsonl
"""
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.exp_routes import first, mixer, replace_first  # noqa: E402
from scripts.exp_selfgen import PROMPTS  # noqa: E402
from wi.steer import Steerable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "heads"
HD = 256
TOP_HEADS = [(15, 8), (15, 2), (11, 0)]


@torch.no_grad()
def collect(s):
    f = OUT / "vectors.pt"
    if f.exists():
        return torch.load(f)
    targets = {d["q"]: d["target"] for d in json.loads((RUNS / "distill/targets.json").read_text())}
    qs = [p["q"] for p in json.loads((ROOT / "data/contrast_pairs.json").read_text())]
    full = [L for L in sorted(s.layers) if hasattr(s.layers[L], "self_attn")]
    sums = {"mixer": {L: 0 for L in s.layers}, "oin": {L: 0 for L in full}}
    for q in qs:
        a = s.tok(targets[q])["input_ids"]
        rec = {}
        for tag, ctx in (("P", s.chat(q, system=PROMPTS["woven"])), ("U", s.chat(q))):
            c = s.tok(ctx)["input_ids"]
            n = len(a)
            r, hooks = {"mixer": {}, "oin": {}}, []
            for L in s.layers:
                hooks.append(mixer(s, L).register_forward_hook(
                    lambda _m, _i, o, L=L: r["mixer"].__setitem__(L, first(o)[0, -n:].float().mean(0))))
            for L in full:
                hooks.append(s.layers[L].self_attn.o_proj.register_forward_pre_hook(
                    lambda _m, x, L=L: r["oin"].__setitem__(L, x[0][0, -n:].float().mean(0))))
            try:
                s.model(input_ids=torch.tensor([c + a], device=s.device))
            finally:
                for h in hooks:
                    h.remove()
            rec[tag] = r
        for k in sums:
            for L in sums[k]:
                sums[k][L] = sums[k][L] + (rec["P"][k][L] - rec["U"][k][L]) / len(qs)
    vec = {k: {L: v.cpu() for L, v in d.items()} for k, d in sums.items()}
    torch.save(vec, f)
    return vec


@contextlib.contextmanager
def add_at(s, heads=(), layers=(), vec=None, g=1.0):
    hooks = []
    by = {}
    for L, h in heads:
        by.setdefault(L, []).append(h)
    for L, hs in by.items():
        add = torch.zeros(s.layers[L].self_attn.o_proj.in_features)
        for h in hs:
            add[h * HD:(h + 1) * HD] = vec["oin"][L][h * HD:(h + 1) * HD]
        add = (g * add).to(s.device, torch.bfloat16)
        hooks.append(s.layers[L].self_attn.o_proj.register_forward_pre_hook(
            lambda _m, x, add=add: (x[0] + add,) + tuple(x[1:])))
    for L in layers:
        add = (g * vec["mixer"][L]).to(s.device, torch.bfloat16)
        hooks.append(mixer(s, L).register_forward_hook(lambda _m, _i, o, add=add: replace_first(o, first(o) + add)))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    s = Steerable()
    p50 = {int(k): v for k, v in json.loads((RUNS / "p50.json").read_text()).items()}
    vec = collect(s)
    W = {L: s.layers[L].self_attn.o_proj.weight.float() for L in vec["oin"]}
    for L, h in TOP_HEADS:
        sl = torch.zeros_like(vec["oin"][L]); sl[h * HD:(h + 1) * HD] = vec["oin"][L][h * HD:(h + 1) * HD]
        print(f"head L{L}.{h}: residual push {float((W[L] @ sl.to(W[L].device)).norm().detach()) / p50[L]:.3f}·p50", flush=True)
    for L in (11, 15, 16, 17, 18):
        print(f"mixer L{L}: residual push {float(vec['mixer'][L].norm()) / p50[L]:.3f}·p50", flush=True)

    prompts = json.loads((ROOT / "data/eval_prompts.json").read_text())
    heldout = json.loads((ROOT / "data/heldout_prompts.json").read_text())
    all_prompts = prompts + heldout
    split = ["eval"] * len(prompts) + ["heldout"] * len(heldout)
    chats = [s.chat(p) for p in all_prompts]
    cfgs = [("baseline", {})]
    for g in (1, 2, 4, 8):
        cfgs.append((f"top-3 heads ×{g}", dict(heads=TOP_HEADS, g=g)))
    for g in (1, 2, 4):
        cfgs.append((f"attention L11+L15, all heads ×{g}", dict(heads=[(L, h) for L in (11, 15) for h in range(16)], g=g)))
    for g in (1, 2, 4):
        cfgs.append((f"DeltaNet L16–18 ×{g}", dict(layers=[16, 17, 18], g=g)))
    for g in (0.5, 1, 2):
        cfgs.append((f"all 32 token mixers ×{g}", dict(layers=sorted(s.layers), g=g)))

    if args.round == 2:
        d = torch.load(RUNS / "diffmeans.pt")[8].float()
        cfgs = [("baseline", {}), ("prompt woven", "prompt"), ("diff-means L8 α0.5", ("steer", {8: 0.5 * p50[8] * d / d.norm()}))]
        cfgs += [(f"all 32 token mixers ×{g}", dict(layers=sorted(s.layers), g=g)) for g in (1.5, 2, 2.5, 3)]
    out = OUT / ("gen.jsonl" if args.round == 1 else "gen_round2.jsonl")
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()
    for i, (name, kw) in enumerate(cfgs):
        if name in done:
            continue
        t0 = time.time()
        if kw == "prompt":
            outs = s.generate([s.chat(p, system=PROMPTS["woven"]) for p in all_prompts], max_new_tokens=120)
        elif isinstance(kw, tuple):
            with s.steer([], kw[1]):
                outs = s.generate(chats, max_new_tokens=120)
        else:
            with add_at(s, vec=vec, **kw):
                outs = s.generate(chats, max_new_tokens=120)
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": "baseline" if not kw else "steer", "prompts": all_prompts,
                                "split": split, "outputs": outs}) + "\n")
        print(f"[{i + 1}/{len(cfgs)}] {name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
