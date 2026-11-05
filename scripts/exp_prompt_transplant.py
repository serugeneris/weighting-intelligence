"""Experiment F: is the prompt's effect carried by a few positions at the end of the prompt? (LITERATURE.md H1, H8)

The woven system prompt (experiment B) beats steering. Two literature leads say a prompt mostly acts through a few
positions where the model "decides" what kind of answer to write. With the chat template, the 9 tokens after the
question (<|im_end|> \\n <|im_start|> assistant \\n <think> \\n\\n </think> \\n\\n) are identical for every prompt,
so they are a natural decision point.

  transplant   run the *prompted* model, record the residual stream at the last k prompt positions, and paste it
               into the *unprompted* run at the same positions (all layers, or one layer). The answer is generated
               without the system prompt. If the k positions carry the decision, the answer should look prompted.
  prompt vector  the mean (prompted - unprompted) residual at layer L and the last k positions, over the 32
               contrast questions (never evaluation prompts). Added at those positions only, with no system prompt:
               a prompt-free steering vector that acts only at the decision point (gain g scales it).

Outputs under runs/prompt_transplant/: prompt_vector.pt, gen.jsonl. Then:
  scripts/claude_judge.py --input prompt_transplant/gen.jsonl --output prompt_transplant/judge.jsonl
"""
import contextlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.exp_selfgen import PROMPTS  # noqa: E402
from wi.steer import Steerable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "prompt_transplant"
SYS = PROMPTS["woven"]


def layer_out(o):
    return o[0] if isinstance(o, tuple) else o


def with_layer_out(o, h):
    return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h


@torch.no_grad()
def record_tail(s, texts, k, layers):
    """Residual after each layer at the last k positions, for left-padded batch prefill. -> {L: [B, k, d]}"""
    s.tok.padding_side = "left"
    enc = s.tok(texts, return_tensors="pt", padding=True).to(s.device)
    rec, hooks = {}, []
    for L in layers:
        hooks.append(s.layers[L].register_forward_hook(
            lambda _m, _i, o, L=L: rec.__setitem__(L, layer_out(o)[:, -k:].detach().clone())))
    try:
        s.model(**enc)
    finally:
        for h in hooks:
            h.remove()
    return rec


@contextlib.contextmanager
def patch_tail(s, k, values: dict, add: bool = False):
    """During prefill only, replace (or add to) the residual after layer L at the last k positions."""
    hooks = []
    for L, v in values.items():
        def hook(_m, _i, o, L=L, v=v):
            h = layer_out(o)
            if h.shape[1] <= 1:                       # decode step: leave alone
                return o
            h = h.clone()
            h[:, -k:] = (h[:, -k:] + v.to(h.dtype)) if add else v.to(h.dtype)
            return with_layer_out(o, h)
        hooks.append(s.layers[L].register_forward_hook(hook))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = Steerable()
    prompts = json.loads((ROOT / "data/eval_prompts.json").read_text())
    heldout = json.loads((ROOT / "data/heldout_prompts.json").read_text())
    all_prompts = prompts + heldout
    split = ["eval"] * len(prompts) + ["heldout"] * len(heldout)
    U = [s.chat(p) for p in all_prompts]
    P = [s.chat(p, system=SYS) for p in all_prompts]
    tail = s.tok(U[0])["input_ids"][-9:]
    assert all(s.tok(t)["input_ids"][-9:] == tail for t in U + P), "header tokens differ"
    layers = sorted(s.layers)

    # prompt vector from the contrast questions (not evaluation prompts)
    pv_file = OUT / "prompt_vector.pt"
    if pv_file.exists():
        pv = torch.load(pv_file)
    else:
        qs = [p["q"] for p in json.loads((ROOT / "data/contrast_pairs.json").read_text())]
        assert not set(qs) & set(all_prompts)
        pv = {}
        for k in (1, 9):
            a = record_tail(s, [s.chat(q, system=SYS) for q in qs], k, layers)
            b = record_tail(s, [s.chat(q) for q in qs], k, layers)
            pv[k] = {L: (a[L] - b[L]).float().mean(0).cpu() for L in layers}
        torch.save(pv, pv_file)
    for k in (1, 9):
        print(f"prompt vector k={k}: |v|/p50 at L8 = "
              f"{pv[k][8].norm(dim=-1).mean() / json.loads((RUNS / 'p50.json').read_text())['8']:.2f}", flush=True)

    rec = {k: record_tail(s, P, k, layers) for k in (1, 9, 16)}
    cfgs = [("baseline", None), ("prompt woven", "prompt")]
    for k in (1, 9, 16):
        cfgs.append((f"transplant last {k}, all layers", ("patch", k, {L: rec[k][L] for L in layers})))
    for L in (8, 16):
        cfgs.append((f"transplant last 9, L{L} only", ("patch", 9, {L: rec[9][L]})))
    for L in (8, 16):
        for g in (1.0, 2.0):
            cfgs.append((f"prompt vector last 9, L{L} ×{g}", ("add", 9, {L: g * pv[9][L].to(s.device)})))
    cfgs.append(("prompt vector last 1, L8 ×2.0", ("add", 1, {8: 2.0 * pv[1][8].to(s.device)})))

    out = OUT / "gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()
    for i, (name, c) in enumerate(cfgs):
        if name in done:
            continue
        t0 = time.time()
        if c is None:
            outs = s.generate(U, max_new_tokens=120)
        elif c == "prompt":
            outs = s.generate(P, max_new_tokens=120)
        else:
            mode, k, vals = c
            with patch_tail(s, k, vals, add=(mode == "add")):
                outs = s.generate(U, max_new_tokens=120)
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": "baseline" if c is None else "steer", "prompts": all_prompts,
                                "split": split, "outputs": outs}) + "\n")
        print(f"[{i + 1}/{len(cfgs)}] {name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
