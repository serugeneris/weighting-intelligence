"""Experiment G: through which route does the "philosophical mode" travel? (LITERATURE.md H3, the attention hypothesis)

Qwen3.5-4B mixes information between tokens in two ways: 8 full-attention layers (3, 7, ..., 31; 16 heads each)
and 24 Gated DeltaNet linear-attention layers, which carry a recurrent state instead of attending. Two tests:

Part 1, mediation of the system prompt (forward passes only). For 16 contrast questions we have the prompted model's
woven answer A_P (runs/distill/targets.json) and the unprompted answer A_U. The preference for the philosophical
answer in a context is  pref = mean logprob/token(A_P) - mean logprob/token(A_U).  The woven system prompt raises it:
E = pref(prompted) - pref(unprompted). A system prompt can only reach the answer through the token mixers (attention
or DeltaNet) reading the system-prompt positions. So, in the prompted run, we restore one mixer's output, at every
position after the system block, to its value in the unprompted run, and measure the fraction of E that disappears:
  - each of the 32 layers' mixer (self_attn or linear_attn)
  - each of the 8 x 16 full-attention heads (the head's slice of o_proj's input)
  - all 8 full-attention layers together vs all 24 DeltaNet layers together
The attention hypothesis predicts the full-attention layers (and a few heads) carry most of E. It is falsified if the
DeltaNet layers carry as much or more, or if E is spread thinly over many heads.

Part 2, routes of prompt-only steering (experiment C). Pushing only the prompt tokens at L8 made the whole answer
philosophical. We build the cache for the prompt (all but its last token) twice, with and without the push, then mix:
only the full-attention keys/values from the steered run, only the DeltaNet states, both, or neither, and generate
from the mixed cache with no push. Whichever cache keeps the effect is the route.

Outputs under runs/routes/: mediation.json, answers_plain.json, gen.jsonl. Then:
  scripts/claude_judge.py --input routes/gen.jsonl --output routes/judge.jsonl
"""
import copy
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
OUT = RUNS / "routes"
N_Q = 16


def mixer(s, L):
    layer = s.layers[L]
    return layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn


def first(o):
    return o[0] if isinstance(o, tuple) else o


def replace_first(o, h):
    return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h


@torch.no_grad()
def answer_logprob(s, ctx_ids, ans_ids, hooks_factory=None):
    ids = torch.tensor([ctx_ids + ans_ids], device=s.device)
    hooks = hooks_factory() if hooks_factory else []
    try:
        logits = s.model(input_ids=ids).logits[0, len(ctx_ids) - 1:-1].float()
    finally:
        for h in hooks:
            h.remove()
    lp = logits.log_softmax(-1).gather(1, torch.tensor(ans_ids, device=s.device)[:, None])
    return float(lp.mean())


def part1(s):
    full = [L for L in sorted(s.layers) if hasattr(s.layers[L], "self_attn")]
    lin = [L for L in sorted(s.layers) if L not in full]
    targets = {d["q"]: d["target"] for d in json.loads((RUNS / "distill/targets.json").read_text())}
    qs = [p["q"] for p in json.loads((ROOT / "data/contrast_pairs.json").read_text())][:N_Q]
    pf = OUT / "answers_plain.json"
    if pf.exists():
        plain = json.loads(pf.read_text())
    else:
        plain = dict(zip(qs, [o.strip() for o in s.generate([s.chat(q) for q in qs], max_new_tokens=160)]))
        pf.write_text(json.dumps(plain, indent=1, ensure_ascii=False))
    n_heads, hd = 16, 256
    units = [("layer", L) for L in sorted(s.layers)] + [("head", (L, h)) for L in full for h in range(n_heads)] \
        + [("group", "full-attention layers"), ("group", "DeltaNet layers")]
    frac = {str(u): [] for u in units}
    E_all = []
    for qi, q in enumerate(qs):
        cU, cP = s.tok(s.chat(q))["input_ids"], s.tok(s.chat(q, system=PROMPTS["woven"]))["input_ids"]
        n_tail = len(cU) - 2                            # positions after "<|im_start|>" ... aligned from the end
        assert cU[-n_tail:] == cP[-n_tail:]
        aP, aU = s.tok(targets[q])["input_ids"], s.tok(plain[q])["input_ids"]

        # record unprompted mixer outputs and o_proj inputs over (tail + answer) for both answers
        def record(ans_ids):
            rec_m, rec_o, hooks = {}, {}, []
            for L in sorted(s.layers):
                hooks.append(mixer(s, L).register_forward_hook(
                    lambda _m, _i, o, L=L: rec_m.__setitem__(L, first(o).detach().clone())))
            for L in full:
                hooks.append(s.layers[L].self_attn.o_proj.register_forward_pre_hook(
                    lambda _m, a, L=L: rec_o.__setitem__(L, a[0].detach().clone())))
            ids = torch.tensor([cU + ans_ids], device=s.device)
            try:
                s.model(input_ids=ids)
            finally:
                for h in hooks:
                    h.remove()
            return rec_m, rec_o
        recs = {id(a): record(a) for a in (aP, aU)}

        def pref(ctx, factory_for=None):
            return answer_logprob(s, ctx, aP, factory_for(aP) if factory_for else None) - \
                answer_logprob(s, ctx, aU, factory_for(aU) if factory_for else None)

        p_un, p_pr = pref(cU), pref(cP)
        E = p_pr - p_un
        E_all.append(E)

        def factory(layers_m=(), heads=()):
            def for_ans(a):
                rec_m, rec_o = recs[id(a)]
                m = n_tail + len(a)                         # positions to restore, aligned from the end

                def make():
                    hooks = []
                    for L in layers_m:
                        def hm(_m, _i, o, L=L):
                            h = first(o).clone()
                            h[:, -m:] = rec_m[L][:, -m:]
                            return replace_first(o, h)
                        hooks.append(mixer(s, L).register_forward_hook(hm))
                    by_layer = {}
                    for L, hh in heads:
                        by_layer.setdefault(L, []).append(hh)
                    for L, hs in by_layer.items():
                        def ho(_m, a, L=L, hs=hs):
                            x = a[0].clone()
                            for hh in hs:
                                x[:, -m:, hh * hd:(hh + 1) * hd] = rec_o[L][:, -m:, hh * hd:(hh + 1) * hd]
                            return (x,) + tuple(a[1:])
                        hooks.append(s.layers[L].self_attn.o_proj.register_forward_pre_hook(ho))
                    return hooks
                return make
            return for_ans

        for u in units:
            kind, x = u
            if kind == "layer":
                f = factory(layers_m=[x])
            elif kind == "head":
                f = factory(heads=[x])
            else:
                f = factory(layers_m=full if x.startswith("full") else lin)
            frac[str(u)].append((p_pr - pref(cP, f)) / E if abs(E) > 1e-6 else float("nan"))
        print(f"q{qi + 1}/{len(qs)} E={E:.3f}", flush=True)

    mean = lambda v: float(torch.tensor(v).nanmean())
    res = {"E_mean": mean(E_all), "E_per_question": E_all,
           "fraction_removed": {k: round(mean(v), 4) for k, v in frac.items()},
           "full_attention_layers": full}
    (OUT / "mediation.json").write_text(json.dumps(res, indent=1))
    fr = res["fraction_removed"]
    print(f"E mean = {res['E_mean']:.3f}")
    print("groups:", {k: v for k, v in fr.items() if "group" in k})
    print("layers:", {k.split(", ")[1].rstrip(")"): v for k, v in fr.items() if k.startswith("('layer'")})
    top = sorted(((v, k) for k, v in fr.items() if k.startswith("('head'")), reverse=True)[:10]
    print("top heads:", top)


@torch.no_grad()
def part2(s):
    p50 = json.loads((RUNS / "p50.json").read_text())
    s.p50 = {int(k): v for k, v in p50.items()}
    d = torch.load(RUNS / "diffmeans.pt")[8].float()
    push = {8: 1.5 * s.p50[8] * d / d.norm()}
    full = {L for L in s.layers if hasattr(s.layers[L], "self_attn")}
    prompts = json.loads((ROOT / "data/eval_prompts.json").read_text())
    heldout = json.loads((ROOT / "data/heldout_prompts.json").read_text())
    all_prompts = prompts + heldout
    split = ["eval"] * len(prompts) + ["heldout"] * len(heldout)
    out = OUT / "gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()
    routes = {"cache: none steered (= baseline)": set(), "cache: all steered": set(s.layers),
              "cache: full-attention KV steered only": full, "cache: DeltaNet states steered only": set(s.layers) - full}
    gens = {name: [] for name in routes}
    if all(n in done for n in routes):
        return
    gen_kw = dict(max_new_tokens=120, do_sample=False, repetition_penalty=1.1,
                  pad_token_id=s.tok.pad_token_id or s.tok.eos_token_id)
    same = 0
    for p in all_prompts:
        ids = s.tok(s.chat(p), return_tensors="pt")["input_ids"].to(s.device)
        caches = {}
        for steered in (False, True):
            with s.steer([], push if steered else None):
                caches[steered] = s.model(input_ids=ids[:, :-1], use_cache=True).past_key_values
        for name, from_steered in routes.items():
            cache = copy.deepcopy(caches[False])
            for L in from_steered:
                cache.layers[L] = copy.deepcopy(caches[True].layers[L])
            o = s.model.generate(input_ids=ids, past_key_values=cache, **gen_kw)
            gens[name].append(s.tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True))
        ref = s.model.generate(input_ids=ids, **gen_kw)          # same prompt, no cache surgery
        same += s.tok.decode(ref[0, ids.shape[1]:], skip_special_tokens=True) == gens["cache: none steered (= baseline)"][-1]
    print(f"sanity: unsteered rebuilt cache reproduces plain generation on {same}/{len(all_prompts)} prompts", flush=True)
    with out.open("a") as f:
        for name in routes:
            f.write(json.dumps({"name": name, "kind": "route", "prompts": all_prompts, "split": split,
                                "outputs": gens[name]}) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = Steerable()
    t0 = time.time()
    part1(s)
    print(f"part 1 done ({time.time() - t0:.0f}s)", flush=True)
    part2(s)
    print(f"part 2 done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
