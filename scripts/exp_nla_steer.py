"""Pilot AM, stage 3 (Q3, optional; run because Stage 2's P2b held): steering with an NLA-derived direction.

Question and prior (docs/plans/nla-pilot.md, copied before the run):
  Q3. Write paired explanations (plain plan vs "answer, then connect it to a philosophical idea"), map both through
      the AR and use the mean difference as a steering vector at layer 21. Compare with diff-means at layer 21 on the
      philosophy + helpfulness frontier (16 eval prompts: data/eval_prompts.json + data/heldout_prompts.json,
      Sonnet judge, the scripts/claude_judge.py rubric). Prior from E: in the 4B only layers 8-16 respond to modest
      pushes, and layer 21 of 32 may be too late for any direction; the layer-21 diff-means arm is the control.

Pairs: the 16 greedy base explanations of Stage 2 (first 16 confirmation prompts, which are disjoint from the eval
prompts) are the plain side; Sonnet rewrites each one, changing only the planned register to "answer the request,
then connect it to a philosophical idea", so both sides stay in the AR's distribution.

Arms (4-bit 9B, vector added to the layer-21 output at every token, alpha x p50_21 x unit direction, as in
scripts/sweep.py): base; AR difference at alpha 0.25/0.5/1; layer-21 diff-means at 0.25/0.5/1; a random direction
at 1; and the Stage 2 layer-8 diff-means push at 0.5 as a known-working reference. Greedy, 120 tokens, cut at
<|im_end|>.

  .venv-mlx/bin/python scripts/exp_nla_steer.py pairs     -> runs/nla/s3_pairs.json   (claude -p rewrite)
  .venv-mlx/bin/python scripts/exp_nla_steer.py calib     -> runs/nla/s3_dm21.npz     (bf16 target)
  .venv-mlx/bin/python scripts/exp_nla_steer.py ar        -> runs/nla/s3_ar.npz       (AR)
  .venv-mlx/bin/python scripts/exp_nla_steer.py gen       -> runs/nla/s3_gen.jsonl    (4-bit 9B)
  .venv-mlx/bin/python scripts/exp_nla_steer.py judge     -> runs/nla/s3_judge.jsonl
  .venv-mlx/bin/python scripts/exp_nla_steer.py analyze   -> runs/nla/s3_results.json
"""
import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.exp_nla_retention import chat_text, push_vectors  # noqa: E402
from wi import nla  # noqa: E402

OUT = nla.ROOT / "runs" / "nla"
LAYER = nla.LAYER
N_PAIRS = 16
ALPHAS = [0.25, 0.5, 1.0]
MAX_NEW = 120

REWRITE = """Below is a text written by a decoder that describes a language model's internal state at the moment it starts
answering a user's request. It describes the genre, the request, and the planned opening of the answer.

Rewrite it so that it describes a model that is about to answer the same request helpfully AND then connect it to a
philosophical idea (a reflection on meaning, time, habit, value, knowledge or the self). Keep the same request, topic,
length, line structure and telegraphic style; change only what the text says about the register, tone and planned
opening. Do not add anything else.

Return ONLY the rewritten text, nothing before or after it.

TEXT:
"""


def configs():
    cfgs = [("base", None, 0.0)]
    cfgs += [(f"ar@{a}", "ar", a) for a in ALPHAS]
    cfgs += [(f"dm21@{a}", "dm21", a) for a in ALPHAS]
    cfgs += [("random21@1.0", "random21", 1.0), ("dm8@0.5 (stage 2 push)", "dm8", 0.5)]
    return cfgs


def phase_pairs(model="sonnet"):
    from scripts.claude_judge import call_claude
    rows = json.loads((OUT / "s2_rows.json").read_text())
    ex = [json.loads(line) for line in (OUT / "s2_expl.jsonl").read_text().splitlines() if line]
    plain = {rows[r["i"]]["p"]: r["expl"] for r in ex
             if r["seed"] == 0 and rows[r["i"]]["config"] == "base" and rows[r["i"]]["p"] < N_PAIRS}

    def rewrite(p):
        return {"p": p, "plain": plain[p], "philo": call_claude(REWRITE + plain[p], model).strip()}

    with ThreadPoolExecutor(4) as pool:
        pairs = list(pool.map(rewrite, sorted(plain)))
    (OUT / "s3_pairs.json").write_text(json.dumps(pairs, indent=1))
    print(pairs[0]["plain"], "\n=>\n", pairs[0]["philo"])


def phase_calib():
    target = nla.Target()
    tok, mx = target.tok, target.mx
    pairs = json.loads((nla.ROOT / "data" / "contrast_pairs.json").read_text())
    norms, diffs = [], []
    for p in pairs:
        prompt = chat_text(tok, p["q"])
        n0 = len(tok.encode(prompt, add_special_tokens=False))
        means = {}
        for kind in ("philo", "plain"):
            ids = tok.encode(prompt + p[kind], add_special_tokens=False)
            h = target.acts(ids, positions=range(len(ids)))
            means[kind] = h[n0:].mean(0)
            if kind == "plain":
                norms.append(np.linalg.norm(h, axis=1))
        diffs.append(means["philo"] - means["plain"])
    d = np.mean(diffs, 0)
    p50 = float(np.median(np.concatenate(norms)))
    np.savez(OUT / "s3_dm21.npz", dm=d, p50=p50)
    print(f"layer {LAYER}: |dm| {np.linalg.norm(d):.2f}, p50 {p50:.2f}")
    del mx


def phase_ar():
    ar = nla.Reconstructor()
    pairs = json.loads((OUT / "s3_pairs.json").read_text())
    pp = ar.predict([p["philo"] for p in pairs])
    pl = ar.predict([p["plain"] for p in pairs])
    d = (pp - pl).mean(0)
    per = pp - pl
    cos_pairs = per @ d / np.linalg.norm(per, axis=1) / np.linalg.norm(d)
    np.savez(OUT / "s3_ar.npz", d=d, per=per)
    print(f"|AR diff| {np.linalg.norm(d):.2f} (AR outputs normalized to {np.sqrt(nla.D_MODEL):.0f}); "
          f"per-pair cos to mean: median {np.median(cos_pairs):.2f}, min {cos_pairs.min():.2f}")


def unit(v):
    return v / np.linalg.norm(v)


def directions():
    dm = np.load(OUT / "s3_dm21.npz")
    ar = np.load(OUT / "s3_ar.npz")
    r = np.random.default_rng(1).normal(size=dm["dm"].shape)
    return {"ar": unit(ar["d"]), "dm21": unit(dm["dm"]), "random21": unit(r)}, float(dm["p50"])


def geometry():
    dirs, _ = directions()
    acts = np.load(OUT / "s2_acts.npy")
    rows = json.loads((OUT / "s2_rows.json").read_text())
    by = {(r["p"], r["config"]): acts[i] for i, r in enumerate(rows)}
    n = max(r["p"] for r in rows) + 1
    woven = unit(np.mean([by[(p, "woven")] - by[(p, "base")] for p in range(n)], 0))
    push = unit(np.mean([by[(p, "push")] - by[(p, "base")] for p in range(n)], 0))
    return {"cos(ar, dm21)": float(dirs["ar"] @ dirs["dm21"]),
            "cos(ar, stage2 woven-base at L21)": float(dirs["ar"] @ woven),
            "cos(ar, stage2 push-base at L21)": float(dirs["ar"] @ push),
            "cos(dm21, stage2 woven-base at L21)": float(dirs["dm21"] @ woven),
            "cos(dm21, stage2 push-base at L21)": float(dirs["dm21"] @ push),
            "cos(ar, random21)": float(dirs["ar"] @ dirs["random21"])}


def phase_gen():
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    print(json.dumps(geometry(), indent=1))
    dirs, p50 = directions()
    dm8 = push_vectors()["push"] / 0.5           # unit x p50_8 (Stage 2 scaling); alpha applied below
    model, tok = load("mlx-community/Qwen3.5-9B-4bit")
    inner = model.language_model.model if hasattr(model, "language_model") else model.model
    state = {}

    def wrap(idx):
        layer = inner.layers[idx]
        base_cls = type(layer)

        class Pushed(base_cls):
            def __call__(self_, x, mask=None, cache=None):
                out = base_cls.__call__(self_, x, mask=mask, cache=cache)
                v = state.get(idx)
                return out if v is None else out + v.astype(out.dtype)

        layer.__class__ = Pushed

    wrap(LAYER)
    wrap(8)
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    prompts = [(q, "eval") for q in json.loads((nla.ROOT / "data" / "eval_prompts.json").read_text())]
    prompts += [(q, "heldout") for q in json.loads((nla.ROOT / "data" / "heldout_prompts.json").read_text())]
    path = OUT / "s3_gen.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["name"] for line in path.read_text().splitlines() if line}
    with path.open("a") as f:
        for name, kind, a in configs():
            if name in done:
                continue
            state.clear()
            if kind == "dm8":
                state[8] = mx.array((a * dm8).astype(np.float32))
            elif kind:
                state[LAYER] = mx.array((a * p50 * dirs[kind]).astype(np.float32))
            outs = []
            for q, _ in prompts:
                ids = tok.encode(chat_text(tok, q), add_special_tokens=False)
                cache = make_prompt_cache(model)
                logits = model(mx.array(ids)[None], cache=cache)[0, -1]
                out = []
                for _ in range(MAX_NEW):
                    nxt = int(mx.argmax(logits).item())
                    if nxt in stop:
                        break
                    out.append(nxt)
                    logits = model(mx.array([[nxt]]), cache=cache)[0, -1]
                outs.append(tok.decode(out))
            f.write(json.dumps({"name": name, "kind": kind or "base", "alpha": a, "prompts": [q for q, _ in prompts],
                                "split": [s for _, s in prompts], "outputs": outs}) + "\n")
            f.flush()
            print(f"[{name}] {outs[0][:160]!r}", flush=True)


def phase_judge(model="sonnet", batch=16, workers=4):
    from scripts.claude_judge import RUBRIC, call_claude, parse
    recs = [json.loads(line) for line in (OUT / "s3_gen.jsonl").read_text().splitlines() if line]
    items = [{"cfg": r["name"], "i": i, "split": s, "prompt": p, "output": o}
             for r in recs for i, (p, o, s) in enumerate(zip(r["prompts"], r["outputs"], r["split"]))]
    random.Random(7).shuffle(items)
    for j, it in enumerate(items):
        it["id"] = f"x{j:04d}"
    batches = [items[k:k + batch] for k in range(0, len(items), batch)]

    def run(b):
        body = "\n\n".join(f'### id={it["id"]}\nUSER: {it["prompt"]}\nRESPONSE: """{it["output"].strip()}"""'
                           for it in b)
        for attempt in range(3):
            try:
                res = {d["id"]: d for d in parse(call_claude(RUBRIC + body, model))}
                if any(it["id"] not in res for it in b):
                    raise ValueError("missing ids")
                return [{**it, **{k: res[it["id"]][k] for k in ("philo", "coherence", "helpful", "degenerate")}}
                        for it in b]
            except Exception as e:  # noqa: BLE001
                print("retry", attempt, str(e)[:200], flush=True)
        return []

    with (OUT / "s3_judge.jsonl").open("w") as f, ThreadPoolExecutor(workers) as pool:
        for res in pool.map(run, batches):
            for r in res:
                f.write(json.dumps(r) + "\n")
    print(len(items), "items judged")


def phase_analyze():
    rows = [json.loads(line) for line in (OUT / "s3_judge.jsonl").read_text().splitlines() if line]
    res = {"geometry": geometry(), "configs": {}}
    for name, _, _ in configs():
        rs = [r for r in rows if r["cfg"] == name]
        if not rs:
            continue
        res["configs"][name] = {
            "n": len(rs),
            "philo": round(float(np.mean([r["philo"] for r in rs])), 2),
            "coherence": round(float(np.mean([r["coherence"] for r in rs])), 2),
            "helpful": round(float(np.mean([r["helpful"] for r in rs])), 2),
            "degenerate": int(sum(bool(r["degenerate"]) for r in rs)),
            "philo_and_helpful_ge5": int(sum(r["philo"] >= 5 and r["helpful"] >= 5 for r in rs)),
        }
    (OUT / "s3_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["pairs", "calib", "ar", "gen", "judge", "analyze"])
    args = ap.parse_args()
    {"pairs": phase_pairs, "calib": phase_calib, "ar": phase_ar, "gen": phase_gen, "judge": phase_judge,
     "analyze": phase_analyze}[args.phase]()


if __name__ == "__main__":
    main()
