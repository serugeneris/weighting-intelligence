"""Pilot AM, stage 2 (Q2): does a register push erase the task from the 9B's layer-21 state?

Questions and predictions (docs/plans/nla-pilot.md, copied before the run):
  Q2. At the first assistant position of the 48 confirmation prompts (data/confirm_prompts.json), under base, the woven
      system prompt and a 9B diff-means push at layer 8 (alpha 0.5, scaled as in scripts/sweep.py: alpha x median
      layer-8 residual norm x unit direction, added to the layer-8 output at every token), plus a random direction of
      equal norm as a push control:
  P2a. A blind judge picks which of 4 user requests an explanation came from (the true one + 3 random distractors,
       fixed per prompt) at the same rate for base and woven, and less often under the push.
  P2b. "Reflective / philosophical" content appears more often under woven and push than base. This measures the
       NLA's sensitivity, not a finding.

Direction: mean layer-8 output over answer tokens, philo minus plain, over the 32 pairs of data/contrast_pairs.json
(the recipe of scripts/find_neurons.py at 4B); p50 = median layer-8 output norm over all tokens of chat(q) + plain.
Targets read in bf16 (wi.nla.Target, blocks 0-21); the gen phase checks by eye what the same push does to the
4-bit 9B's generations.

Phases (one model in memory at a time):
  .venv-mlx/bin/python scripts/exp_nla_retention.py calib    -> runs/nla/s2_dir.npz
  .venv-mlx/bin/python scripts/exp_nla_retention.py acts     -> runs/nla/s2_acts.npy, s2_rows.json
  .venv-mlx/bin/python scripts/exp_nla_retention.py gen      -> runs/nla/s2_gen.jsonl  (4-bit 9B, 8 prompts, by eye)
  .venv-mlx/bin/python scripts/exp_nla_retention.py explain  -> runs/nla/s2_expl.jsonl
  .venv-mlx/bin/python scripts/exp_nla_retention.py judge    -> runs/nla/s2_judge.jsonl
  .venv-mlx/bin/python scripts/exp_nla_retention.py analyze  -> runs/nla/s2_results.json
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wi import nla  # noqa: E402

OUT = nla.ROOT / "runs" / "nla"
PUSH_LAYER = 8
ALPHA = 0.5
CONFIGS = ["base", "woven", "push", "random"]
SAMPLES = [(0.0, 0), (1.0, 1), (1.0, 2), (1.0, 3)]
WOVEN = ("You are a thoughtful assistant with a philosophical turn of mind. Answer the user's request fully and helpfully, "
         "weaving brief philosophical reflections into the answer itself.")   # = exp_selfgen.PROMPTS["woven"]
N_GEN = 8


def chat_text(tok, user, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def prompts():
    return json.loads((nla.ROOT / "data" / "confirm_prompts.json").read_text())


def phase_calib():
    target = nla.Target()
    tok, mx = target.tok, target.mx
    pairs = json.loads((nla.ROOT / "data" / "contrast_pairs.json").read_text())
    grab = {}

    def hook(i, h):
        if i == PUSH_LAYER:
            grab["h"] = h
        return h

    def layer8(text):
        target.acts(tok.encode(text, add_special_tokens=False), hook=hook)
        return np.array(grab["h"][0].astype(mx.float32))

    norms, diffs = [], []
    for p in pairs:
        prompt = chat_text(tok, p["q"])
        n0 = len(tok.encode(prompt, add_special_tokens=False))
        means = {}
        for kind in ("philo", "plain"):
            ids = tok.encode(prompt + p[kind], add_special_tokens=False)
            assert ids[:n0] == tok.encode(prompt, add_special_tokens=False)
            h = layer8(prompt + p[kind])
            means[kind] = h[n0:].mean(0)
            if kind == "plain":
                norms.append(np.linalg.norm(h, axis=1))
        diffs.append(means["philo"] - means["plain"])
    d = np.mean(diffs, 0)
    p50 = float(np.median(np.concatenate(norms)))
    cos_pairs = [float(x @ d / np.linalg.norm(x) / np.linalg.norm(d)) for x in diffs]
    np.savez(OUT / "s2_dir.npz", dm=d, p50=p50)
    print(f"layer {PUSH_LAYER}: |dm| {np.linalg.norm(d):.2f}, p50 {p50:.2f}, push norm {ALPHA * p50:.2f}; "
          f"per-pair cos to mean: median {np.median(cos_pairs):.2f}, min {min(cos_pairs):.2f}")


def push_vectors():
    z = np.load(OUT / "s2_dir.npz")
    d, p50 = z["dm"], float(z["p50"])
    r = np.random.default_rng(0).normal(size=d.shape)
    return {"push": ALPHA * p50 * d / np.linalg.norm(d), "random": ALPHA * p50 * r / np.linalg.norm(r)}


def phase_acts():
    target = nla.Target()
    tok, mx = target.tok, target.mx
    vecs = {k: mx.array(v.astype(np.float32)) for k, v in push_vectors().items()}
    rows, acts = [], []
    for pi, q in enumerate(prompts()):
        for cfg in CONFIGS:
            text = chat_text(tok, q, WOVEN if cfg == "woven" else None)
            v = vecs.get(cfg)

            def hook(i, h, v=v):
                return h + v.astype(h.dtype) if (v is not None and i == PUSH_LAYER) else h

            ids = tok.encode(text, add_special_tokens=False)
            acts.append(target.acts(ids, hook=hook)[0])
            rows.append({"p": pi, "config": cfg, "request": q, "context": text, "token": tok.decode([ids[-1]])})
    acts = np.stack(acts)
    np.save(OUT / "s2_acts.npy", acts)
    (OUT / "s2_rows.json").write_text(json.dumps(rows, indent=1))
    base = {r["p"]: acts[i] for i, r in enumerate(rows) if r["config"] == "base"}
    for cfg in CONFIGS[1:]:
        c = [float(acts[i] @ base[r["p"]] / np.linalg.norm(acts[i]) / np.linalg.norm(base[r["p"]]))
             for i, r in enumerate(rows) if r["config"] == cfg]
        print(f"{cfg}: layer-21 cos to base, mean {np.mean(c):.3f}, min {np.min(c):.3f}")
    print(acts.shape, "activations")


def phase_gen(max_tokens=100):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    model, tok = load("mlx-community/Qwen3.5-9B-4bit")
    inner = model.language_model.model if hasattr(model, "language_model") else model.model
    layer = inner.layers[PUSH_LAYER]
    state = {"v": None}
    base_cls = type(layer)

    class Pushed(base_cls):
        def __call__(self_, x, mask=None, cache=None):
            out = base_cls.__call__(self_, x, mask=mask, cache=cache)
            return out if state["v"] is None else out + state["v"].astype(out.dtype)

    layer.__class__ = Pushed
    vecs = {k: mx.array(v.astype(np.float32)) for k, v in push_vectors().items()}
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    with (OUT / "s2_gen.jsonl").open("w") as f:
        for q in prompts()[:N_GEN]:
            for cfg in CONFIGS:
                state["v"] = vecs.get(cfg)
                ids = tok.encode(chat_text(tok, q, WOVEN if cfg == "woven" else None), add_special_tokens=False)
                cache = make_prompt_cache(model)
                logits = model(mx.array(ids)[None], cache=cache)[0, -1]
                out = []
                for _ in range(max_tokens):
                    nxt = int(mx.argmax(logits).item())
                    if nxt in stop:
                        break
                    out.append(nxt)
                    logits = model(mx.array([[nxt]]), cache=cache)[0, -1]
                text = tok.decode(out)
                f.write(json.dumps({"request": q, "config": cfg, "text": text}) + "\n")
                print(f"[{cfg}] {q}\n  {text[:300]!r}", flush=True)


def phase_explain():
    av = nla.Verbalizer()
    acts = np.load(OUT / "s2_acts.npy")
    path = OUT / "s2_expl.jsonl"
    done = set()
    if path.exists():
        done = {(r["i"], r["seed"]) for r in map(json.loads, path.read_text().splitlines()) if r}
    todo = [(i, t, s) for i in range(len(acts)) for t, s in SAMPLES if (i, s) not in done]
    t0 = time.time()
    with path.open("a") as f:
        for k, (i, temp, seed) in enumerate(todo, 1):
            r = av.explain(acts[i], temp=temp, seed=seed)
            f.write(json.dumps({"i": i, "temp": temp, "seed": seed, **r}) + "\n")
            f.flush()
            if k % 20 == 0:
                el = time.time() - t0
                print(f"{k}/{len(todo)} explanations, {el / k:.1f}s each, eta {(len(todo) - k) * el / k / 60:.0f} min",
                      flush=True)


def options(n_prompts, seed=0):
    """Per prompt: 4 requests (true + 3 random distractors) in a fixed random order, and the true option's index."""
    rng = random.Random(seed)
    out = []
    for p in range(n_prompts):
        opts = [p] + rng.sample([j for j in range(n_prompts) if j != p], 3)
        rng.shuffle(opts)
        out.append(opts)
    return out


JUDGE_MATCH = """You are helping with an interpretability experiment. Each item below is a text that describes the internal
state of a language model at the moment it starts answering a user's request. The texts were written by a decoder of
the model's activations; they may be noisy, partly wrong, or drift to related topics. Each item lists four candidate
user requests, numbered 1-4; exactly one of them is the request the model was answering.

For each item, pick the candidate request the text most likely came from. Always give your best guess (1, 2, 3 or 4),
even when the evidence is weak. Judge each item independently.

Return ONLY a JSON array, one object per item, in any order: [{"id": "...", "answer": 1|2|3|4}, ...]

ITEMS:
"""

JUDGE_REFLECT = """You are helping with an interpretability experiment. Each item below is a text that describes the internal
state of a language model at the moment it starts answering a user's request. The texts were written by a decoder of
the model's activations and may be noisy.

For each item, answer "yes" if the text says or suggests that the model is about to bring in reflection, philosophy,
meaning, or abstract ideas (for example: musing on why something matters, a philosophical aside, a reflective or
contemplative register), and "no" otherwise. A purely practical, factual, or instructional answer is "no".
Judge each item independently.

Return ONLY a JSON array, one object per item, in any order: [{"id": "...", "answer": "yes"|"no"}, ...]

ITEMS:
"""


def judge_items():
    rows = json.loads((OUT / "s2_rows.json").read_text())
    ex = [json.loads(line) for line in (OUT / "s2_expl.jsonl").read_text().splitlines() if line]
    reqs = prompts()
    opts = options(len(reqs))
    items = []
    for r in ex:
        row = rows[r["i"]]
        text = r["expl"] or r["text"]
        cands = "\n".join(f"{n}. {reqs[j]}" for n, j in enumerate(opts[row["p"]], 1))
        for cond, t in (("expl", text), ("stripped", nla.strip_quotes(text, row["context"]))):
            items.append({"key": f"{r['i']}|{r['seed']}|{cond}|match", "kind": "match",
                          "body": f"{t}\n\nCANDIDATE REQUESTS:\n{cands}"})
        items.append({"key": f"{r['i']}|{r['seed']}|expl|reflect", "kind": "reflect", "body": text})
    return items


def phase_judge(model="sonnet", batch=20, workers=4):
    from scripts.claude_judge import call_claude, parse
    path = OUT / "s2_judge.jsonl"
    done = set()
    if path.exists():
        done = {r["key"] for r in map(json.loads, path.read_text().splitlines()) if r}
    items = [it for it in judge_items() if it["key"] not in done]
    rng = random.Random(0)
    rng.shuffle(items)
    batches = []
    for kind in ("match", "reflect"):
        its = [it for it in items if it["kind"] == kind]
        batches += [(kind, its[k:k + batch]) for k in range(0, len(its), batch)]
    print(f"{len(items)} items in {len(batches)} batches", flush=True)

    def run(b):
        kind, its = b
        ids = {f"t{random.randrange(10**6):06d}{n}": it for n, it in enumerate(its)}
        body = "\n\n".join(f'<item id="{k}">\n{it["body"]}\n</item>' for k, it in ids.items())
        head = JUDGE_MATCH if kind == "match" else JUDGE_REFLECT
        for attempt in range(3):
            try:
                got = {g["id"]: g["answer"] for g in parse(call_claude(head + body, model))}
                return [{"key": it["key"], "answer": got.get(k)} for k, it in ids.items()]
            except Exception as e:  # noqa: BLE001
                print("retry", attempt, str(e)[:200], flush=True)
        return []

    with path.open("a") as f, ThreadPoolExecutor(workers) as pool:
        for n, res in enumerate(pool.map(run, batches), 1):
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f"{n}/{len(batches)} batches", flush=True)


def _boot(vals, rng, n_boot=2000):
    vals = np.asarray(vals, float)
    bs = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n_boot)]
    return [round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])]


def phase_analyze():
    rows = json.loads((OUT / "s2_rows.json").read_text())
    judged = {r["key"]: r["answer"] for r in map(json.loads, (OUT / "s2_judge.jsonl").read_text().splitlines()) if r}
    reqs = prompts()
    opts = options(len(reqs))
    acts = np.load(OUT / "s2_acts.npy")
    rng = np.random.default_rng(0)
    idx = {(r["p"], r["config"]): i for i, r in enumerate(rows)}

    def pick(i, seed, cond):
        a = judged.get(f"{i}|{seed}|{cond}|match")
        try:
            return opts[rows[i]["p"]][int(a) - 1]
        except (TypeError, ValueError, IndexError):
            return None

    def reflect(i, seed):
        a = judged.get(f"{i}|{seed}|expl|reflect")
        return None if a is None else str(a).strip().lower() == "yes"

    res = {"n_prompts": len(reqs), "alpha": ALPHA, "push_layer": PUSH_LAYER, "chance_match": 0.25}
    per = {}
    for cfg in CONFIGS:
        out = {}
        for cond in ("expl", "stripped"):
            # per prompt: fraction of the 4 samples that pick the true request; and the majority-of-4 pick
            frac, vote, greedy = [], [], []
            for p in range(len(reqs)):
                i = idx[(p, cfg)]
                picks = [pick(i, s, cond) for _, s in SAMPLES]
                frac.append(np.mean([x == p for x in picks]))
                c = Counter(x for x in picks if x is not None).most_common()
                vote.append(bool(c) and c[0][0] == p and (len(c) == 1 or c[0][1] > c[1][1]))
                greedy.append(picks[0] == p)
            out[cond] = {"per_sample": round(float(np.mean(frac)), 3), "per_sample_ci95": _boot(frac, rng),
                         "vote": round(float(np.mean(vote)), 3), "greedy": round(float(np.mean(greedy)), 3)}
            per[(cfg, cond)] = np.array(frac)
        refl = [np.mean([x for x in (reflect(idx[(p, cfg)], s) for _, s in SAMPLES) if x is not None] or [np.nan])
                for p in range(len(reqs))]
        out["reflect_rate"] = round(float(np.nanmean(refl)), 3)
        out["reflect_ci95"] = _boot([x for x in refl if not np.isnan(x)], rng)
        per[(cfg, "reflect")] = np.array(refl)
        if cfg != "base":
            c = [float(acts[idx[(p, cfg)]] @ acts[idx[(p, "base")]] / np.linalg.norm(acts[idx[(p, cfg)]])
                       / np.linalg.norm(acts[idx[(p, "base")]])) for p in range(len(reqs))]
            out["l21_cos_to_base"] = round(float(np.mean(c)), 3)
        res[cfg] = out
    diffs = {}
    for cfg in CONFIGS[1:]:
        for m in ("expl", "reflect"):
            d = per[(cfg, m)] - per[("base", m)]
            d = d[~np.isnan(d)]
            diffs[f"{cfg}-base:{m}"] = {"mean": round(float(d.mean()), 3), "ci95": _boot(d, rng)}
    res["paired_vs_base"] = diffs
    (OUT / "s2_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["calib", "acts", "gen", "explain", "judge", "analyze"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"calib": phase_calib, "acts": phase_acts, "gen": phase_gen, "explain": phase_explain,
     "judge": phase_judge, "analyze": phase_analyze}[args.phase]()


if __name__ == "__main__":
    main()
