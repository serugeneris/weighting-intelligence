"""Pilot AM, stage 1 (Q1): does the 9B's layer-21 state, read by the NLA, carry its validity verdict?

Items: AL `verdict` prompts (scripts/philbench_items.verdict) for the 22 original arguments and their 87 corrupted
versions, both letter orders: 218 prompts, rendered exactly as in scripts/exp_real_text_mlx.py (thinking off,
"Answer:" prefill). Paraphrases are left out.

Positions per prompt: "pre" = last token of the "Answer:" prefill (main measure); "passage" = last token of the user
turn (is the verdict there before the question is answered?). Activations come from bf16 Qwen3.5-9B blocks 0-21
(the NLA's training distribution); the model's own verdict comes from AL's 9B-4bit:base scores
(runs/real_text/scores.jsonl), and `acts4` measures how far the 4-bit model's activations are from the bf16 ones.

Per activation: 1 greedy + 3 sampled (T=1) explanations. A blind `claude -p` judge (Sonnet) sees one text at a time,
never the passage or the condition, pooled and shuffled across conditions:
  - pre:     which answer letter is the model about to give? A / B / unclear
  - passage: does the argument's conclusion follow? follows / does not follow / unclear
Conditions judged: the explanation; the explanation with copied spans (>= 5 tokens) removed; tail-only (the last 80
characters of the context). The shuffled-activation control and the majority baseline need no extra judging: the
shuffled control scores another prompt's judged explanation against this prompt's labels.

Predictions (written before the run, from docs/plans/nla-pilot.md):
  P1a  Agreement of the judged explanation with the model's verdict exceeds tail-only, shuffled and majority.
  P1b  On prompts the 9B gets wrong, the explanation agrees with the model's (wrong) verdict more than with the key.
       If it agrees with the key, the AV is solving the task from quoted text and Q1 cannot be read as the model's
       belief.
  P1c  A logistic probe on the same activations (grouped CV by argument) sets the ceiling for what is linearly
       present; the NLA is expected below it.
Stop rule: if P1a fails, report that the NLA does not read validity from these items at layer 21; do not read P1b.

  .venv-mlx/bin/python scripts/exp_nla_validity.py acts       -> runs/nla/s1_acts.npy, s1_rows.json
  .venv-mlx/bin/python scripts/exp_nla_validity.py acts4      -> runs/nla/s1_acts4_cos.json
  .venv-mlx/bin/python scripts/exp_nla_validity.py probe      -> runs/nla/s1_probe.json
  .venv-mlx/bin/python scripts/exp_nla_validity.py explain    -> runs/nla/s1_expl.jsonl   (resumable, ~1.8 h)
  .venv-mlx/bin/python scripts/exp_nla_validity.py judge      -> runs/nla/s1_judge.jsonl  (resumable)
  .venv-mlx/bin/python scripts/exp_nla_validity.py analyze    -> runs/nla/s1_results.json
"""
import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import philbench_items as items_mod  # noqa: E402
from wi import nla  # noqa: E402

OUT = nla.ROOT / "runs" / "nla"
SAMPLES = [(0.0, 0), (1.0, 1), (1.0, 2), (1.0, 3)]   # (temperature, seed): greedy + 3 sampled
TAIL = 80


def prompts():
    scores = {}
    for line in (nla.ROOT / "runs" / "real_text" / "scores.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["config"] == "9B-4bit:base" and r["task"] == "verdict":
            scores[(r["item"], r["variant"])] = r["probs"]
    out = []
    for it in items_mod.verdict():
        if it["form"] == "paraphrase":
            continue
        p = scores[(it["item"], it["variant"])]
        letter = "AB"[int(np.argmax(p))] if p[0] != p[1] else None
        out.append({"item": it["item"], "variant": it["variant"], "form": it["form"], "user": it["user"],
                    "key_letter": it["expected"], "follows_letter": it["follows_letter"],
                    "key_follows": not it["form"].startswith("corrupted"),
                    "p_model": p, "model_letter": letter,
                    "model_follows": None if letter is None else letter == it["follows_letter"]})
    return out


def render(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) + items_mod.PREFILL


def positions(tok, ids):
    im_end, im_start = tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|im_start|>")
    return {"pre": len(ids) - 1, "passage": ids.index(im_end, ids.index(im_start)) - 1}


def phase_acts():
    target = nla.Target()
    rows, acts = [], []
    t0 = time.time()
    for p in prompts():
        ids = target.tok.encode(render(target.tok, p["user"]), add_special_tokens=False)
        pos = positions(target.tok, ids)
        a = target.acts(ids, list(pos.values()))
        for (name, k), v in zip(pos.items(), a):
            rows.append({**p, "position": name, "pos": k, "context_tail": target.tok.decode(ids[: k + 1])[-TAIL:],
                         "context": target.tok.decode(ids[: k + 1])})
            acts.append(v)
    np.save(OUT / "s1_acts.npy", np.stack(acts))
    (OUT / "s1_rows.json").write_text(json.dumps(rows))
    print(f"{len(rows)} activations in {time.time() - t0:.0f}s")


def phase_acts4():
    """Cosine between bf16 and 4-bit block-21 activations at the same positions (verdicts come from the 4-bit)."""
    import mlx.core as mx
    from mlx_lm import load
    model, tok = load("mlx-community/Qwen3.5-9B-4bit", lazy=True)
    tm = model.language_model.model
    tm.layers = tm.layers[: nla.LAYER + 1]
    mx.eval(model.parameters())
    rows = json.loads((OUT / "s1_rows.json").read_text())
    acts = np.load(OUT / "s1_acts.npy")
    cos = []
    for i in range(0, len(rows), 2):
        ids = tok.encode(render(tok, rows[i]["user"]), add_special_tokens=False)
        h = nla._blocks_forward(tm, tm.embed_tokens(mx.array(ids)[None]), tm.layers)
        for j in (i, i + 1):
            v = np.array(h[0, rows[j]["pos"]].astype(mx.float32))
            cos.append(float(v @ acts[j] / np.linalg.norm(v) / np.linalg.norm(acts[j])))
    res = {"n": len(cos), "mean": float(np.mean(cos)), "min": float(np.min(cos)),
           "p05": float(np.percentile(cos, 5))}
    (OUT / "s1_acts4_cos.json").write_text(json.dumps(res, indent=1))
    print(res)


def _logreg(X, y, lam=1.0, iters=500):
    """L2 logistic regression by Newton steps on standardized PCA features (small n, no sklearn here)."""
    w = np.zeros(X.shape[1] + 1)
    Xb = np.hstack([X, np.ones((len(X), 1))])
    reg = lam * np.eye(len(w))
    reg[-1, -1] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) + reg @ w
        H = Xb.T @ (Xb * (p * (1 - p))[:, None]) + reg
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def phase_probe(k_pca=32, lam=10.0):
    rows = json.loads((OUT / "s1_rows.json").read_text())
    acts = np.load(OUT / "s1_acts.npy")
    groups = sorted({r["item"] for r in rows})
    rng = np.random.default_rng(0)
    folds = {g: f for f, gs in enumerate(np.array_split(rng.permutation(groups), 5)) for g in gs}
    res = {}
    for position in ("pre", "passage"):
        for target in ("model_letter", "key_letter", "model_follows", "key_follows"):
            idx = [i for i, r in enumerate(rows) if r["position"] == position and r[target] is not None]
            y = np.array([float(rows[i][target] in ("A", True)) for i in idx])
            X = acts[idx]
            pred = np.zeros(len(idx))
            for f in range(5):
                te = np.array([folds[rows[i]["item"]] == f for i in idx])
                mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-6
                Z = (X - mu) / sd
                _, _, Vt = np.linalg.svd(Z[~te], full_matrices=False)
                P = Z @ Vt[:k_pca].T
                w = _logreg(P[~te], y[~te], lam)
                pred[te] = (np.hstack([P[te], np.ones((te.sum(), 1))]) @ w) > 0
            acc = float((pred == y).mean())
            res[f"{position}:{target}"] = {"acc": round(acc, 3), "n": len(idx),
                                           "majority": round(float(max(y.mean(), 1 - y.mean())), 3)}
            print(position, target, res[f"{position}:{target}"])
    (OUT / "s1_probe.json").write_text(json.dumps(res, indent=1))


def phase_explain():
    av = nla.Verbalizer()
    acts = np.load(OUT / "s1_acts.npy")
    path = OUT / "s1_expl.jsonl"
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


JUDGE_PRE = """You are helping with an interpretability experiment. Each item below is a short text that describes the internal
state of a language model at the moment just before it answers a two-option multiple-choice question whose answer
is a single letter, A or B. Some texts are explanations written by a decoder of the model's activations; some are
fragments of raw text. They may be noisy, partly wrong, or about unrelated topics.

For each item, decide which answer letter the text indicates the model is about to give. Answer "A" or "B" only if
the text gives some indication; otherwise "unclear". Judge each item independently.

Return ONLY a JSON array, one object per item, in any order: [{"id": "...", "answer": "A"|"B"|"unclear"}, ...]

ITEMS:
"""

JUDGE_PASSAGE = """You are helping with an interpretability experiment. Each item below is a short text that describes the internal
state of a language model that has just read a short argument (a few sentences ending in a conclusion). Some texts
are explanations written by a decoder of the model's activations; some are fragments of raw text. They may be noisy,
partly wrong, or about unrelated topics.

For each item, decide whether the text indicates that the argument's final conclusion FOLLOWS from what comes before
it, or DOES NOT FOLLOW (the reasoning is flawed, a premise is missing, the inference is invalid, the conclusion
overreaches). Answer "follows" or "does not follow" only if the text gives some indication; otherwise "unclear".
Judge each item independently.

Return ONLY a JSON array, one object per item, in any order:
[{"id": "...", "answer": "follows"|"does not follow"|"unclear"}, ...]

ITEMS:
"""


def judge_items():
    rows = json.loads((OUT / "s1_rows.json").read_text())
    ex = [json.loads(line) for line in (OUT / "s1_expl.jsonl").read_text().splitlines() if line]
    items = []
    for r in ex:
        row = rows[r["i"]]
        text = r["expl"] or r["text"]
        items.append({"key": f"{r['i']}|{r['seed']}|expl", "position": row["position"], "text": text})
        items.append({"key": f"{r['i']}|{r['seed']}|stripped", "position": row["position"],
                      "text": nla.strip_quotes(text, row["context"])})
    for i, row in enumerate(rows):
        items.append({"key": f"{i}|-|tail", "position": row["position"], "text": row["context_tail"]})
    return items


def phase_judge(model="sonnet", batch=20, workers=4):
    from scripts.claude_judge import call_claude, parse
    path = OUT / "s1_judge.jsonl"
    done = set()
    if path.exists():
        done = {r["key"] for r in map(json.loads, path.read_text().splitlines()) if r}
    items = [it for it in judge_items() if it["key"] not in done]
    rng = random.Random(0)
    rng.shuffle(items)
    batches = []
    for position in ("pre", "passage"):
        its = [it for it in items if it["position"] == position]
        batches += [(position, its[k:k + batch]) for k in range(0, len(its), batch)]
    print(f"{len(items)} items in {len(batches)} batches")

    def run(b):
        position, its = b
        ids = {f"t{rng.randrange(10**6):06d}{n}": it for n, it in enumerate(its)}
        body = "\n\n".join(f'<item id="{k}">\n{it["text"]}\n</item>' for k, it in ids.items())
        head = JUDGE_PRE if position == "pre" else JUDGE_PASSAGE
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


def _norm_answer(position, a):
    if a is None:
        return None
    a = a.strip().lower()
    if position == "pre":
        return {"a": "A", "b": "B"}.get(a)
    return {"follows": True, "does not follow": False}.get(a)


def phase_analyze(n_boot=2000):
    rows = json.loads((OUT / "s1_rows.json").read_text())
    judged = {r["key"]: r["answer"] for r in map(json.loads, (OUT / "s1_judge.jsonl").read_text().splitlines()) if r}
    rng = np.random.default_rng(0)
    res = {}
    for position in ("pre", "passage"):
        lab_m, lab_k = ("model_letter", "key_letter") if position == "pre" else ("model_follows", "key_follows")
        idx = [i for i, r in enumerate(rows) if r["position"] == position and r[lab_m] is not None]

        def votes(cond, i):
            """Judged answers for activation i under cond; for expl/stripped a majority over the 4 samples."""
            if cond == "tail":
                a = _norm_answer(position, judged.get(f"{i}|-|tail"))
                return a
            got = [_norm_answer(position, judged.get(f"{i}|{s}|{cond}")) for _, s in SAMPLES]
            got = [g for g in got if g is not None]
            if not got:
                return None
            c = Counter(got).most_common()
            return c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else None

        def greedy(cond, i):
            return _norm_answer(position, judged.get(f"{i}|0|{cond}"))

        perm = {i: j for i, j in zip(idx, rng.permutation(idx))}
        conds = {
            "expl_vote": lambda i: votes("expl", i),
            "expl_greedy": lambda i: greedy("expl", i),
            "stripped_vote": lambda i: votes("stripped", i),
            "tail": lambda i: votes("tail", i),
            "shuffled_vote": lambda i: votes("expl", perm[i]),
        }
        maj_m = Counter(rows[i][lab_m] for i in idx).most_common(1)[0][0]
        out = {"n": len(idx), "majority_model_label": str(maj_m),
               "majority_agree_model": round(np.mean([rows[i][lab_m] == maj_m for i in idx]), 3),
               "majority_agree_key": round(np.mean([rows[i][lab_k] == maj_m for i in idx]), 3),
               "model_correct": round(np.mean([rows[i][lab_m] == rows[i][lab_k] for i in idx]), 3)}
        for name, f in conds.items():
            pred = {i: f(i) for i in idx}
            dec = [i for i in idx if pred[i] is not None]
            am = np.array([pred[i] == rows[i][lab_m] for i in dec], float)
            ak = np.array([pred[i] == rows[i][lab_k] for i in dec], float)
            wrong = [i for i in dec if rows[i][lab_m] != rows[i][lab_k]]
            # agreement over ALL prompts counting "unclear" as a miss (an honest coverage-adjusted rate)
            all_m = np.array([pred[i] == rows[i][lab_m] for i in idx], float)
            boot = [all_m[rng.integers(0, len(all_m), len(all_m))].mean() for _ in range(n_boot)]
            out[name] = {
                "decided": len(dec),
                "agree_model_of_decided": round(am.mean(), 3) if len(dec) else None,
                "agree_key_of_decided": round(ak.mean(), 3) if len(dec) else None,
                "agree_model_all": round(all_m.mean(), 3),
                "agree_model_all_ci95": [round(float(x), 3) for x in np.percentile(boot, [2.5, 97.5])],
                "model_wrong_decided": len(wrong),
                "on_model_wrong_agree_model": int(sum(pred[i] == rows[i][lab_m] for i in wrong)),
                "on_model_wrong_agree_key": int(sum(pred[i] == rows[i][lab_k] for i in wrong)),
                "pred_counts": dict(Counter(str(pred[i]) for i in idx)),
            }
        res[position] = out
    (OUT / "s1_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["acts", "acts4", "probe", "explain", "judge", "analyze"])
    ap.add_argument("--model", default="sonnet")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"acts": phase_acts, "acts4": phase_acts4, "probe": phase_probe, "explain": phase_explain,
     "judge": lambda: phase_judge(args.model), "analyze": phase_analyze}[args.phase]()


if __name__ == "__main__":
    main()
