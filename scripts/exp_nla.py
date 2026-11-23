"""Pilot AM, stage 0: does the Qwen3.5-9B layer-21 NLA (therem/qwen3.5-9b-nla-L21-{av,ar}-rl) work in MLX here?

Gate (docs/plans/nla-pilot.md): on held-out rows of the NLA's own corpus (ceselder/easynla-dsv4-warmstart-opus5,
corpus/av_val.parquet, document-disjoint from SFT and RL), the round trip AV (4-bit, greedy) -> AR (bf16) must reach
FVE >= 0.70 (card: 0.786 on 512 rows with vLLM/bf16). Reported alongside:
  - no-injection FVE (AV sees the bare marker): what the AR gets from the prompt alone;
  - tail-only FVE (AR reads the last 80 characters of the source instead of an explanation; card: ~0.58);
  - gold FVE (AR reads the Claude-written training explanation);
  - CJK rate (a silently skipped injection makes the AV write Chinese) and the hook fire count.

Phases run as separate processes so that only one model is in memory at a time:
  .venv-mlx/bin/python scripts/exp_nla.py convert            -> runs/nla/models/av-4bit
  .venv-mlx/bin/python scripts/exp_nla.py acts    --n 128     -> runs/nla/stage0_acts.npy, stage0_rows.json
  .venv-mlx/bin/python scripts/exp_nla.py explain --n 128     -> runs/nla/stage0_expl.jsonl
  .venv-mlx/bin/python scripts/exp_nla.py score               -> runs/nla/stage0.json

Smoke test on our chat format (8 AL verdict prompts: 4 units x intact/corrupted; 8 held-out prompts), read by hand:
  .venv-mlx/bin/python scripts/exp_nla.py smoke-acts          -> runs/nla/smoke_acts.npy, smoke_rows.json
  .venv-mlx/bin/python scripts/exp_nla.py smoke-explain       -> runs/nla/smoke_expl.jsonl
Positions: verdict "pre" = last token of the "Answer:" prefill, "passage" = last token of the user turn;
held-out "assistant" = last token of the generation prompt (thinking off, as everywhere in this repo).
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wi import nla  # noqa: E402

OUT = nla.ROOT / "runs" / "nla"
DATASET = "ceselder/easynla-dsv4-warmstart-opus5"
N_NOINJ = 32


def rows(n):
    import pyarrow.parquet as pq
    path = nla._snapshot(DATASET, "dataset") / "corpus" / "av_val.parquet"
    return pq.read_table(path).slice(0, n).to_pylist()


def phase_acts(n):
    target = nla.Target()
    rs = rows(n)
    acts, meta = [], []
    t0 = time.time()
    for r in rs:
        ids = target.tok.encode(r["detokenized_text_truncated"], add_special_tokens=True)
        acts.append(target.acts(ids)[0])
        meta.append({"doc_id": r["doc_id"], "text": r["detokenized_text_truncated"],
                     "gold": r["api_explanation"], "ntok": len(ids)})
    np.save(OUT / "stage0_acts.npy", np.stack(acts))
    (OUT / "stage0_rows.json").write_text(json.dumps(meta))
    print(f"acts {np.stack(acts).shape} in {time.time() - t0:.0f}s; mean norm {np.linalg.norm(acts, axis=1).mean():.1f}")


def phase_explain(n):
    av = nla.Verbalizer()
    acts = np.load(OUT / "stage0_acts.npy")[:n]
    path = OUT / "stage0_expl.jsonl"
    done = set()
    if path.exists():
        done = {(r["i"], r["cond"]) for r in map(json.loads, path.read_text().splitlines()) if r}
    t0, k = time.time(), 0
    with path.open("a") as f:
        for i, v in enumerate(acts):
            for cond in ("inject", "noinj") if i < N_NOINJ else ("inject",):
                if (i, cond) in done:
                    continue
                r = av.explain(v, temp=0.0, inject=cond == "inject")
                f.write(json.dumps({"i": i, "cond": cond, **r}) + "\n")
                f.flush()
                k += 1
                if k % 8 == 0:
                    print(f"{k} explanations, {(time.time() - t0) / k:.1f}s each", flush=True)


def phase_score():
    ar = nla.Reconstructor()
    gold = np.load(OUT / "stage0_acts.npy")
    meta = json.loads((OUT / "stage0_rows.json").read_text())
    ex = [json.loads(line) for line in (OUT / "stage0_expl.jsonl").read_text().splitlines() if line]
    res = {"n_rows": len(gold)}

    def score(name, idx, texts):
        idx = list(idx)
        pred = np.full((len(idx), gold.shape[1]), np.nan)
        ok = [j for j, t in enumerate(texts) if t]
        if ok:
            pred[ok] = ar.predict([texts[j] for j in ok])
        f, mse, _ = nla.fve(pred, gold[idx])
        res[name] = {"fve": round(f, 4), "n": len(ok), "n_unparsed": len(idx) - len(ok)}
        print(name, res[name], flush=True)
        return mse

    inj = sorted((r for r in ex if r["cond"] == "inject"), key=lambda r: r["i"])
    noinj = sorted((r for r in ex if r["cond"] == "noinj"), key=lambda r: r["i"])
    mse_inj = score("av_inject", [r["i"] for r in inj], [r["expl"] for r in inj])
    score("av_inject_quotes_stripped", [r["i"] for r in inj],
          [nla.strip_quotes(r["expl"], meta[r["i"]]["text"]) if r["expl"] else None for r in inj])
    score("av_noinject", [r["i"] for r in noinj], [r["expl"] for r in noinj])
    score("av_inject_same_rows_as_noinject", [r["i"] for r in noinj], [inj[r["i"]]["expl"] for r in noinj])
    score("tail80", range(len(gold)), [m["text"][-80:] for m in meta])
    score("gold_explanation", range(len(gold)), [m["gold"] for m in meta])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(inj))
    score("av_inject_shuffled", [r["i"] for r in inj], [inj[p]["expl"] for p in perm])
    res["cjk"] = sum(nla.has_cjk(r["text"]) for r in inj)
    res["fired_not_once"] = sum(r["fired"] != 1 for r in inj)
    res["noinj_fired"] = sum(r["fired"] != 0 for r in noinj)
    res["truncated"] = sum(r["finish"] == "length" for r in inj)
    res["mean_tokens"] = round(float(np.mean([r["ntok"] for r in inj])), 1)
    res["quoted_fraction"] = round(float(np.mean([nla.quoted_fraction(r["expl"], meta[r["i"]]["text"])
                                                   for r in inj if r["expl"]])), 3)
    _, _, base = nla.fve(gold, gold)
    m = mse_inj[~np.isnan(mse_inj)]
    bs = [1 - m[rng.integers(0, len(m), len(m))].mean() / base for _ in range(2000)]
    res["av_inject"]["ci95"] = [round(float(x), 4) for x in np.percentile(bs, [2.5, 97.5])]
    res["gate_pass"] = res["av_inject"]["fve"] >= 0.70
    (OUT / "stage0.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def smoke_prompts(tok):
    from scripts import philbench_items as items_mod
    out, seen = [], []
    for it in items_mod.all_items(("verdict",)):
        if not it["variant"].endswith("follows-first") or it["form"] not in ("original", "corrupted"):
            continue
        if it["item"] not in seen:
            if len(seen) == 4:
                continue
            seen.append(it["item"])
        text = tok.apply_chat_template([{"role": "user", "content": it["user"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) + items_mod.PREFILL
        out.append({"kind": "verdict", "item": it["item"], "form": it["form"], "expected": it["expected"],
                    "user": it["user"], "text": text})
    for q in json.loads((nla.ROOT / "data" / "heldout_prompts.json").read_text())[:8]:
        text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        out.append({"kind": "heldout", "item": q, "form": None, "user": q, "text": text})
    return out


def phase_smoke_acts():
    target = nla.Target()
    im_end = target.tok.convert_tokens_to_ids("<|im_end|>")
    rows_, acts = [], []
    for p in smoke_prompts(target.tok):
        ids = target.tok.encode(p["text"], add_special_tokens=False)
        user_end = ids.index(im_end, ids.index(target.tok.convert_tokens_to_ids("<|im_start|>"))) - 1
        positions = {"pre" if p["kind"] == "verdict" else "assistant": len(ids) - 1}
        if p["kind"] == "verdict":
            positions["passage"] = user_end
        a = target.acts(ids, list(positions.values()))
        for (name, pos), v in zip(positions.items(), a):
            rows_.append({**p, "position": name, "pos": pos, "token": target.tok.decode([ids[pos]]),
                          "context": target.tok.decode(ids[: pos + 1])})
            acts.append(v)
    np.save(OUT / "smoke_acts.npy", np.stack(acts))
    (OUT / "smoke_rows.json").write_text(json.dumps(rows_, indent=1))
    print(len(rows_), "smoke activations")


def phase_smoke_explain():
    av = nla.Verbalizer()
    acts = np.load(OUT / "smoke_acts.npy")
    meta = json.loads((OUT / "smoke_rows.json").read_text())
    with (OUT / "smoke_expl.jsonl").open("w") as f:
        for i, (v, m) in enumerate(zip(acts, meta)):
            r = av.explain(v, temp=0.0)
            r["quoted_fraction"] = round(nla.quoted_fraction(r["expl"] or "", m["context"]), 3)
            f.write(json.dumps({"i": i, **{k: m[k] for k in ("kind", "item", "form", "position", "token")}, **r})
                    + "\n")
            f.flush()
            print(i, m["kind"], m["form"], m["position"], repr(m["token"]), "->", (r["expl"] or r["text"])[:200],
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["convert", "acts", "explain", "score", "smoke-acts", "smoke-explain"])
    ap.add_argument("--n", type=int, default=128)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "convert":
        print(nla.prepare_mlx_dirs())
    elif args.phase == "acts":
        phase_acts(args.n)
    elif args.phase == "explain":
        phase_explain(args.n)
    elif args.phase == "score":
        phase_score()
    elif args.phase == "smoke-acts":
        phase_smoke_acts()
    else:
        phase_smoke_explain()
    gc.collect()


if __name__ == "__main__":
    main()
