"""Experiment AI: is "philosophical" the reflective register, or something else? (docs/plans/experiment-ae-philbench.md)

Everything in FINDINGS §1-33 measured philosophy as a judge's 0-9 score for reflective sentences in practical
answers. This experiment varies that register with the levers already built (prompts, layer-8 pushes along
diff-means and the learned direction B, random pushes of the same size, the rank-1 and Claude-taught students) and
measures philosophical skill on externally defined tasks scored without a judge: forced choice over the 2020
PhilPapers Survey questions (stability under paraphrase and option order, consistency with logical constraints,
holding a position under bare disagreement), an adapted forced-choice version of philosophy-bench, and the
in-house argument items of pilots Z and AB.

Predictions, written before any run:
  If philosophy is the register: skill metrics rise with register score across configurations, and rise with dose
    inside each lever family (B 0.4 -> 0.8, diff-means 0.35 -> 0.5, students x1.0 -> x1.5/x2).
  If it is something else: skill is flat or falls as register rises; a matched random push does as much damage as a
    register push; skill grows with model size while register does not (§33: 2B and 4B students equally reflective).
Agreement with the philosopher plurality is descriptive, not correctness. Stability, constraint consistency and
holding under pressure are the skill measures.

Stages (all outputs under runs/philbench/):
  data-check  item counts, single-token option letters, prompt lengths, constraint coverage
  score       forced choice after an "Answer:" assistant prefill, one item per forward pass (no padding, §28),
              resumable per configuration
              -> scores.jsonl; on base also the layer-8/16 projections of the last prompt token
  generate    PhilPapers baseline variant, 160 tokens, "2-3 sentences then ANSWER: <letter>" -> generate.jsonl
  pressure    second turn "I disagree. The answer is <other option>." with no reasons -> pressure.jsonl
  register    the 48 confirmation prompts under every configuration, 120 tokens -> register_gen.jsonl, then
                scripts/claude_judge.py --input philbench/register_gen.jsonl --output philbench/register_judge.jsonl
                scripts/claude_judge.py --model opus --input philbench/register_gen.jsonl --output philbench/register_judge_opus.jsonl
  analyze     -> summary.json (scripts/exp_philbench_analyze.py)

  .venv/bin/python -m scripts.exp_philbench --stage data-check score generate pressure register
  .venv/bin/python -m scripts.exp_philbench --model 2B --stage score pressure          (Stage 5, forced choice only)
  .venv-mlx/bin/python scripts/exp_philbench_mlx.py --model 4B   (then 9B)             (Stage 5, 4-bit MLX)
  .venv/bin/python -m scripts.exp_philbench --stage analyze
"""
import argparse
import json
import re
import time
from collections import defaultdict

import torch

from scripts import philbench_items as items_mod
from scripts.philosophy_pilot import REGISTRY, RUNS, configured, generate, messages, render, unit_directions
from wi.steer import Steerable

OUT = RUNS / "philbench"
GRID = ["base", "woven", "reflect-first", "diffmeans-0.35", "diffmeans-0.5", "B-0.4", "B-0.6", "B-0.8",
        "random0-0.6", "random1-0.6", "random2-0.6", "rank1-x1", "rank1-x2", "claude384-x1.0", "claude384-x1.5",
        "claude128-x1.5"]
GRID_2B = ["2B:base", "2B:woven", "2B:student"]
PROJ_LAYERS = (8, 16)


def read_jsonl(name):
    path = OUT / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def append_jsonl(name, rows):
    with (OUT / name).open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def limited(items, limit):
    """First `limit` distinct items per task (all variants of each)."""
    if not limit:
        return items
    keep, seen = [], defaultdict(list)
    for it in items:
        names = seen[it["task"]]
        if it["item"] not in names:
            if len(names) >= limit:
                continue
            names.append(it["item"])
        keep.append(it)
    return keep


def letter_ids(model, n, prefixed=True):
    """Token ids of the option letters; after the "Answer:" prefill they carry a leading space."""
    ids = [model.tok.encode((" " if prefixed else "") + c, add_special_tokens=False) for c in items_mod.LETTERS[:n]]
    assert all(len(t) == 1 for t in ids), "every option letter must be a single token"
    return [t[0] for t in ids]


@torch.no_grad()
def forced_choice(model, history, n, prefill=True):
    """Option-letter probabilities (softmax over the letters) and the full-vocabulary mass on them.

    The assistant turn is prefilled with "Answer:" so the next token is the choice. Without it, configurations that
    open with prose (reflect-first put ~1% of its next-token mass on the letters) were scored on noise; that first
    run is kept as scores_noprefill.jsonl / pressure_noprefill.jsonl, where letter_mass measures answering at once."""
    text = render(model, history) + (items_mod.PREFILL if prefill else "")

    def next_probs(t):
        enc = model.tok(t, return_tensors="pt").to(model.device)
        logits = model.model(**enc, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        return logits.softmax(-1), enc["input_ids"].shape[1]

    probs, length = next_probs(text)
    letters = probs[letter_ids(model, n, prefixed=prefill)]
    bold = 0.0
    if prefill:
        # Strongly steered configurations answer in bold ("Answer: **A**"): add that path,
        # P(" **") * P("A" | "Answer: **"), whenever it carries more than 2% of the mass.
        bold = float(probs[model.tok.convert_tokens_to_ids("Ġ**")])
        if bold > 0.02:
            second, _ = next_probs(text + " **")
            letters = letters + bold * second[letter_ids(model, n, prefixed=False)]
    mass = float(letters.sum())
    return {"probs": [round(p, 6) for p in (letters / mass).tolist()], "letter_mass": round(mass, 6),
            "bold_mass": round(bold, 6), "prompt_tokens": length}


@torch.no_grad()
def projections(model, history):
    """Last-prompt-token residual (layer output) projections onto unit diff-means_L and unit B (base only)."""
    units = unit_directions(model)
    dms = torch.load(RUNS / "diffmeans.pt")
    store, hooks = {}, []
    for L in PROJ_LAYERS:
        def hook(_m, _i, out, L=L):
            h = out[0] if isinstance(out, tuple) else out
            store[L] = h[0, -1].float().cpu()
        hooks.append(model.layers[L].register_forward_hook(hook))
    try:
        enc = model.tok(render(model, history) + items_mod.PREFILL, return_tensors="pt").to(model.device)
        model.model(**enc, use_cache=False, logits_to_keep=1)
    finally:
        for h in hooks:
            h.remove()
    out = {}
    for L in PROJ_LAYERS:
        dm = dms[L].float()
        out[f"L{L}_diffmeans"] = round(float(store[L] @ (dm / dm.norm())), 4)
        out[f"L{L}_B"] = round(float(store[L] @ units["B"]), 4)
    return out


def stage_data_check(model, args):
    items = items_mod.all_items()
    counts = defaultdict(int)
    for it in items:
        counts[it["task"]] += 1
    print("items per task:", dict(counts))
    assert counts["philpapers"] == 98 * 3 and counts["philbench"] == 100
    assert counts["critique"] == 24 * 2 and counts["validity"] == 48 * 2
    letter_ids(model, max(len(it["labels"]) for it in items))
    lengths = defaultdict(list)
    for it in items:
        lengths[it["task"]].append(len(model.tok(render(model, messages(it["user"])))["input_ids"]))
    for task, ls in lengths.items():
        print(f"{task}: prompt tokens min {min(ls)} median {sorted(ls)[len(ls) // 2]} max {max(ls)}")
    topics = {q["topic"] for q in items_mod.philpapers_data()["questions"]}
    cons = items_mod.philpapers_constraints()
    assert all(c["a"]["topic"] in topics and c["b"]["topic"] in topics for c in cons)
    print(f"{len(cons)} constraints ({sum(c['contestable'] for c in cons)} contestable)")
    for name in args.configs:
        assert name in REGISTRY, name
    print("data-check passed")


def stage_score(model, args):
    items = limited(items_mod.all_items(), args.limit)
    done = {(r["config"], r["task"], r["item"], r["variant"]) for r in read_jsonl("scores.jsonl")}
    for name in args.configs:
        todo = [it for it in items if (name, it["task"], it["item"], it["variant"]) not in done]
        if not todo:
            continue
        started, rows = time.time(), []
        with configured(model, name) as system:
            for it in todo:
                history = messages(it["user"], system)
                row = {"config": name, **{k: v for k, v in it.items() if k != "user"},
                       **forced_choice(model, history, len(it["labels"]))}
                if name == "base" and it["task"] == "philpapers" and it["variant"] == "baseline":
                    row["proj"] = projections(model, history)
                rows.append(row)
        append_jsonl("scores.jsonl", rows)
        print(f"score {name}: {len(rows)} items ({time.time() - started:.0f}s)", flush=True)


def baseline_choices(config):
    rows = [r for r in read_jsonl("scores.jsonl")
            if r["config"] == config and r["task"] == "philpapers" and r["variant"] == "baseline"]
    return {r["item"]: r for r in rows}


ANSWER = re.compile(r"ANSWER:\s*\**\s*([A-L])\b")


def stage_generate(model, args):
    items = limited(items_mod.philpapers(("baseline",)), args.limit)
    done = {(r["config"], r["item"]) for r in read_jsonl("generate.jsonl")}
    for name in args.configs:
        todo = [it for it in items if (name, it["item"]) not in done]
        if not todo:
            continue
        started = time.time()
        with configured(model, name) as system:
            outs = generate(model, [messages(items_mod.generation_user(it), system) for it in todo], 160)
        forced = baseline_choices(name)
        rows = []
        for it, out in zip(todo, outs):
            found = ANSWER.findall(out["answer"])
            letter = found[-1] if found else None
            valid = letter is not None and items_mod.LETTERS.index(letter) < len(it["labels"])
            probs = forced.get(it["item"], {}).get("probs")
            fc = items_mod.LETTERS[max(range(len(probs)), key=probs.__getitem__)] if probs else None
            rows.append({"config": name, "item": it["item"], "labels": it["labels"], **out,
                         "stated": letter if valid else None, "compliant": valid, "forced_choice": fc,
                         "match": valid and letter == fc, "words": len(out["answer"].split())})
        append_jsonl("generate.jsonl", rows)
        print(f"generate {name}: {len(rows)} ({time.time() - started:.0f}s)", flush=True)


def stage_pressure(model, args):
    items = {it["item"]: it for it in limited(items_mod.philpapers(("baseline",)), args.limit)}
    done = {(r["config"], r["item"]) for r in read_jsonl("pressure.jsonl")}
    for name in args.configs:
        first = baseline_choices(name)
        todo = [k for k in items if (name, k) not in done and k in first]
        if not todo:
            continue
        started, rows = time.time(), []
        with configured(model, name) as system:
            for key in todo:
                it, probs = items[key], first[key]["probs"]
                chosen = max(range(len(probs)), key=probs.__getitem__)
                # the most probable substantive option the model did not pick
                subst = [i for i, kind in enumerate(it["kinds"]) if kind == "substantive" and i != chosen]
                target = max(subst, key=probs.__getitem__)
                history = ([{"role": "system", "content": system}] if system else []) + \
                    items_mod.pressure_turns(it, chosen, target)
                after = forced_choice(model, history, len(it["labels"]))
                new = max(range(len(after["probs"])), key=after["probs"].__getitem__)
                rows.append({"config": name, "item": key, "chosen": chosen, "target": target,
                             "chosen_kind": it["kinds"][chosen], "p_before": probs[chosen],
                             "after": after["probs"], "new": new, "flip": new != chosen, "to_target": new == target,
                             "letter_mass": after["letter_mass"]})
        append_jsonl("pressure.jsonl", rows)
        print(f"pressure {name}: {len(rows)} ({time.time() - started:.0f}s)", flush=True)


def stage_register(model, args):
    prompts = json.loads((items_mod.ROOT / "data/confirm_prompts.json").read_text())
    if args.limit:
        prompts = prompts[:args.limit]
    out = "register_gen.jsonl" if not args.limit else "register_gen_smoke.jsonl"
    done = {r["name"] for r in read_jsonl(out)}
    for name in args.configs:
        if name in done:
            continue
        started = time.time()
        with configured(model, name) as system:
            chats = [model.chat(p, system=system) for p in prompts]
            outs = []
            for i in range(0, len(chats), 16):
                outs += model.generate(chats[i:i + 16], max_new_tokens=120)
        append_jsonl(out, [{"name": name, "kind": name.split(":")[0] if ":" in name else "4B",
                            "prompts": prompts, "split": ["heldout"] * len(prompts), "outputs": outs}])
        print(f"register {name} ({time.time() - started:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", nargs="+", default=["data-check"],
                    choices=["data-check", "score", "generate", "pressure", "register", "analyze"])
    ap.add_argument("--model", choices=["4B", "2B"], default="4B")
    ap.add_argument("--configs", nargs="+")
    ap.add_argument("--limit", type=int, default=0, help="first N items per task (smoke runs)")
    ap.add_argument("--out", default="philbench", help="output folder under runs/ (smoke runs use their own)")
    args = ap.parse_args()
    global OUT
    OUT = RUNS / args.out
    OUT.mkdir(parents=True, exist_ok=True)
    args.configs = args.configs or (GRID if args.model == "4B" else GRID_2B)
    if args.stage == ["analyze"]:
        from scripts.exp_philbench_analyze import analyze
        analyze(OUT)
        return
    model = Steerable("Qwen/Qwen3.5-4B" if args.model == "4B" else "Qwen/Qwen3.5-2B")
    print(f"Device: {model.device}, layers: {model.n_layers}", flush=True)
    stages = {"data-check": stage_data_check, "score": stage_score, "generate": stage_generate,
              "pressure": stage_pressure, "register": stage_register}
    for stage in args.stage:
        if stage == "analyze":
            from scripts.exp_philbench_analyze import analyze
            analyze(OUT)
        else:
            stages[stage](model, args)


if __name__ == "__main__":
    main()
