"""Pilot AL: a philosophical-skill measure built from real texts (docs/plans/real-text-skill-pilot.md).

Two forced-choice tasks whose content comes from public-domain philosophical texts (data/sources/) and whose keys come
from the text or from how the item was built, so no grader is needed:
  corrupt   which of two passages has a conclusion that follows (original or paraphrase vs a one-sentence corruption;
            original vs paraphrase as an unkeyed familiarity probe), data/pilot-real-corruption.json
  objreply  which of four texts from the same exchange is the reply to an objection (or the objection a reply
            answers), in original and fully paraphrased wording, data/pilot-real-objreply.json
Sources: Descartes (Meditations; Hobbes' Third Objections with replies), Leibniz-Clarke, Hume's Dialogues, Berkeley's
Three Dialogues, Anselm with Gaunilo, Vitoria (De Indis, De Jure Belli), Galileo (Two New Sciences). Mill was dropped
and Anselm, Vitoria and Galileo added at the author's request, to reduce the tilt towards modern northern Europe.

Predictions, written before any run:
  P1 (does the measure measure skill). Accuracy rises 2B -> 4B -> 9B, and the human graders score at or above the
     9B. If P1 fails, stop and report: the tasks measure something other than skill, and P2 cannot be read.
  P2 (the main claim). Prompts and trained students (woven, rank1-x1, claude384-x1.0) score like 4B base; constant
     pushes (diffmeans-0.5, B-0.6, random0-0.6) score lower.
  P3 (memorisation). Accuracy is the same whether the correct option is the original passage or its paraphrase
     (orig-corr = para-corr), and the unkeyed original-vs-paraphrase pairs sit near 50/50. If the original wins,
     the model is using familiarity with the text.
The pilot works if human agreement with the keys is >= 85%, the 2B -> 9B gap is about 15 points or more with z > 2
(4-bit stack caveat), and fewer than a third of items are at ceiling or floor for every model.

Scoring is AI's unchanged: exp_philbench.forced_choice, "Answer:" prefill plus the bold path, one item per forward
pass, no padding, both orders. The 9B (4-bit MLX, scripts/exp_real_text_mlx.py) is compared only with the 4-bit 4B.

  .venv/bin/python -m scripts.exp_real_text --stage data-check score                  (4B bf16: base + register grid)
  .venv/bin/python -m scripts.exp_real_text --model 2B --stage score                  (2B bf16 base)
  .venv-mlx/bin/python scripts/exp_real_text_mlx.py --model 4B                        (then --model 9B)
  .venv/bin/python -m scripts.exp_real_text --stage analyze        -> runs/real_text/summary.json, summary.md
"""
import argparse
import json
import math
import statistics as st
import time
from collections import defaultdict

from scripts import philbench_items as items_mod

TASKS = ("corrupt", "objreply", "verdict")
GRID_4B = ["base", "woven", "rank1-x1", "claude384-x1.0", "diffmeans-0.5", "B-0.6", "random0-0.6"]
GRID_2B = ["2B:base"]
SCALE = [("2B:base", "base", "2B -> 4B (bf16)"), ("4B-4bit:base", "9B-4bit:base", "4B -> 9B (4-bit)"),
         ("2B:base", "9B-4bit:base", "2B (bf16) -> 9B (4-bit), across stacks")]
OUT = None


def read_jsonl(name):
    path = OUT / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def append_jsonl(name, rows):
    with (OUT / name).open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pilot_items(limit=0):
    items = items_mod.all_items(TASKS)
    if limit:
        from scripts.exp_philbench import limited
        items = limited(items, limit)
    return items


def stage_data_check(model, args):
    from scripts.exp_philbench import letter_ids
    from scripts.philosophy_pilot import REGISTRY, messages, render
    items = pilot_items()
    counts = defaultdict(int)
    for it in items:
        counts[it["task"]] += 1
    print("items per task:", dict(counts))
    letter_ids(model, 4)
    for task in TASKS:
        lengths = sorted(len(model.tok(render(model, messages(it["user"])))["input_ids"])
                         for it in items if it["task"] == task)
        print(f"{task}: prompt tokens min {lengths[0]} median {lengths[len(lengths) // 2]} max {lengths[-1]}")
    for name in args.configs:
        assert name in REGISTRY, name
    print("data-check passed")


def stage_score(model, args):
    from scripts.exp_philbench import forced_choice
    from scripts.philosophy_pilot import configured, messages
    items = pilot_items(args.limit)
    done = {(r["config"], r["task"], r["item"], r["variant"]) for r in read_jsonl("scores.jsonl")}
    for name in args.configs:
        todo = [it for it in items if (name, it["task"], it["item"], it["variant"]) not in done]
        if not todo:
            continue
        started, rows = time.time(), []
        with configured(model, name) as system:
            for it in todo:
                rows.append({"config": name, **{k: v for k, v in it.items() if k != "user"},
                             **forced_choice(model, messages(it["user"], system), len(it["labels"]))})
        append_jsonl("scores.jsonl", rows)
        print(f"score {name}: {len(rows)} items ({time.time() - started:.0f}s)", flush=True)


# ---------------------------------------------------------------- analysis

def mean_se(xs):
    xs = list(xs)
    if not xs:
        return None
    m = st.mean(xs)
    return {"mean": round(m, 4), "se": round(st.stdev(xs) / math.sqrt(len(xs)), 4) if len(xs) > 1 else None,
            "n": len(xs)}


def per_unit(rows, keep, value):
    """Average `value(row)` over the rows of each argument / objection pair that pass `keep` (both orders etc.)."""
    acc = defaultdict(list)
    for r in rows:
        if keep(r):
            acc[r["item"]].append(value(r))
    return {k: st.mean(v) for k, v in acc.items()}


def correct(r):
    return float(items_mod.LETTERS[max(range(len(r["probs"])), key=r["probs"].__getitem__)] == r["expected"])


def metrics(rows):
    """Per-unit values for each metric (units: arguments for corrupt, objection-reply pairs for objreply)."""
    c = [r for r in rows if r["task"] == "corrupt"]
    o = [r for r in rows if r["task"] == "objreply"]
    out = {
        "corrupt_keyed": per_unit(c, lambda r: r["expected"], correct),
        "corrupt_orig_corr": per_unit(c, lambda r: r["pair"] == "orig-corr", correct),
        "corrupt_para_corr": per_unit(c, lambda r: r["pair"] == "para-corr", correct),
        # probability mass on the original in the unkeyed original-vs-paraphrase pairs (0.5 = no preference)
        "corrupt_orig_pref": per_unit(c, lambda r: r["pair"] == "orig-para",
                                      lambda r: r["probs"][items_mod.LETTERS.index(r["original_letter"])]),
        "objreply_all": per_unit(o, lambda r: True, correct),
        "objreply_original": per_unit(o, lambda r: r["form"] == "original", correct),
        "objreply_paraphrase": per_unit(o, lambda r: r["form"] == "paraphrase", correct),
        "objreply_obj_to_reply": per_unit(o, lambda r: r["direction"] == "obj->reply", correct),
        "objreply_reply_to_obj": per_unit(o, lambda r: r["direction"] == "reply->obj", correct),
        "objreply_no_echo": per_unit(o, lambda r: not r["echo"], correct),
    }
    v = [r for r in rows if r["task"] == "verdict"]
    if v:
        pf = defaultdict(lambda: defaultdict(list))   # P(follows), averaged over the two letter orders
        kind_of = {}
        for r in v:
            pf[r["item"]][r["form"]].append(r["probs"][items_mod.LETTERS.index(r["follows_letter"])])
            if r["form"].startswith("corrupted"):
                kind_of[(r["item"], r["form"])] = r["corruption_type"]
        pf = {k: {f: st.mean(x) for f, x in d.items()} for k, d in pf.items()}
        # units: (argument, corrupted version); the original and paraphrase are shared across an argument's versions
        units = {(k, f): (d["original"], d["paraphrase"], d[f]) for k, d in pf.items() for f in d
                 if f.startswith("corrupted") and "original" in d and "paraphrase" in d}
        out["verdict_acc"] = per_unit(v, lambda r: True, correct)
        out["verdict_orig_gt_corr"] = {u: float(o > c) for u, (o, p, c) in units.items()}
        out["verdict_para_gt_corr"] = {u: float(p > c) for u, (o, p, c) in units.items()}
        out["verdict_gap_orig"] = {u: o - c for u, (o, p, c) in units.items()}
        out["verdict_gap_para"] = {u: p - c for u, (o, p, c) in units.items()}
        out["verdict_order_debiased"] = {u: st.mean([o > .5, p > .5, c < .5]) for u, (o, p, c) in units.items()}
        out["verdict_orig_minus_para"] = {k: d["original"] - d["paraphrase"] for k, d in pf.items()}
        for kind in sorted(set(kind_of.values())):
            out[f"verdict_disc_{kind}"] = {u: st.mean([o > c, p > c]) for u, (o, p, c) in units.items()
                                           if kind_of[u] == kind}
    for kind in sorted({r["corruption_type"] for r in c}):
        out[f"corrupt_type_{kind}"] = per_unit(c, lambda r, k=kind: r["expected"] and r["corruption_type"] == k,
                                               correct)
    return out


def paired(a, b):
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return None
    d = [a[k] - b[k] for k in keys]
    m, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
    return {"diff": round(m, 4), "se": round(se, 4), "z": round(m / se, 2) if se > 0 else None, "n": len(d)}


def analyze(out):
    rows = [json.loads(line) for line in (out / "scores.jsonl").read_text().splitlines() if line.strip()]
    current = {(it["task"], it["item"], it["variant"]) for it in pilot_items()}   # dropped items leave the analysis
    rows = [r for r in rows if (r["task"], r["item"], r["variant"]) in current]
    by_config = defaultdict(list)
    for r in rows:
        by_config[r["config"]].append(r)
    values = {c: metrics(rs) for c, rs in by_config.items()}
    summary = {"configs": {c: {m: mean_se(v.values()) for m, v in ms.items()} for c, ms in values.items()},
               "letter_mass": {c: round(st.mean(r["letter_mass"] for r in rs), 4) for c, rs in by_config.items()},
               "paired_vs_base": {}, "scale": {}, "memorisation": {}, "ceiling_floor": {}}
    main = ("verdict_orig_gt_corr", "verdict_para_gt_corr", "verdict_gap_orig", "verdict_gap_para",
            "verdict_order_debiased", "corrupt_keyed", "objreply_all", "objreply_original", "objreply_paraphrase")
    for c in values:
        if c in GRID_4B and c != "base" and "base" in values:
            summary["paired_vs_base"][c] = {m: paired(values[c][m], values["base"][m]) for m in main}
    for small, big, label in SCALE:
        if small in values and big in values:
            summary["scale"][label] = {m: paired(values[big][m], values[small][m]) for m in main}
    for c, v in values.items():   # P3: original vs paraphrase as the keyed option, and the unkeyed preference
        summary["memorisation"][c] = {
            "orig_corr_minus_para_corr": paired(v["corrupt_orig_corr"], v["corrupt_para_corr"]),
            "orig_pref_minus_half": paired(v["corrupt_orig_pref"], {k: 0.5 for k in v["corrupt_orig_pref"]}),
            "objreply_original_minus_paraphrase": paired(v["objreply_original"], v["objreply_paraphrase"]),
            "verdict_orig_minus_para": paired(v["verdict_orig_minus_para"], {k: 0 for k in v["verdict_orig_minus_para"]})
            if "verdict_orig_minus_para" in v else None}
    for task_metric in ("corrupt_keyed", "objreply_all", "verdict_order_debiased"):   # units at ceiling or floor under every configuration
        units = set.intersection(*(set(v[task_metric]) for v in values.values()))
        ceil = [u for u in units if all(v[task_metric][u] == 1.0 for v in values.values())]
        floor = [u for u in units if all(v[task_metric][u] == 0.0 for v in values.values())]
        summary["ceiling_floor"][task_metric] = {"units": len(units), "ceiling": sorted(ceil), "floor": sorted(floor),
                                                 "share": round((len(ceil) + len(floor)) / max(len(units), 1), 3)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "summary.md").write_text(markdown(summary) + "\n")
    print(markdown(summary))


def markdown(s):
    def f(x):
        return "-" if x is None else f"{100 * x['mean']:.1f} ± {100 * x['se']:.1f}" if x.get("se") else \
            f"{100 * x['mean']:.1f}"
    cols = ["verdict_orig_gt_corr", "verdict_para_gt_corr", "verdict_order_debiased", "verdict_acc",
            "corrupt_keyed", "corrupt_orig_pref", "objreply_all", "objreply_original", "objreply_paraphrase"]
    lines = ["# Pilot AL: real-text skill", "", "Accuracy in % over units (± SE); orig_pref is mass on the original "
             "in unkeyed original-vs-paraphrase pairs.", "", "| config | " + " | ".join(cols) + " | letter mass |",
             "|" + "---|" * (len(cols) + 2)]
    for c, ms in s["configs"].items():
        lines.append(f"| {c} | " + " | ".join(f(ms.get(m)) for m in cols) + f" | {s['letter_mass'][c]:.3f} |")

    def z_table(title, block):
        out = ["", f"## {title}", "", "| comparison | " + " | ".join(block and next(iter(block.values())) or []) +
               " |", "|---|" + "---|" * len(next(iter(block.values()), {}))]
        for name, ms in block.items():
            out.append(f"| {name} | " + " | ".join("-" if v is None else f"{100 * v['diff']:+.1f} (z {v['z']})"
                                                   for v in ms.values()) + " |")
        return out
    if s["scale"]:
        lines += z_table("Scale (base models, paired over units)", s["scale"])
    if s["paired_vs_base"]:
        lines += z_table("Register grid vs 4B base (paired over units)", s["paired_vs_base"])
    lines += z_table("Memorisation (P3)", s["memorisation"])
    lines += ["", "## Ceiling / floor under every configuration", ""]
    for m, cf in s["ceiling_floor"].items():
        lines.append(f"- {m}: {len(cf['ceiling'])} at ceiling, {len(cf['floor'])} at floor of {cf['units']} "
                     f"(share {cf['share']:.0%})")
    return "\n".join(lines)


def main():
    global OUT
    from scripts.philosophy_pilot import RUNS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=["data-check"], choices=["data-check", "score", "analyze"])
    ap.add_argument("--model", choices=["4B", "2B"], default="4B")
    ap.add_argument("--configs", nargs="+")
    ap.add_argument("--limit", type=int, default=0, help="first N units per task (smoke runs)")
    ap.add_argument("--out", default="real_text", help="output folder under runs/ (smoke runs use their own)")
    args = ap.parse_args()
    OUT = RUNS / args.out
    OUT.mkdir(parents=True, exist_ok=True)
    args.configs = args.configs or (GRID_4B if args.model == "4B" else GRID_2B)
    if args.stage == ["analyze"]:
        analyze(OUT)
        return
    from wi.steer import Steerable
    model = Steerable("Qwen/Qwen3.5-4B" if args.model == "4B" else "Qwen/Qwen3.5-2B")
    print(f"Device: {model.device}, layers: {model.n_layers}", flush=True)
    for stage in args.stage:
        if stage == "analyze":
            analyze(OUT)
        else:
            {"data-check": stage_data_check, "score": stage_score}[stage](model, args)


if __name__ == "__main__":
    main()
