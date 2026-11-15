"""Experiment AI, Stage 4: analysis of runs/philbench/ -> summary.json and summary.md.

Per configuration: every metric with its standard error over items; paired differences against the same model's
base on the same items (bootstrap over items); across the 4B grid, Spearman correlations between the judged register
(philosophy score, mean of Sonnet and Opus) and each skill metric; dose response inside each lever family; the
register pushes against the random pushes of the same size; and, on base, whether the last-prompt-token projection
onto diff-means or B predicts item-level stability.

PhilPapers choices are position-averaged: each question's option probabilities are averaged by label over the
baseline and reversed orders before taking a choice for the constraint and plurality metrics (the stability metrics
compare raw single-order choices).
"""
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LETTERS = "ABCDEFGHIJKL"
GRID = ["base", "woven", "reflect-first", "diffmeans-0.35", "diffmeans-0.5", "B-0.4", "B-0.6", "B-0.8",
        "random0-0.6", "random1-0.6", "random2-0.6", "rank1-x1", "rank1-x2", "claude384-x1.0", "claude384-x1.5",
        "claude128-x1.5"]
RANDOM = ["random0-0.6", "random1-0.6", "random2-0.6"]
FAMILIES = {"learned direction B": ["base", "B-0.4", "B-0.6", "B-0.8"],
            "diff-means": ["base", "diffmeans-0.35", "diffmeans-0.5"],
            "rank-1 student": ["base", "rank1-x1", "rank1-x2"],
            "Claude-taught 384 student": ["base", "claude384-x1.0", "claude384-x1.5"]}
SCALE = {"2B": ["2B:base", "2B:woven", "2B:student"], "4B-4bit": ["4B-4bit:base", "4B-4bit:woven"],
         "9B-4bit": ["9B-4bit:base", "9B-4bit:woven"]}
# metric -> (label, higher is better for skill?, is it a skill measure?)
METRICS = {
    "pp_paraphrase_stable": ("PhilPapers: same choice under paraphrase", True, True),
    "pp_order_stable": ("PhilPapers: same choice under reversed order", True, True),
    "pp_violation": ("PhilPapers: constraint violated (all 37)", False, True),
    "pp_violation_firm": ("PhilPapers: constraint violated (28 firm)", False, True),
    "pp_violation_soft": ("PhilPapers: soft violation p(a)p(b), firm", False, True),
    "pressure_flip": ("Pressure: flips after bare disagreement", False, True),
    "pressure_to_user": ("Pressure: p(the user's option) after", False, True),
    "pressure_p_drop": ("Pressure: drop in p(own answer)", False, False),
    "critique_correct": ("Pilot Z objections: correct", True, True),
    "validity_correct": ("Pilot AB validity: correct", True, True),
    "pp_meta_mass": ("PhilPapers: mass on meta options", None, False),
    "pp_pick_meta": ("PhilPapers: position-averaged pick is meta", None, False),
    "pp_plurality": ("PhilPapers: agrees with philosopher plurality", None, False),
    "pp_entropy": ("PhilPapers: answer entropy (nats)", None, False),
    "pp_first_letter": ("PhilPapers: p(first option letter)", None, False),
    "pb_cd": ("philosophy-bench (adapted): expected cd_score", None, False),
    "pb_botched_mass": ("philosophy-bench (adapted): mass on botched", None, False),
    "gen_compliant": ("Generated: ANSWER line present and valid", None, False),
    "gen_match": ("Generated: stated answer = forced choice", True, True),
    "gen_match_eos": ("Generated, ended naturally: stated = forced choice", True, True),
    "gen_eos": ("Generated: ended before the 160-token cap", None, False),
    "gen_words": ("Generated: words written", None, False),
    "letter_mass": ("Prefilled: next-token mass on option letters", None, False),
    "answers_at_once": ("No prefill: next-token mass on option letters", None, False),
}


def read(out, name):
    path = out / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def entropy(ps):
    return -sum(p * math.log(p) for p in ps if p > 0)


def argmax(ps):
    return max(range(len(ps)), key=ps.__getitem__)


def per_item(out, min_mass=0.0):
    """config -> metric -> {item key: value}. min_mass > 0 drops forced-choice rows whose prefilled next-token mass
    on the option letters is below it (robustness check: the letter distribution of such rows is a small residue)."""
    pp = json.loads((ROOT / "data/philpapers2020.json").read_text())
    plural = {q["topic"]: max(q["options"], key=lambda o: o["accept_exclusive"])["label"]
              for q in pp["questions"] if q["kind"] == "single_choice"}
    constraints = json.loads((ROOT / "data/philpapers_constraints.json").read_text())["constraints"]
    values = defaultdict(lambda: defaultdict(dict))
    by = defaultdict(lambda: defaultdict(dict))
    for r in read(out, "scores.jsonl"):
        values[r["config"]]["letter_mass"][f"{r['task']}|{r['item']}|{r['variant']}"] = r["letter_mass"]
        if r["letter_mass"] >= min_mass:
            by[r["config"]][r["task"]][(r["item"], r["variant"])] = r
    for r in read(out, "scores_noprefill.jsonl"):
        values[r["config"]]["answers_at_once"][f"{r['task']}|{r['item']}|{r['variant']}"] = r["letter_mass"]
    avgdist = {}
    for config, tasks in by.items():
        v = values[config]
        rows = tasks.get("philpapers", {})
        topics = sorted({k[0] for k in rows})
        avgdist[config] = {}
        for t in topics:
            variants = {var: rows[(t, var)] for var in ("baseline", "paraphrase", "reversed") if (t, var) in rows}
            if len(variants) < 3:
                continue
            label = {var: r["labels"][argmax(r["probs"])] for var, r in variants.items()}
            v["pp_paraphrase_stable"][t] = float(label["baseline"] == label["paraphrase"])
            v["pp_order_stable"][t] = float(label["baseline"] == label["reversed"])
            v["pp_meta_mass"][t] = st.mean(sum(p for p, k in zip(r["probs"], r["kinds"]) if k == "meta")
                                           for r in variants.values())
            v["pp_entropy"][t] = st.mean(entropy(r["probs"]) for r in variants.values())
            v["pp_first_letter"][t] = st.mean(r["probs"][0] for r in variants.values())
            dist = defaultdict(float)
            for var in ("baseline", "reversed"):
                r = variants[var]
                for lab, p in zip(r["labels"], r["probs"]):
                    dist[lab] += p / 2
            kinds = dict(zip(variants["baseline"]["labels"], variants["baseline"]["kinds"]))
            avgdist[config][t] = (dict(dist), kinds)
            pick = max(dist, key=dist.get)
            v["pp_pick_meta"][t] = float(kinds[pick] == "meta")
            subst = {lab: p for lab, p in dist.items() if kinds[lab] == "substantive"}
            v["pp_plurality"][t] = float(max(subst, key=subst.get) == plural[t])
        for c in constraints:
            ta, tb = c["a"]["topic"], c["b"]["topic"]
            if ta not in avgdist[config] or tb not in avgdist[config]:
                continue
            (da, ka), (db, kb) = avgdist[config][ta], avgdist[config][tb]
            hard = float(max(da, key=da.get) == c["a"]["option"] and max(db, key=db.get) == c["b"]["option"])
            sa = da[c["a"]["option"]] / sum(p for lab, p in da.items() if ka[lab] == "substantive")
            sb = db[c["b"]["option"]] / sum(p for lab, p in db.items() if kb[lab] == "substantive")
            v["pp_violation"][c["id"]] = hard
            if not c["contestable"]:
                v["pp_violation_firm"][c["id"]] = hard
                v["pp_violation_soft"][c["id"]] = sa * sb
        for (item, _), r in tasks.get("philbench", {}).items():
            keep = [(p, cd) for p, cd, b in zip(r["probs"], r["cd"], r["botched"]) if not b]
            v["pb_cd"][item] = sum(p * cd for p, cd in keep) / sum(p for p, _ in keep)
            v["pb_botched_mass"][item] = sum(p for p, b in zip(r["probs"], r["botched"]) if b)
        for task in ("critique", "validity"):
            for (item, variant), r in tasks.get(task, {}).items():
                v[f"{task}_correct"][f"{item}|{variant}"] = float(r["labels"][argmax(r["probs"])] == r["expected"])
    for r in read(out, "pressure.jsonl"):
        if r["letter_mass"] < min_mass:
            continue
        v = values[r["config"]]
        v["pressure_flip"][r["item"]] = float(r["flip"])
        v["pressure_p_drop"][r["item"]] = r["p_before"] - r["after"][r["chosen"]]   # depends on starting confidence
        v["pressure_to_user"][r["item"]] = r["after"][r["target"]]
    for r in read(out, "generate.jsonl"):
        v = values[r["config"]]
        # compare with the prefilled forced choice on the baseline variant (the stored "match" used the first scoring)
        fc = by[r["config"]].get("philpapers", {}).get((r["item"], "baseline"))
        forced = LETTERS[argmax(fc["probs"])] if fc else None
        match = float(r["stated"] is not None and r["stated"] == forced)
        v["gen_compliant"][r["item"]] = float(r["compliant"])
        v["gen_eos"][r["item"]] = float(r["stop"] == "eos")
        v["gen_words"][r["item"]] = float(r["words"])
        if forced:
            v["gen_match"][r["item"]] = match
            if r["stop"] == "eos":
                v["gen_match_eos"][r["item"]] = match
    return values, avgdist


def mean_se(xs):
    xs = list(xs)
    if not xs:
        return None
    m = st.mean(xs)
    return {"mean": round(m, 4), "se": round(st.stdev(xs) / math.sqrt(len(xs)), 4) if len(xs) > 1 else None,
            "n": len(xs)}


def paired(a, b, reps=2000, seed=0):
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return None
    d = [a[k] - b[k] for k in keys]
    rng = random.Random(seed)
    boots = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(reps))
    m, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
    return {"diff": round(m, 4), "se": round(se, 4), "ci95": [round(boots[int(0.025 * reps)], 4),
            round(boots[int(0.975 * reps)], 4)], "z": round(m / se, 2) if se > 0 else None, "n": len(d)}


def ranks(xs):
    order = sorted(range(len(xs)), key=xs.__getitem__)
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def spearman_boot(x, y, reps=2000, seed=0):
    rng = random.Random(seed)
    n, vals = len(x), []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        xs, ys = [x[i] for i in idx], [y[i] for i in idx]
        if len(set(xs)) > 1 and len(set(ys)) > 1:
            vals.append(spearman(xs, ys))
    vals.sort()
    if len(set(y)) < 2 or not vals:
        return {"rho": None, "ci95": [None, None], "n": n, "note": "metric constant across configurations"}
    return {"rho": round(spearman(x, y), 3), "ci95": [round(vals[int(0.025 * len(vals))], 3),
            round(vals[int(0.975 * len(vals))], 3)], "n": n}


def register_scores(out):
    scores = defaultdict(lambda: defaultdict(list))
    for judge, name in (("sonnet", "register_judge.jsonl"), ("opus", "register_judge_opus.jsonl")):
        for r in read(out, name):
            scores[r["cfg"]][f"philo_{judge}"].append(r["philo"])
            scores[r["cfg"]][f"helpful_{judge}"].append(r["helpful"])
            scores[r["cfg"]][f"coherence_{judge}"].append(r["coherence"])
            scores[r["cfg"]][f"degenerate_{judge}"].append(float(bool(r["degenerate"])))
    result = {}
    for cfg, s in scores.items():
        result[cfg] = {k: round(st.mean(v), 3) for k, v in s.items()}
        for m in ("philo", "helpful", "coherence"):
            both = [result[cfg][f"{m}_{j}"] for j in ("sonnet", "opus") if f"{m}_{j}" in result[cfg]]
            result[cfg][m] = round(st.mean(both), 3) if both else None
    return result


def analyze(out):
    out = Path(out)
    values, avgdist = per_item(out)
    configs = [c for c in GRID + [c for g in SCALE.values() for c in g] if c in values]
    summary = {"configs": {}, "paired_vs_base": {}, "notes": []}
    for c in configs:
        summary["configs"][c] = {m: mean_se(values[c][m].values()) for m in METRICS if values[c].get(m)}
    base_of = {c: (c.split(":")[0] + ":base" if ":" in c else "base") for c in configs}
    for c in configs:
        b = base_of[c]
        if c == b or b not in values:
            continue
        summary["paired_vs_base"][c] = {m: paired(values[c][m], values[b][m]) for m in METRICS
                                        if values[c].get(m) and values[b].get(m)}
    reg = register_scores(out)
    summary["register"] = reg
    robust, _ = per_item(out, min_mass=0.5)
    summary["robust_min_mass_0.5"] = {
        c: {m: mean_se(robust[c][m].values()) for m in METRICS if METRICS[m][2] and robust[c].get(m)}
        for c in configs if c in robust}
    summary["robust_paired_vs_base"] = {
        c: {m: paired(robust[c][m], robust[base_of[c]][m]) for m in METRICS
            if METRICS[m][2] and robust[c].get(m) and robust[base_of[c]].get(m)}
        for c in configs if c in robust and c != base_of[c] and base_of[c] in robust}

    # across configurations (4B grid): register philosophy vs each metric
    grid = [c for c in GRID if c in values and c in reg and reg[c].get("philo") is not None]
    summary["spearman_register"] = {}
    if len(grid) >= 5:
        x = [reg[c]["philo"] for c in grid]
        for m in METRICS:
            if all(summary["configs"][c].get(m) for c in grid):
                y = [summary["configs"][c][m]["mean"] for c in grid]
                summary["spearman_register"][m] = spearman_boot(x, y)
        summary["spearman_register"]["_configs"] = grid

    # dose response inside each family
    summary["dose"] = {}
    for fam, members in FAMILIES.items():
        members = [c for c in members if c in values]
        fam_out = {"members": members, "register_philo": [reg.get(c, {}).get("philo") for c in members]}
        for m in METRICS:
            seq = [summary["configs"][c].get(m, {}) for c in members]
            if all(seq):
                means = [s["mean"] for s in seq]
                inc = all(b >= a for a, b in zip(means, means[1:]))
                dec = all(b <= a for a, b in zip(means, means[1:]))
                fam_out[m] = {"means": means, "monotonic": "up" if inc else "down" if dec else "no"}
        summary["dose"][fam] = fam_out

    # register pushes against the random pushes of the same size (and the random seeds against each other)
    rand = [c for c in RANDOM if c in values]
    summary["random_control"] = {}
    if rand:
        for m in METRICS:
            if not all(values[c].get(m) for c in rand):
                continue
            seeds = [summary["configs"][c][m]["mean"] for c in rand]
            pooled = defaultdict(list)
            for c in rand:
                for k, val in values[c][m].items():
                    pooled[k].append(val)
            pooled = {k: st.mean(v) for k, v in pooled.items()}
            row = {"random_seeds": seeds, "random_spread": round(max(seeds) - min(seeds), 4)}
            for c in ("B-0.6", "B-0.8", "diffmeans-0.5", "B-0.4", "diffmeans-0.35"):
                if values[c].get(m):
                    row[f"{c} minus random"] = paired(values[c][m], pooled)
            row["random minus base"] = paired(pooled, values["base"][m]) if values["base"].get(m) else None
            summary["random_control"][m] = row

    # projection check on base
    proj = {}
    for r in read(out, "scores.jsonl"):
        if r["config"] == "base" and r.get("proj"):
            proj[r["item"]] = r["proj"]
    summary["projection"] = {}
    if proj:
        for target in ("pp_paraphrase_stable", "pp_order_stable", "pp_meta_mass"):
            vals = values["base"][target]
            keys = sorted(set(vals) & set(proj))
            for p in next(iter(proj.values())):
                summary["projection"][f"{p} vs {target}"] = spearman_boot([proj[k][p] for k in keys],
                                                                         [vals[k] for k in keys])
        # do topics involved in a base violation project differently?
        cons = json.loads((ROOT / "data/philpapers_constraints.json").read_text())["constraints"]
        bad = {c[s]["topic"] for c in cons for s in ("a", "b") if values["base"]["pp_violation"].get(c["id"]) == 1}
        summary["projection"]["topics_in_base_violations"] = sorted(bad)

    # sanity checks
    sanity = {}
    if values.get("base", {}).get("pp_order_stable"):
        sanity["base_reversed_same_as_baseline"] = summary["configs"]["base"]["pp_order_stable"]
        sanity["base_first_letter_mass"] = summary["configs"]["base"]["pp_first_letter"]
    for task in ("critique", "validity"):
        for c in ("base",) + tuple(configs):
            v = values[c].get(f"{task}_correct")
            if not v:
                continue
            orders = defaultdict(list)
            for k, val in v.items():
                orders[k.rsplit("|", 1)[1]].append(val)
            sanity.setdefault(f"{task}_accuracy_by_order", {})[c] = {o: round(st.mean(x), 3) for o, x in orders.items()}
    summary["sanity"] = sanity

    # scale (Stage 5): base models only, paired over items; 9B only against the 4-bit 4B (same stack, §33)
    summary["scale"] = {}
    for bigger, smaller in (("base", "2B:base"), ("9B-4bit:base", "4B-4bit:base"), ("9B-4bit:woven", "4B-4bit:woven")):
        if bigger in values and smaller in values:
            summary["scale"][f"{bigger} minus {smaller}"] = {
                m: paired(values[bigger][m], values[smaller][m]) for m in METRICS
                if values[bigger].get(m) and values[smaller].get(m)}
    # the same 4B in two stacks: how often do bf16 PyTorch and 4-bit MLX make the same forced choice?
    same = defaultdict(list)
    rows = defaultdict(dict)
    for r in read(out, "scores.jsonl"):
        if r["config"] in ("base", "4B-4bit:base"):
            rows[(r["task"], r["item"], r["variant"])][r["config"]] = argmax(r["probs"])
    for (task, *_), pair in rows.items():
        if len(pair) == 2:
            same[task].append(float(pair["base"] == pair["4B-4bit:base"]))
    summary["stack_agreement_4B_bf16_vs_4bit"] = {t: mean_se(v) for t, v in same.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "summary.md").write_text(markdown(summary, configs))
    print((out / "summary.md").read_text())


def fmt(s, pct=False):
    if not s:
        return "—"
    m, se = s["mean"], s["se"] or 0
    return f"{100 * m:.0f}% ± {100 * se:.0f}" if pct else f"{m:.2f} ± {se:.2f}"


def markdown(summary, configs):
    pct = {m for m in METRICS if m not in ("pp_entropy", "pb_cd", "gen_words", "pressure_p_drop", "pressure_to_user", "pp_violation_soft",
                                           "letter_mass", "answers_at_once")}
    lines = ["# Experiment AI summary", ""]
    groups = [("Skill", [m for m in METRICS if METRICS[m][2]]), ("Descriptive", [m for m in METRICS if not METRICS[m][2]])]
    for title, ms in groups:
        lines += [f"## {title} metrics", "", "| Config | register philo/help | " + " | ".join(METRICS[m][0] for m in ms) + " |",
                  "|---|---|" + "---:|" * len(ms)]
        for c in configs:
            r = summary["register"].get(c, {})
            reg = f"{r['philo']:.2f} / {r['helpful']:.2f}" if r.get("philo") is not None else "—"
            lines.append(f"| {c} | {reg} | " + " | ".join(fmt(summary["configs"][c].get(m), m in pct) for m in ms) + " |")
        lines.append("")
    lines += ["## Robustness: skill metrics on rows with prefilled letter mass >= 0.5", "",
              "| Config | " + " | ".join(METRICS[m][0] for m in groups[0][1]) + " |", "|---|" + "---:|" * len(groups[0][1])]
    for c in configs:
        rs = summary["robust_min_mass_0.5"].get(c, {})
        lines.append(f"| {c} | " + " | ".join(fmt(rs.get(m), m in pct) + (f" (n {rs[m]['n']})" if rs.get(m) else "")
                                              for m in groups[0][1]) + " |")
    lines += ["", "## Paired difference vs base (z = diff / paired SE)", ""]
    for c, ms in summary["paired_vs_base"].items():
        cells = [f"{m}: {v['diff']:+.3f} (z {v['z']})" for m, v in ms.items() if v and v["z"] is not None and abs(v["z"]) >= 2]
        lines.append(f"- **{c}**: " + ("; ".join(cells) if cells else "no metric with |z| >= 2"))
    lines += ["", "## Spearman: register philosophy vs metric, across the 4B grid", ""]
    for m, v in summary["spearman_register"].items():
        if m != "_configs":
            lines.append(f"- {METRICS[m][0]}: " + (f"rho {v['rho']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]" if v["rho"] is not None else "constant"))
    lines += ["", "## Dose response", ""]
    for fam, d in summary["dose"].items():
        lines.append(f"- **{fam}** ({' → '.join(d['members'])}), register philo {d['register_philo']}")
        for m in METRICS:
            if m in d:
                lines.append(f"  - {METRICS[m][0]}: {[round(x, 3) for x in d[m]['means']]} ({d[m]['monotonic']})")
    lines += ["", "## Random control", ""]
    for m, row in summary["random_control"].items():
        cells = [f"seeds {row['random_seeds']}"]
        for k, v in row.items():
            if isinstance(v, dict) and v:
                cells.append(f"{k} {v['diff']:+.3f} (z {v['z']})")
        lines.append(f"- {METRICS[m][0]}: " + "; ".join(cells))
    lines += ["", "## Projection check (base)", ""]
    for k, v in summary["projection"].items():
        if isinstance(v, dict):
            lines.append(f"- {k}: rho {v['rho']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")
    lines += ["", "## Scale (base models, paired over items)", ""]
    for k, ms in summary.get("scale", {}).items():
        cells = [f"{m}: {v['diff']:+.3f} (z {v['z']})" for m, v in ms.items() if v and v["z"] is not None]
        lines.append(f"- **{k}**: " + "; ".join(cells))
    lines += ["", f"4B bf16 vs 4-bit same forced choice: {json.dumps(summary.get('stack_agreement_4B_bf16_vs_4bit'))}"]
    lines += ["", "## Sanity", "", "```", json.dumps(summary["sanity"], indent=1), "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    analyze(ROOT / "runs/philbench")
