"""Summarize a sweep: paired per-prompt philo gains vs baseline, coherence cost, and random-control null.

wins   = prompts where philo rose by >= 1.5 points vs baseline
breaks = prompts where coherence fell by >= 2 points vs baseline (or ppl > 20)
A config is "beats null" if its mean philo gain exceeds the max gain of any random-neuron control.
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "sweep.jsonl")
recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
base = next(r for r in recs if r["kind"] == "baseline")


def summarize(r):
    dp = [a - b for a, b in zip(r["philo_each"], base["philo_each"])]
    dc = [a - b for a, b in zip(r["coh_each"], base["coh_each"])]
    return {
        "name": r["name"], "kind": r["kind"],
        "dphilo": st.mean(dp), "dcoh": st.mean(dc), "ppl": r["ppl"],
        "wins": sum(d >= 1.5 for d in dp), "breaks": sum(d <= -2 for d in dc) + (r["ppl"] > 20),
        "n": len(dp),
    }


rows = [summarize(r) for r in recs if r["kind"] != "baseline"]
MAX_BREAKS = 2  # a config with more broken prompts than this can't count as a win (or set the null)
null = max((x["dphilo"] for x in rows if x["kind"] == "random" and x["breaks"] <= MAX_BREAKS), default=0.0)
print(f"baseline philo={base['philo']:.2f} coh={base['coh']:.2f}   "
      f"random-control max Δphilo (≤{MAX_BREAKS} breaks)={null:+.2f}\n")
print(f"{'config':<34}{'kind':<10}{'Δphilo':>8}{'Δcoh':>7}{'ppl':>7}{'wins':>6}{'breaks':>8}  beats-null")
for x in sorted(rows, key=lambda x: -(x["dphilo"] - 0.5 * max(0, -x["dcoh"]))):
    print(f"{x['name']:<34}{x['kind']:<10}{x['dphilo']:>+8.2f}{x['dcoh']:>+7.2f}{x['ppl']:>7.1f}"
          f"{x['wins']:>4}/{x['n']}{x['breaks']:>6}    {'yes' if x['dphilo'] > null and x['breaks'] <= MAX_BREAKS else ''}")
