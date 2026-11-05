"""Helpfulness ceiling: build a judge input with the natural contrast-pair answers (the steering target)
pooled with four steered configs from the dose sweep as anchors.

Writes runs/natural_ceiling.jsonl; score with:
  scripts/claude_judge.py --input natural_ceiling.jsonl --output claude_judge_natural.jsonl
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
ANCHORS = {"baseline", "diffmeans L8@0.5", "omp* L8 k=256 a=0.35", "omp* L8 k=256 a=0.5"}


def main():
    pairs = json.loads((ROOT / "data/contrast_pairs.json").read_text())
    recs = [{"name": f"natural {k}", "kind": "natural", "prompts": [p["q"] for p in pairs],
             "split": ["pairs"] * len(pairs), "outputs": [p[k] for p in pairs]} for k in ("plain", "philo")]
    for line in (RUNS / "dose_sweep.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["name"] in ANCHORS:
            recs.append({**r, "name": "anchor " + r["name"], "kind": "anchor"})
    with (RUNS / "natural_ceiling.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print([(r["name"], len(r["outputs"])) for r in recs])


if __name__ == "__main__":
    main()
