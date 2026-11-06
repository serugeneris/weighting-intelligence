"""Experiment Y: stacking levers, and how much headroom is in the student's own samples.

Two inference-only questions about the 384-answer Claude-taught student (experiment P):
  stacking   does the student plus a system prompt go past the frontier, or is it the same push twice? Configs:
             the woven prompt on the student; a light prompt ("where it fits, one brief reflective thought") alone
             and on the student.
  headroom   best-of-4: the student (and, for comparison, the woven prompt) samples 4 answers per prompt at T 0.7;
             Opus picks the one with the best philosophy + helpfulness; Sonnet scores the picked answer in a blind
             pool next to a random single sample (the control). The gap is what a perfect selection signal could
             still extract from what the student already produces, i.e. the headroom any preference training
             (experiments M, S) or reranker could hope to capture. Opus picks and Sonnet scores, to limit the
             winner's curse; the Sonnet score of picked answers is still optimistic (the two judges correlate).

  stage gen     -> runs/stack/gen.jsonl (greedy configs), samples_gen.jsonl (4 samples x 48 x 2)
  (judge)       scripts/claude_judge.py --input stack/samples_gen.jsonl --output stack/samples_judge_opus.jsonl --model opus
  stage select  -> appends "best-of-4" and "one sample" rows to runs/stack/gen.jsonl
Then:
  scripts/claude_judge.py --input stack/gen.jsonl --output stack/judge.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "stack"
LIGHT = "Answer the user's request fully and practically. Where it fits naturally, add one brief reflective thought."
STUDENT = RUNS / "more_teacher/mlp-r8_claude384.pt"


def stage_gen():
    import scripts.exp_distill as D
    from scripts.exp_selfgen import PROMPTS
    from scripts.exp_teacher import load_lora
    from wi.steer import Steerable
    s = Steerable()
    prompts = json.loads((ROOT / "data/confirm_prompts.json").read_text())
    out, sout = OUT / "gen.jsonl", OUT / "samples_gen.jsonl"
    done = {json.loads(l)["name"] for f in (out, sout) if f.exists() for l in f.read_text().splitlines()}

    def gen(c, **kw):
        o = []
        for i in range(0, len(c), 16):
            o += s.generate(c[i:i + 16], max_new_tokens=120, **({"seed": kw["seed"] + i} if "seed" in kw else {}),
                            **{k: v for k, v in kw.items() if k != "seed"})
        return o

    def write(f, name, outs):
        with f.open("a") as fh:
            fh.write(json.dumps({"name": name, "kind": "x", "prompts": prompts, "split": ["heldout"] * len(prompts), "outputs": outs}) + "\n")
        print(f"generated: {name}", flush=True)

    plain = [s.chat(p) for p in prompts]
    woven = [s.chat(p, system=PROMPTS["woven"]) for p in prompts]
    light = [s.chat(p, system=LIGHT) for p in prompts]
    with torch.no_grad():
        for name, c in (("baseline", plain), ("prompt woven", woven), ("light prompt", light)):
            if name not in done:
                write(out, name, gen(c))
        for k in range(4):
            if f"woven sample {k}" not in done:
                write(sout, f"woven sample {k}", gen(woven, temperature=0.7, seed=1000 * k))
        added = load_lora(s, D.VARIANTS["mlp-r8"], STUDENT)
        for name, c in (("student", plain), ("student + woven prompt", woven), ("student + light prompt", light)):
            if name not in done:
                write(out, name, gen(c))
        for k in range(4):
            if f"student sample {k}" not in done:
                write(sout, f"student sample {k}", gen(plain, temperature=0.7, seed=1000 * k))
        D.detach(added)


def stage_select():
    rows = [json.loads(l) for l in (OUT / "samples_judge_opus.jsonl").read_text().splitlines()]
    prompts = json.loads((ROOT / "data/confirm_prompts.json").read_text())
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        src = r["cfg"].rsplit(" sample", 1)[0]
        by[src][r["i"]].append(r)
    out = OUT / "gen.jsonl"
    stats = {}
    for src, items in by.items():
        best, one = [], []
        for i in range(len(prompts)):
            v = sorted(items[i], key=lambda r: r["cfg"])
            ok = [r for r in v if not r["degenerate"]] or v
            best.append(max(ok, key=lambda r: r["philo"] + r["helpful"])["output"])
            one.append(v[0]["output"])                                     # sample 0: an unselected draw
        allr = [r for v in items.values() for r in v]
        pick = [max([r for r in items[i] if not r["degenerate"]] or items[i], key=lambda r: r["philo"] + r["helpful"]) for i in range(len(prompts))]
        stats[src] = {"opus mean sum, all samples": round(sum(r["philo"] + r["helpful"] for r in allr) / len(allr), 2),
                      "opus mean sum, picked": round(sum(r["philo"] + r["helpful"] for r in pick) / len(pick), 2)}
        with out.open("a") as f:
            for name, outs in ((f"{src} best-of-4 (Opus-picked)", best), (f"{src} one sample T0.7", one)):
                f.write(json.dumps({"name": name, "kind": "x", "prompts": prompts, "split": ["heldout"] * len(prompts), "outputs": outs}) + "\n")
    (OUT / "select_stats.json").write_text(json.dumps(stats, indent=1))
    print(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "select"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stage_gen() if args.stage == "gen" else stage_select()


if __name__ == "__main__":
    main()
