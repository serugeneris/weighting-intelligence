"""Experiment J: distil a better teacher than the woven prompt.

Experiment D showed a LoRA copies the prompted model's balance without the prompt, but no better: the student's
ceiling is the teacher. Two ways to raise the teacher, both using only training questions (never evaluation prompts):

  stage prompts   six system prompts (the two best from experiment B and four new ones) on 16 training questions,
                  judged blind; the prompt with the highest mean philo + helpful (non-degenerate) is the new teacher.
  stage samples   with that prompt, 4 sampled answers (temperature 0.8, top-p 0.9, 160 tokens) to each of the 128
                  training questions, judged blind; for each question keep the answer with the highest
                  philo + helpful among non-degenerate ones with coherence >= 7 (best-of-4, "rejection sampling").
  stage train     LoRAs trained exactly as in experiment D on (a) the new prompt's greedy answers and (b) the
                  best-of-4 answers; evaluated with no prompt on the 16 evaluation prompts, in one blind pool with
                  the baseline, both prompts and experiment D's rank-1 adapter (reloaded from runs/distill/).

Run each stage, then judge its output before the next:
  scripts/exp_teacher.py --stage prompts  && scripts/claude_judge.py --input teacher/prompts_gen.jsonl --output teacher/prompts_judge.jsonl
  scripts/exp_teacher.py --stage samples  && scripts/claude_judge.py --input teacher/samples_gen.jsonl --output teacher/samples_judge.jsonl
  scripts/exp_teacher.py --stage train    && scripts/claude_judge.py --input teacher/gen.jsonl --output teacher/judge.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.exp_distill as D  # noqa: E402
from scripts.exp_selfgen import PROMPTS  # noqa: E402
from wi.steer import Steerable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "teacher"
CANDIDATES = {
    "woven": PROMPTS["woven"],
    "reflect-first": PROMPTS["reflect-first"],
    "practical-first": "Answer the user's request fully, with every practical detail they need. Where it fits naturally, "
                       "add one or two short sentences of genuine philosophical insight (about meaning, time, knowledge, "
                       "ethics, mind or value) tied to the topic. Never let the reflection crowd out useful detail.",
    "frame": "Begin with one sentence that frames the request through a genuine philosophical idea. Then give a "
             "complete, concrete, practical answer, and let that idea reappear briefly where it fits.",
    "practical-philosopher": "You are a practical philosopher: you give excellent, concrete advice, and you see the deeper "
                             "questions in everyday tasks. Answer helpfully and specifically, and let your philosophical "
                             "perspective show in a few precise remarks, never at the expense of useful detail.",
    "named-idea": "Answer the user's request fully and helpfully. Along the way, connect it to one specific philosophical "
                  "idea (for example from Stoicism, phenomenology, ethics or epistemology), explained in plain words.",
}
N_SAMPLES = 4


def score(r):
    return r["philo"] + r["helpful"]


def chats_for(s, qs, system):
    return [s.chat(q, system=system) for q in qs]


def stage_prompts(s, train_qs):
    out = OUT / "prompts_gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()
    dev = train_qs[:16]
    for name, sysmsg in CANDIDATES.items():
        if f"prompt {name}" in done:
            continue
        outs = s.generate(chats_for(s, dev, sysmsg), max_new_tokens=120)
        with out.open("a") as f:
            f.write(json.dumps({"name": f"prompt {name}", "kind": "prompt", "prompts": dev, "split": ["eval"] * 16,
                                "outputs": outs}) + "\n")
        print(f"prompt {name} done", flush=True)


def best_prompt():
    rows = [json.loads(l) for l in (OUT / "prompts_judge.jsonl").read_text().splitlines()]
    by = defaultdict(list)
    for r in rows:
        by[r["cfg"]].append(r)
    table = {c: (sum(score(r) for r in v if not r["degenerate"]) / len(v), sum(r["philo"] for r in v) / len(v),
                 sum(r["helpful"] for r in v) / len(v)) for c, v in by.items()}
    for c, (sc, p, h) in sorted(table.items(), key=lambda kv: -kv[1][0]):
        print(f"  {c:32s} philo+help {sc:.2f}  philo {p:.2f}  help {h:.2f}")
    name = max(table, key=lambda c: table[c][0]).removeprefix("prompt ")
    return name, CANDIDATES[name]


def cut(o):
    o = o.strip()
    c = max(o.rfind(". "), o.rfind(".\n"), o.rfind("!"), o.rfind("?"))
    return o[:c + 1] if c > len(o) // 2 else o


def stage_samples(s, qs):
    name, sysmsg = best_prompt()
    print(f"teacher prompt: {name}", flush=True)
    (OUT / "teacher_prompt.json").write_text(json.dumps({"name": name, "system": sysmsg}, indent=1))
    out = OUT / "samples_gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()
    for k in range(N_SAMPLES):
        if f"sample {k}" in done:
            continue
        outs = []
        for i in range(0, len(qs), 32):
            outs += s.generate(chats_for(s, qs[i:i + 32], sysmsg), max_new_tokens=160, temperature=0.8, seed=100 * k + i)
        with out.open("a") as f:
            f.write(json.dumps({"name": f"sample {k}", "kind": "sample", "prompts": qs, "split": ["eval"] * len(qs),
                                "outputs": [cut(o) for o in outs]}) + "\n")
        print(f"sample {k} done", flush=True)


def load_lora(s, cfg, path):
    added = D.attach(s, cfg)
    w = torch.load(path)
    for L, _, mod, lo in added:
        e = w[f"layers.{L}.{'mlp.' + mod if '.' not in mod else mod}"]
        lo.A.data.copy_(e["A"].to(lo.A.device))
        lo.B.data.copy_(e["B"].to(lo.B.device))
    return added


def stage_train(s, qs):
    tp = json.loads((OUT / "teacher_prompt.json").read_text())
    # (a) greedy answers with the new prompt
    fa = OUT / "targets_prompt.json"
    if not fa.exists():
        outs = []
        for i in range(0, len(qs), 32):
            outs += s.generate(chats_for(s, qs[i:i + 32], tp["system"]), max_new_tokens=160)
        fa.write_text(json.dumps([{"q": q, "target": cut(o)} for q, o in zip(qs, outs)], indent=1, ensure_ascii=False))
    # (b) best-of-4 by the judge
    fb = OUT / "targets_best_of_4.json"
    if not fb.exists():
        rows = [json.loads(l) for l in (OUT / "samples_judge.jsonl").read_text().splitlines()]
        by = defaultdict(list)
        for r in rows:
            if not r["degenerate"] and r["coherence"] >= 7:
                by[r["prompt"]].append(r)
        data = [{"q": q, "target": max(by[q], key=score)["output"], "score": score(max(by[q], key=score))}
                for q in qs if by[q]]
        fb.write_text(json.dumps(data, indent=1, ensure_ascii=False))
        print(f"best-of-{N_SAMPLES}: {len(data)}/{len(qs)} questions kept; mean philo+help of kept "
              f"{sum(d['score'] for d in data) / len(data):.2f}", flush=True)

    prompts = json.loads((ROOT / "data/eval_prompts.json").read_text())
    heldout = json.loads((ROOT / "data/heldout_prompts.json").read_text())
    all_prompts = prompts + heldout
    assert not set(all_prompts) & set(qs), "evaluation prompt in training data"
    split = ["eval"] * len(prompts) + ["heldout"] * len(heldout)
    chats = [s.chat(p) for p in all_prompts]
    out = OUT / "gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()

    def write(name, kind, outs):
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": kind, "prompts": all_prompts, "split": split, "outputs": outs}) + "\n")
        print(f"generated: {name}", flush=True)

    if "baseline" not in done:
        write("baseline", "baseline", s.generate(chats, max_new_tokens=120))
    if "prompt woven (no LoRA)" not in done:
        write("prompt woven (no LoRA)", "prompt", s.generate(chats_for(s, all_prompts, PROMPTS["woven"]), max_new_tokens=120))
    if f"prompt {tp['name']} (no LoRA)" not in done and tp["name"] != "woven":
        write(f"prompt {tp['name']} (no LoRA)", "prompt", s.generate(chats_for(s, all_prompts, tp["system"]), max_new_tokens=120))
    if "LoRA down8-r1 from woven (exp D), no prompt" not in done:
        added = load_lora(s, D.VARIANTS["down8-r1"], RUNS / "distill/down8-r1.pt")
        write("LoRA down8-r1 from woven (exp D), no prompt", "lora", s.generate(chats, max_new_tokens=120))
        D.detach(added)
    D.OUT = OUT                                           # train() saves its weights under runs/teacher/
    runs = [("prompt", fa, "down8-r1"), ("best-of-4", fb, "down8-r1"), ("best-of-4", fb, "mlp-r8")]
    for tag, f, v in runs:
        name = f"LoRA {v} from {tag}, no prompt"
        if name in done:
            continue
        data = json.loads(f.read_text())
        added = D.train(s, data, f"{v}_{tag}", D.VARIANTS[v])
        with torch.no_grad():
            write(name, "lora", s.generate(chats, max_new_tokens=120))
        D.detach(added)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prompts", "samples", "train"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    s = Steerable()
    qs = json.loads((ROOT / "data/train_questions.json").read_text())
    qs += [p["q"] for p in json.loads((ROOT / "data/contrast_pairs.json").read_text())]
    {"prompts": lambda: stage_prompts(s, qs), "samples": lambda: stage_samples(s, qs),
     "train": lambda: stage_train(s, qs)}[args.stage]()


if __name__ == "__main__":
    main()
