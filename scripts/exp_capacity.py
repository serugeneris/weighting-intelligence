"""Experiment R: is the student's capacity the bottleneck? A 4x larger adapter on both Claude teachers.

Experiment Q found that a more philosophical teacher (named, task-specific ideas) gave a student that copied the
surface and lost helpfulness, with a higher final loss: the rank-8 MLP adapter (9M parameters) may be too small to
fit the denser answers. Here a rank-32 MLP adapter (36M parameters, same layers, same training recipe) is trained on
each teacher's 384 answers (plain: experiment P; deeper: experiment Q) and compared with the rank-8 students on the
48 confirmation prompts, in one pool with the baseline and the woven prompt, scored by Sonnet and Opus.

Outputs under runs/capacity/: mlp-r32_{plain,deeper}.pt, gen.jsonl. Then:
  scripts/claude_judge.py --input capacity/gen.jsonl --output capacity/judge.jsonl
  scripts/claude_judge.py --input capacity/gen.jsonl --output capacity/judge_opus.jsonl --model opus
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.exp_distill as D  # noqa: E402
from scripts.exp_selfgen import PROMPTS  # noqa: E402
from scripts.exp_teacher import load_lora  # noqa: E402
from wi.steer import Steerable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "capacity"
R32 = dict(D.VARIANTS["mlp-r8"], r=32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = Steerable()
    plain = json.loads((RUNS / "claude_teacher/targets.json").read_text()) + json.loads((RUNS / "more_teacher/targets.json").read_text())
    deeper = json.loads((RUNS / "deeper_teacher/targets.json").read_text())
    prompts = json.loads((ROOT / "data/confirm_prompts.json").read_text())
    chats = [s.chat(p) for p in prompts]
    out = OUT / "gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()

    def gen(c):
        o = []
        for i in range(0, len(c), 16):
            o += s.generate(c[i:i + 16], max_new_tokens=120)
        return o

    def write(name, outs):
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": "x", "prompts": prompts, "split": ["heldout"] * len(prompts), "outputs": outs}) + "\n")
        print(f"generated: {name}", flush=True)

    if "baseline" not in done:
        write("baseline", gen(chats))
    if "prompt woven" not in done:
        write("prompt woven", gen([s.chat(p, system=PROMPTS["woven"]) for p in prompts]))
    for tag, f in (("plain", RUNS / "more_teacher/mlp-r8_claude384.pt"), ("deeper", RUNS / "deeper_teacher/mlp-r8_deeper384.pt")):
        name = f"rank 8, {tag} teacher"
        if name not in done:
            added = load_lora(s, D.VARIANTS["mlp-r8"], f)
            with torch.no_grad():
                write(name, gen(chats))
            D.detach(added)
    D.OUT = OUT
    for tag, data in (("plain", plain), ("deeper", deeper)):
        name = f"rank 32, {tag} teacher"
        if name in done:
            continue
        added = D.train(s, data, f"mlp-r32_{tag}", R32)
        with torch.no_grad():
            write(name, gen(chats))
        D.detach(added)


if __name__ == "__main__":
    main()
