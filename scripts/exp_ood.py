"""Experiment X: does the prompt-free student hold up off the beaten path?

Every evaluation so far is a single-turn everyday request. Here the best prompt-free students and the woven prompt
face three situations they were not built for (data/ood_prompts.json):
  two-turn       16 conversations: an everyday request, the config's own answer, then a practical follow-up.
                 Does the reflective register persist (or pile up) in turn 2, and does turn 2 stay useful?
  terse system   the first 16 confirmation prompts under a competing system prompt ("as briefly and literally as
                 possible"). The students carry no prompt, so they simply get the terse one; the woven prompt has to
                 be concatenated with it. Who obeys the user's operator?
  checkable      16 fact / arithmetic / code questions with a regex-checkable answer: does philosophising cost
                 correctness, and does the student reflect where nobody wants it?
Configs: baseline, woven prompt, the Claude-taught 128-answer student x1.5 (experiment O), the 384-answer student x1.0
(experiment P). 160 new tokens. Correctness is computed here; philosophy, helpfulness and coherence by the judge.

  scripts/exp_ood.py            -> runs/ood/gen.jsonl, accuracy.json
Then:
  scripts/claude_judge.py --input ood/gen.jsonl --output ood/judge.jsonl
"""
import json
import re
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
OUT = RUNS / "ood"
CFGS = [("baseline", None, 1.0, False), ("prompt woven", None, 1.0, True),
        ("128-answer student ×1.5", RUNS / "claude_teacher/mlp-r8_claude.pt", 1.5, False),
        ("384-answer student ×1.0", RUNS / "more_teacher/mlp-r8_claude384.pt", 1.0, False)]


def render(s, msgs):
    return s.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = Steerable()
    data = json.loads((ROOT / "data/ood_prompts.json").read_text())
    confirm = json.loads((ROOT / "data/confirm_prompts.json").read_text())[:16]
    out = OUT / "gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()
    acc = json.loads((OUT / "accuracy.json").read_text()) if (OUT / "accuracy.json").exists() else {}

    def gen(c):
        o = []
        for i in range(0, len(c), 16):
            o += s.generate(c[i:i + 16], max_new_tokens=160)
        return o

    def write(name, prompts, outs):
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": "x", "prompts": prompts, "split": ["heldout"] * len(prompts), "outputs": outs},
                               ensure_ascii=False) + "\n")
        print(f"generated: {name}", flush=True)

    for name, path, gain, woven in CFGS:
        if f"{name} | checkable" in done:
            continue
        added = load_lora(s, D.VARIANTS["mlp-r8"], path) if path else None
        if added:
            for *_, lo in added:
                lo.scale *= gain
        sysm = [{"role": "system", "content": PROMPTS["woven"]}] if woven else []
        with torch.no_grad():
            t1q = [a for a, _ in data["two_turn"]]
            t1 = gen([render(s, sysm + [{"role": "user", "content": q}]) for q in t1q])
            t2 = gen([render(s, sysm + [{"role": "user", "content": a}, {"role": "assistant", "content": r.strip()},
                                         {"role": "user", "content": b}]) for (a, b), r in zip(data["two_turn"], t1)])
            write(f"{name} | turn 1", t1q, t1)
            write(f"{name} | turn 2", [f"(Follow-up. Earlier the user asked: \"{a}\") {b}" for a, b in data["two_turn"]], t2)
            terse = data["terse_system"] if not woven else PROMPTS["woven"] + "\n\n" + data["terse_system"]
            write(f"{name} | terse system prompt", confirm,
                  gen([render(s, [{"role": "system", "content": terse}, {"role": "user", "content": q}]) for q in confirm]))
            cq = [d["q"] for d in data["checkable"]]
            co = gen([render(s, sysm + [{"role": "user", "content": q}]) for q in cq])
            write(f"{name} | checkable", cq, co)
        ok = [bool(re.search(d["a"], o.lower())) for d, o in zip(data["checkable"], co)]
        acc[name] = {"correct": sum(ok), "of": len(ok), "words": round(sum(len(o.split()) for o in co) / len(co), 1),
                     "missed": [d["q"] for d, k in zip(data["checkable"], ok) if not k]}
        (OUT / "accuracy.json").write_text(json.dumps(acc, indent=1))
        if added:
            D.detach(added)


if __name__ == "__main__":
    main()
