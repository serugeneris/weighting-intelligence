"""Experiment Q: a more philosophical teacher. Same 384 questions as experiment P, deeper reflections.

Experiment P's teacher answers (Claude, instruction in exp_claude_teacher.INSTRUCT) are helpful-heavy: philosophy
~2.5 with helpfulness ~8.8, and the student inherits that balance. Here Claude re-answers the same 384 training
questions with an instruction that asks for more philosophical depth (a named idea, a question that opens up the
topic) while keeping every practical detail. A rank-8 MLP LoRA is trained on them exactly as in P and compared with
P's 384-answer student (same questions, plainer teacher) on the 48 confirmation prompts, in one pool with the
baseline and the woven prompt, scored by Sonnet and Opus. The teacher answers of both styles are judged too.

  stage targets  -> runs/deeper_teacher/targets.json
  stage train    -> runs/deeper_teacher/gen.jsonl
Then:
  scripts/claude_judge.py --input deeper_teacher/gen.jsonl --output deeper_teacher/judge.jsonl
  scripts/claude_judge.py --input deeper_teacher/gen.jsonl --output deeper_teacher/judge_opus.jsonl --model opus
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.exp_distill as D  # noqa: E402
from scripts.claude_judge import call_claude, parse  # noqa: E402
from scripts.exp_selfgen import PROMPTS  # noqa: E402
from scripts.exp_teacher import load_lora  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "deeper_teacher"
INSTRUCT = """You are writing reference answers for training a small assistant model to be a genuinely philosophical
but fully useful assistant. For each user request below, write the answer such an assistant would give:
- keep every practical detail a good plain answer would have (concrete steps, numbers, facts, cautions);
- make the philosophy substantive, not decorative: weave in two or three short reflections, at least one of which
  names a specific idea or thinker (e.g. Aristotle on habit, Heidegger on tools, Stoic attention, Zen beginner's mind,
  Ricoeur on narrative, Arendt on work) and connects it precisely to this task; you may close with one open question;
- the reflections must clarify or enrich the practical advice, never replace it;
- plain prose or a short list, at most 120 words, no headings, no preamble.
Return ONLY a JSON array: [{"id": "...", "answer": "..."}, ...]

REQUESTS:
"""


def questions():
    a = json.loads((RUNS / "claude_teacher/targets.json").read_text())
    b = json.loads((RUNS / "more_teacher/targets.json").read_text())
    return [d["q"] for d in a + b], a + b


def stage_targets():
    f = OUT / "targets.json"
    if f.exists():
        return
    qs, _ = questions()
    data = {}
    for i in range(0, len(qs), 16):
        batch = {f"q{j}": q for j, q in enumerate(qs[i:i + 16], start=i)}
        prompt = INSTRUCT + "\n".join(json.dumps({"id": k, "request": v}) for k, v in batch.items())
        for attempt in range(3):
            try:
                rows = parse(call_claude(prompt, "sonnet"))
                got = {r["id"]: r["answer"].strip() for r in rows if r.get("id") in batch}
                assert len(got) == len(batch)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  retry {attempt + 1}: {e}", flush=True)
        data.update({batch[k]: v for k, v in got.items()})
        print(f"  answers {len(data)}/{len(qs)}", flush=True)
    f.write_text(json.dumps([{"q": q, "target": data[q]} for q in qs], indent=1, ensure_ascii=False))


def stage_train():
    from wi.steer import Steerable
    s = Steerable()
    data = json.loads((OUT / "targets.json").read_text())
    _, plain_targets = questions()
    prompts = json.loads((ROOT / "data/confirm_prompts.json").read_text())
    assert not {p.lower() for p in prompts} & {d["q"].lower() for d in data}
    chats = [s.chat(p) for p in prompts]
    out = OUT / "gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()

    def gen(c):
        o = []
        for i in range(0, len(c), 16):
            o += s.generate(c[i:i + 16], max_new_tokens=120)
        return o

    def write(name, kind, outs, ps=prompts):
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": kind, "prompts": ps, "split": ["heldout"] * len(ps), "outputs": outs}) + "\n")
        print(f"generated: {name}", flush=True)

    rng = random.Random(0)
    idx = sorted(rng.sample(range(len(data)), 64))                # judge 64 of the 384 teacher answers of each style
    if "teacher answers: deeper (64 training questions)" not in done:
        write("teacher answers: deeper (64 training questions)", "target", [data[i]["target"] for i in idx], [data[i]["q"] for i in idx])
        write("teacher answers: plain, exp P (64 training questions)", "target", [plain_targets[i]["target"] for i in idx],
              [plain_targets[i]["q"] for i in idx])
    if "baseline" not in done:
        write("baseline", "baseline", gen(chats))
    if "prompt woven" not in done:
        write("prompt woven", "prompt", gen([s.chat(p, system=PROMPTS["woven"]) for p in prompts]))
    if "plain-teacher student (exp P) ×1.0" not in done:
        added = load_lora(s, D.VARIANTS["mlp-r8"], RUNS / "more_teacher/mlp-r8_claude384.pt")
        with torch.no_grad():
            write("plain-teacher student (exp P) ×1.0", "lora", gen(chats))
        D.detach(added)
    if "deeper-teacher student ×1.0" not in done:
        D.OUT = OUT
        added = D.train(s, data, "mlp-r8_deeper384", D.VARIANTS["mlp-r8"])
        with torch.no_grad():
            write("deeper-teacher student ×1.0", "lora", gen(chats))
            for *_, lo in added:
                lo.scale *= 1.2
            write("deeper-teacher student ×1.2", "lora", gen(chats))
        D.detach(added)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["targets", "train"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stage_targets() if args.stage == "targets" else stage_train()


if __name__ == "__main__":
    main()
