"""Experiment O: a stronger teacher. Distil answers written by Claude instead of Qwen's own prompted answers.

Experiments D and J found that a student copies its teacher's balance, and Qwen's prompted answers sit on a
frontier of philosophy + helpfulness ~ 9.2. A stronger model can write answers that are both fully helpful and
genuinely reflective. Here Claude (Sonnet, via the local `claude -p` CLI) writes one answer per training question
(the same 128 questions as experiments D/J; no evaluation or confirmation prompt), short enough to fit the 160-token
training window. A LoRA is trained on them exactly as in experiment D (no system prompt).

  stage targets   Claude writes the answers -> runs/claude_teacher/targets.json
  stage train     rank-8 MLP and rank-1 L8 students; generations on the 48 confirmation prompts, in one pool with
                  the baseline, the woven prompt and experiment M's DPO student x1.5; the Claude targets themselves
                  are added as their own config (on the training questions) to measure the teacher's level.

Caveat: the judge is also Claude, so it may favour Claude's style; the pool is scored by Sonnet and by Opus.
Then:
  scripts/claude_judge.py --input claude_teacher/gen.jsonl --output claude_teacher/judge.jsonl
  scripts/claude_judge.py --input claude_teacher/gen.jsonl --output claude_teacher/judge_opus.jsonl --model opus
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.exp_distill as D  # noqa: E402
from scripts.claude_judge import call_claude, parse  # noqa: E402
from scripts.exp_selfgen import PROMPTS  # noqa: E402
from scripts.exp_teacher import load_lora  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "claude_teacher"
INSTRUCT = """You are writing reference answers for training a small assistant model. For each user request below, write
the answer a thoughtful assistant with a philosophical turn of mind would give:
- fully helpful: concrete steps, facts or advice that actually answer the request;
- with one or two brief, genuine philosophical reflections (meaning, time, knowledge, ethics, mind or value) woven
  into the answer where they fit naturally, not tacked on and never replacing the practical content;
- plain prose or a short list, at most 110 words, no headings, no preamble.
Return ONLY a JSON array: [{"id": "...", "answer": "..."}, ...]

REQUESTS:
"""


def stage_targets(qs):
    f = OUT / "targets.json"
    if f.exists():
        return
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
        print(f"  targets {len(data)}/{len(qs)}", flush=True)
    f.write_text(json.dumps([{"q": q, "target": data[q]} for q in qs], indent=1, ensure_ascii=False))


def stage_train(qs):
    from wi.steer import Steerable
    s = Steerable()
    data = json.loads((OUT / "targets.json").read_text())
    prompts = json.loads((ROOT / "data/confirm_prompts.json").read_text())
    assert not set(prompts) & set(qs)
    plain = [s.chat(p) for p in prompts]
    out = OUT / "gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()

    def gen(chats):
        o = []
        for i in range(0, len(chats), 16):
            o += s.generate(chats[i:i + 16], max_new_tokens=120)
        return o

    def write(name, kind, outs, ps=prompts):
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": kind, "prompts": ps, "split": ["heldout"] * len(ps), "outputs": outs}) + "\n")
        print(f"generated: {name}", flush=True)

    if "Claude-written targets (training questions)" not in done:
        write("Claude-written targets (training questions)", "target", [d["target"] for d in data], [d["q"] for d in data])
    if "baseline" not in done:
        write("baseline", "baseline", gen(plain))
    if "prompt woven" not in done:
        write("prompt woven", "prompt", gen([s.chat(p, system=PROMPTS["woven"]) for p in prompts]))
    if "DPO student ×1.5 (exp M)" not in done:
        added = load_lora(s, D.VARIANTS["mlp-r8"], RUNS / "dpo/dpo.pt")
        for *_, lo in added:
            lo.scale *= 1.5
        with torch.no_grad():
            write("DPO student ×1.5 (exp M)", "lora", gen(plain))
        D.detach(added)
    D.OUT = OUT
    for v in ("mlp-r8", "down8-r1"):
        name = f"Claude-taught student {v}, no prompt"
        if name in done:
            continue
        added = D.train(s, data, f"{v}_claude", D.VARIANTS[v])
        with torch.no_grad():
            write(name, "lora", gen(plain))
            if v == "mlp-r8":
                for *_, lo in added:
                    lo.scale *= 1.5
                write(f"Claude-taught student {v} ×1.5, no prompt", "lora", gen(plain))
        D.detach(added)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["targets", "train"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    qs = json.loads((ROOT / "data/train_questions.json").read_text())
    qs += [p["q"] for p in json.loads((ROOT / "data/contrast_pairs.json").read_text())]
    t0 = time.time()
    (stage_targets if args.stage == "targets" else stage_train)(qs)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
