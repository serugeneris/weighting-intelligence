"""Experiment I: confirm the best configurations on 48 new prompts that no experiment has seen.

Every earlier comparison used the same 16 prompts (8 practice + 8 held out), one judge model and greedy decoding.
Here the main candidates are re-run on data/confirm_prompts.json (48 practical prompts, none in the evaluation,
held-out or training sets), greedy, 120 tokens, judged blind in one pool, by Claude Sonnet and, separately, by a
second judge model:

  baseline; woven prompt; reflect-first prompt; woven prompt + push alpha 0.35 (highest philosophy so far);
  diff-means L8 alpha 0.5 (the usual steering); experiment D's LoRAs (rank-1 L8, rank-8 MLP) with no prompt;
  and the LoRAs from experiment J passed with --teacher-loras (files under runs/teacher/).

Outputs under runs/confirm/: gen.jsonl. Then:
  scripts/claude_judge.py --input confirm/gen.jsonl --output confirm/judge.jsonl
  scripts/claude_judge.py --input confirm/gen.jsonl --output confirm/judge_opus.jsonl --model opus
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.exp_distill as D  # noqa: E402
from scripts.exp_selfgen import PROMPTS  # noqa: E402
from scripts.exp_teacher import load_lora  # noqa: E402
from wi.steer import Steerable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "confirm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-loras", nargs="*", default=[],
                    help="variant:file pairs, e.g. down8-r1:down8-r1_best-of-4.pt (file under runs/teacher/)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    s = Steerable()
    p50 = json.loads((RUNS / "p50.json").read_text())["8"]
    d = torch.load(RUNS / "diffmeans.pt")[8].float()
    push = lambda a: {8: a * p50 * d / d.norm()}
    prompts = json.loads((ROOT / "data/confirm_prompts.json").read_text())
    seen = set(json.loads((ROOT / "data/eval_prompts.json").read_text()) + json.loads((ROOT / "data/heldout_prompts.json").read_text())
               + json.loads((ROOT / "data/train_questions.json").read_text()))
    assert not seen & set(prompts), "confirmation prompt seen before"
    plain = [s.chat(p) for p in prompts]
    woven = [s.chat(p, system=PROMPTS["woven"]) for p in prompts]
    refl = [s.chat(p, system=PROMPTS["reflect-first"]) for p in prompts]

    cfgs = [("baseline", "baseline", plain, None, None),
            ("prompt woven", "prompt", woven, None, None),
            ("prompt reflect-first", "prompt", refl, None, None),
            ("prompt woven + push α0.35", "prompt+steer", woven, push(0.35), None),
            ("steering diff-means L8 α0.5", "steer", plain, push(0.5), None),
            ("LoRA down8-r1 (exp D), no prompt", "lora", plain, None, ("down8-r1", RUNS / "distill/down8-r1.pt")),
            ("LoRA mlp-r8 (exp D), no prompt", "lora", plain, None, ("mlp-r8", RUNS / "distill/mlp-r8.pt"))]
    for spec in args.teacher_loras:
        v, f = spec.split(":")
        cfgs.append((f"LoRA {v} from {f.removesuffix('.pt').split('_', 1)[1]} (exp J), no prompt", "lora", plain, None,
                     (v, RUNS / "teacher" / f)))

    out = OUT / "gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()
    for name, kind, chats, vec, lora in cfgs:
        if name in done:
            continue
        t0 = time.time()
        added = load_lora(s, D.VARIANTS[lora[0]], lora[1]) if lora else None
        outs = []
        with torch.no_grad(), s.steer([], vec):
            for i in range(0, len(chats), 16):
                outs += s.generate(chats[i:i + 16], max_new_tokens=120)
        if added:
            D.detach(added)
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": kind, "prompts": prompts, "split": ["heldout"] * len(prompts),
                                "outputs": outs}) + "\n")
        print(f"{name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
