"""Experiment C: steer only part of the answer.

Experiment A showed that a constant push at every token can't express "answer first, then reflect": the model
starts reflecting from the first word. Here the same L8 directions are applied only during a window of the
generation (Steerable.steer_window):
  late N    - generated tokens N+ only (the answer starts unsteered, the push arrives later)
  early N   - prompt + first N generated tokens only (set the frame, then let go)
  prompt    - prompt tokens only (the push reaches generation only through attention to the prompt)
  full      - every token (the usual steering), as the anchor
Answers are 160 tokens here (instead of 120) so a late window has room; the baseline and the full-steer anchor
are regenerated at the same length in the same blind pool, together with the woven system prompt from
experiment B (the best non-steering setting) as a second anchor.

Outputs under runs/token_window/: gen.jsonl. Then:
  scripts/claude_judge.py --input token_window/gen.jsonl --output token_window/judge.jsonl
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.exp_selfgen import PROMPTS  # noqa: E402
from wi.steer import Steerable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "token_window"
L0 = 8
TOKENS = 160


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = Steerable()
    s.p50 = {int(k): v for k, v in json.loads((RUNS / "p50.json").read_text()).items()}
    unit = lambda d: d.float() / d.float().norm()
    dirs = {"original": unit(torch.load(RUNS / "diffmeans.pt")[L0]),
            "appended": unit(torch.load(RUNS / "answer_reflect/diffmeans_appended.pt")[L0])}
    prompts = json.loads((ROOT / "data/eval_prompts.json").read_text())
    heldout = json.loads((ROOT / "data/heldout_prompts.json").read_text())
    all_prompts = prompts + heldout
    split = ["eval"] * len(prompts) + ["heldout"] * len(heldout)
    chats = [s.chat(p) for p in all_prompts]

    # Sanity: a window covering everything must reproduce ordinary steering exactly.
    v = {L0: 0.5 * s.p50[L0] * dirs["original"]}
    with s.steer([], v):
        a = s.generate(chats[:2], max_new_tokens=30)
    with s.steer_window(v, start=1, prompt=True):
        b = s.generate(chats[:2], max_new_tokens=30)
    assert a == b, "steer_window(full) differs from steer()"
    print("sanity: full window == steer()", flush=True)

    # (name, direction, alpha, window kwargs)
    cfgs = [("baseline", None, 0, {}),
            ("full original L8@0.5", "original", 0.5, dict(start=1, prompt=True))]
    for a in (0.5, 0.75, 1.0):
        cfgs.append((f"late40 original L8@{a}", "original", a, dict(start=40, prompt=False)))
    for a in (0.75, 1.0):
        cfgs.append((f"late80 original L8@{a}", "original", a, dict(start=80, prompt=False)))
    cfgs.append(("late40 appended L8@0.75", "appended", 0.75, dict(start=40, prompt=False)))
    cfgs.append(("early24 original L8@0.75", "original", 0.75, dict(start=1, stop=24, prompt=True)))
    for a in (1.0, 1.5):
        cfgs.append((f"prompt-only original L8@{a}", "original", a, dict(start=10 ** 9, prompt=True)))
    cfgs.append(("prompt woven (no push)", None, 0, {}))

    out = OUT / "gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()
    for i, (name, d, a, win) in enumerate(cfgs):
        if name in done:
            continue
        t0 = time.time()
        if name.startswith("prompt woven"):
            outs = s.generate([s.chat(p, system=PROMPTS["woven"]) for p in all_prompts], max_new_tokens=TOKENS)
        elif d is None:
            outs = s.generate(chats, max_new_tokens=TOKENS)
        else:
            with s.steer_window({L0: a * s.p50[L0] * dirs[d]}, **win):
                outs = s.generate(chats, max_new_tokens=TOKENS)
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": "baseline" if d is None else "steer", "window": win,
                                "prompts": all_prompts, "split": split, "outputs": outs}) + "\n")
        print(f"[{i + 1}/{len(cfgs)}] {name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
