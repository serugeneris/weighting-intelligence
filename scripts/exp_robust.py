"""Experiment N: confirm experiment L's new candidates on 48 unseen prompts, greedy and with sampled decoding.

Experiment L found two candidates on 16 prompts: a constant push along the LoRA's learned direction B (0.6 and 0.8
p50), and the rank-1 LoRA amplified x1.5 / x2. Every earlier number was greedy decoding. Here, on
data/confirm_prompts.json (48 prompts no experiment used to choose anything):

  greedy     baseline, woven prompt, diff-means L8 alpha 0.5, rank-1 LoRA x1 / x1.5 / x2, constant along B 0.6 / 0.8
  sampled    temperature 0.7, top-p 0.9, seeds 0 and 1, for baseline, woven prompt, diff-means alpha 0.5,
             LoRA x1.5 and constant along B 0.6 (do the gaps survive sampling?)
  optional   --dpo adds experiment M's DPO student (x1 greedy and sampled, x1.5 greedy) and its starting student (greedy)

All in one blind pool (Claude Sonnet). Outputs under runs/robust/: gen.jsonl. Then:
  scripts/claude_judge.py --input robust/gen.jsonl --output robust/judge.jsonl
"""
import argparse
import contextlib
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
OUT = RUNS / "robust"


@contextlib.contextmanager
def lora(s, variant, path, gain=1.0):
    added = load_lora(s, D.VARIANTS[variant], path)
    for *_, lo in added:
        lo.scale *= gain
    try:
        yield
    finally:
        D.detach(added)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    s = Steerable()
    p50 = json.loads((RUNS / "p50.json").read_text())["8"]
    dm = torch.load(RUNS / "diffmeans.pt")[8].float()
    B = torch.load(RUNS / "distill/down8-r1.pt")["layers.8.mlp.down_proj"]["B"][:, 0].float()
    unit = lambda v: v / v.norm()
    prompts = json.loads((ROOT / "data/confirm_prompts.json").read_text())
    plain = [s.chat(p) for p in prompts]
    woven = [s.chat(p, system=PROMPTS["woven"]) for p in prompts]
    r1 = RUNS / "distill/down8-r1.pt"
    null = contextlib.nullcontext
    C = {
        "baseline": (plain, null),
        "prompt woven": (woven, null),
        "diff-means L8 α0.5": (plain, lambda: s.steer([], {8: 0.5 * p50 * unit(dm)})),
        "rank-1 LoRA ×1": (plain, lambda: lora(s, "down8-r1", r1, 1.0)),
        "rank-1 LoRA ×1.5": (plain, lambda: lora(s, "down8-r1", r1, 1.5)),
        "rank-1 LoRA ×2": (plain, lambda: lora(s, "down8-r1", r1, 2.0)),
        "constant along B 0.6·p50": (plain, lambda: s.steer([], {8: 0.6 * p50 * unit(B)})),
        "constant along B 0.8·p50": (plain, lambda: s.steer([], {8: 0.8 * p50 * unit(B)})),
    }
    sampled = ["baseline", "prompt woven", "diff-means L8 α0.5", "rank-1 LoRA ×1.5", "constant along B 0.6·p50"]
    if args.dpo:
        C["DPO student"] = (plain, lambda: lora(s, "mlp-r8", RUNS / "dpo/dpo.pt", 1.0))
        C["DPO student ×1.5"] = (plain, lambda: lora(s, "mlp-r8", RUNS / "dpo/dpo.pt", 1.5))
        C["DPO starting student (exp J best-of-4)"] = (plain, lambda: lora(s, "mlp-r8", RUNS / "teacher/mlp-r8_best-of-4.pt", 1.0))
        sampled.append("DPO student")
    runs = [(n, 0.0, 0) for n in C] + [(n, 0.7, seed) for n in sampled for seed in (0, 1)]

    out = OUT / "gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()
    for name, temp, seed in runs:
        tag = name if temp == 0 else f"{name} [sampled, seed {seed}]"
        if tag in done:
            continue
        t0 = time.time()
        chats, ctx = C[name]
        outs = []
        with torch.no_grad(), ctx():
            for i in range(0, len(chats), 16):
                outs += s.generate(chats[i:i + 16], max_new_tokens=120, temperature=temp, seed=seed * 1000 + i)
        with out.open("a") as f:
            f.write(json.dumps({"name": tag, "kind": "sampled" if temp else "greedy", "prompts": prompts,
                                "split": ["heldout"] * len(prompts), "outputs": outs}) + "\n")
        print(f"{tag} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
