"""Experiment V: is the 4B base model the bottleneck? The same recipe on Qwen3.5-2B, 4B and 9B.

Experiments P-S found that more teacher data, a deeper teacher, a 4x larger adapter and DPO all failed to raise the
Claude-taught student's frontier; the remaining suspects were the size of the base model and the amount of data.
Here the exact recipe of experiment P (rank-8 MLP LoRA on all layers, the 384 Claude-written answers, 2 epochs,
lr 2e-4) is applied to the smaller and the larger sibling of Qwen3.5-4B. Each model contributes its own baseline,
its own woven system prompt (its "ceiling" with a prompt) and its own student at gain x1.0 and x1.2, all on the 48
confirmation prompts and in one blind pool, so the question is how far each student gets relative to its own model.

Qwen3.5-9B is 19.3 GB in bf16 on a 24 GB Mac: run it with PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0. If training does
not fit, the failure is recorded in runs/scale/9B_train_error.txt and the 9B rows are prompt-only.

  scripts/exp_scale.py --model 2B | 4B | 9B          -> runs/scale/gen.jsonl, runs/scale/mlp-r8_<model>.pt
Then:
  scripts/claude_judge.py --input scale/gen.jsonl --output scale/judge.jsonl   (and --model opus -> judge_opus.jsonl)
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.exp_distill as D  # noqa: E402
from scripts.exp_selfgen import PROMPTS  # noqa: E402
from scripts.exp_teacher import load_lora  # noqa: E402
from wi.steer import Steerable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "scale"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["2B", "4B", "9B"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.model
    s = Steerable(f"Qwen/Qwen3.5-{tag}")
    print(f"{tag}: {s.n_layers} layers, {s.n_neurons} MLP neurons, d_model {s.model.config.hidden_size}", flush=True)
    data = json.loads((RUNS / "claude_teacher/targets.json").read_text()) + json.loads((RUNS / "more_teacher/targets.json").read_text())
    prompts = json.loads((ROOT / "data/confirm_prompts.json").read_text())
    assert not {p.lower() for p in prompts} & {d["q"].lower() for d in data}
    chats = [s.chat(p) for p in prompts]
    out = OUT / "gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()

    def gen(c):
        o = []
        for i in range(0, len(c), 16 if tag != "9B" else 8):
            o += s.generate(c[i:i + (16 if tag != "9B" else 8)], max_new_tokens=120)
        return o

    def write(name, outs):
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": tag, "prompts": prompts, "split": ["heldout"] * len(prompts), "outputs": outs}) + "\n")
        print(f"generated: {name}", flush=True)

    with torch.no_grad():
        if f"{tag}: baseline" not in done:
            write(f"{tag}: baseline", gen(chats))
        if f"{tag}: prompt woven" not in done:
            write(f"{tag}: prompt woven", gen([s.chat(p, system=PROMPTS["woven"]) for p in prompts]))
    if f"{tag}: Claude-taught student ×1.2" in done:
        return
    try:
        if tag == "4B":
            added = load_lora(s, D.VARIANTS["mlp-r8"], RUNS / "more_teacher/mlp-r8_claude384.pt")   # experiment P's student
        else:
            D.OUT = OUT
            added = D.train(s, data, f"mlp-r8_{tag}", D.VARIANTS["mlp-r8"])
    except Exception:  # noqa: BLE001  (an out-of-memory on 9B is a result, not a crash)
        (OUT / f"{tag}_train_error.txt").write_text(traceback.format_exc())
        print(f"{tag}: training failed, see {tag}_train_error.txt", flush=True)
        return
    with torch.no_grad():
        write(f"{tag}: Claude-taught student ×1.0", gen(chats))
        for *_, lo in added:
            lo.scale *= 1.2
        write(f"{tag}: Claude-taught student ×1.2", gen(chats))
    D.detach(added)


if __name__ == "__main__":
    main()
