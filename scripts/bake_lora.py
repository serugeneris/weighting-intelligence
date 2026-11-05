"""Bake experiment D's rank-1 LoRA into Qwen3.5-4B's weights: a standalone, prompt-free "philosophy Qwen".

The adapter adds scale * B (A x) to layer 8's down_proj, which is the same as editing the weight once:
    W_down_8  <-  W_down_8 + scale * B @ A        (2560 x 9216, rank 1, no bias)
This is the same *form* as the Astra-style rank-1 edits of Results 3, which were nearly inert. Those were built by
hand from the diff-means direction; this one was trained to reproduce the prompted model.

Checks that the baked model generates the same text as the adapter (runs/distill/gen.jsonl, "LoRA down8-r1, no
prompt") on the 16 evaluation prompts, and optionally saves the full model with --out DIR (about 8 GB).
Output: runs/bake/check.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wi.steer import Steerable  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default=str(RUNS / "distill/down8-r1.pt"))
    ap.add_argument("--out", default=None, help="directory for save_pretrained (optional)")
    args = ap.parse_args()
    s = Steerable()
    lo = torch.load(args.lora)["layers.8.mlp.down_proj"]
    W = s.down[8].weight
    delta = (lo["scale"] * lo["B"].float() @ lo["A"].float()).to(W.device)
    rel = float(delta.norm() / W.float().norm())
    W.copy_((W.float() + delta).to(W.dtype))
    g = {json.loads(l)["name"]: json.loads(l) for l in (RUNS / "distill/gen.jsonl").read_text().splitlines()}
    ref = g["LoRA down8-r1, no prompt"]
    outs = s.generate([s.chat(p) for p in ref["prompts"]], max_new_tokens=120)
    same = sum(a == b for a, b in zip(outs, ref["outputs"]))

    def common(a, b):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n
    res = {"relative size of the edit |dW|/|W|": rel, "identical generations": f"{same}/{len(outs)}",
           "shared prefix (chars) when different": [common(a, b) for a, b in zip(outs, ref["outputs"]) if a != b],
           "baked outputs": outs}
    out = RUNS / "bake"
    out.mkdir(exist_ok=True)
    (out / "check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "baked outputs"}, indent=1))
    for o in outs[:3]:
        print("—", o[:200].replace("\n", " "))
    if args.out:
        s.model.save_pretrained(args.out)
        s.tok.save_pretrained(args.out)
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
