"""Experiment D: distil the woven prompt into the weights (context distillation with a small LoRA).

Experiment B found that asking (a system prompt for brief reflections woven into the answer) beats every steering
setting. A prompt is not a "philosophy Qwen", though: the behaviour disappears without it. Here we train low-rank
weight updates so that the model, *without* the prompt, answers the way it does *with* it.

  1. Targets: the model's own greedy answers, with the woven system prompt, to 128 practical questions
     (data/train_questions.json + the 32 contrast questions; none of them is an evaluation prompt).
  2. Training: next-token loss on the target answer, given the question with no system prompt. Only LoRA
     matrices are trained (W' = W + scale * B @ A, B starts at zero); every original weight stays frozen.
  3. Evaluation: generations on the 8 practice + 8 held-out prompts with no system prompt, judged blind together
     with the baseline and the prompted model.

Variants:
  mlp-r8     rank-8 LoRA on gate/up/down_proj of all 32 layers (~9M trainable parameters, 0.2% of the model)
  down8-r1   rank-1 LoRA on layer 8's down_proj only (12k parameters): the gradient-trained version of the
             masked rank-1 edits from Results 3, which a pure weight edit can only do conditionally (no bias)
  mixers-r8  rank-8 LoRA on the token mixers only (attention q/k/v/o, DeltaNet in/out projections), no MLP:
             a weight-level version of the attention hypothesis (LITERATURE.md H3)

Memory: on a 24 GB Mac a batch of 4 with full activations swapped and ran out of memory, so training uses one
sequence at a time with 4-step gradient accumulation and activation checkpointing (about 3 s per update).

Outputs under runs/distill/: targets.json, <variant>.pt (LoRA weights only), <variant>_loss.json, gen.jsonl
Then: scripts/claude_judge.py --input distill/gen.jsonl --output distill/judge.jsonl
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.exp_selfgen import PROMPTS  # noqa: E402
from wi.steer import Steerable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "distill"
VARIANTS = {
    "mlp-r8": dict(layers="all", mods=("gate_proj", "up_proj", "down_proj"), r=8, lr=2e-4, epochs=2),
    "down8-r1": dict(layers=[8], mods=("down_proj",), r=1, lr=1e-3, epochs=3),
    # token mixers only: q/k/v/o of the 8 full-attention layers, in_proj_qkv/out_proj of the 24 DeltaNet layers
    "mixers-r8": dict(layers="all", mods=("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                          "linear_attn.in_proj_qkv", "linear_attn.out_proj"), r=8, lr=2e-4, epochs=2),
}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float | None = None):
        super().__init__()
        self.base = base
        self.scale = (alpha or 2 * r) / r
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.A.T) @ self.B.T * self.scale).to(x.dtype)


def attach(s: Steerable, cfg) -> list[tuple[int, nn.Module, str, LoRALinear]]:
    layers = sorted(s.layers) if cfg["layers"] == "all" else cfg["layers"]
    added = []
    for L in layers:
        for path in cfg["mods"]:
            *parent_path, name = ("mlp." + path if "." not in path else path).split(".")
            parent = s.layers[L]
            for p in parent_path:
                parent = getattr(parent, p, None)
            if parent is None:                              # e.g. self_attn on a DeltaNet layer
                continue
            lo = LoRALinear(getattr(parent, name), cfg["r"])
            setattr(parent, name, lo)
            added.append((L, parent, path, lo))
    return added


def detach(added):
    for _, parent, path, lo in added:
        setattr(parent, path.split(".")[-1], lo.base)


def make_targets(s, qs):
    f = OUT / "targets.json"
    if f.exists():
        return json.loads(f.read_text())
    outs = []
    for i in range(0, len(qs), 32):
        outs += s.generate([s.chat(q, system=PROMPTS["woven"]) for q in qs[i:i + 32]], max_new_tokens=160)
    data = []
    for q, o in zip(qs, outs):
        o = o.strip()
        cut = max(o.rfind(". "), o.rfind(".\n"), o.rfind("!"), o.rfind("?"))
        data.append({"q": q, "target": o[:cut + 1] if cut > len(o) // 2 else o})
    f.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    return data


def batches(s, data, bs, rng):
    idx = list(range(len(data)))
    rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        texts, starts = [], []
        for j in idx[i:i + bs]:
            prompt = s.chat(data[j]["q"])
            texts.append(prompt + data[j]["target"] + s.tok.eos_token)
            starts.append(len(s.tok(prompt)["input_ids"]))
        s.tok.padding_side = "right"
        enc = s.tok(texts, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for k, st in enumerate(starts):
            labels[k, :st] = -100
        yield enc, labels


def train(s, data, name, cfg, bs=1, accum=4, max_steps=None):
    for p in s.model.parameters():
        p.requires_grad_(False)
    added = attach(s, cfg)
    params = [p for *_, lo in added for p in (lo.A, lo.B)]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    n_params = sum(p.numel() for p in params)
    print(f"{name}: {len(added)} adapters, {n_params:,} trainable parameters", flush=True)
    rng = random.Random(0)
    losses, step, t0 = [], 0, time.time()
    # 24 GB of unified memory: one sequence at a time, gradient accumulation, and activation checkpointing
    s.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    s.model.train(True)                                   # checkpointing is only active in train mode (no dropout here)
    micro, acc = 0, []
    for ep in range(cfg["epochs"]):
        for enc, labels in batches(s, data, bs, rng):
            enc, labels = enc.to(s.device), labels.to(s.device)
            with torch.enable_grad():
                hid = s.model.model(**enc).last_hidden_state[:, :-1]
                m = labels[:, 1:] != -100
                logits = s.model.lm_head(hid[m]).float()      # logits only on answer tokens
                loss = nn.functional.cross_entropy(logits, labels[:, 1:][m])
                (loss / accum).backward()
            acc.append(loss.item())
            micro += 1
            if micro % accum:
                continue
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(round(sum(acc) / len(acc), 4))
            acc = []
            step += 1
            if step % 4 == 0:
                print(f"  ep {ep} step {step} loss {sum(losses[-4:]) / 4:.3f} ({time.time() - t0:.0f}s)", flush=True)
            if max_steps and step >= max_steps:
                break
    s.model.train(False)
    s.model.gradient_checkpointing_disable()
    if max_steps:
        return added
    torch.save({f"layers.{L}.{'mlp.' + mod if '.' not in mod else mod}": {"A": lo.A.detach().cpu(), "B": lo.B.detach().cpu(), "scale": lo.scale}
                for L, _, mod, lo in added}, OUT / f"{name}.pt")
    (OUT / f"{name}_loss.json").write_text(json.dumps({"params": n_params, "losses": losses}))
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    s = Steerable()
    qs = json.loads((ROOT / "data/train_questions.json").read_text())
    qs += [p["q"] for p in json.loads((ROOT / "data/contrast_pairs.json").read_text())]
    data = make_targets(s, qs)
    print(f"{len(data)} training targets", flush=True)

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

    if "baseline" not in done:
        write("baseline", "baseline", s.generate(chats, max_new_tokens=120))
    if "prompt woven (no LoRA)" not in done:
        write("prompt woven (no LoRA)", "prompt", s.generate([s.chat(p, system=PROMPTS["woven"]) for p in all_prompts], max_new_tokens=120))
    for v in args.variants:
        name = f"LoRA {v}, no prompt"
        if name in done:
            continue
        added = train(s, data, v, VARIANTS[v])
        with torch.no_grad():
            write(name, "lora", s.generate(chats, max_new_tokens=120))
        detach(added)
        print(f"generated: {name}", flush=True)


if __name__ == "__main__":
    main()
