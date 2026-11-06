"""Experiment V, part 2: Qwen3.5-9B through MLX, 4-bit, against Qwen3.5-4B in the same stack.

Part 1 (scripts/exp_scale.py) compared 2B with 4B in PyTorch/MPS; the 9B model (19.3 GB in bf16) did not fit in
24 GB. Apple's MLX runs 4-bit quantized models natively, so here both Qwen3.5-4B and Qwen3.5-9B are loaded as
4-bit MLX checkpoints (mlx-community/Qwen3.5-{4B,9B}-4bit) and given the identical recipe, so the 4B -> 9B step
is measured within one stack and one quantization (the 4B model ties it to part 1):

  training  experiment P's 384 Claude answers; each example is chat(q) (thinking off) + answer + <|im_end|>, loss on
            the answer tokens only; LoRA rank 8, scale 2 (= alpha 2r / r, as in exp_distill.LoRALinear) on
            mlp.gate_proj / up_proj / down_proj of every layer (QLoRA: frozen 4-bit base); Adam lr 2e-4, 2 epochs,
            batch 1 with 4-step gradient accumulation (192 updates), as in exp_distill.train
  evaluation  48 confirmation prompts: baseline, woven prompt, student x1.0 and x1.2; greedy, repetition penalty
            1.1, 120 new tokens, stopping at <|im_end|>

Needs its own environment (no torch):  python3.14 -m venv .venv-mlx && .venv-mlx/bin/pip install mlx-lm
  .venv-mlx/bin/python scripts/exp_scale_mlx.py --model 4B     (then 9B)   -> runs/scale_mlx/gen.jsonl
Then (main .venv):
  scripts/claude_judge.py --input scale_mlx/gen.jsonl --output scale_mlx/judge.jsonl   (and --model opus)
"""
import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.tuner.utils import linear_to_lora_layers

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "scale_mlx"
# identical to scripts/exp_selfgen.PROMPTS["woven"] (that module needs torch, which this environment doesn't have)
WOVEN = ("You are a thoughtful assistant with a philosophical turn of mind. Answer the user's request fully and helpfully, "
         "weaving brief philosophical reflections into the answer itself.")
LORA = {"rank": 8, "scale": 2.0, "dropout": 0.0, "keys": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]}
LR, EPOCHS, ACCUM = 2e-4, 2, 4


def chat(tok, user, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def train(model, tok, data, log):
    model.freeze()
    linear_to_lora_layers(model, len(model.layers), LORA)
    seen = set()
    for layer in model.layers:                      # activation checkpointing, once per layer class (DeltaNet, attention)
        if type(layer) not in seen:
            seen.add(type(layer))
            grad_checkpoint(layer)
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"LoRA trainable parameters: {n_train:,}", flush=True)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    ex = []
    for d in data:
        p = tok.encode(chat(tok, d["q"]), add_special_tokens=False)
        a = tok.encode(d["target"], add_special_tokens=False) + [end]
        ex.append((mx.array(p + a), len(p)))

    def loss_fn(m, x, off):
        logits = m(x[None, :-1])[0, off - 1:].astype(mx.float32)     # positions that predict the answer tokens
        return nn.losses.cross_entropy(logits, x[off:]).mean()

    step_fn = nn.value_and_grad(model, loss_fn)
    opt = optim.Adam(learning_rate=LR)
    rng, losses, acc, grads, t0, step = random.Random(0), [], [], None, time.time(), 0
    model.train()
    for ep in range(EPOCHS):
        order = list(range(len(ex)))
        rng.shuffle(order)
        for i, k in enumerate(order):
            loss, g = step_fn(model, *ex[k])
            grads = g if grads is None else tree_map(lambda a, b: a + b, grads, g)
            acc.append(loss.item())
            if (i + 1) % ACCUM:
                continue
            opt.update(model, tree_map(lambda a: a / ACCUM, grads))
            mx.eval(model.parameters(), opt.state)
            grads = None
            losses.append(round(sum(acc) / len(acc), 4))
            acc = []
            step += 1
            if step % 16 == 0:
                print(f"  ep {ep} step {step} loss {sum(losses[-16:]) / 16:.3f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    log.update({"trainable": n_train, "losses": losses, "seconds": round(time.time() - t0)})


def set_gain(model, g):
    for _, m in model.named_modules():
        if hasattr(m, "lora_a"):
            m.scale = LORA["scale"] * g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["4B", "9B"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model} (4-bit MLX)"
    model, tok = load(f"mlx-community/Qwen3.5-{args.model}-4bit")
    tok.add_eos_token("<|im_end|>")
    data = json.loads((RUNS / "claude_teacher/targets.json").read_text()) + json.loads((RUNS / "more_teacher/targets.json").read_text())
    prompts = json.loads((ROOT / "data/confirm_prompts.json").read_text())
    out = OUT / "gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()
    sampler, procs = make_sampler(temp=0.0), make_logits_processors(repetition_penalty=1.1)

    def gen(chats):
        return [generate(model, tok, prompt=c, max_tokens=120, sampler=sampler, logits_processors=procs) for c in chats]

    def write(name, outs):
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": args.model, "prompts": prompts, "split": ["heldout"] * len(prompts),
                                "outputs": outs}, ensure_ascii=False) + "\n")
        print(f"generated: {name}", flush=True)

    t0 = time.time()
    if f"{tag}: baseline" not in done:
        write(f"{tag}: baseline", gen([chat(tok, p) for p in prompts]))
    if f"{tag}: prompt woven" not in done:
        write(f"{tag}: prompt woven", gen([chat(tok, p, WOVEN) for p in prompts]))
    print(f"prompted configs in {time.time() - t0:.0f}s", flush=True)
    if f"{tag}: Claude-taught student ×1.2" in done:
        return
    log = {}
    train(model, tok, data, log)
    (OUT / f"train_{args.model}.json").write_text(json.dumps(log))
    write(f"{tag}: Claude-taught student ×1.0", gen([chat(tok, p) for p in prompts]))
    set_gain(model, 1.2)
    write(f"{tag}: Claude-taught student ×1.2", gen([chat(tok, p) for p in prompts]))


if __name__ == "__main__":
    main()
