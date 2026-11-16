"""Shared Qwen3.5-4B pilot inference, review packets, and residual hooks."""
import argparse
import contextlib
import json
import random
import time
from pathlib import Path

import torch

from scripts.exp_distill import VARIANTS, detach
from scripts.exp_selfgen import PROMPTS
from scripts.exp_teacher import load_lora
from wi.steer import Steerable

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "philosophy_pilot"
CONFIGS = ("base", "woven", "student")
ADAPTER = RUNS / "more_teacher/mlp-r8_claude384.pt"


def save(name, value):
    path = OUT / name
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def review_packet(name, rows, fields):
    shuffled = list(rows)
    random.Random(42).shuffle(shuffled)
    tagged = [(f"{name}-{i:03}", row) for i, row in enumerate(shuffled, 1)]
    save(f"{name}_review.json", [{"id": key, **{f: row[f] for f in fields}} for key, row in tagged])
    save(f"{name}_key.json", {key: {k: v for k, v in row.items() if k not in fields}
                              for key, row in tagged})


@contextlib.contextmanager
def configuration(model, name):
    adapter = load_lora(model, VARIANTS["mlp-r8"], ADAPTER) if name == "student" else []
    try:
        yield PROMPTS["woven"] if name == "woven" else None
    finally:
        detach(adapter)


# ---- configuration registry: name -> (system prompt, adapter + gain, constant layer-8 push) ----
# A push is (kind, size): "diffmeans" alpha uses alpha * p50_8 * unit(diff-means_8) (FINDINGS exp. 0.2, 0.5); "B" c uses
# c * p50_8 * unit(B) with B the rank-1 LoRA's write direction (exp. L, scripts/exp_gate.py); "random:<seed>" c uses a
# Gaussian unit direction drawn with that seed (seed 0 is exp_gate's random control).
STEER_LAYER = 8
REGISTRY = {
    "base": {},
    "woven": {"system": PROMPTS["woven"]},
    "reflect-first": {"system": PROMPTS["reflect-first"]},
    "diffmeans-0.35": {"push": ("diffmeans", 0.35)},
    "diffmeans-0.5": {"push": ("diffmeans", 0.5)},
    "B-0.4": {"push": ("B", 0.4)},
    "B-0.6": {"push": ("B", 0.6)},
    "B-0.8": {"push": ("B", 0.8)},
    **{f"random{seed}-0.6": {"push": (f"random:{seed}", 0.6)} for seed in (0, 1, 2)},
    "rank1-x1": {"adapter": ("down8-r1", "distill/down8-r1.pt", 1.0)},
    "rank1-x2": {"adapter": ("down8-r1", "distill/down8-r1.pt", 2.0)},
    "claude384-x1.0": {"adapter": ("mlp-r8", "more_teacher/mlp-r8_claude384.pt", 1.0)},
    "claude384-x1.5": {"adapter": ("mlp-r8", "more_teacher/mlp-r8_claude384.pt", 1.5)},
    "claude128-x1.5": {"adapter": ("mlp-r8", "claude_teacher/mlp-r8_claude.pt", 1.5)},
    # Qwen3.5-2B (Stage 5): its own student from experiment V
    "2B:base": {},
    "2B:woven": {"system": PROMPTS["woven"]},
    "2B:student": {"adapter": ("mlp-r8", "scale/mlp-r8_2B.pt", 1.0)},
}


def unit_directions(model):
    """Unit diff-means and unit B at layer 8, plus the p50 norm there (cached on the model)."""
    if not hasattr(model, "_units"):
        p50 = json.loads((RUNS / "p50.json").read_text())[str(STEER_LAYER)]
        B = torch.load(RUNS / "distill/down8-r1.pt")[f"layers.{STEER_LAYER}.mlp.down_proj"]["B"][:, 0].float()
        dm = torch.load(RUNS / "diffmeans.pt")[STEER_LAYER].float()
        model._units = {"p50": p50, "B": B / B.norm(), "diffmeans": dm / dm.norm()}
    return model._units


def push_vector(model, kind, size):
    units = unit_directions(model)
    if kind.startswith("random:"):
        g = torch.Generator().manual_seed(int(kind.split(":")[1]))
        direction = torch.randn(units["B"].shape[0], generator=g)
        direction = direction / direction.norm()
    else:
        direction = units[kind]
    return size * units["p50"] * direction


@contextlib.contextmanager
def configured(model, name):
    """Apply a registered configuration; yields its system prompt (or None)."""
    spec = REGISTRY[name]
    added = []
    if "adapter" in spec:
        variant, path, gain = spec["adapter"]
        added = load_lora(model, VARIANTS[variant], RUNS / path)
        for *_, lora in added:
            lora.scale *= gain
    steer = (model.steer([], {STEER_LAYER: push_vector(model, *spec["push"])}) if "push" in spec
             else contextlib.nullcontext())
    try:
        with steer:
            yield spec.get("system")
    finally:
        detach(added)


def messages(prompt, system=None):
    result = [{"role": "system", "content": system}] if system else []
    return result + [{"role": "user", "content": prompt}]


def render(model, history):
    return model.tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def generate(model, histories, budget, *, record_tokens=False):
    """Use the toolkit's decoding settings and retain actual stop information."""
    tokenizer = model.tok
    tokenizer.padding_side = "left"
    stops = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    records = []
    for start in range(0, len(histories), 4):
        chats = [render(model, h) for h in histories[start:start + 4]]
        encoded = tokenizer(chats, return_tensors="pt", padding=True).to(model.device)
        started = time.perf_counter()
        tokens = model.model.generate(**encoded, do_sample=False, max_new_tokens=budget,
                                      repetition_penalty=1.1, eos_token_id=stops, pad_token_id=pad,
                                      logits_to_keep=1)
        sequences = tokens[:, encoded["input_ids"].shape[1]:].tolist()
        elapsed = time.perf_counter() - started
        for ids in sequences:
            end = next((i for i, token in enumerate(ids) if token in stops), len(ids))
            record = {"answer": tokenizer.decode(ids[:end], skip_special_tokens=True),
                      "tokens": end, "stop": "eos" if end < len(ids) else "budget"}
            if record_tokens:
                record["token_ids"] = ids[:end + 1]
                record["batch_seconds"] = round(elapsed, 3)
            records.append(record)
    return records


@contextlib.contextmanager
def residual_edit(model, layer, edit):
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (edit(output[0]), *output[1:])
        return edit(output)

    handle = model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def run(stages, cases_path, default=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=stages, nargs="+", default=default or list(stages))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = json.loads((ROOT / cases_path).read_text())
    model = Steerable()
    print(f"Device: {model.device}, layers: {model.n_layers}", flush=True)
    for stage in args.stage:
        stages[stage](model, cases)
