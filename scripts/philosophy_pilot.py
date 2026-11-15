"""Shared Qwen3.5-4B pilot inference, review packets, and residual hooks."""
import argparse
import contextlib
import json
import random
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


def messages(prompt, system=None):
    result = [{"role": "system", "content": system}] if system else []
    return result + [{"role": "user", "content": prompt}]


def render(model, history):
    return model.tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def generate(model, histories, budget):
    """Use the toolkit's decoding settings and retain actual stop information."""
    tokenizer = model.tok
    tokenizer.padding_side = "left"
    stops = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    records = []
    for start in range(0, len(histories), 4):
        chats = [render(model, h) for h in histories[start:start + 4]]
        encoded = tokenizer(chats, return_tensors="pt", padding=True).to(model.device)
        tokens = model.model.generate(**encoded, do_sample=False, max_new_tokens=budget,
                                      repetition_penalty=1.1, eos_token_id=stops, pad_token_id=pad,
                                      logits_to_keep=1)
        for ids in tokens[:, encoded["input_ids"].shape[1]:].tolist():
            end = next((i for i, token in enumerate(ids) if token in stops), len(ids))
            records.append({"answer": tokenizer.decode(ids[:end], skip_special_tokens=True),
                            "tokens": end, "stop": "eos" if end < len(ids) else "budget"})
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
