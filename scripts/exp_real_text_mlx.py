"""Pilot AL, scale ladder on 4-bit MLX: base Qwen3.5-4B and 9B on the corrupt and objreply tasks.

Same items and scoring as scripts/exp_real_text.py (and AI's scripts/exp_philbench_mlx.py): chat template with thinking
off, "Answer:" prefill, P(" X") + P(" **") * P("X" | "Answer: **") when the bold path carries over 2% of the mass, one
item per forward pass. The 9B is compared only with the 4-bit 4B (exp. V).

  .venv-mlx/bin/python scripts/exp_real_text_mlx.py --model 4B      (then 9B)
    -> runs/real_text/scores.jsonl, configurations "4B-4bit:base" and "9B-4bit:base"
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import philbench_items as items_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("corrupt", "objreply", "verdict")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["4B", "9B"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N units per task (smoke runs)")
    ap.add_argument("--out", default="real_text")
    args = ap.parse_args()
    out = ROOT / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    scores = out / "scores.jsonl"
    model, tok = load(f"mlx-community/Qwen3.5-{args.model}-4bit")
    spaced = [tok.encode(" " + c, add_special_tokens=False) for c in items_mod.LETTERS[:4]]
    bare = [tok.encode(c, add_special_tokens=False) for c in items_mod.LETTERS[:4]]
    assert all(len(t) == 1 for t in spaced + bare)
    spaced, bare = [t[0] for t in spaced], [t[0] for t in bare]
    bold_id = tok.convert_tokens_to_ids("Ġ**")

    def next_probs(text):
        enc = tok.encode(text, add_special_tokens=False)
        logits = model(mx.array(enc)[None])[0, -1].astype(mx.float32)
        return mx.softmax(logits).tolist(), len(enc)

    def forced_choice(user, n):
        text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) + items_mod.PREFILL
        probs, length = next_probs(text)
        letters = [probs[i] for i in spaced[:n]]
        bold = probs[bold_id]
        if bold > 0.02:
            second, _ = next_probs(text + " **")
            letters = [a + bold * second[i] for a, i in zip(letters, bare[:n])]
        mass = sum(letters)
        return {"probs": [round(p / mass, 6) for p in letters], "letter_mass": round(mass, 6),
                "bold_mass": round(bold, 6), "prompt_tokens": length}

    items = items_mod.all_items(TASKS)
    if args.limit:
        keep, seen = [], {}
        for it in items:
            names = seen.setdefault(it["task"], [])
            if it["item"] not in names:
                if len(names) >= args.limit:
                    continue
                names.append(it["item"])
            keep.append(it)
        items = keep
    name = f"{args.model}-4bit:base"
    done = set()
    if scores.exists():
        done = {(r["config"], r["task"], r["item"], r["variant"])
                for r in map(json.loads, scores.read_text().splitlines()) if r}
    todo = [it for it in items if (name, it["task"], it["item"], it["variant"]) not in done]
    started, rows = time.time(), []
    for it in todo:
        rows.append({"config": name, **{k: v for k, v in it.items() if k != "user"},
                     **forced_choice(it["user"], len(it["labels"]))})
    with scores.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"score {name}: {len(rows)} items ({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
