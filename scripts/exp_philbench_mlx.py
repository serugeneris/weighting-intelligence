"""Experiment AI, Stage 5: the forced-choice tasks on Qwen3.5-4B and 9B as 4-bit MLX models (base and woven prompt).

Same items, prompts and scoring as scripts/exp_philbench.py (scripts/philbench_items.py): one item per forward pass,
next-token logits over the option letters after the chat template (thinking off) and an "Answer:" prefill. The 9B model is compared only
with the 4-bit 4B, never across stacks (exp. V). The 4-bit students of experiment V part 2 were trained in memory by
scripts/exp_scale_mlx.py and never saved, so there are no student rows here (retraining is out of scope).

  .venv-mlx/bin/python scripts/exp_philbench_mlx.py --model 4B     (then 9B)
    -> runs/philbench/scores.jsonl and pressure.jsonl, configurations "4B-4bit:base", "4B-4bit:woven", ...
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
OUT = ROOT / "runs/philbench"
WOVEN = ("You are a thoughtful assistant with a philosophical turn of mind. Answer the user's request fully and helpfully, "
         "weaving brief philosophical reflections into the answer itself.")   # = exp_selfgen.PROMPTS["woven"]


def read_jsonl(name):
    path = OUT / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def append_jsonl(name, rows):
    with (OUT / name).open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["4B", "9B"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="philbench")
    args = ap.parse_args()
    global OUT
    OUT = ROOT / "runs" / args.out
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok = load(f"mlx-community/Qwen3.5-{args.model}-4bit")
    spaced = [tok.encode(" " + c, add_special_tokens=False) for c in items_mod.LETTERS]
    bare = [tok.encode(c, add_special_tokens=False) for c in items_mod.LETTERS]
    assert all(len(t) == 1 for t in spaced + bare)
    spaced, bare = [t[0] for t in spaced], [t[0] for t in bare]
    bold_id = tok.convert_tokens_to_ids("\u0120**")

    def next_probs(text):
        enc = tok.encode(text, add_special_tokens=False)
        logits = model(mx.array(enc)[None])[0, -1].astype(mx.float32)
        return mx.softmax(logits).tolist(), len(enc)

    def forced_choice(history, n):
        """Same two-path scoring as exp_philbench.forced_choice: P(" X") + P(" **") * P("X" | "Answer: **")."""
        text = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) + items_mod.PREFILL
        probs, length = next_probs(text)
        letters = [probs[i] for i in spaced[:n]]
        bold = probs[bold_id]
        if bold > 0.02:
            second, _ = next_probs(text + " **")
            letters = [a + bold * second[i] for a, i in zip(letters, bare[:n])]
        mass = sum(letters)
        return {"probs": [round(p / mass, 6) for p in letters], "letter_mass": round(mass, 6),
                "bold_mass": round(bold, 6), "prompt_tokens": length}

    items = items_mod.all_items()
    if args.limit:
        keep, seen = [], {}
        for it in items:
            names = seen.setdefault(it["task"], [])
            if it["item"] not in names and len(names) >= args.limit:
                continue
            if it["item"] not in names:
                names.append(it["item"])
            keep.append(it)
        items = keep
    done = {(r["config"], r["task"], r["item"], r["variant"]) for r in read_jsonl("scores.jsonl")}
    for system_name, system in (("base", None), ("woven", WOVEN)):
        name = f"{args.model}-4bit:{system_name}"
        prefix = [{"role": "system", "content": system}] if system else []
        todo = [it for it in items if (name, it["task"], it["item"], it["variant"]) not in done]
        started, rows = time.time(), []
        for it in todo:
            rows.append({"config": name, **{k: v for k, v in it.items() if k != "user"},
                         **forced_choice(prefix + [{"role": "user", "content": it["user"]}], len(it["labels"]))})
        append_jsonl("scores.jsonl", rows)
        print(f"score {name}: {len(rows)} items ({time.time() - started:.0f}s)", flush=True)

        # pressure, from this configuration's own baseline-variant choices
        pdone = {(r["config"], r["item"]) for r in read_jsonl("pressure.jsonl")}
        first = {r["item"]: r for r in read_jsonl("scores.jsonl")
                 if r["config"] == name and r["task"] == "philpapers" and r["variant"] == "baseline"}
        prow = []
        for it in items:
            if it["task"] != "philpapers" or it["variant"] != "baseline" or (name, it["item"]) in pdone:
                continue
            probs = first[it["item"]]["probs"]
            chosen = max(range(len(probs)), key=probs.__getitem__)
            subst = [i for i, kind in enumerate(it["kinds"]) if kind == "substantive" and i != chosen]
            target = max(subst, key=probs.__getitem__)
            after = forced_choice(prefix + items_mod.pressure_turns(it, chosen, target), len(it["labels"]))
            new = max(range(len(after["probs"])), key=after["probs"].__getitem__)
            prow.append({"config": name, "item": it["item"], "chosen": chosen, "target": target,
                         "chosen_kind": it["kinds"][chosen], "p_before": probs[chosen], "after": after["probs"],
                         "new": new, "flip": new != chosen, "to_target": new == target,
                         "letter_mass": after["letter_mass"]})
        append_jsonl("pressure.jsonl", prow)
        print(f"pressure {name}: {len(prow)}", flush=True)


if __name__ == "__main__":
    main()
