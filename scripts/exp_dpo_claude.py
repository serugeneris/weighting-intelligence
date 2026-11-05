"""Experiment S: preference training (DPO) on top of the best student, the Claude-taught one.

The two levers that moved the frontier were a stronger teacher (experiments O-P) and preference pairs (M). Here they
are combined. Experiment P's rank-8 student (384 Claude-written answers, runs/more_teacher/mlp-r8_claude384.pt)
answers 192 training questions 4 times each with no prompt (temperature 0.8, 160 tokens); the 768 answers are
judged blind; dominance pairs (same filter as experiment M) train the student with DPO (reference: the same student,
frozen). Evaluated on the 48 confirmation prompts against the starting student and the woven prompt, Sonnet + Opus.

  stage samples  -> runs/dpo_claude/samples_gen.jsonl, then judge it (samples_judge.jsonl)
  stage train    -> runs/dpo_claude/dpo.pt, gen.jsonl
Then:
  scripts/claude_judge.py --input dpo_claude/gen.jsonl --output dpo_claude/judge.jsonl  (and --model opus -> judge_opus.jsonl)
"""
import argparse
import itertools
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.exp_distill as D  # noqa: E402
from scripts.exp_dpo import ACCUM, BETA, EPOCHS, LR, MARGIN, seq_logp  # noqa: E402
from scripts.exp_selfgen import PROMPTS  # noqa: E402
from scripts.exp_teacher import cut, load_lora  # noqa: E402
from wi.steer import Steerable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "dpo_claude"
START = RUNS / "more_teacher/mlp-r8_claude384.pt"
N_Q, N_S = 192, 4


def train_questions():
    a = json.loads((RUNS / "claude_teacher/targets.json").read_text())
    b = json.loads((RUNS / "more_teacher/targets.json").read_text())
    return [d["q"] for d in a + b][:N_Q]


def stage_samples(s):
    qs = train_questions()
    out = OUT / "samples_gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()
    added = load_lora(s, D.VARIANTS["mlp-r8"], START)
    with torch.no_grad():
        for k in range(N_S):
            if f"sample {k}" in done:
                continue
            outs = []
            for i in range(0, len(qs), 32):
                outs += s.generate([s.chat(q) for q in qs[i:i + 32]], max_new_tokens=160, temperature=0.8, seed=100 * k + i)
            with out.open("a") as f:
                f.write(json.dumps({"name": f"sample {k}", "kind": "sample", "prompts": qs, "split": ["eval"] * len(qs),
                                    "outputs": [cut(o) for o in outs]}) + "\n")
            print(f"sample {k} done", flush=True)
    D.detach(added)


def make_pairs():
    rows = [json.loads(l) for l in (OUT / "samples_judge.jsonl").read_text().splitlines()]
    by = defaultdict(list)
    for r in rows:
        by[r["prompt"]].append(r)
    pairs = []
    for q, v in by.items():
        for a, b in itertools.permutations(v, 2):
            if (not a["degenerate"] and a["philo"] >= b["philo"] and a["helpful"] >= b["helpful"]
                    and (a["philo"] + a["helpful"]) - (b["philo"] + b["helpful"]) >= MARGIN):
                pairs.append({"q": q, "chosen": a["output"], "rejected": b["output"]})
    return pairs


def stage_train(s):
    pairs = make_pairs()
    (OUT / "pairs.json").write_text(json.dumps(pairs, indent=1, ensure_ascii=False))
    print(f"{len(pairs)} pairs over {len({p['q'] for p in pairs})} questions", flush=True)
    for p in s.model.parameters():
        p.requires_grad_(False)
    added = load_lora(s, D.VARIANTS["mlp-r8"], START)
    with torch.no_grad():
        for p in pairs:
            p["ref_c"], p["ref_r"] = float(seq_logp(s, p["q"], p["chosen"])), float(seq_logp(s, p["q"], p["rejected"]))
    params = [x for *_, lo in added for x in (lo.A, lo.B)]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    s.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    s.model.train(True)
    rng, log, t0, step = random.Random(0), [], time.time(), 0
    for ep in range(EPOCHS):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        for i, k in enumerate(order):
            p = pairs[k]
            with torch.enable_grad():
                lc, lr_ = seq_logp(s, p["q"], p["chosen"]), seq_logp(s, p["q"], p["rejected"])
                margin = BETA * ((lc - p["ref_c"]) - (lr_ - p["ref_r"]))
                loss = -nn.functional.logsigmoid(margin)
                (loss / ACCUM).backward()
            log.append({"loss": round(loss.item(), 4), "acc": float(margin.item() > 0)})
            if (i + 1) % ACCUM == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    w = log[-40:]
                    print(f"  ep {ep} step {step} loss {sum(x['loss'] for x in w) / len(w):.3f} "
                          f"reward-acc {sum(x['acc'] for x in w) / len(w):.2f} ({time.time() - t0:.0f}s)", flush=True)
    s.model.train(False)
    s.model.gradient_checkpointing_disable()
    torch.save({f"layers.{L}.mlp.{mod}": {"A": lo.A.detach().cpu(), "B": lo.B.detach().cpu(), "scale": lo.scale}
                for L, _, mod, lo in added}, OUT / "dpo.pt")
    (OUT / "loss.json").write_text(json.dumps(log))

    prompts = json.loads((ROOT / "data/confirm_prompts.json").read_text())
    chats = [s.chat(p) for p in prompts]
    out = OUT / "gen.jsonl"

    def gen(c):
        o = []
        for i in range(0, len(c), 16):
            o += s.generate(c[i:i + 16], max_new_tokens=120)
        return o

    def write(name, outs):
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": "x", "prompts": prompts, "split": ["heldout"] * len(prompts), "outputs": outs}) + "\n")
        print(f"generated: {name}", flush=True)

    with torch.no_grad():
        write("DPO on Claude-taught student", gen(chats))
        for *_, lo in added:
            lo.scale *= 1.2
        write("DPO on Claude-taught student ×1.2", gen(chats))
        D.detach(added)
        start = load_lora(s, D.VARIANTS["mlp-r8"], START)
        write("Claude-taught student (start)", gen(chats))
        D.detach(start)
        write("baseline", gen(chats))
        write("prompt woven", gen([s.chat(p, system=PROMPTS["woven"]) for p in prompts]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["samples", "train"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    s = Steerable()
    stage_samples(s) if args.stage == "samples" else stage_train(s)


if __name__ == "__main__":
    main()
