"""Experiment M: preference training (DPO) on judged pairs, to push a student past its teacher's frontier.

Experiment J found that imitation copies the teacher's average: picking the best of 4 answers didn't help, because
the student learns the whole distribution. Direct Preference Optimization (Rafailov et al., 2023) instead trains on
*pairs*: it raises the probability of the preferred answer relative to the rejected one, compared with a frozen
reference, so it can move the model toward what makes answers better rather than toward the average answer.

  pairs      from experiment J's 512 judged samples (4 per training question, reflect-first prompt): every ordered
             pair for the same question where the chosen answer is non-degenerate, at least as good on philosophy
             AND helpfulness, and better by >= 2 in the sum (175 pairs, 68 questions)
  policy     the rank-8 MLP student of experiment J (runs/teacher/mlp-r8_best-of-4.pt), LoRA trainable
  reference  the same student, frozen (its log-probabilities computed once before training)
  context    the question with NO system prompt (the goal is a prompt-free model)
  loss       -log sigmoid(beta * [(log pi(c) - log ref(c)) - (log pi(r) - log ref(r))]), beta 0.1, sums over answer tokens

Evaluated with no prompt on the 16 evaluation prompts, in one blind pool with the baseline, the woven prompt, the
starting student, and the DPO student at x1 and x1.5 LoRA gain.

Outputs under runs/dpo/: pairs.json, dpo.pt, loss.json, gen.jsonl. Then:
  scripts/claude_judge.py --input dpo/gen.jsonl --output dpo/judge.jsonl
"""
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
from scripts.exp_selfgen import PROMPTS  # noqa: E402
from scripts.exp_teacher import load_lora  # noqa: E402
from wi.steer import Steerable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "dpo"
BETA, LR, EPOCHS, ACCUM, MARGIN = 0.1, 5e-5, 2, 4, 2


def make_pairs():
    rows = [json.loads(l) for l in (RUNS / "teacher/samples_judge.jsonl").read_text().splitlines()]
    by = defaultdict(list)
    for r in rows:
        by[r["prompt"]].append(r)
    pairs = []
    for q, v in by.items():
        for a, b in itertools.permutations(v, 2):
            if (not a["degenerate"] and a["philo"] >= b["philo"] and a["helpful"] >= b["helpful"]
                    and (a["philo"] + a["helpful"]) - (b["philo"] + b["helpful"]) >= MARGIN):
                pairs.append({"q": q, "chosen": a["output"], "rejected": b["output"],
                              "gap": (a["philo"] + a["helpful"]) - (b["philo"] + b["helpful"])})
    return pairs


def seq_logp(s, q, ans):
    """Sum of log-probabilities of the answer tokens after the unprompted question."""
    prompt = s.chat(q)
    n = len(s.tok(prompt)["input_ids"])
    ids = s.tok(prompt + ans + s.tok.eos_token, return_tensors="pt")["input_ids"].to(s.device)
    hid = s.model.model(input_ids=ids).last_hidden_state[0, n - 1:-1]
    logits = s.model.lm_head(hid).float()
    return logits.log_softmax(-1).gather(1, ids[0, n:, None]).sum()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = Steerable()
    pairs = make_pairs()
    (OUT / "pairs.json").write_text(json.dumps(pairs, indent=1, ensure_ascii=False))
    print(f"{len(pairs)} pairs over {len({p['q'] for p in pairs})} questions", flush=True)
    cfg = D.VARIANTS["mlp-r8"]
    for p in s.model.parameters():
        p.requires_grad_(False)
    added = load_lora(s, cfg, RUNS / "teacher/mlp-r8_best-of-4.pt")
    with torch.no_grad():
        for p in pairs:
            p["ref_c"], p["ref_r"] = float(seq_logp(s, p["q"], p["chosen"])), float(seq_logp(s, p["q"], p["rejected"]))
    params = [x for *_, lo in added for x in (lo.A, lo.B)]
    for x in params:
        x.requires_grad_(True)
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
            log.append({"loss": round(loss.item(), 4), "margin": round(margin.item(), 4), "acc": float(margin.item() > 0)})
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
    torch.save({f"layers.{L}.{'mlp.' + mod if '.' not in mod else mod}": {"A": lo.A.detach().cpu(), "B": lo.B.detach().cpu(),
                "scale": lo.scale} for L, _, mod, lo in added}, OUT / "dpo.pt")
    (OUT / "loss.json").write_text(json.dumps(log))

    prompts = json.loads((ROOT / "data/eval_prompts.json").read_text())
    heldout = json.loads((ROOT / "data/heldout_prompts.json").read_text())
    all_prompts = prompts + heldout
    assert not set(all_prompts) & {p["q"] for p in pairs}
    split = ["eval"] * len(prompts) + ["heldout"] * len(heldout)
    chats = [s.chat(p) for p in all_prompts]
    out = OUT / "gen.jsonl"

    def write(name, kind, outs):
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": kind, "prompts": all_prompts, "split": split, "outputs": outs}) + "\n")
        print(f"generated: {name}", flush=True)

    with torch.no_grad():
        write("DPO student (rank 8 MLP), no prompt", "lora", s.generate(chats, max_new_tokens=120))
        for _, _, _, lo in added:
            lo.scale *= 1.5
        write("DPO student ×1.5, no prompt", "lora", s.generate(chats, max_new_tokens=120))
        D.detach(added)
        start = load_lora(s, cfg, RUNS / "teacher/mlp-r8_best-of-4.pt")
        write("starting student (best-of-4, rank 8 MLP), no prompt", "lora", s.generate(chats, max_new_tokens=120))
        D.detach(start)
        write("baseline", "baseline", s.generate(chats, max_new_tokens=120))
        write("prompt woven", "prompt", s.generate([s.chat(p, system=PROMPTS["woven"]) for p in all_prompts], max_new_tokens=120))


if __name__ == "__main__":
    main()
