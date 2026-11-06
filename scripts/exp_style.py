"""Experiment W: style or content? What the Claude-taught student actually learns from its teacher's answers.

The 384 Claude-written answers of experiment P are mostly a practical answer followed by one or two reflective
sentences. Claude (Sonnet) splits each answer into its practical part (reflective sentences removed, nothing else
changed) and the removed reflection, verbatim. Three rank-8 MLP students are trained exactly as in P on rebuilt
answers:
  practical only        the practical part alone: what do the reflections add, and do they cost helpfulness?
  own reflection        practical part + its own reflection appended (the matched control, built by the same pipeline)
  mismatched reflection practical part + the reflection written for a *different* question (fixed derangement)
If "mismatched" scores like "own", the student learns a register (a reflective coda of any kind), not reflection
tied to the task; if it is clearly worse on philosophy or helpfulness, the content binding matters and data
quality is a lever. One blind pool, 48 confirmation prompts, with the baseline, the woven prompt and P's original
student; Sonnet + Opus.

  stage split  -> runs/style/split.json      (Claude, no GPU)
  stage train  -> runs/style/mlp-r8_<variant>.pt, gen.jsonl
Then:
  scripts/claude_judge.py --input style/gen.jsonl --output style/judge.jsonl   (and --model opus -> judge_opus.jsonl)
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.exp_distill as D  # noqa: E402
from scripts.claude_judge import call_claude, parse  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "style"
SPLIT = """Each item below is a short answer to a user's practical request. Most answers end (or are interspersed) with
one or two reflective, philosophical sentences. Split each answer into:
- "practical": the answer with every reflective/philosophical sentence removed and NOTHING else changed (keep all
  practical sentences verbatim, in order; do not add words);
- "reflection": the removed sentences, verbatim, in order ("" if there are none).
Return ONLY a JSON array: [{"id": "...", "practical": "...", "reflection": "..."}, ...]

ITEMS:
"""


def teacher():
    return json.loads((RUNS / "claude_teacher/targets.json").read_text()) + json.loads((RUNS / "more_teacher/targets.json").read_text())


def stage_split():
    f = OUT / "split.json"
    if f.exists():
        return
    data = teacher()
    res = {}
    for i in range(0, len(data), 16):
        batch = {f"a{j}": d["target"] for j, d in enumerate(data[i:i + 16], start=i)}
        prompt = SPLIT + "\n".join(json.dumps({"id": k, "answer": v}) for k, v in batch.items())
        for attempt in range(3):
            try:
                rows = parse(call_claude(prompt, "sonnet"))
                got = {r["id"]: r for r in rows if r.get("id") in batch}
                assert len(got) == len(batch)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  retry {attempt + 1}: {e}", flush=True)
        res.update(got)
        print(f"  split {len(res)}/{len(data)}", flush=True)
    out = []
    for j, d in enumerate(data):
        r = res[f"a{j}"]
        out.append({"q": d["q"], "target": d["target"], "practical": r["practical"].strip(), "reflection": r["reflection"].strip()})
    f.write_text(json.dumps(out, indent=1, ensure_ascii=False))


def variants():
    sp = json.loads((OUT / "split.json").read_text())
    keep = [d for d in sp if d["reflection"] and d["practical"]]
    rng = random.Random(0)
    perm = list(range(len(keep)))
    while any(i == p for i, p in enumerate(perm)):                     # a derangement: no answer keeps its own
        rng.shuffle(perm)
    return keep, {
        "practical only": [{"q": d["q"], "target": d["practical"]} for d in keep],
        "own reflection": [{"q": d["q"], "target": d["practical"] + " " + d["reflection"]} for d in keep],
        "mismatched reflection": [{"q": d["q"], "target": d["practical"] + " " + keep[p]["reflection"]} for d, p in zip(keep, perm)],
    }


def stage_train():
    from scripts.exp_selfgen import PROMPTS
    from scripts.exp_teacher import load_lora
    from wi.steer import Steerable
    s = Steerable()
    keep, V = variants()
    stats = {"answers": len(teacher()), "with reflection and practical part": len(keep),
             "mean words": {k: round(sum(len(d["target"].split()) for d in v) / len(v), 1) for k, v in V.items()}}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(stats, flush=True)
    prompts = json.loads((ROOT / "data/confirm_prompts.json").read_text())
    chats = [s.chat(p) for p in prompts]
    out = OUT / "gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()

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
        if "baseline" not in done:
            write("baseline", gen(chats))
        if "prompt woven" not in done:
            write("prompt woven", gen([s.chat(p, system=PROMPTS["woven"]) for p in prompts]))
        if "original answers (exp P student) ×1.0" not in done:
            added = load_lora(s, D.VARIANTS["mlp-r8"], RUNS / "more_teacher/mlp-r8_claude384.pt")
            write("original answers (exp P student) ×1.0", gen(chats))
            D.detach(added)
    D.OUT = OUT
    for name, data in V.items():
        if f"{name} ×1.2" in done:
            continue
        added = D.train(s, data, "mlp-r8_" + name.replace(" ", "_"), D.VARIANTS["mlp-r8"])
        with torch.no_grad():
            write(f"{name} ×1.0", gen(chats))
            for *_, lo in added:
                lo.scale *= 1.2
            write(f"{name} ×1.2", gen(chats))
        D.detach(added)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["split", "train"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stage_split() if args.stage == "split" else stage_train()


if __name__ == "__main__":
    main()
