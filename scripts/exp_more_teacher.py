"""Experiment P: more teacher data. Does a 3x larger set of Claude-written answers close the student-teacher gap?

Experiment O's student (128 Claude-written answers) matched the woven prompt's philosophy only at 1.5x gain, and
stayed far below its teacher (Claude's answers: philosophy ~2.5 with helpfulness ~8.8). Here Claude writes 256 new
everyday practical questions (data/train_questions_extra.json, checked against every existing prompt set) and
answers them with the same instruction as experiment O. A rank-8 MLP LoRA is trained on all 384 answers, exactly as
before, and compared with experiment O's 128-answer student on the 48 confirmation prompts, in one pool with the
baseline and the woven prompt, scored by Sonnet and Opus.

  stage data    questions + answers  -> data/train_questions_extra.json, runs/more_teacher/targets.json
  stage train   student + generations -> runs/more_teacher/gen.jsonl
  stage gains   the 384-answer student broke at x1.5, so x1.0 / 1.2 / 1.3 are compared in a second pool with the
                baseline, the woven prompt and the 128-answer student x1.5 -> runs/more_teacher/gen_gains.jsonl
Then:
  scripts/claude_judge.py --input more_teacher/gen.jsonl --output more_teacher/judge.jsonl
  scripts/claude_judge.py --input more_teacher/gen.jsonl --output more_teacher/judge_opus.jsonl --model opus
  (and the same two commands for gen_gains.jsonl -> judge_gains.jsonl / judge_gains_opus.jsonl)
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.exp_distill as D  # noqa: E402
from scripts.claude_judge import call_claude, parse  # noqa: E402
from scripts.exp_claude_teacher import INSTRUCT  # noqa: E402
from scripts.exp_selfgen import PROMPTS  # noqa: E402
from scripts.exp_teacher import load_lora  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "more_teacher"
QFILE = ROOT / "data/train_questions_extra.json"
TOPICS = ["cooking and food", "home repair and cleaning", "gardening and plants", "personal finance basics",
          "travel and packing", "computers and phones", "fitness and everyday health habits", "work and study skills",
          "writing emails and messages", "cars and bikes", "pets", "hobbies and crafts", "parenting and family logistics",
          "science explanations for curious adults", "shopping and comparing products", "moving and organizing"]


def existing():
    s = set()
    for f in ("eval_prompts", "heldout_prompts", "confirm_prompts", "train_questions"):
        s |= {q.strip().lower() for q in json.loads((ROOT / f"data/{f}.json").read_text())}
    s |= {p["q"].strip().lower() for p in json.loads((ROOT / "data/contrast_pairs.json").read_text())}
    return s


def stage_data():
    if not QFILE.exists():
        seen, qs = existing(), []
        for topic in TOPICS:
            prompt = (f"Write 20 distinct, everyday practical questions a person might ask an assistant about {topic}. "
                      "Each is one sentence, specific, answerable in under 100 words, with no personal data and no "
                      "philosophy. Return ONLY a JSON array of strings.")
            got = parse(call_claude(prompt, "sonnet"))
            fresh = [q.strip() for q in got if isinstance(q, str) and q.strip().lower() not in seen]
            for q in fresh[:16]:
                seen.add(q.lower())
                qs.append(q)
            print(f"  {topic}: {len(fresh[:16])}", flush=True)
        QFILE.write_text(json.dumps(qs, indent=1, ensure_ascii=False))
    qs = json.loads(QFILE.read_text())
    f = OUT / "targets.json"
    if f.exists():
        return
    data = {}
    for i in range(0, len(qs), 16):
        batch = {f"q{j}": q for j, q in enumerate(qs[i:i + 16], start=i)}
        prompt = INSTRUCT + "\n".join(json.dumps({"id": k, "request": v}) for k, v in batch.items())
        for attempt in range(3):
            try:
                rows = parse(call_claude(prompt, "sonnet"))
                got = {r["id"]: r["answer"].strip() for r in rows if r.get("id") in batch}
                assert len(got) == len(batch)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  retry {attempt + 1}: {e}", flush=True)
        data.update({batch[k]: v for k, v in got.items()})
        print(f"  answers {len(data)}/{len(qs)}", flush=True)
    f.write_text(json.dumps([{"q": q, "target": data[q]} for q in qs], indent=1, ensure_ascii=False))


def stage_train():
    from wi.steer import Steerable
    s = Steerable()
    data = json.loads((RUNS / "claude_teacher/targets.json").read_text()) + json.loads((OUT / "targets.json").read_text())
    prompts = json.loads((ROOT / "data/confirm_prompts.json").read_text())
    assert not {p.lower() for p in prompts} & {d["q"].lower() for d in data}
    print(f"{len(data)} training answers", flush=True)
    plain = [s.chat(p) for p in prompts]
    out = OUT / "gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()

    def gen(chats):
        o = []
        for i in range(0, len(chats), 16):
            o += s.generate(chats[i:i + 16], max_new_tokens=120)
        return o

    def write(name, kind, outs):
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": kind, "prompts": prompts, "split": ["heldout"] * len(prompts),
                                "outputs": outs}) + "\n")
        print(f"generated: {name}", flush=True)

    if "baseline" not in done:
        write("baseline", "baseline", gen(plain))
    if "prompt woven" not in done:
        write("prompt woven", "prompt", gen([s.chat(p, system=PROMPTS["woven"]) for p in prompts]))
    for gain in (1.0, 1.5):
        name = f"128-answer student (exp O) ×{gain}"
        if name not in done:
            added = load_lora(s, D.VARIANTS["mlp-r8"], RUNS / "claude_teacher/mlp-r8_claude.pt")
            for *_, lo in added:
                lo.scale *= gain
            with torch.no_grad():
                write(name, "lora", gen(plain))
            D.detach(added)
    if "384-answer student ×1.0" not in done:
        D.OUT = OUT
        added = D.train(s, data, "mlp-r8_claude384", D.VARIANTS["mlp-r8"])
        with torch.no_grad():
            write("384-answer student ×1.0", "lora", gen(plain))
            for *_, lo in added:
                lo.scale *= 1.5
            write("384-answer student ×1.5", "lora", gen(plain))
        D.detach(added)


def stage_gains():
    from wi.steer import Steerable
    s = Steerable()
    prompts = json.loads((ROOT / "data/confirm_prompts.json").read_text())
    plain = [s.chat(p) for p in prompts]
    out = OUT / "gen_gains.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()

    def gen(chats):
        o = []
        for i in range(0, len(chats), 16):
            o += s.generate(chats[i:i + 16], max_new_tokens=120)
        return o

    runs = [("baseline", None, 1.0), ("prompt woven", "prompt", 1.0),
            ("128-answer student ×1.5", RUNS / "claude_teacher/mlp-r8_claude.pt", 1.5)]
    runs += [(f"384-answer student ×{g}", OUT / "mlp-r8_claude384.pt", g) for g in (1.0, 1.2, 1.3)]
    for name, src, g in runs:
        if name in done:
            continue
        with torch.no_grad():
            if src is None:
                outs = gen(plain)
            elif src == "prompt":
                outs = gen([s.chat(p, system=PROMPTS["woven"]) for p in prompts])
            else:
                added = load_lora(s, D.VARIANTS["mlp-r8"], src)
                for *_, lo in added:
                    lo.scale *= g
                outs = gen(plain)
                D.detach(added)
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": "x", "prompts": prompts, "split": ["heldout"] * len(prompts),
                                "outputs": outs}) + "\n")
        print(f"generated: {name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["data", "train", "gains"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"data": stage_data, "train": stage_train, "gains": stage_gains}[args.stage]()


if __name__ == "__main__":
    main()
