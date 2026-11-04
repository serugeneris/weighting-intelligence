"""Blind external judge: score runs/validate.jsonl outputs with Claude via the `claude -p` CLI.

All (config, prompt) outputs are pooled, shuffled and given anonymous ids, so the judge never sees which
config produced what. Only the user prompt and the generated text are sent. Scores per item:
  philo 0-9, coherence 0-9, helpful 0-9 (addresses the user's request), degenerate true/false.
Writes runs/claude_judge.jsonl (resumable by batch) and prints an aggregate table.
"""
import argparse
import json
import random
import re
import statistics as st
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

RUBRIC = """You are grading assistant responses from a language-model experiment. Each item has the user's request and the
response (responses are truncated at ~120 tokens, so don't penalize abrupt endings). Score every item independently:

- philo (0-9): how much the response engages in genuine philosophical reflection (meaning, existence, knowledge,
  ethics, mind, time, value). 0 = purely practical/factual. 9 = substantive philosophical thinking.
  Mere repetition of words like "being" or "truth", or incoherent mystical-sounding fragments, is NOT philosophy.
- coherence (0-9): fluent, grammatical, logically connected prose. 0 = gibberish/loops; 9 = fully coherent.
- helpful (0-9): how well it actually addresses the user's request. 0 = ignores it; 9 = fully useful answer.
- degenerate (true/false): repetition loops, gibberish, or identity confusion (speaking as if it were the user,
  or confused about who/what it is).

Return ONLY a JSON array, one object per item, in any order:
[{"id": "...", "philo": n, "coherence": n, "helpful": n, "degenerate": true|false}, ...]

ITEMS:
"""


def call_claude(prompt: str, model: str) -> str:
    r = subprocess.run(
        ["claude", "-p", "--model", model, "--output-format", "json", "--no-session-persistence",
         "--tools", "", "--strict-mcp-config"],
        input=prompt, capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])
    return json.loads(r.stdout)["result"]


def parse(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    return json.loads(m.group(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-batches", type=int, default=0, help="judge only the first N batches (smoke test)")
    ap.add_argument("--input", default="validate.jsonl")
    ap.add_argument("--output", default="claude_judge.jsonl")
    args = ap.parse_args()

    recs = [json.loads(l) for l in (RUNS / args.input).read_text().splitlines() if l.strip()]
    items = []
    for r in recs:
        for i, (p, o, sp) in enumerate(zip(r["prompts"], r["outputs"], r["split"])):
            items.append({"cfg": r["name"], "kind": r["kind"], "matches": r.get("matches"), "i": i, "split": sp,
                          "prompt": p, "output": o})
    random.Random(7).shuffle(items)
    for j, it in enumerate(items):
        it["id"] = f"x{j:04d}"
    batches = [items[i:i + args.batch] for i in range(0, len(items), args.batch)]
    if args.limit_batches:
        batches = batches[:args.limit_batches]

    out = RUNS / args.output
    scored = {}
    if out.exists():
        for l in out.read_text().splitlines():
            d = json.loads(l)
            scored[d["id"]] = d
    todo = [b for b in batches if not all(it["id"] in scored for it in b)]
    print(f"{len(items)} items, {len(batches)} batches, {len(todo)} to judge with {args.model}")

    def run_batch(b):
        body = "\n\n".join(f'### id={it["id"]}\nUSER: {it["prompt"]}\nRESPONSE: """{it["output"].strip()}"""' for it in b)
        for attempt in range(3):
            try:
                res = {d["id"]: d for d in parse(call_claude(RUBRIC + body, args.model))}
                missing = [it["id"] for it in b if it["id"] not in res]
                if missing:
                    raise ValueError(f"missing {len(missing)} ids")
                return [{**it, **{k: res[it["id"]][k] for k in ("philo", "coherence", "helpful", "degenerate")}}
                        for it in b]
            except Exception as e:  # noqa: BLE001
                print(f"  batch retry {attempt + 1}: {e}", flush=True)
        return []

    with ThreadPoolExecutor(args.workers) as ex:
        for rows in ex.map(run_batch, todo):
            with out.open("a") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
                    scored[row["id"]] = row
            print(f"  judged batch ({len(scored)}/{len(items)})", flush=True)

    aggregate(list(scored.values()))


def aggregate(rows):
    by = {}
    for r in rows:
        by.setdefault(r["cfg"], []).append(r)

    def stats(rs):
        return {"philo": st.mean(r["philo"] for r in rs), "coh": st.mean(r["coherence"] for r in rs),
                "help": st.mean(r["helpful"] for r in rs), "degen": sum(bool(r["degenerate"]) for r in rs), "n": len(rs)}

    print(f"\n{'config':<30}{'philo':>7}{'coh':>6}{'help':>6}{'degen':>8}   heldout philo/coh/degen")
    order = sorted(by, key=lambda c: (c.startswith("ctrl"), c))
    for c in order:
        s, h = stats(by[c]), stats([r for r in by[c] if r["split"] == "heldout"] or by[c])
        print(f"{c:<30}{s['philo']:>7.2f}{s['coh']:>6.2f}{s['help']:>6.2f}{s['degen']:>5}/{s['n']:<3}"
              f"   {h['philo']:.2f}/{h['coh']:.2f}/{h['degen']}")

    print("\nGroup vs its matched random controls (philo, on non-degenerate items):")
    for c in order:
        if not c.startswith("omp"):
            continue
        g = [r for r in by[c] if not r["degenerate"]]
        ctrl = [r for k, v in by.items() if k.endswith(f"of {c}") for r in v if not r["degenerate"]]
        if g and ctrl:
            gp, cp = st.mean(r["philo"] for r in g), st.mean(r["philo"] for r in ctrl)
            print(f"  {c:<26} group={gp:.2f} (n={len(g)})  controls={cp:.2f} (n={len(ctrl)})  Δ={gp - cp:+.2f}")


if __name__ == "__main__":
    main()
