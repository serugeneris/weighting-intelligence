"""Blind human regrade of the revision and transfer pilot (experiment AC).

  python3 scripts/human_regrade.py build                 -> docs/reviews/human-regrade/{packet.json,key.json,grade.html}
  python3 scripts/human_regrade.py score GRADES.json     -> agreement with the author and Codex grades

`build` shuffles the 18 revision and 18 transfer records with a fixed seed and writes them without the
configuration name. packet.json is what the grader sees; key.json maps each packet id back to configuration and
case and must stay away from the grader. `score` takes the file the grading page exports and compares it, label
by label, with the paired grades of experiment AK. Standard library only.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs/reviews/2026-11-14/revision-transfer"
PAIRED = ROOT / "docs/reviews/2026-11-16/codex-regrade/grades.jsonl"
OUT = ROOT / "docs/reviews/human-regrade"
PACKET_ID = "ac-human-regrade-v1"
SEED = 20261117

CRITERIA = {
    "revision": ("targeted_question", "relevant_fact", "irrelevant_fact"),
    "transfer": ("transfers_core_distinction",),
}
# what the model was asked at each step, as in scripts/exp_revision_transfer.py
ASK_QUESTION = "Ask me exactly one question that probes its most consequential unstated assumption."
ASK_FACT = "Say whether this fact changes the force of the original argument and why, in at most two sentences."
ASK_CONCEPT = "Give a reusable distinction that clarifies the original argument. Use at most 45 words."
ASK_TRANSFER = "Apply the same distinction to this fresh case: {case}. Sort both sides in at most 45 words."


def kappa(a, b):
    """Cohen's kappa for two lists of booleans; None when chance agreement is 1 (no variation)."""
    n = len(a)
    if n == 0 or n != len(b):
        raise ValueError("need two non-empty lists of equal length")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def _rows(name):
    return [json.loads(line) for line in (SRC / name).read_text().splitlines() if line.strip()]


def _turn(part):
    return {"text": part["answer"], "cut_off": part["stop"] == "budget"}


def build_packet():
    """Return (packet, key). Deterministic: same inputs and SEED give the same ids and order."""
    rng = random.Random(SEED)
    items, key = [], {}
    for set_name, prefix, rows in (("revision", "R", _rows("revision.jsonl")),
                                   ("transfer", "T", _rows("transfer.jsonl"))):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for n, i in enumerate(order, 1):
            r, item_id = rows[i], f"{prefix}{n:02d}"
            item = {"id": item_id, "set": set_name, "argument": r["argument"], "assumption": r["reference"],
                    "question": _turn(r["question"])}
            if set_name == "revision":
                item.update(relevant_fact=r["relevant_fact"], relevant=_turn(r["relevant"]),
                            irrelevant_fact=r["irrelevant_fact"], irrelevant=_turn(r["irrelevant"]))
            else:
                item.update(relevant_fact=r["relevant_fact"], concept=_turn(r["concept"]),
                            transfer_case=r["transfer_case"], transfer=_turn(r["transfer"]))
            items.append(item)
            key[item_id] = {"set": set_name, "configuration": r["config"], "case": r["case"]}
    packet = {"packet": PACKET_ID, "seed": SEED, "criteria": CRITERIA,
              "prompts": {"question": ASK_QUESTION, "fact": ASK_FACT, "concept": ASK_CONCEPT,
                          "transfer": ASK_TRANSFER},
              "items": items}
    return packet, key


def write_packet():
    packet, key = build_packet()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "packet.json").write_text(json.dumps(packet, indent=1, ensure_ascii=False) + "\n")
    (OUT / "key.json").write_text(json.dumps(key, indent=1) + "\n")
    # the grading page is the template with the packet inlined; "</" is escaped so the JSON cannot end the script tag
    inline = json.dumps(packet, ensure_ascii=False).replace("</", "<\\/")
    template = (OUT / "grade.template.html").read_text()
    (OUT / "grade.html").write_text(template.replace("__PACKET__", inline))
    print(f"{len(packet['items'])} items -> {OUT}")


def paired_grades():
    """{(set, configuration, case number): {"author": {...}, "codex": {...}}} from experiment AK."""
    case_no = {r["argument"]: r["case"] for r in _rows("revision.jsonl")}
    out = {}
    for line in PAIRED.read_text().splitlines():
        g = json.loads(line)
        if g["set"] not in CRITERIA:
            continue
        out[(g["set"], g["configuration"], g["source_line"])] = {"author": g["original"], "codex": g["codex"]}
    # source_line is the one-based line in the response file; turn it into the case number
    by_line = {}
    for set_name in CRITERIA:
        for n, r in enumerate(_rows(f"{set_name}.jsonl"), 1):
            by_line[(set_name, n)] = (r["config"], case_no[r["argument"]])
    fixed = {}
    for (set_name, config, line), v in out.items():
        cfg, case = by_line[(set_name, line)]
        if cfg != config:
            raise ValueError(f"{set_name} line {line}: grades say {config}, responses say {cfg}")
        fixed[(set_name, config, case)] = v
    return fixed


def score(grades, key=None, paired=None):
    """Compare an exported grades file with the author and Codex labels. Returns a dict of results."""
    key = key or json.loads((OUT / "key.json").read_text())
    paired = paired or paired_grades()
    if grades.get("packet") != PACKET_ID:
        raise ValueError(f"grades are for packet {grades.get('packet')!r}, expected {PACKET_ID!r}")
    labels = {}       # criterion -> list of (human, author, codex, configuration)
    missing = []
    for item_id, k in key.items():
        g = grades["grades"].get(item_id) or {}
        ref = paired[(k["set"], k["configuration"], k["case"])]
        for c in CRITERIA[k["set"]]:
            if not isinstance(g.get(c), bool):
                missing.append(f"{item_id}.{c}")
                continue
            labels.setdefault(c, []).append((g[c], ref["author"][c], ref["codex"][c], k["configuration"]))
    result = {"grader": grades.get("grader"), "missing": missing, "criteria": {}}
    for c, rows in labels.items():
        h, a, x = [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]
        passes = {}
        for cfg in ("base", "woven", "student"):
            sub = [r for r in rows if r[3] == cfg]
            passes[cfg] = {"human": sum(r[0] for r in sub), "author": sum(r[1] for r in sub),
                           "codex": sum(r[2] for r in sub), "n": len(sub)}
        result["criteria"][c] = {
            "n": len(rows),
            "agree_author": sum(p == q for p, q in zip(h, a)), "kappa_author": kappa(h, a),
            "agree_codex": sum(p == q for p, q in zip(h, x)), "kappa_codex": kappa(h, x),
            "passes": passes,
        }
    return result


def _fmt(k):
    return "  n/a" if k is None else f"{k:5.2f}"


def main(argv):
    if len(argv) == 2 and argv[1] == "build":
        write_packet()
    elif len(argv) == 3 and argv[1] == "score":
        res = score(json.loads(Path(argv[2]).read_text()))
        print(f"grader: {res['grader'] or '(not given)'}")
        if res["missing"]:
            print(f"{len(res['missing'])} labels not graded: {', '.join(res['missing'])}")
        print(f"{'criterion':28s} {'n':>3s}  {'= author':>8s} {'kappa':>5s}  {'= codex':>8s} {'kappa':>5s}")
        for c, r in res["criteria"].items():
            print(f"{c:28s} {r['n']:3d}  {r['agree_author']:5d}/{r['n']:<2d} {_fmt(r['kappa_author'])}"
                  f"  {r['agree_codex']:5d}/{r['n']:<2d} {_fmt(r['kappa_codex'])}")
        print("\npasses per configuration (human / author / codex)")
        for c, r in res["criteria"].items():
            cells = "   ".join(f"{cfg} {p['human']}/{p['author']}/{p['codex']} of {p['n']}"
                               for cfg, p in r["passes"].items())
            print(f"{c:28s} {cells}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
