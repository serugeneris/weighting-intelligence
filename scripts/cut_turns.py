"""Cut generated answers at the end of the assistant's turn (FINDINGS section 29).

Until the fix in wi/steer.py, generate() did not stop at <|im_end|> (the model config's eos is <|endoftext|>), so an
answer that ended before the token limit ran on into an invented next turn: "...keep old accounts open.\\nuser\\nWhat is
the best way...". The tokens before the turn end do not depend on when generation stops (greedy or sampled), so
cutting a saved output at the first "\\nuser\\n" / "\\nassistant\\n" gives exactly what a correct harness would have
returned.

  scripts/cut_turns.py runs/<exp>/gen.jsonl            -> runs/<exp>/gen_cut.jsonl (prints how many were cut per config)
"""
import json
import re
import sys
from pathlib import Path

TURN = re.compile(r"\n(user|assistant)\n")


def cut(o):
    m = TURN.search(o)
    return (o[:m.start()].rstrip(), True) if m else (o, False)


def main():
    src = Path(sys.argv[1])
    dst = src.with_name(src.stem + "_cut" + src.suffix)
    with dst.open("w") as f:
        for line in src.read_text().splitlines():
            d = json.loads(line)
            res = [cut(o) for o in d["outputs"]]
            d["outputs"] = [o for o, _ in res]
            n = sum(c for _, c in res)
            print(f"{d['name'][:60]:60s} cut {n}/{len(res)}")
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
