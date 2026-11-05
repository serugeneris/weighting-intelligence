"""How does the rank-1 LoRA at L8 (experiment D, down8-r1) push? Forward passes only.

For the 16 evaluation prompts, with the model's own baseline answer (runs/distill/gen.jsonl) and with the LoRA's
answer, record the LoRA's added output  scale * B (A a)  at every token (a = the 9216 MLP activations at L8) and report
its size relative to p50 and its cosine with the diff-means direction, separately for prompt and answer tokens.
Output: runs/distill/probe.json
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wi.steer import Steerable  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs"


@torch.no_grad()
def main():
    s = Steerable()
    p50 = json.loads((RUNS / "p50.json").read_text())["8"]
    lo = torch.load(RUNS / "distill/down8-r1.pt")["layers.8.mlp.down_proj"]
    A, B, sc = lo["A"].to(s.device).float(), lo["B"].to(s.device).float(), lo["scale"]
    d = torch.load(RUNS / "diffmeans.pt")[8].float().to(s.device)
    d = d / d.norm()
    Bu = (B[:, 0] / B.norm()).float()
    gens = {json.loads(l)["name"]: json.loads(l) for l in (RUNS / "distill/gen.jsonl").read_text().splitlines()}
    res = {"cos(B, diffmeans L8)": float(Bu @ d), "p50": p50}
    for which in ("baseline", "LoRA down8-r1, no prompt"):
        g = gens[which]
        rows = {"prompt": [], "answer": [], "answer first 10": []}
        for p, o in zip(g["prompts"], g["outputs"]):
            prompt = s.chat(p)
            n = len(s.tok(prompt)["input_ids"])
            ids = s.tok(prompt + o, return_tensors="pt")["input_ids"].to(s.device)
            cap = {}
            h = s.down[8].register_forward_pre_hook(lambda _m, a: cap.__setitem__("a", a[0][0].float()))
            s.model(input_ids=ids)
            h.remove()
            z = cap["a"] @ A[0] * sc * B.norm()          # signed size of the push along unit(B), per token
            rows["prompt"].append(z[:n])
            rows["answer"].append(z[n:])
            rows["answer first 10"].append(z[n:n + 10])
        res[which] = {k: {"mean |push|/p50": float(torch.cat(v).abs().mean() / p50),
                          "mean push/p50 (signed)": float(torch.cat(v).mean() / p50),
                          "share of tokens with |push| > 0.1 p50": float((torch.cat(v).abs() > 0.1 * p50).float().mean())}
                      for k, v in rows.items()}
    (RUNS / "distill/probe.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
