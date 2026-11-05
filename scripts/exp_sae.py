"""Experiment T: train a small sparse autoencoder (SAE) at layer 8 and steer with its features (LITERATURE.md H6).

STATUS: drafted, NOT RUN YET (listed under "To do" in FINDINGS.md). Untested code: check each stage before trusting it.

There is no public SAE for Qwen3.5-4B, so we train one. An SAE rewrites a hidden state h (2560 numbers) as a sparse
sum of learned feature directions:  h ~ b + sum_i z_i d_i  with only k of the 8192 z_i non-zero (TopK SAE). The
hope (2505.20063) is that some feature means "philosophical register" more cleanly than the diff-means direction.

  stage acts   residual stream after layer 8 on ~2,000 texts this project already generated (all runs/*gen*.jsonl,
               plain, reflective and broken answers, with their chat prompts), up to 400k tokens, fp16 on disk.
               Tokens with very large norm (attention-sink positions, > 4x the median) are dropped.
  stage train  TopK SAE, 8192 latents, k = 32, inputs centred and scaled to unit mean norm; Adam, batch 4096.
  stage steer  features ranked by mean activation on philosophical minus plain contrast answers (32 pairs, answer
               tokens). Steering at L8 along the top feature's decoder direction, and along the activation-weighted
               sum of the top 8, at 0.5 / 0.8 p50; one blind pool with diff-means alpha 0.5 and the learned direction
               B at 0.6 p50 (experiment L), 16 evaluation prompts.

Outputs under runs/sae/: acts.pt, sae.pt, train.json, features.json, gen.jsonl. Then:
  scripts/claude_judge.py --input sae/gen.jsonl --output sae/judge.jsonl
"""
import argparse
import glob
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "sae"
L, N_LAT, K, MAX_TOK = 8, 8192, 32, 400_000


class TopKSAE(nn.Module):
    def __init__(self, d, n, k):
        super().__init__()
        self.k = k
        self.enc = nn.Linear(d, n)
        self.dec = nn.Parameter(self.enc.weight.data.T.clone())  # tied init, untied training
        self.b_dec = nn.Parameter(torch.zeros(d))
        with torch.no_grad():
            self.dec.data /= self.dec.data.norm(dim=1, keepdim=True)

    def encode(self, x):
        pre = self.enc(x - self.b_dec)
        v, i = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, i, v.relu())

    def forward(self, x):
        z = self.encode(x)
        return z @ self.dec + self.b_dec, z


def stage_acts(s):
    f = OUT / "acts.pt"
    if f.exists():
        return
    texts = []
    for fn in sorted(glob.glob(str(RUNS / "**/*gen*.jsonl"), recursive=True)):
        for line in open(fn):
            d = json.loads(line)
            sysmsg = None
            for p, o in zip(d["prompts"], d["outputs"]):
                texts.append(s.chat(p, system=sysmsg) + o)
    random.Random(0).shuffle(texts)
    print(f"{len(texts)} texts available", flush=True)
    chunks, n = [], 0
    cap = {}
    h = s.layers[L].register_forward_hook(lambda _m, _i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o)))
    try:
        with torch.no_grad():
            for i in range(0, len(texts), 8):
                s.tok.padding_side = "right"
                enc = s.tok(texts[i:i + 8], return_tensors="pt", padding=True, truncation=True, max_length=256).to(s.device)
                s.model(**enc)
                x = cap["h"][enc["attention_mask"].bool()].float()
                nrm = x.norm(dim=-1)
                x = x[nrm < 4 * nrm.median()]
                chunks.append(x.half().cpu())
                n += len(x)
                if n >= MAX_TOK:
                    break
    finally:
        h.remove()
    acts = torch.cat(chunks)[:MAX_TOK]
    torch.save(acts, f)
    print(f"saved {len(acts)} tokens", flush=True)


def stage_train(device):
    acts = torch.load(OUT / "acts.pt").float()
    mu = acts.mean(0)
    scale = (acts - mu).norm(dim=-1).mean()
    X = ((acts - mu) / scale).to(device)
    sae = TopKSAE(X.shape[1], N_LAT, K).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=2e-4)
    log, t0, bs = [], time.time(), 4096
    steps = 3000
    for step in range(steps):
        idx = torch.randint(0, len(X), (bs,), device=device)
        x = X[idx]
        xh, z = sae(x)
        loss = ((xh - x) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            sae.dec.data /= sae.dec.data.norm(dim=1, keepdim=True)
        if step % 250 == 0 or step == steps - 1:
            with torch.no_grad():
                xv = X[:20000]
                xh, z = sae(xv)
                fvu = float(((xh - xv) ** 2).sum() / ((xv - xv.mean(0)) ** 2).sum())
                alive = int((z > 0).any(0).sum())
            log.append({"step": step, "loss": round(loss.item(), 4), "fvu": round(fvu, 4), "alive": alive})
            print(f"  step {step} loss {loss.item():.4f} FVU {fvu:.3f} alive {alive}/{N_LAT} ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"state": sae.state_dict(), "mu": mu, "scale": scale}, OUT / "sae.pt")
    (OUT / "train.json").write_text(json.dumps(log, indent=1))


def stage_steer(s):
    ck = torch.load(OUT / "sae.pt")
    sae = TopKSAE(2560, N_LAT, K)
    sae.load_state_dict(ck["state"])
    sae = sae.to(s.device).eval()
    mu, scale = ck["mu"].to(s.device), ck["scale"].to(s.device)
    pairs = json.loads((ROOT / "data/contrast_pairs.json").read_text())
    feats = {"philo": [], "plain": []}
    with torch.no_grad():
        for p in pairs:
            for tag, key in (("philo", "philosophical"), ("plain", "plain")):
                prompt = s.chat(p["q"])
                ids = s.tok(prompt + p[key], return_tensors="pt")["input_ids"].to(s.device)
                n = len(s.tok(prompt)["input_ids"])
                cap = {}
                h = s.layers[L].register_forward_hook(lambda _m, _i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o)))
                s.model(input_ids=ids)
                h.remove()
                x = (cap["h"][0, n:].float() - mu) / scale
                feats[tag].append(sae.encode(x).mean(0))
    P, Q = torch.stack(feats["philo"]), torch.stack(feats["plain"])
    diff = (P - Q).mean(0)
    consist = ((P - Q) > 0).float().mean(0)
    top = diff.argsort(descending=True)[:20].tolist()
    p50 = json.loads((RUNS / "p50.json").read_text())[str(L)]
    dm = torch.load(RUNS / "diffmeans.pt")[L].float().to(s.device)
    B = torch.load(RUNS / "distill/down8-r1.pt")[f"layers.{L}.mlp.down_proj"]["B"][:, 0].float().to(s.device)
    unit = lambda v: v / v.norm()
    D = sae.dec.data                                         # decoder rows, in the scaled space: direction is what matters
    info = [{"feature": i, "mean diff": round(float(diff[i]), 4), "consistency": round(float(consist[i]), 3),
             "cos with diff-means": round(float(unit(D[i]) @ unit(dm)), 3), "cos with B": round(float(unit(D[i]) @ unit(B)), 3)}
            for i in top]
    w = diff[top[:8]].clamp(min=0)
    top8 = unit((w[:, None] * D[top[:8]]).sum(0))
    info_extra = {"cos(top-8 combo, diff-means)": round(float(top8 @ unit(dm)), 3), "cos(top-8 combo, B)": round(float(top8 @ unit(B)), 3)}
    (OUT / "features.json").write_text(json.dumps({"top": info, **info_extra}, indent=1))
    print(json.dumps(info[:8], indent=0), info_extra, flush=True)

    prompts = json.loads((ROOT / "data/eval_prompts.json").read_text())
    heldout = json.loads((ROOT / "data/heldout_prompts.json").read_text())
    all_prompts = prompts + heldout
    split = ["eval"] * len(prompts) + ["heldout"] * len(heldout)
    chats = [s.chat(p) for p in all_prompts]
    cfgs = [("baseline", None), ("diff-means α0.5", 0.5 * p50 * unit(dm)), ("learned direction B 0.6·p50", 0.6 * p50 * unit(B))]
    for a in (0.5, 0.8):
        cfgs.append((f"SAE top-1 feature {top[0]} {a}·p50", a * p50 * unit(D[top[0]])))
        cfgs.append((f"SAE top-8 combination {a}·p50", a * p50 * top8))
    out = OUT / "gen.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()
    for name, v in cfgs:
        if name in done:
            continue
        with s.steer([], {L: v.cpu()} if v is not None else None):
            outs = s.generate(chats, max_new_tokens=120)
        with out.open("a") as f:
            f.write(json.dumps({"name": name, "kind": "baseline" if v is None else "steer", "prompts": all_prompts,
                                "split": split, "outputs": outs}) + "\n")
        print(f"generated: {name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["acts", "train", "steer"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "train":
        from wi.steer import pick_device
        stage_train(pick_device())
        return
    from wi.steer import Steerable
    s = Steerable()
    stage_acts(s) if args.stage == "acts" else stage_steer(s)


if __name__ == "__main__":
    main()
