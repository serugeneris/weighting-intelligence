"""Experiment AN, stage 0: does Qwen3.5-4B have a J-space? (gate for the J-lens experiments)

Replicates, at small scale, the checks of arXiv 2607.15495 that a workspace exists before any of the project's
questions are asked of it:

  corpus      pretraining-like contexts: windows of the public-domain texts in data/sources plus free samples
              of the base model from varied seeds (fit set and a disjoint held-out set)
  fit         the averaged Jacobian J_l for every layer (wi/jlens.py), checkpointed so it can resume
  signatures  per layer, J-lens against the logit lens: top-10 next-token accuracy, excess kurtosis of the readout,
              top-1 autocorrelation over positions against a shuffled baseline (paper section 4.1)
  twohop      two-hop prompts (data/jlens_twohop.json): is the unspoken intermediate in the lens, and does swapping its
              lens coordinates for a same-category alternative move the answer (paper section 3.3)? Controls: the same
              swap with logit-lens vectors, and a swap between two uninvolved same-category tokens
  ablate      project out the top-10 J-lens vectors per position over a band of layers (paper section 3.5.2):
              two-hop accuracy against next-token agreement on held-out text, with a random-subspace control

Gate: the experiments that use the lens go ahead only if the swap moves two-hop answers well above both controls and
ablation hurts two-hop accuracy more than ordinary next-token prediction. Outputs go to runs/jlens/.
"""
import argparse
import json
import random
from pathlib import Path

import torch

from wi.jlens import JLens, capture, decoder_parts, edit_layers, excess_kurtosis, fit
from wi.steer import MODEL_ID, Steerable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/jlens"
SEEDS = [
    "The recipe calls for", "In 1987, the company", "def parse_args(", "Dear Sir or Madam,", "The patient presented with",
    "Breaking news:", "Chapter 3\n\n", "The mitochondria", "According to the survey,", "Once upon a time",
    "import numpy as np\n", "The court ruled that", "Step 1:", "Q: How do I", "The weather in", "Abstract. We",
    "The best way to learn", "She opened the door and", "Product description:", "The history of the Roman",
    "To install the package,", "In the second half,", "The following table shows", "My favorite thing about",
    "The theorem states that", "Welcome to our", "The election results", "Ingredients:\n-", "He was born in",
    "The main reason why", "Error: could not", "Thanks for your email.",
]


def sample_corpus(model, n_fit, n_heldout, length, seed=0):
    rng = random.Random(seed)
    tok = model.tok
    windows = []
    for path in sorted((ROOT / "data/sources").glob("*.txt")):
        ids = tok(path.read_text(errors="ignore"), return_tensors="pt")["input_ids"][0]
        for _ in range((n_fit + n_heldout) // 2 // 8 + 2):
            start = rng.randrange(0, max(1, ids.shape[0] - length))
            windows.append(ids[start:start + length])
    rng.shuffle(windows)
    sources = windows[:(n_fit + n_heldout) // 2]
    samples = []
    torch.manual_seed(seed)
    need = n_fit + n_heldout - len(sources)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        while len(samples) < need:
            prompts = [rng.choice(SEEDS) for _ in range(8)]
            tok.padding_side = "left"
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            out = model.model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=length,
                                       pad_token_id=pad)
            for row, mask in zip(out, enc["attention_mask"]):
                ids = row[int((mask == 0).sum()):].cpu()   # drop left padding
                if ids.shape[0] >= length:
                    samples.append(ids[:length])
    contexts = sources + samples[:need]
    rng.shuffle(contexts)
    return {"fit": contexts[:n_fit], "heldout": contexts[n_fit:], "length": length}


def stage_corpus(model, args):
    corpus = sample_corpus(model, args.n_fit, args.n_heldout, args.length)
    torch.save(corpus, OUT / "corpus.pt")
    preview = [model.tok.decode(c) for c in corpus["heldout"][:4]]
    print(f"corpus: {len(corpus['fit'])} fit, {len(corpus['heldout'])} held out, {args.length} tokens each")
    for p in preview:
        print("  |", p[:160].replace("\n", " "))


def stage_fit(model, args):
    corpus = torch.load(OUT / "corpus.pt")
    contexts = corpus["fit"][:args.n_fit]
    ckpt = OUT / "fit_checkpoint.pt"
    state = torch.load(ckpt) if ckpt.exists() else {"done": 0, "J_sum": None, "pairs": 0}
    print(f"fitting on {len(contexts)} contexts, resuming after {state['done']}", flush=True)
    while state["done"] < len(contexts):
        chunk = contexts[state["done"]:state["done"] + args.chunk]
        J = fit(model.model, chunk, copies=args.copies)
        T = chunk[0].shape[0]
        pairs = len(chunk) * T * (T + 1) // 2
        state["J_sum"] = J * pairs if state["J_sum"] is None else state["J_sum"] + J * pairs
        state["pairs"] += pairs
        state["done"] += len(chunk)
        torch.save(state, ckpt)
        print(f"checkpoint: {state['done']}/{len(contexts)}", flush=True)
    J = state["J_sum"] / state["pairs"]
    torch.save(J, OUT / "J.pt")
    print("saved", OUT / "J.pt", tuple(J.shape))


def lenses(model):
    return {"jlens": JLens.load(model.model, OUT / "J.pt"), "logit": JLens.logit_lens(model.model)}


@torch.no_grad()
def stage_signatures(model, args):
    corpus = torch.load(OUT / "corpus.pt")
    contexts = corpus["heldout"][:args.n_sig]
    layers_all, _, _ = decoder_parts(model.model)
    layers = list(range(0, len(layers_all), args.layer_step))
    if len(layers_all) - 1 not in layers:
        layers.append(len(layers_all) - 1)
    stats = {name: {l: {"top10": [], "kurt": [], "auto": [], "auto_null": []} for l in layers}
             for name in ("jlens", "logit")}
    lens = lenses(model)
    gen = torch.Generator().manual_seed(0)
    for n, ids in enumerate(contexts, 1):
        with capture(layers_all) as acts:
            out = model.model(input_ids=ids[None].to(model.device), use_cache=False)
        top1 = out.logits[0].argmax(-1)
        for name, L in lens.items():
            for l in layers:
                logits = L.logits(l, acts[l][0])
                top10 = logits.topk(10, -1).indices
                s = stats[name][l]
                s["top10"].append((top10 == top1[:, None]).any(-1).float().mean().item())
                s["kurt"].append(excess_kurtosis(logits).mean().item())
                t1 = top10[:, 0].cpu()
                d = args.lag
                s["auto"].append((t1[:-d] == t1[d:]).float().mean().item())
                perm = t1[torch.randperm(t1.shape[0], generator=gen)]
                s["auto_null"].append((t1 == perm).float().mean().item())
        print(f"  signatures: {n}/{len(contexts)}", flush=True)
    summary = {name: {l: {k: sum(v) / len(v) for k, v in s.items()} for l, s in per.items()}
               for name, per in stats.items()}
    (OUT / "signatures.json").write_text(json.dumps(summary, indent=1))
    print(f"{'layer':>5} | {'J top10':>7} {'J kurt':>7} {'J auto-null':>11} | {'LL top10':>8} {'LL kurt':>7} "
          f"{'LL auto-null':>12}")
    for l in layers:
        j, g = summary["jlens"][l], summary["logit"][l]
        print(f"{l:>5} | {j['top10']:7.3f} {j['kurt']:7.1f} {j['auto'] - j['auto_null']:11.3f} | "
              f"{g['top10']:8.3f} {g['kurt']:7.1f} {g['auto'] - g['auto_null']:12.3f}")


def first_ids(tok, words):
    return sorted({tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words})


def single_id(tok, word):
    ids = tok(" " + word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None


def load_items(tok):
    items = json.loads((ROOT / "data/jlens_twohop.json").read_text())["items"]
    pool = {}
    for it in items:
        pool.setdefault(it["template"].split("-")[0], set()).update({it["intermediate"], it["swap_to"]})
    kept = []
    for it in items:
        s, t = single_id(tok, it["intermediate"]), single_id(tok, it["swap_to"])
        if s is None or t is None:
            print(f"  skip {it['id']}: intermediate or swap target is not one token")
            continue
        others = sorted(w for w in pool[it["template"].split("-")[0]]
                        if w not in (it["intermediate"], it["swap_to"]) and single_id(tok, w) is not None)
        a, b = random.Random(it["id"]).sample(others, 2)
        kept.append({**it, "s": s, "t": t, "ans": first_ids(tok, it["answers"]),
                     "alt": first_ids(tok, it["swap_answers"]), "ctrl": (single_id(tok, a), single_id(tok, b))})
    return kept


@torch.no_grad()
def last_logits(model, prompt, edits=None):
    ids = model.tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    layers, _, _ = decoder_parts(model.model)
    with edit_layers(layers, edits or {}):
        return model.model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()


def parse_band(text):
    a, b = (int(x) for x in text.split("-"))
    return list(range(a, b + 1))


@torch.no_grad()
def stage_twohop(model, args):
    items = load_items(model.tok)
    lens = lenses(model)
    layers_all, _, _ = decoder_parts(model.model)
    band = parse_band(args.band)
    rows = []
    for it in items:
        clean = last_logits(model, it["prompt"])
        row = {"id": it["id"], "template": it["template"], "clean_top1": model.tok.decode(clean.argmax()),
               "clean_correct": int(clean.argmax()) in it["ans"]}
        # presence: best rank of the intermediate over positions, per layer, for both lenses
        ids = model.tok(it["prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
        with capture(layers_all) as acts:
            model.model(input_ids=ids, use_cache=False, logits_to_keep=1)
        for name, L in lens.items():
            ranks = []
            for l in range(len(layers_all)):
                logits = L.logits(l, acts[l][0])
                rank = (logits > logits[:, it["s"]:it["s"] + 1]).sum(-1)
                ranks.append(int(rank.min()))
            row[f"rank_{name}"] = ranks
        for name, L in lens.items():
            for alpha in args.alphas:
                swapped = last_logits(model, it["prompt"],
                                      {l: L.swap_edit(l, it["s"], it["t"], alpha) for l in band})
                row[f"swap_{name}_a{alpha}"] = int(swapped.argmax()) in it["alt"]
                row[f"swap_{name}_a{alpha}_top1"] = model.tok.decode(swapped.argmax())
            a, b = it["ctrl"]
            ctrl = last_logits(model, it["prompt"], {l: L.swap_edit(l, a, b, max(args.alphas)) for l in band})
            row[f"ctrl_{name}_unchanged"] = int(ctrl.argmax()) == int(clean.argmax())
        rows.append(row)
        print(f"  {it['id']}: clean {row['clean_top1']!r} ok={row['clean_correct']} "
              f"swap J={row[f'swap_jlens_a{max(args.alphas)}_top1']!r} "
              f"LL={row[f'swap_logit_a{max(args.alphas)}_top1']!r}", flush=True)
    with (OUT / f"twohop_{args.band}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in rows if r["clean_correct"]]
    print(f"\nclean correct: {len(ok)}/{len(rows)}; band {band[0]}-{band[-1]}")
    for name in lens:
        for alpha in args.alphas:
            k = sum(r[f"swap_{name}_a{alpha}"] for r in ok)
            print(f"  {name:6} swap alpha {alpha}: {k}/{len(ok)} answers moved to the swapped intermediate's answer")
        k = sum(r[f"ctrl_{name}_unchanged"] for r in ok)
        print(f"  {name:6} control swap (uninvolved tokens): answer unchanged on {k}/{len(ok)}")
    for name in lens:
        best = [min(r[f"rank_{name}"][l] for l in band) for r in ok]
        print(f"  {name:6} intermediate in top-25 at some band layer: {sum(b < 25 for b in best)}/{len(ok)}")


def ablation_edit(L, layer, exclude, k, random_gen=None):
    """Project out, at every position, the k most active J-lens vectors (excluding the clean top-10 output tokens)."""
    def edit(h):
        x = h[0]
        if random_gen is not None:
            V = torch.randn(x.shape[0], k, x.shape[-1], generator=random_gen).to(x.device)
        else:
            logits = L.logits(layer, x)
            logits.scatter_(1, exclude, float("-inf"))
            top = logits.topk(k, -1).indices                              # (T, k)
            V = L.vectors(layer, top.flatten()).view(x.shape[0], k, -1)
        return L.project_out(x, V)[None]
    return edit


@torch.no_grad()
def stage_ablate(model, args):
    lens = lenses(model)["jlens"]
    layers_all, _, _ = decoder_parts(model.model)
    band = parse_band(args.band)
    third = max(1, len(band) // 3)
    strengths = {"light": band[:third], "medium": band[:2 * third], "heavy": band}
    items = [it for it in load_items(model.tok)]
    corpus = torch.load(OUT / "corpus.pt")["heldout"][:args.n_sig]

    def run(ids, layers, mode):
        clean = model.model(input_ids=ids, use_cache=False).logits[0].float()
        exclude = clean.topk(10, -1).indices
        gen = torch.Generator().manual_seed(0) if mode == "random" else None
        edits = {l: ablation_edit(lens, l, exclude, args.k, gen) for l in layers} if mode != "clean" else {}
        with edit_layers(layers_all, edits):
            out = model.model(input_ids=ids, use_cache=False).logits[0].float()
        return clean, out

    results = {}
    for strength, layers in strengths.items():
        for mode in ("jspace", "random"):
            correct = total = 0
            for it in items:
                ids = model.tok(it["prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"]
                clean, out = run(ids.to(model.device), layers, mode)
                if int(clean[-1].argmax()) in it["ans"]:
                    total += 1
                    correct += int(out[-1].argmax()) in it["ans"]
            agree = n = 0
            for ids in corpus:
                clean, out = run(ids[None].to(model.device), layers, mode)
                agree += (clean.argmax(-1) == out.argmax(-1)).sum().item()
                n += clean.shape[0]
            results[f"{strength}/{mode}"] = {"layers": layers, "twohop": correct / max(total, 1),
                                             "twohop_n": total, "next_token_agreement": agree / n}
            r = results[f"{strength}/{mode}"]
            print(f"  {strength:6} {mode:6} layers {layers[0]}-{layers[-1]}: two-hop {correct}/{total}, "
                  f"next-token agreement {r['next_token_agreement']:.3f}", flush=True)
    (OUT / f"ablate_{args.band}.json").write_text(json.dumps(results, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", nargs="+", default=["corpus", "fit", "signatures"],
                        choices=["corpus", "fit", "signatures", "twohop", "ablate"])
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--n-fit", type=int, default=128)
    parser.add_argument("--n-heldout", type=int, default=64)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--copies", type=int, default=64, help="backward passes packed per batch in the fit")
    parser.add_argument("--chunk", type=int, default=8, help="contexts per fit checkpoint")
    parser.add_argument("--n-sig", type=int, default=16, help="held-out contexts for signatures and ablation")
    parser.add_argument("--layer-step", type=int, default=1)
    parser.add_argument("--lag", type=int, default=4)
    parser.add_argument("--band", default="11-28", help="layers for swaps and ablation, as first-last")
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model = Steerable(args.model)
    print(f"Device: {model.device}, layers: {model.n_layers}", flush=True)
    stages = {"corpus": stage_corpus, "fit": stage_fit, "signatures": stage_signatures,
              "twohop": stage_twohop, "ablate": stage_ablate}
    for s in args.stage:
        stages[s](model, args)


if __name__ == "__main__":
    main()
