"""Reproduce the prose and validity pilot."""

import contextlib

import torch

from scripts.exp_sae import K, L, N_LAT, TopKSAE, resid
from scripts.philosophy_pilot import RUNS, residual_edit, run, save


def sae_cases(items):
    rows = []
    for index, (title, valid, conclusion, invalid) in enumerate(items):
        for elaborate in (False, True):
            for sound in (False, True):
                premises = valid if sound else invalid
                argument = (f"Consider, then, the following: {premises} Are we not thereby compelled to "
                            f"affirm that {conclusion}?" if elaborate else
                            f"Take these as the premises: {premises} The stated conclusion is that {conclusion}.")
                valid_first = index % 2 == 0
                options = ["follows", "does not follow"]
                if not valid_first:
                    options.reverse()
                question = ("From the stated premises alone, does the conclusion follow? "
                            f"Reply with one letter. A = {options[0]}; B = {options[1]}.\n\n")
                rows.append({"case": title, "index": index, "elaborate": elaborate, "sound": sound,
                             "question": question, "argument": argument,
                             "expected": "A" if sound == valid_first else "B"})
    return rows


def sae(model, items):
    ckpt = torch.load(RUNS / "sae/sae.pt", map_location="cpu")
    autoencoder = TopKSAE(len(ckpt["mu"]), N_LAT, K)
    autoencoder.load_state_dict(ckpt["state"])
    autoencoder = autoencoder.to(model.device).eval()
    mu, scale = ckpt["mu"].to(model.device), ckpt["scale"].to(model.device)
    rows = sae_cases(items)
    means = []
    for row in rows:
        p, a = row["question"], row["argument"]
        text = model.chat(p + a)
        start = text.index(p + a) + len(p)
        enc = model.tok(text, return_tensors="pt", return_offsets_mapping=True)
        off = enc.pop("offset_mapping")[0]
        mask = ((off[:, 0] >= start) & (off[:, 0] < start + len(a))).to(model.device)
        h = resid(model, enc.to(model.device))[0, mask].float()
        means.append(autoencoder.encode((h - mu) / scale).mean(0).cpu())
    # Rank within discovery cases, then measure the fixed features on held-out cases.
    features = torch.stack(means).reshape(len(items), 2, 2, N_LAT)
    split = len(items) // 2
    contrasts = {"style": features[:, 1].mean(1) - features[:, 0].mean(1),
                 "soundness": features[:, :, 1].mean(1) - features[:, :, 0].mean(1)}
    summary = {}
    for factor, contrast in contrasts.items():
        mean = contrast[:split].mean(0)
        top = mean.abs().topk(3).indices.tolist()
        summary[factor] = [{"feature": k, "discovery_delta": float(mean[k]),
                            "heldout_delta": float(contrast[split:, k].mean())} for k in top]
    selected = {factor: group[0]["feature"] for factor, group in summary.items()}
    selected["stock_phrase_7238"] = 7238
    heldout = [r for r in rows if r["index"] >= split]
    model.tok.padding_side = "left"
    ids = [model.tok.encode(c, add_special_tokens=False) for c in ("A", "B")]
    assert all(len(t) == 1 for t in ids)
    ids = [t[0] for t in ids]
    for label, latent in {"none": None, **selected}.items():
        print("sae", label, flush=True)
        seen = 0

        def remove(h):
            nonlocal seen
            hf = h.float()
            activations = autoencoder.encode((hf - mu) / scale)[..., latent]
            seen += int((activations > 0).sum())
            return (hf - activations[..., None] * autoencoder.dec[latent] * scale).to(h.dtype)

        hook = residual_edit(model, L, remove) if latent is not None else contextlib.nullcontext()
        with hook:
            for start in range(0, len(heldout), 4):
                batch = heldout[start:start + 4]
                prompts = [model.chat(r["question"] + r["argument"]) for r in batch]
                enc = model.tok(prompts, padding=True, return_tensors="pt").to(model.device)
                logits = model.model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, ids].float().tolist()
                for row, (a, b) in zip(batch, logits):
                    row.setdefault("scores", {})[label] = {
                        "choice": "A" if a > b else "B",
                        "margin": (a - b) * (1 if row["expected"] == "A" else -1)}
        summary.setdefault("ablation", {})[label] = {"feature": latent, "active_positions": seen}
    save("sae.json", {"summary": summary, "rows": rows})


if __name__ == "__main__":
    run({"sae": sae}, "data/pilot-prose-validity.json")
