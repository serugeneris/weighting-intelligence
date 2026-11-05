"""Minimal neuron-steering toolkit for Qwen3.5-4B (or any HF decoder with mlp.down_proj).

Steering convention (same as neuron-explorer / EvilQwen control vectors):
    push(L, n, dose) adds  dose * p50_L * unit(W_down_L[:, n])  to the MLP output of layer L
at every token, where p50_L is the median residual-stream norm leaving layer L on neutral text.
This is equivalent to adding a constant to neuron n's activation; stored weights are untouched.
"""
from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-4B"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Push:
    layer: int
    neuron: int
    dose: float


class Steerable:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self.device = device or pick_device()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(self.device).eval()
        self.layers: dict[int, torch.nn.Module] = {}
        self.down: dict[int, torch.nn.Linear] = {}
        for name, mod in self.model.named_modules():
            if "visual" in name or "mtp" in name:
                continue
            m = re.search(r"layers\.(\d+)$", name)
            if m:
                self.layers[int(m.group(1))] = mod
            m = re.search(r"layers\.(\d+)\.mlp\.down_proj$", name)
            if m:
                self.down[int(m.group(1))] = mod
        assert self.down, "no mlp.down_proj modules found"
        self.n_layers = len(self.down)
        self.n_neurons = self.down[0].weight.shape[1]
        self.p50: dict[int, float] = {}

    # ---------- prompt helpers ----------
    def chat(self, user: str, assistant: str | None = None, system: str | None = None) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        prompt = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return prompt + (assistant or "")

    # ---------- calibration ----------
    @torch.no_grad()
    def calibrate(self, texts: list[str]) -> dict[int, float]:
        norms: dict[int, list[torch.Tensor]] = {i: [] for i in self.layers}
        hooks = []
        for i, layer in self.layers.items():
            def hook(_m, _inp, out, i=i):
                h = out[0] if isinstance(out, tuple) else out
                norms[i].append(h[0].float().norm(dim=-1).cpu())
            hooks.append(layer.register_forward_hook(hook))
        try:
            for t in texts:
                ids = self.tok(t, return_tensors="pt").to(self.device)
                self.model(**ids)
        finally:
            for h in hooks:
                h.remove()
        self.p50 = {i: torch.cat(v).median().item() for i, v in norms.items()}
        return self.p50

    # ---------- activation capture ----------
    @torch.no_grad()
    def neuron_acts(self, text: str, start_char: int = 0, layers: list[int] | None = None) -> dict[int, torch.Tensor]:
        """Mean MLP neuron activation (input to down_proj) over tokens at/after start_char."""
        layers = layers if layers is not None else sorted(self.down)
        enc = self.tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0]
        mask = (offs[:, 0] >= start_char).to(self.device)
        enc = enc.to(self.device)
        out: dict[int, torch.Tensor] = {}
        hooks = []
        for i in layers:
            def pre(_m, inp, i=i):
                a = inp[0][0].float()
                out[i] = a[mask].mean(0).cpu()
            hooks.append(self.down[i].register_forward_pre_hook(pre))
        try:
            self.model(**enc)
        finally:
            for h in hooks:
                h.remove()
        return out

    @torch.no_grad()
    def resid_mean(self, text: str, start_char: int, layers: list[int]) -> dict[int, torch.Tensor]:
        enc = self.tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0]
        mask = (offs[:, 0] >= start_char).to(self.device)
        enc = enc.to(self.device)
        out: dict[int, torch.Tensor] = {}
        hooks = []
        for i in layers:
            def hook(_m, _inp, o, i=i):
                h = (o[0] if isinstance(o, tuple) else o)[0].float()
                out[i] = h[mask].mean(0).cpu()
            hooks.append(self.layers[i].register_forward_hook(hook))
        try:
            self.model(**enc)
        finally:
            for h in hooks:
                h.remove()
        return out

    @torch.no_grad()
    def capture(self, text: str, start_char: int, resid_layers: list[int], act_layers: list[int]
                ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """One forward pass: mean residual (layer output) and mean MLP neuron activation over tokens
        at/after start_char. Respects any active steer() hooks."""
        enc = self.tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0]
        mask = (offs[:, 0] >= start_char).to(self.device)
        enc = enc.to(self.device)
        res: dict[int, torch.Tensor] = {}
        act: dict[int, torch.Tensor] = {}
        hooks = []
        for i in resid_layers:
            def hook(_m, _inp, o, i=i):
                h = (o[0] if isinstance(o, tuple) else o)[0].float()
                res[i] = h[mask].mean(0).cpu()
            hooks.append(self.layers[i].register_forward_hook(hook))
        for i in act_layers:
            def pre(_m, inp, i=i):
                act[i] = inp[0][0].float()[mask].mean(0).cpu()
            hooks.append(self.down[i].register_forward_pre_hook(pre))
        try:
            self.model(**enc)
        finally:
            for h in hooks:
                h.remove()
        return res, act

    # ---------- steering ----------
    def neuron_vector(self, p: Push) -> torch.Tensor:
        col = self.down[p.layer].weight[:, p.neuron].float()
        return p.dose * self.p50[p.layer] * col / col.norm()

    @contextlib.contextmanager
    def steer(self, pushes: list[Push] = (), vectors: dict[int, torch.Tensor] | None = None):
        """Add neuron pushes and/or raw residual vectors (keyed by layer) to MLP outputs."""
        add: dict[int, torch.Tensor] = {}
        for p in pushes:
            add[p.layer] = add.get(p.layer, 0) + self.neuron_vector(p)
        for L, v in (vectors or {}).items():
            add[L] = add.get(L, 0) + v.float()
        hooks = []
        for L, v in add.items():
            v = v.to(self.device, torch.bfloat16)
            hooks.append(self.down[L].register_forward_hook(lambda _m, _i, o, v=v: o + v))
        try:
            yield
        finally:
            for h in hooks:
                h.remove()

    @torch.no_grad()
    def generate(self, prompts: list[str], max_new_tokens: int = 120, temperature: float = 0.0, seed: int = 0) -> list[str]:
        self.tok.padding_side = "left"
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        torch.manual_seed(seed)
        kw = dict(do_sample=temperature > 0, max_new_tokens=max_new_tokens, repetition_penalty=1.1,
                  pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        if temperature > 0:
            kw.update(temperature=temperature, top_p=0.9)
        out = self.model.generate(**enc, **kw)
        return self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    # ---------- scoring ----------
    @torch.no_grad()
    def judge(self, texts: list[str], question: str) -> list[float]:
        """Expected 0-9 rating from the *unsteered* model's next-digit distribution."""
        digit_ids = [self.tok.convert_tokens_to_ids(str(d)) for d in range(10)]
        scores = []
        for t in texts:
            p = self.chat(f"{question}\n\nText:\n\"\"\"\n{t.strip()[:1500]}\n\"\"\"\n\nAnswer with a single digit from 0 to 9.")
            ids = self.tok(p, return_tensors="pt").to(self.device)
            logits = self.model(**ids).logits[0, -1, digit_ids].float()
            probs = logits.softmax(-1)
            scores.append(float((probs * torch.arange(10, device=probs.device)).sum()))
        return scores

    @torch.no_grad()
    def ppl(self, prompt: str, completion: str) -> float:
        """Perplexity of a completion under the unsteered model (coherence proxy)."""
        enc = self.tok(prompt + completion, return_tensors="pt", return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0]
        enc = enc.to(self.device)
        logits = self.model(**enc).logits[0, :-1].float()
        tgt = enc["input_ids"][0, 1:]
        nll = torch.nn.functional.cross_entropy(logits, tgt, reduction="none")
        m = (offs[1:, 0] >= len(prompt)).to(self.device)
        return float(nll[m].mean().exp()) if m.any() else float("nan")


PHILO_Q = ("How philosophical is this text? 0 = purely practical/factual, no reflection; "
           "9 = deeply philosophical (questions of meaning, existence, knowledge, ethics, mind, time).")
COHERENCE_Q = ("How coherent and well-formed is this text as English prose? 0 = gibberish, loops or broken "
               "fragments; 9 = fluent, grammatical and on-topic.")
