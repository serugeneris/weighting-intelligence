"""Jacobian lens (J-lens) for a decoder language model.

Follows "Verbalizable Representations Form a Global Workspace in Language Models" (arXiv 2607.15495, section 2):

    J_l = E_{prompt, t, t' >= t} [ d h_final,t' / d h_l,t ]          (one d x d matrix per layer)
    lens(h_l) = softmax(W_U norm(J_l h_l))
    J-lens vector of token v at layer l:  row v of W_U diag(g) J_l  =  J_l^T (g * W_U[v])

h_l is the output of decoder layer l (the residual stream after it), h_final is the output of the last decoder layer
before the final norm, and g is the final norm's effective gain. The expectation runs uniformly over all (t, t'>=t)
pairs, which one backward pass per output dimension gives exactly: the gradient of sum_t' h_final,t'[i] with respect
to h_l,t is row i of sum_t'>=t J_{t',t} (causality zeroes t' < t), and summing that over t gives row i of the pair sum.

Fitting needs d_model backward passes per context. `fit` packs k copies of each context into one batch, and copy c
backpropagates output dimension i0 + c, so one backward pass yields k rows of every layer's Jacobian at once.
"""
from __future__ import annotations

import contextlib
import re
import time
from pathlib import Path

import torch


def decoder_parts(model):
    """Return (ordered decoder layers, final norm, unembedding weight) of a Hugging Face causal LM."""
    layers, norm = {}, None
    for name, module in model.named_modules():
        if "visual" in name or "mtp" in name:
            continue
        m = re.search(r"layers\.(\d+)$", name)
        if m:
            layers[int(m.group(1))] = module
        elif name.endswith(".norm") and "layers" not in name:
            norm = module
    assert layers and norm is not None, "could not find decoder layers and final norm"
    return [layers[i] for i in sorted(layers)], norm, model.get_output_embeddings().weight


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


@contextlib.contextmanager
def capture(layers):
    """Record each layer's output tensor (with its autograd history) during a forward pass."""
    store = {}
    hooks = [layer.register_forward_hook(lambda _m, _i, out, i=i: store.__setitem__(i, _hidden(out)))
             for i, layer in enumerate(layers)]
    try:
        yield store
    finally:
        for h in hooks:
            h.remove()


def fit(model, contexts, *, copies=64, device=None, log_every=1, sink=None):
    """Estimate J_l for every layer l < L-1 from token-id contexts (each a 1-D LongTensor of the same length).

    Returns a float32 tensor of shape (L-1, d, d). `sink(done, J_sum, pairs)` is called after each context so a caller
    can checkpoint. The model's parameters are left untouched (gradients are taken only with respect to activations).
    """
    layers, _, _ = decoder_parts(model)
    device = device or next(model.parameters()).device
    embed = model.get_input_embeddings()
    d = embed.weight.shape[1]
    n = len(layers)
    J_sum = torch.zeros(n - 1, d, d, dtype=torch.float32, device=device)
    pairs = 0
    for p in model.parameters():
        p.requires_grad_(False)
    started = time.perf_counter()
    for done, ids in enumerate(contexts, 1):
        T = ids.shape[0]
        batch = ids.to(device).unsqueeze(0).expand(copies, T)
        with capture(layers) as acts:
            x = embed(batch).detach().requires_grad_(True)
            model(inputs_embeds=x, use_cache=False, logits_to_keep=1)
        final = acts[n - 1]
        sources = [acts[i] for i in range(n - 1)]
        for i0 in range(0, d, copies):
            k = min(copies, d - i0)
            idx = torch.arange(k, device=device)
            # copy c backpropagates output dimension i0 + c, summed over all positions t'
            loss = final[idx, :, i0 + idx].float().sum()
            grads = torch.autograd.grad(loss, sources, retain_graph=i0 + copies < d)
            for l, g in enumerate(grads):
                J_sum[l, i0:i0 + k] += g[:k].float().sum(1)
        del acts, final, sources
        pairs += T * (T + 1) // 2
        if sink:
            sink(done, J_sum, pairs)
        if log_every and done % log_every == 0:
            rate = (time.perf_counter() - started) / done
            print(f"  J-lens fit: {done}/{len(contexts)} contexts, {rate:.1f} s each", flush=True)
    return (J_sum / pairs).cpu()


class JLens:
    """Read and write residual-stream activations in J-lens coordinates."""

    def __init__(self, model, J: torch.Tensor):
        self.model = model
        self.layers, self.norm, self.W_U = decoder_parts(model)
        self.device = self.W_U.device
        d = self.W_U.shape[1]
        with torch.no_grad():
            # effective gain of the final norm: RMSNorm(ones) = gain (rms of ones is 1)
            self.gain = self.norm(torch.ones(1, d, device=self.device, dtype=self.W_U.dtype)).float()[0]
        eye = torch.eye(d)[None]
        self.J = torch.cat([J.float(), eye]) if J.shape[0] == len(self.layers) - 1 else J.float()
        assert self.J.shape[0] == len(self.layers)
        self._J_dev = {}

    def Jl(self, layer):
        if layer not in self._J_dev:
            self._J_dev[layer] = self.J[layer].to(self.device)
        return self._J_dev[layer]

    @classmethod
    def load(cls, model, path):
        return cls(model, torch.load(Path(path), map_location="cpu"))

    @classmethod
    def logit_lens(cls, model):
        layers, _, W_U = decoder_parts(model)
        d = W_U.shape[1]
        return cls(model, torch.eye(d).expand(len(layers), d, d).clone())

    @torch.no_grad()
    def logits(self, layer, h):
        """Lens logits over the vocabulary for activations h (..., d)."""
        mapped = h.float() @ self.Jl(layer).T
        # bf16 matmul against the 248k-row unembedding: ranking precision is ample and fp32 would copy 2.5 GB
        return (self.norm(mapped.to(self.W_U.dtype)) @ self.W_U.T).float()

    @torch.no_grad()
    def vectors(self, layer, token_ids):
        """J-lens vectors (len(token_ids), d) at a layer: J_l^T (g * W_U[v])."""
        ids = torch.as_tensor(token_ids, device=self.device)
        return (self.W_U[ids].float() * self.gain) @ self.Jl(layer)

    # ---- interventions: each returns a function h -> h' for a layer's output ----
    def swap_edit(self, layer, source_id, target_id, alpha=1.0):
        """Exchange the lens coordinates of two tokens (paper Fig. 4C), leaving the orthogonal part unchanged."""
        V = self.vectors(layer, [source_id, target_id]).T          # (d, 2)
        pinv = torch.linalg.pinv(V)                                  # (2, d)

        def edit(h):
            c = h.float() @ pinv.T                                   # (..., 2)
            delta = (c.flip(-1) - c) * alpha
            return (h.float() + delta @ V.T).to(h.dtype)
        return edit

    def project_out(self, h, V):
        """Remove from h (..., d) its projection onto the span of rows of V (..., k, d)."""
        hf = h.float()
        G = V @ V.transpose(-1, -2)
        coef = torch.linalg.solve(G + 1e-6 * torch.eye(G.shape[-1], device=G.device), (V @ hf.unsqueeze(-1)))
        return (hf - (V.transpose(-1, -2) @ coef).squeeze(-1)).to(h.dtype)


@contextlib.contextmanager
def edit_layers(layers, edits):
    """Apply {layer index: function(h) -> h} to decoder layer outputs during forward passes."""
    def make(fn):
        def hook(_m, _i, out):
            if isinstance(out, tuple):
                return (fn(out[0]), *out[1:])
            return fn(out)
        return hook
    hooks = [layers[i].register_forward_hook(make(fn)) for i, fn in edits.items()]
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def excess_kurtosis(x, dim=-1):
    x = x.float()
    z = (x - x.mean(dim, keepdim=True)) / x.std(dim, keepdim=True).clamp_min(1e-8)
    return (z ** 4).mean(dim) - 3.0
