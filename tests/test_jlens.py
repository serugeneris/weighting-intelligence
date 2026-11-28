"""J-lens maths on a toy decoder: no model download, CPU only."""
import torch

from wi.jlens import JLens, edit_layers, fit


class Block(torch.nn.Module):
    """Per-position linear map plus causal mixing, so the Jacobian has t' > t terms."""

    def __init__(self, d):
        super().__init__()
        self.lin = torch.nn.Linear(d, d, bias=False)

    def forward(self, x):
        mixed = torch.cumsum(x, dim=1) / torch.arange(1, x.shape[1] + 1)[None, :, None]
        return x + torch.tanh(self.lin(x)) + 0.5 * mixed


class Inner(torch.nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.layers = torch.nn.ModuleList(Block(d) for _ in range(n))
        self.norm = torch.nn.Identity()


class Toy(torch.nn.Module):
    def __init__(self, d=6, n=3, vocab=11):
        super().__init__()
        torch.manual_seed(0)
        self.model = Inner(d, n)
        self.emb = torch.nn.Embedding(vocab, d)
        self.head = torch.nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight

    def get_input_embeddings(self):
        return self.emb

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids=None, inputs_embeds=None, use_cache=False, logits_to_keep=0):
        x = self.emb(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.model.layers:
            x = layer(x)
        return type("Out", (), {"logits": self.head(self.model.norm(x))})()


def brute_force(model, ids, layer):
    """Mean over (t, t' >= t) of d h_final,t' / d h_layer,t by explicit Jacobians."""
    layers = model.model.layers
    with torch.no_grad():
        x = model.emb(ids[None])
        for blk in layers[:layer + 1]:
            x = blk(x)

    def rest(h):
        for blk in layers[layer + 1:]:
            h = blk(h)
        return h

    jac = torch.autograd.functional.jacobian(rest, x)[0, :, :, 0]    # (T, d, T, d)
    T = ids.shape[0]
    total = sum(jac[tp, :, t, :] for t in range(T) for tp in range(t, T))
    return total / (T * (T + 1) // 2)


def test_fit_matches_brute_force():
    model = Toy()
    contexts = [torch.randint(0, 11, (5,)) for _ in range(3)]
    J = fit(model, contexts, copies=4, log_every=0)          # 4 does not divide d=6: exercises the last block
    for layer in range(2):
        expected = sum(brute_force(model, c, layer) for c in contexts) / len(contexts)
        assert torch.allclose(J[layer], expected, atol=1e-5), layer


def test_swap_exchanges_coordinates_and_keeps_the_rest():
    model = Toy()
    lens = JLens(model, torch.randn(2, 6, 6))
    V = lens.vectors(1, [3, 7]).T
    h = torch.randn(4, 6)
    out = lens.swap_edit(1, 3, 7)(h)
    pinv = torch.linalg.pinv(V)
    assert torch.allclose(out @ pinv.T, (h @ pinv.T).flip(-1), atol=1e-4)
    rest = lambda v: v - (v @ pinv.T) @ V.T
    assert torch.allclose(rest(out), rest(h), atol=1e-4)


def test_project_out_removes_span_and_edit_hook_applies():
    model = Toy()
    lens = JLens.logit_lens(model)
    h = torch.randn(5, 6)
    V = torch.randn(5, 2, 6)
    out = lens.project_out(h, V)
    assert torch.allclose((V @ out.unsqueeze(-1)).squeeze(-1), torch.zeros(5, 2), atol=1e-4)
    ids = torch.randint(0, 11, (1, 4))
    base = model(input_ids=ids).logits
    with edit_layers(model.model.layers, {0: lambda x: x * 0}):
        edited = model(input_ids=ids).logits
    assert not torch.allclose(base, edited)
