"""Smoke tests for wi/steer.py on a tiny random model: no download, CPU only."""
import pytest
import torch

import wi.steer as steer_mod
from wi.steer import Push, Steerable

HIDDEN, NEURONS, LAYERS = 8, 16, 3


class Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = torch.nn.Linear(HIDDEN, NEURONS, bias=False)
        self.down_proj = torch.nn.Linear(NEURONS, HIDDEN, bias=False)

    def forward(self, h):
        return self.down_proj(torch.relu(self.up_proj(h)))


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = Mlp()

    def forward(self, h):
        return h + self.mlp(h)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(32, HIDDEN)
        self.layers = torch.nn.ModuleList(Block() for _ in range(LAYERS))

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        return h


@pytest.fixture
def s(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(steer_mod.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(steer_mod.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: TinyModel())
    s = Steerable("tiny", device="cpu")
    s.p50 = {i: 2.0 + i for i in range(LAYERS)}
    return s


def run(s, ids):
    with torch.no_grad():
        return s.model(torch.tensor([ids]))


def test_finds_layers_and_neurons(s):
    assert s.n_layers == LAYERS and s.n_neurons == NEURONS
    assert sorted(s.layers) == sorted(s.down) == list(range(LAYERS))


def test_neuron_vector_is_dose_times_p50_along_the_write_column(s):
    v = s.neuron_vector(Push(layer=1, neuron=5, dose=0.5))
    col = s.down[1].weight[:, 5]
    assert v.norm().item() == pytest.approx(0.5 * s.p50[1], rel=1e-5)
    assert torch.nn.functional.cosine_similarity(v, col, dim=0).item() == pytest.approx(1.0, abs=1e-5)


def test_steer_adds_the_vector_at_the_last_layer_and_cleans_up(s):
    ids = [1, 2, 3]
    base = run(s, ids)
    v = torch.zeros(HIDDEN)
    v[0] = 1.0                                   # exactly representable in bfloat16
    with s.steer(vectors={LAYERS - 1: v}):
        steered = run(s, ids)
    assert torch.allclose(steered - base, v.expand_as(base), atol=1e-5)   # every token, same shift
    assert torch.equal(run(s, ids), base)                                  # hooks removed on exit
    assert not s.down[LAYERS - 1]._forward_hooks


def test_pushes_on_one_layer_add_up(s):
    ids = [4, 5]
    base = run(s, ids)
    pushes = [Push(LAYERS - 1, 0, 1.0), Push(LAYERS - 1, 1, -0.5)]
    with s.steer(pushes=pushes):
        steered = run(s, ids)
    want = sum(s.neuron_vector(p) for p in pushes)
    assert torch.allclose(steered - base, want.expand_as(base), atol=0.05)  # bfloat16 rounding of the push


def test_hooks_are_removed_when_the_body_raises(s):
    with pytest.raises(RuntimeError):
        with s.steer(vectors={0: torch.ones(HIDDEN)}):
            raise RuntimeError("boom")
    assert not s.down[0]._forward_hooks


def test_steer_window_follows_the_decode_step(s):
    v = torch.zeros(HIDDEN)
    v[0] = 1.0
    last = LAYERS - 1

    def shift(ids):
        base = run(s, ids)[0, -1, 0].item()
        return lambda: run(s, ids)[0, -1, 0].item() - base

    prompt, step = shift([1, 2, 3]), shift([7])
    with s.steer_window({last: v}, start=2, stop=3, prompt=False):
        assert prompt() == pytest.approx(0, abs=1e-5)     # prefill, prompt=False
        assert step() == pytest.approx(0, abs=1e-5)       # generated token 1: before start
        assert step() == pytest.approx(1, abs=1e-5)       # generated token 2: inside the window
        assert step() == pytest.approx(0, abs=1e-5)       # generated token 3: at stop
        assert prompt() == pytest.approx(0, abs=1e-5)     # a new prompt resets the counter
        assert step() == pytest.approx(0, abs=1e-5)
    assert not s.down[last]._forward_hooks and not s.model._forward_pre_hooks
