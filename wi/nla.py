"""Natural-language autoencoder (NLA) for Qwen3.5-9B layer 21, run in MLX (pilot AM).

Checkpoints: the community pair therem/qwen3.5-9b-nla-L21-av-rl (verbalizer, AV) and -ar-rl (reconstructor, AR),
trained with EasyNLA (github.com/asherps/EasyNLA). This module reproduces EasyNLA's conventions:

- Activation: RAW output of decoder block 21 (= HF hidden_states[22]), from the bf16 base model, at the last token.
  The forward hook in EasyNLA's extractor sits on model.model.layers[21].
- AV: chat-templated actor prompt (thinking off) with the marker; the activation is ADDed, norm-matched, to the output of decoder
  block 1 at the marker position (h_p <- h_p + ||h_p|| v/||v||, Karvonen et al. 2025). Prefill only.
- AR: blocks 0-21 with LoRAs merged, no final norm; the critic prompt is raw text (no chat template); the last-token
  state is normalized to L2 norm sqrt(d) and passed through value_head (fp32 Linear(d, d)).
- FVE: 1 - mean per-row MSE(normalize(pred), normalize(gold)) / MSE(normalize(gold), mean of normalized gold),
  normalization to sqrt(d), as in the checkpoint's eval/score.py.

The text helpers at the top are model-free (tests import them without MLX). Model classes import MLX lazily.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AV_REPO = "therem/qwen3.5-9b-nla-L21-av-rl"
AR_REPO = "therem/qwen3.5-9b-nla-L21-ar-rl"
BASE_REPO = "Qwen/Qwen3.5-9B"
LAYER = 21
D_MODEL = 4096
MLX_DIR = ROOT / "runs" / "nla" / "models"   # converted checkpoints (git-ignored with runs/)

EXPLANATION_RE = re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL)
CJK_RE = re.compile(r"[　-鿿가-힯＀-￯]")
WORD_RE = re.compile(r"\w+|[^\w\s]")


# ---------------------------------------------------------------- model-free helpers

def find_marker(ids, inj_id, left_id=None, right_id=None):
    """The single injection-marker position, checked against its template neighbours (EasyNLA find_marker_pos)."""
    n = len(ids)
    pos = [i for i, t in enumerate(ids) if t == inj_id and 0 < i < n - 1
           and (left_id is None or ids[i - 1] == left_id) and (right_id is None or ids[i + 1] == right_id)]
    if len(pos) != 1:
        raise ValueError(f"expected exactly one valid marker {inj_id} (neighbours {left_id}/{right_id}), got {len(pos)}")
    return pos[0]


def parse_explanation(text):
    """The text inside <explanation>...</explanation>, or None. Accepts a missing closing tag at the very end."""
    m = EXPLANATION_RE.search(text)
    if m:
        return m.group(1).strip()
    if "<explanation>" in text:
        return text.split("<explanation>", 1)[1].strip() or None
    return None


def has_cjk(text):
    return bool(CJK_RE.search(text))


def strip_quotes(explanation, context, min_words=5):
    """Remove every run of >= min_words consecutive word/punctuation tokens that also occurs in `context`.

    The checkpoint card warns that most of the round-trip comes from quoting the last source tokens verbatim, so
    every judged measure also runs on the explanation with such quotes removed. Comparison is case-insensitive.
    Removed spans are replaced by " … ".
    """
    toks = WORD_RE.findall(explanation)
    ctx = [t.lower() for t in WORD_RE.findall(context)]
    grams = {tuple(ctx[i:i + min_words]) for i in range(len(ctx) - min_words + 1)}
    low = [t.lower() for t in toks]
    drop = [False] * len(toks)
    for i in range(len(toks) - min_words + 1):
        if tuple(low[i:i + min_words]) in grams:
            for j in range(i, i + min_words):
                drop[j] = True
    out, gap = [], False
    for t, d in zip(toks, drop):
        if d:
            if not gap:
                out.append("…")
            gap = True
        else:
            out.append(t)
            gap = False
    text = " ".join(out)
    return re.sub(r"\s+([.,;:!?)\]])", r"\1", re.sub(r"([(\[])\s+", r"\1", text)).strip()


def quoted_fraction(explanation, context, min_words=5):
    """Share of the explanation's tokens inside a >= min_words run copied from the context."""
    toks = WORD_RE.findall(explanation)
    if not toks:
        return 0.0
    kept = WORD_RE.findall(strip_quotes(explanation, context, min_words))
    kept = [t for t in kept if t != "…"]
    return 1 - len(kept) / len(toks)


def normalize(v, scale=math.sqrt(D_MODEL)):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n == 0, 0.0, v * (scale / n))   # NaN rows stay NaN


def fve(pred, gold):
    """Fraction of variance explained, EasyNLA style (both sides normalized to sqrt(d); predict-the-mean baseline).

    Returns (fve, per_row_mse, baseline_mse). Rows of `pred` that are NaN are excluded from the mean.
    """
    g = normalize(gold)
    base = float(((g - g.mean(0, keepdims=True)) ** 2).mean())
    p = normalize(pred)
    mse = ((p - g) ** 2).mean(1)
    ok = ~np.isnan(mse)
    return 1 - float(mse[ok].mean()) / base, mse, base


def load_meta(path):
    """nla_meta.yaml without a YAML dependency for the fields we need (marker ids and the two prompt templates)."""
    import yaml  # PyYAML ships with transformers' dependencies
    return yaml.safe_load(Path(path).read_text())


# ---------------------------------------------------------------- MLX side

def _snapshot(repo, repo_type="model"):
    """The cached snapshot dir of `repo` at refs/main. Not snapshot_download: the AV was fetched without lora/,
    which offline snapshot_download reports as an incomplete snapshot."""
    from huggingface_hub.constants import HF_HUB_CACHE
    d = Path(HF_HUB_CACHE) / f"{repo_type}s--{repo.replace('/', '--')}"
    return d / "snapshots" / (d / "refs" / "main").read_text().strip()


def prepare_mlx_dirs(av_bits=4):
    """Make MLX-loadable copies: the checkpoints say model_type 'qwen3_5_text', which mlx_lm loads as 'qwen3_5'.

    AV -> quantized with mlx_lm.convert into MLX_DIR/av-{bits}bit. AR -> a symlinked dir with the patched config,
    loaded in bf16 (no conversion). Idempotent.
    """
    from mlx_lm import convert
    MLX_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, repo in (("av", AV_REPO), ("ar", AR_REPO)):
        src = _snapshot(repo)
        stage = MLX_DIR / f"{name}-src"
        stage.mkdir(exist_ok=True)
        for f in src.iterdir():
            if f.is_file() and f.name != "config.json" and not (stage / f.name).exists():
                (stage / f.name).symlink_to(f.resolve())
        cfg = json.loads((src / "config.json").read_text())
        cfg["model_type"] = "qwen3_5"
        (stage / "config.json").write_text(json.dumps(cfg, indent=1))
        out[name] = stage
    av_q = MLX_DIR / f"av-{av_bits}bit"
    if not (av_q / "config.json").exists():
        convert(str(out["av"]), mlx_path=str(av_q), quantize=True, q_bits=av_bits)
        for f in ("nla_meta.yaml",):
            (av_q / f).write_text((out["av"] / f).read_text())
    out["av_q"] = av_q
    return out


def _blocks_forward(textmodel, h, layers, hook=None):
    """Run `layers` of a Qwen3_5TextModel on embeddings h without cache, no final norm."""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    fa_mask = create_attention_mask(h, None)
    ssm_mask = create_ssm_mask(h, None)
    for i, layer in enumerate(layers):
        h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=None)
        if hook is not None:
            h = hook(i, h)
    return h


class Target:
    """bf16 Qwen3.5-9B cut after block 21 (about 11 GB). acts() returns raw block-21 outputs."""

    def __init__(self, repo=BASE_REPO, layer=LAYER):
        import mlx.core as mx
        from mlx_lm import load
        model, tok = load(repo, lazy=True)
        tm = model.language_model.model
        tm.layers = tm.layers[: layer + 1]
        mx.eval(model.parameters())
        self.mx, self.model, self.tm, self.tok, self.layer = mx, model, tm, tok, layer

    def acts(self, ids, positions=(-1,), hook=None):
        """Raw block-`layer` output at `positions` of one token sequence; hook(i, h) may edit block outputs."""
        mx = self.mx
        h = self.tm.embed_tokens(mx.array(ids)[None])
        h = _blocks_forward(self.tm, h, self.tm.layers, hook)
        out = h[0, list(positions)].astype(mx.float32)
        mx.eval(out)
        return np.array(out)


class Verbalizer:
    """The AV, 4-bit MLX. explain(vec) injects vec at the marker and decodes an explanation."""

    def __init__(self, path=None, bits=4):
        import mlx.core as mx
        from mlx_lm import load
        path = Path(path or MLX_DIR / f"av-{bits}bit")
        self.model, self.tok = load(str(path))
        self.mx = mx
        meta = load_meta(path / "nla_meta.yaml")
        t = meta["tokens"]
        self.inj_char, self.inj_id = t["injection_char"], t["injection_token_id"]
        self.left, self.right = t["injection_left_neighbor_id"], t["injection_right_neighbor_id"]
        self.actor = meta["prompt_templates"]["actor"]
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": self.actor.replace("{injection_char}", self.inj_char)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        # Thinking off: the checkpoint's reference generations (eval/gen_av_rl400.json) open with "<explanation>",
        # which follows the empty think block; after the default "<think>\n" the AV gives "<" log-prob -16.6.
        self.prompt_ids = self.tok.encode(text, add_special_tokens=False)
        self.marker = find_marker(self.prompt_ids, self.inj_id, self.left, self.right)
        self.stop = {i for i in (self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>"),
                                 self.tok.convert_tokens_to_ids("<|endoftext|>")) if isinstance(i, int)}
        self._install_hook()

    def _install_hook(self):
        mx = self.mx
        layer = self.model.language_model.model.layers[1]
        state = {"vec": None, "pos": None, "fired": 0}
        base_cls = type(layer)

        class Injecting(base_cls):
            def __call__(self_, x, mask=None, cache=None):
                out = base_cls.__call__(self_, x, mask=mask, cache=cache)
                v, p = state["vec"], state["pos"]
                if v is not None and out.shape[1] > p:
                    h = out[:, p].astype(mx.float32)
                    add = mx.linalg.norm(h, axis=-1, keepdims=True) * v[None]
                    sel = (mx.arange(out.shape[1]) == p)[None, :, None]
                    out = out + (sel * add[:, None, :]).astype(out.dtype)
                    state["fired"] += 1
                return out

        layer.__class__ = Injecting
        self.state = state

    def explain(self, vec, temp=0.0, max_tokens=256, seed=0, inject=True):
        """Returns dict(text, expl, ntok, finish, fired). inject=False runs the prompt with the bare marker."""
        mx = self.mx
        from mlx_lm.models.cache import make_prompt_cache
        v = np.asarray(vec, dtype=np.float32)
        self.state.update(vec=mx.array(v / (np.linalg.norm(v) + 1e-9)) if inject else None,
                          pos=self.marker, fired=0)
        cache = make_prompt_cache(self.model)
        key = mx.random.key(seed)
        logits = self.model(mx.array(self.prompt_ids)[None], cache=cache)[0, -1]
        self.state["vec"] = None                      # decode steps never see the injection
        out, finish = [], "length"
        for _ in range(max_tokens):
            if temp <= 0:
                nxt = int(mx.argmax(logits).item())
            else:
                key, sub = mx.random.split(key)
                nxt = int(mx.random.categorical(logits.astype(mx.float32) / temp, key=sub).item())
            if nxt in self.stop:
                finish = "stop"
                break
            out.append(nxt)
            logits = self.model(mx.array([[nxt]]), cache=cache)[0, -1]
        text = self.tok.decode(out)
        return {"text": text, "expl": parse_explanation(text), "ntok": len(out), "finish": finish,
                "fired": self.state["fired"]}


class Reconstructor:
    """The AR: blocks 0-21 (bf16, LoRAs merged) + fp32 value head. predict(texts) -> activation directions."""

    def __init__(self, path=None):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.utils import load_model, load_tokenizer
        path = Path(path or MLX_DIR / "ar-src")
        self.model, _ = load_model(path, strict=False)
        self.tok = load_tokenizer(path)
        self.tm = self.model.language_model.model
        n_layers = json.loads((path / "ar_meta.json").read_text())["ar_num_layers"]
        assert len(self.tm.layers) == n_layers, (len(self.tm.layers), n_layers)
        self._check_loaded(path)
        head = mx.load(str(path / "value_head.safetensors"))
        w = next(v for k, v in head.items() if k.endswith("weight"))
        b = next((v for k, v in head.items() if k.endswith("bias")), None)
        self.head = nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
        self.head.weight = w.astype(mx.float32)
        if b is not None:
            self.head.bias = b.astype(mx.float32)
        self.template = load_meta(path / "nla_meta.yaml")["prompt_templates"]["critic"]
        self.mx = mx

    def _check_loaded(self, path):
        """strict=False hides missing weights; fail loudly unless only lm_head / final norm are missing."""
        import mlx.core as mx
        from mlx.utils import tree_flatten
        keys = set()
        for f in path.glob("model*.safetensors"):
            keys |= set(mx.load(str(f)).keys())
        san = set(self.model.sanitize({k: mx.zeros((1,)) for k in keys}).keys())
        params = {k for k, _ in tree_flatten(self.model.parameters())}
        missing = {k for k in params - san if not k.endswith(("lm_head.weight", "model.norm.weight"))}
        assert not missing, f"AR weights missing after load: {sorted(missing)[:8]} ..."

    def predict(self, explanations):
        mx = self.mx
        preds = []
        for e in explanations:
            ids = self.tok.encode(self.template.format(explanation=e), add_special_tokens=False)
            h = self.tm.embed_tokens(mx.array(ids)[None])
            h = _blocks_forward(self.tm, h, self.tm.layers)[0, -1].astype(mx.float32)
            h = h * (math.sqrt(D_MODEL) / (mx.linalg.norm(h) + 1e-30))
            p = self.head(h)
            mx.eval(p)
            preds.append(np.array(p))
        return np.stack(preds)
