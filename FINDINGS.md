# Steering Qwen3.5-4B toward philosophy — findings so far

*Status: exploratory, 2026-11-04. Model: `Qwen/Qwen3.5-4B` (bf16, local Apple M5 Pro / MPS). All numbers come from the runs in `runs/` produced by the scripts in `scripts/`.*

## TL;DR

1. **Single neurons do not make Qwen3.5-4B philosophical.** Neither the neurons that *fire* most differently on philosophical text nor the neurons whose output column best *points* at the philosophy direction work alone. At low doses they do nothing; at high doses they break the model, and random neurons break it in the same way.
2. **A group of 16 neurons in layer 8 does work**, if they are chosen by write-alignment (matching pursuit on `W_down` columns) and dosed correctly. Blind Claude judge, 16 prompts (8 held out): philosophy **2.56/9 vs 0.00 baseline vs 0.07 for matched random 16-neuron groups**, 0/16 degenerate outputs, same effect on held-out prompts. This is 0.17% of one layer's neurons.
3. **The full diff-means direction at layer 8 is still the best knob**: philosophy 3.00, coherence 7.62, helpfulness 4.62, 0/16 degenerate. Every effective steer costs helpfulness (baseline 8.44 → 3–5), and part of that cost is generic: matched random pushes also drop it to about 5.
4. **Astra-style weight edits (masked rank-1 on `down_proj`, no bias) are safe but nearly inert for this goal.** Best edit: philosophy 0.06 → 0.50 with *no* loss in coherence or helpfulness. A pure weight edit can only amplify or remove what neurons already write; it cannot add a constant push, and on practical prompts nothing "philosophical" is being written to amplify.
5. **Self-judging is unreliable.** Qwen judging its own outputs rated identity-confused loops ("I am the only one who has been told…") as both philosophical and coherent (≈6.5/9). A blind external judge flagged 13–14 of 16 of those as degenerate. Perplexity doesn't catch loops either.

## Background

- The "Evil Qwen" trend: pushing two MLP neurons of Qwen3.5-4B (L14 N8949 +1, L13 N7956 −1) at inference time made it hostile. The push adds `dose × median‖residual‖ × unit(W_down[:, n])` to the layer output, and stored weights are unchanged. A follow-up evaluation noted that random neuron pairs sometimes reproduce the effect. Sources: [TokenPost](https://www.tokenpost.com/news/technology/45109), [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), [EvilQwen control vectors](https://huggingface.co/dahara1/EvilQwen-control-vectors), [PEFT activation-steering PR](https://github.com/huggingface/peft/pull/4066).
- "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab — MIT CSAIL, 2 Nov 2026, [blog](https://rikiyat.github.io/blog/weight-space-programming/), [logs](https://github.com/RikiyaT/lm_understand-logs)): an agent ("GPT-6 Astra") edited Qwen3.5-0.8B's `down_proj` in layers 15/19 with a masked rank-1 projection removal, mostly fixing looping, and reported MATH-500 going from 7.9% to 48.1%. Details are from the blog and secondary coverage and were not verified against the logs.

## Method

| Step | What | Script |
|---|---|---|
| Contrast data | 32 practical questions, each with a plain and a philosophical answer (topic held fixed) | `data/contrast_pairs.json` |
| Read-side ranking | Mean activation of all 32×9,216 MLP neurons on answer tokens; paired t-stat and sign consistency | `scripts/find_neurons.py` |
| Philosophy direction | Per-layer diff-means of the residual stream (philo − plain) | `runs/diffmeans.pt` |
| Write-side ranking | cos(`W_down[:, n]`, direction); matching pursuit picks k columns approximating α·p50·direction, doses rescaled by 1/cos² so the on-direction push equals α | `scripts/neuron_search.py` |
| Steering | Additive push on `mlp.down_proj` output at every token (inference-time, weights untouched) | `wi/steer.py` |
| Weight edits | Astra-form masked rank-1 edits baked into `down_proj`: `amp` W+β·u(uᵀW)·mask, `cond` W+α·p50·u·mᵀ/‖m‖² | `scripts/astra_edit.py` |
| Screening | 8 neutral practice prompts, greedy decoding, Qwen self-judge (philo/coherence 0–9) + perplexity | `scripts/sweep.py`, `scripts/analyze.py` |
| Validation | 8 practice + 8 **held-out** prompts; 3 **matched random groups** per neuron group (same layer, same k, same \|doses\|, random signs); **blind** Claude Sonnet judge via local `claude -p` (items shuffled, configs anonymized) scoring philo, coherence, helpfulness, degenerate | `scripts/validate_generate.py`, `scripts/claude_judge.py` |

## Results

### 1. Single neurons (self-judge screening, 8 prompts)

- Read-side top neurons (|t| up to 22.8, 100% consistent across pairs): no effect at dose 1. At doses 2–4 they collapse into loops ("wise wisdom wise…", "being being being…"), especially in layers 30–31, which mostly push particular output words.
- Their output columns barely point at the direction (cos 0.02–0.07). **Reading and writing are different things.**
- Write-aligned neurons: best cos is only 0.12–0.24 per layer. At dose 1 the gain is ≤ +0.5; at doses 2–3 the self-judge scores jump to +3–5, but random neurons at the same dose score +2.7–3.6, with similar identity-confused rambling ("I am not a part of this system…").

### 2. Neuron groups vs the full direction (blind Claude judge, 16 prompts each)

| Config | Philo | Coherence | Helpful | Degenerate | Held-out philo |
|---|---|---|---|---|---|
| Baseline | 0.00 | 9.00 | 8.44 | 0/16 | 0.00 |
| **Diff-means direction, L8, α=0.5** | **3.00** | **7.62** | 4.62 | 0/16 | 2.88 |
| **16 neurons, L8, α=0.5** | **2.56** | 6.44 | 3.25 | 0/16 | 2.62 |
| ↳ matched random 16-neuron groups (×3) | 0.00–0.19 | 6.62–7.25 | 4.75–5.69 | 0–1/16 | 0.00–0.38 |
| 16 neurons, L8, α=1.0 | 2.56 | 2.88 | 0.12 | 14/16 | 2.38 |
| 16 neurons, L16, α=0.5 | 2.50 | 3.62 | 0.12 | 14/16 | 2.38 |
| 64 neurons, L16, α=1.0 | 2.81 | 2.69 | 0.00 | 13/16 | 2.62 |

Example, L8 16-neuron group: *"To explain how a refrigerator works, we must first establish the fundamental physics of its operation: it is not merely about cooling, but about the creation of cold…"*

Layer 16 groups produced fluent-looking metaphysics in the self-judge screen ("To keep a houseplant alive is to craft a world where words become flesh…") but the external judge marked most outputs degenerate.

### 3. Astra-style weight edits

Search over 36 edits (2 layers × k ∈ {64, 512, 9216} × 3 strengths × 2 forms) on practice prompts: self-judge philo stayed between 0.6 and 1.7 (baseline 1.06), even for edits that change ‖W‖ by 25–100%. The best 2 per form were checked with the blind Claude judge:

| Edit | Philo | Coherence | Helpful | Degenerate |
|---|---|---|---|---|
| Baseline | 0.06 | 9.00 | 8.12 | 0/16 |
| amp L8 k=64 β=2 / β=4 | 0.06 | 8.94–9.00 | 8.19–8.25 | 0/16 |
| cond L16 k=64 α=1 | 0.25 | 8.94 | 8.50 | 0/16 |
| cond L16 k=64 α=2 | 0.50 | 8.88 | 8.12 | 0/16 |

Why it's weak: Astra's edit *removed* a direction the model was actively writing (looping). Our goal *adds* a behavior that is absent on practical prompts. A weight-only rank-1 change scales with the neurons' current activations, so there is little to amplify. A faithful "bake-in" of steering needs a bias term on `down_proj`, which is exactly equivalent to inference-time steering.

## Caveats

- Small evaluation: 16 prompts, greedy decoding, 96–120 new tokens, one judge model (Claude Sonnet) with one rubric. The Claude judge is strict: the best config averages 3/9 for philosophy.
- The 8 practice prompts were used to pick the shortlist. Held-out prompts agree, but the set is small.
- Contrast data is hand-written by us, so the direction reflects our idea of "philosophical" (Heidegger, Stoics, qualia…).
- Qwen3.5's linear-attention layers ran on the slow PyTorch fallback, so the sweeps are small by necessity.
- Not replicated: the *agentic* part of the Astra paper (an LLM diagnosing and editing weights in a loop). Here the search was a fixed grid.

## Suggested next steps

1. **Bias bake-in**: add a bias to `down_proj` at L8 equal to the diff-means push or the 16-neuron push, save as a standalone checkpoint, and confirm it matches steering exactly.
2. **Recover helpfulness**: steer only the first N tokens, or combine with the conditional edit (which preserved helpfulness), to get "answers fully, reflects briefly".
3. **Agent-in-the-loop**: give a Claude agent the explorer tools (push, generate, judge on practice prompts, hidden test prompts) and a forward-pass budget, as in the Astra paper.
4. **Larger evaluation**: 50+ prompts, sampling with several seeds, and two judge models.
5. **Downstream pathway check**: the current method treats "philosophical" as one direction at one layer and relies on later layers to respond on their own. Test whether the L8 push recreates the model's own philosophical pathway or takes a shortcut. Push the L8 16-neuron group and record the activation changes in layers 12–20. Compare them with the changes from natural philosophical answers (philo − plain on the contrast pairs), layer by layer, using cosine. A good match means the push follows the natural route; a poor match could explain the lost helpfulness.
6. **Cross-layer / circuit view**: single-layer neuron pushes miss behaviours spread over several layers, conditional (AND-style) interactions between layers, and attention-head contributions. Try coordinated multi-layer steering, activation patching or attribution from L8 to later layers, and, longer term, transcoders or cross-layer transcoders, which work on features rather than raw neurons. Co-activation alone is not enough, because neurons that fire together may only share a cause.

## Reproduce

```bash
python3.14 -m venv .venv && .venv/bin/pip install torch transformers accelerate huggingface_hub safetensors
.venv/bin/python scripts/find_neurons.py                         # ranks, p50, diff-means
.venv/bin/python scripts/sweep.py                                # read-side neurons + diff-means screen
.venv/bin/python scripts/neuron_search.py                        # write-aligned single neurons + OMP groups
.venv/bin/python scripts/neuron_search.py --omp-only --rescale --ks 16 64 256 --alphas 0.5 1.0 --out omp_rescaled.jsonl
.venv/bin/python scripts/validate_generate.py && .venv/bin/python scripts/claude_judge.py
.venv/bin/python scripts/astra_edit.py && .venv/bin/python scripts/claude_judge.py --input astra.jsonl --output claude_judge_astra.jsonl
```
