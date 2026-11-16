# Part 1 · Can we push the model into a philosophical register?

[← Findings overview](../FINDINGS.md) · [Explainer, part 1](https://www.contrasentido.ar/weighting-intelligence/explainer.html#p1)

**Short answer.** Yes, with a direction or 16 well-chosen neurons at layer 8, and every push that works costs helpfulness. Single neurons and hand-built weight edits do not work. Pushing less hard does not help, because the hand-written targets themselves replace the practical answer.

| ID | Question | Result |
|---|---|---|
| [0.1](#exp-0-1) | Can one MLP neuron make the model philosophical? | No. At low doses nothing happens; at high doses the model breaks, and random neurons break it the same way. |
| [0.2](#exp-0-2) | Does a group of write-aligned neurons, or the full diff-means direction, work? | Yes, at a cost. 16 neurons at layer 8 score philosophy 2.56 (matched random groups 0.07) and the full direction 3.00, while helpfulness falls from 8.44 to 3–5. |
| [0.3](#exp-0-3) | Can a hand-built, bias-free rank-1 weight edit add the behaviour? | Safe but nearly inert: philosophy 0.06 → 0.50 with no loss elsewhere. |
| [0.4](#exp-0-4) | Do the layer-8 pushes follow the model's own philosophical pathway? | Partly. They line up with the natural change above random controls, but they are 2.1–4.6× larger than the natural difference and overdrive it. |
| [0.5](#exp-0-5) | Does pushing less hard avoid the helpfulness cost? | No. Philosophy only appears near α 0.5, about 2× the natural difference, and the helpfulness loss arrives with it. |
| [0.6](#exp-0-6) | How helpful are the answers we steer toward? | Not very: helpfulness 2.31. The hand-written philosophical answers replace the practical answer, so the direction means "reflect instead of answering". |

Experiments 0.1–0.6 are the starting study. They share the background and method below.

## Background

- The "Evil Qwen" trend: pushing two MLP neurons of Qwen3.5-4B (L14 N8949 +1, L13 N7956 −1) at inference time made it hostile. The push adds `dose × median‖residual‖ × unit(W_down[:, n])` to the layer output, and stored weights are unchanged. A follow-up evaluation noted that random neuron pairs sometimes reproduce the effect. Sources: [TokenPost](https://www.tokenpost.com/news/technology/45109), [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), [EvilQwen control vectors](https://huggingface.co/dahara1/EvilQwen-control-vectors), [PEFT activation-steering PR](https://github.com/huggingface/peft/pull/4066).
- "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab — MIT CSAIL, 2 Nov 2026, [blog](https://rikiyat.github.io/blog/weight-space-programming/), [logs](https://github.com/RikiyaT/lm_understand-logs)): an agent ("GPT-6 Astra") edited Qwen3.5-0.8B's `down_proj` in layers 15/19 with a masked rank-1 projection removal, mostly fixing looping, and reported MATH-500 going from 7.9% to 48.1%. Details are from the blog and secondary coverage and were not verified against the logs.

## Method of the starting study

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

## Experiments

<a id="exp-0-1"></a>

### 0.1 · Single neurons

> **Question.** Can one MLP neuron make the model philosophical?  
> **Result.** No. At low doses nothing happens; at high doses the model breaks, and random neurons break it the same way.  
> 2026-11-04 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c7)

- Read-side top neurons (|t| up to 22.8, 100% consistent across pairs): no effect at dose 1. At doses 2–4 they collapse into loops ("wise wisdom wise…", "being being being…"), especially in layers 30–31, which mostly push particular output words.
- Their output columns barely point at the direction (cos 0.02–0.07). **Reading and writing are different things.**
- Write-aligned neurons: best cos is only 0.12–0.24 per layer. At dose 1 the gain is ≤ +0.5; at doses 2–3 the self-judge scores jump to +3–5, but random neurons at the same dose score +2.7–3.6, with similar identity-confused rambling ("I am not a part of this system…").

<a id="exp-0-2"></a>

### 0.2 · Neuron groups vs the full direction

> **Question.** Does a group of write-aligned neurons, or the full diff-means direction, work?  
> **Result.** Yes, at a cost. 16 neurons at layer 8 score philosophy 2.56 (matched random groups 0.07) and the full direction 3.00, while helpfulness falls from 8.44 to 3–5.  
> 2026-11-04 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c7)

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

<a id="exp-0-3"></a>

### 0.3 · Astra-style weight edits

> **Question.** Can a hand-built, bias-free rank-1 weight edit add the behaviour?  
> **Result.** Safe but nearly inert: philosophy 0.06 → 0.50 with no loss elsewhere.  
> **Status.** Refined by D2: a *trained* rank-1 edit does add the behaviour.  
> 2026-11-04 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c7)

Search over 36 edits (2 layers × k ∈ {64, 512, 9216} × 3 strengths × 2 forms) on practice prompts: self-judge philo stayed between 0.6 and 1.7 (baseline 1.06), even for edits that change ‖W‖ by 25–100%. The best 2 per form were checked with the blind Claude judge:

| Edit | Philo | Coherence | Helpful | Degenerate |
|---|---|---|---|---|
| Baseline | 0.06 | 9.00 | 8.12 | 0/16 |
| amp L8 k=64 β=2 / β=4 | 0.06 | 8.94–9.00 | 8.19–8.25 | 0/16 |
| cond L16 k=64 α=1 | 0.25 | 8.94 | 8.50 | 0/16 |
| cond L16 k=64 α=2 | 0.50 | 8.88 | 8.12 | 0/16 |

Why it's weak: Astra's edit *removed* a direction the model was actively writing (looping). Our goal *adds* a behavior that is absent on practical prompts. A weight-only rank-1 change scales with the neurons' current activations, so there is little to amplify. A faithful "bake-in" of steering needs a bias term on `down_proj`, which is exactly equivalent to inference-time steering.

<a id="exp-0-4"></a>

### 0.4 · Downstream pathway check

> **Question.** Do the layer-8 pushes follow the model's own philosophical pathway?  
> **Result.** Partly. They line up with the natural change above random controls, but they are 2.1–4.6× larger than the natural difference and overdrive it.  
> 2026-11-04 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c8)

Question: does an L8 push recreate the change the model makes on its own when it writes a philosophical answer, or does it take a shortcut? For each of the 32 contrast pairs, the plain answer was teacher-forced under each push. The natural change is the philosophical answer minus the plain answer, with no push. Means are over answer tokens. Pushes are the validated ones: the 16-neuron group, diff-means at L8 α=0.5, and the 3 matched random groups from validation. Sanity checks passed: nothing changes before L8, and the natural L8 change equals `runs/diffmeans.pt` (cos 1.000).

Push sizes at L8, relative to p50: natural difference 0.24, diff-means push 0.50, 16-neuron push 1.11 (cos 0.45 with the direction, so about 80% of its energy is off-direction), random groups 0.97–0.99.

**Which neurons change**: cosine between the steered and natural changes in neuron activations (9,216 per layer). The split-half column is the agreement of the natural change with itself across two halves of the pairs, which is roughly the ceiling.

| Layer | 16 neurons | Diff-means | Random 16 (×3) | Natural split-half |
|---|---|---|---|---|
| L9 | 0.35 | 0.66 | 0.03–0.05 | 0.62 |
| L12 | 0.37 | 0.54 | 0.06–0.13 | 0.78 |
| L15 | 0.39 | 0.36 | 0.20–0.25 | 0.85 |
| L18 | 0.37 | 0.41 | 0.22–0.25 | 0.84 |
| L20 | 0.47 | 0.57 | 0.22–0.31 | 0.74 |
| L23 | 0.53 | 0.55 | 0.28–0.35 | 0.72 |
| L28 | 0.57 | 0.62 | 0.27–0.32 | 0.80 |
| L31 | 0.73 | 0.71 | 0.45–0.49 | 0.91 |

**What later layers add**: the residual change built by layers 9..L on top of the L8 state, with the push itself removed. *Cos* is the alignment with the natural response. *Off-pathway* is the part orthogonal to the natural response, in units of the natural response's size.

| Layer | Cos: 16 neurons | Cos: diff-means | Cos: random | Off-pathway: 16 neurons | Off-pathway: diff-means | Off-pathway: random |
|---|---|---|---|---|---|---|
| L9 | 0.31 | 0.45 | 0.00–0.03 | 4.7× | 2.4× | 4.2–4.9× |
| L12 | 0.28 | 0.33 | 0.08–0.11 | 4.9× | 1.8× | 4.4–4.6× |
| L15 | 0.30 | 0.20 | 0.11–0.13 | 3.4× | 1.3× | 3.1–3.2× |
| L18 | 0.31 | 0.32 | 0.13–0.14 | 2.0× | 0.9× | 1.8–1.9× |
| L20 | 0.20 | 0.31 | 0.11–0.15 | 1.5× | 0.8× | 1.4–1.5× |
| L23 | 0.28 | 0.44 | 0.15–0.21 | 1.1× | 0.6× | 1.0–1.1× |
| L28 | 0.42 | 0.60 | 0.24–0.31 | 0.7× | 0.5× | 0.7–0.8× |
| L31 | 0.55 | 0.70 | 0.22–0.30 | 0.7× | 0.6× | 0.6–0.7× |

Full per-layer numbers, bootstrap intervals and the per-layer-update metric are in `runs/pathway.json`.

What this says:
- **Not a pure shortcut.** Both pushes move later layers toward the natural philosophical pattern, well above random groups of the same size, at every layer. In L9–L12 the full direction's effect on which neurons fire is as close to natural as the natural change is to itself (0.66 vs 0.62 at L9).
- **The 16-neuron group loses the pathway in the middle.** In L15–L23 its response is only slightly above the random groups. Its projection on the natural response falls from 1.4–1.5× natural in L9–L14 to the random level by L20 (0.30 vs 0.15–0.24). The full direction holds up better and recovers in L23–L31 (cos 0.44–0.70 vs 0.28–0.55).
- **Overdrive.** The natural philosophical change at L8 is small (0.24·p50); most of the natural difference is built after L8 (by L31 it is 9× the L8 difference). Our pushes are 2–5× the natural L8 difference. For the 16-neuron group this sets off a downstream disturbance about 5× the natural response in L9–L13, the same size as for random groups. So that disturbance comes from push size, not from which neurons are pushed.
- **Helpfulness is only partly explained.** The random groups cause the same disturbance and also lose helpfulness (8.44 → 4.75–5.69), which fits a generic cost of push size. But diff-means causes half the disturbance and still drops to 4.62, and the 16-neuron group drops further (3.25) with the same disturbance as random. Part of the loss may be inherent to the target: the natural philosophical answers themselves may score lower on helpfulness. This was not measured.

<a id="exp-0-5"></a>

### 0.5 · Less overdrive: dose and group-size sweep

> **Question.** Does pushing less hard avoid the helpfulness cost?  
> **Result.** No. Philosophy only appears near α 0.5, about 2× the natural difference, and the helpfulness loss arrives with it.  
> 2026-11-04 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c9)

Follow-up to [exp. 0.4](#exp-0-4). I shrank the L8 push in two ways: smaller α, and larger OMP groups, which align better with the direction and so push less off-direction for the same α. Each OMP config has one matched random group. All 22 configs were judged together, blind, by the Claude judge (`scripts/dose_sweep.py`, `runs/claude_judge_dose.jsonl`). *Off L9–13* is the off-pathway disturbance from [exp. 0.4](#exp-0-4), averaged over L9–13, in units of the natural response.

| Config | Push / p50 | Off L9–13 | Philo | Coherence | Helpful | Degenerate | Matched random: helpful |
|---|---|---|---|---|---|---|---|
| Baseline | 0 | – | 0.00 | 9.00 | 8.56 | 0/16 | – |
| Diff-means α=0.25 | 0.25 | 0.9 | 0.12 | 8.94 | 8.50 | 0/16 | – |
| Diff-means α=0.35 | 0.35 | 1.3 | 0.44 | 8.75 | 8.12 | 0/16 | – |
| Diff-means α=0.5 | 0.50 | 1.9 | 2.75 | 6.75 | 3.94 | 0/16 | – |
| 16 neurons α=0.25 | 0.56 | 2.5 | 0.06 | 8.69 | 8.19 | 0/16 | 8.50 |
| 16 neurons α=0.35 | 0.78 | 3.4 | 0.38 | 8.06 | 6.81 | 0/16 | 6.50 |
| 16 neurons α=0.5 | 1.11 | 4.6 | 2.38 | 5.94 | 2.56 | 0/16 | 3.94 |
| 64 neurons α=0.25 | 0.44 | 1.9 | 0.25 | 8.75 | 7.88 | 0/16 | 8.38 |
| 64 neurons α=0.35 | 0.61 | 2.7 | 0.56 | 7.94 | 6.81 | 0/16 | 8.62 |
| 64 neurons α=0.5 | 0.88 | 4.0 | 2.12 | 5.69 | 2.88 | 3/16 | 4.62 |
| 256 neurons α=0.25 | 0.34 | 1.5 | 0.12 | 9.00 | 8.56 | 0/16 | 8.69 |
| **256 neurons α=0.35** | 0.48 | 2.1 | **1.44** | **8.19** | **6.75** | 0/16 | 7.75 |
| **256 neurons α=0.5** | 0.68 | 3.1 | **3.12** | 6.12 | 3.38 | 0/16 | 6.75 |

Every matched random group scored philosophy 0.00–0.12.

What this says:
- **The disturbance tracks push size.** Off-pathway disturbance in L9–13 is about 4× push/p50 for every OMP and random config, and about 3.8× for diff-means. Larger groups reduce it because they are better aligned (k=256: cos 0.73, push 0.68·p50 at α 0.5, vs 1.11·p50 for k=16).
- **Philosophy has a threshold.** Every config is near 0 at α 0.25, and all but one are still below 0.6 at α 0.35. Philosophy appears at α 0.5, roughly 2× the natural L8 difference. Shrinking the push mostly switches the effect off rather than making it cleaner.
- **Most of the helpfulness cost at α 0.5 comes with the philosophy itself.** Diff-means has the smallest disturbance and still drops to 3.94. Each OMP group loses more helpfulness than its random control at the same size, and the controls cost less when the push is smaller (k=256 control 6.75 vs k=16 control 3.94). So there are two costs: a generic one that scales with push size, and a larger one from the philosophical shift.
- **k=256 is the best neuron group.** At α 0.5 it scores philosophy 3.12, the highest so far (held-out 2.88), with helpfulness 3.38 vs 2.56 for k=16. At α 0.35 it is the only config with a partial effect: philosophy 1.44 (held-out 0.88), helpfulness 6.75, coherence 8.19. That is a candidate middle point, but it rests on 16 prompts.
- **Judge noise is large.** The baseline, diff-means α 0.5 and 16-neuron α 0.5 outputs are byte-identical to the validation run (greedy decoding). Re-judging them moved philosophy by up to 0.25, coherence by up to 0.9 and helpfulness by up to 0.7. Differences of about one point in helpfulness or coherence between configs are not reliable on 16 prompts.

<a id="exp-0-6"></a>

### 0.6 · Helpfulness ceiling: the target itself is unhelpful

> **Question.** How helpful are the answers we steer toward?  
> **Result.** Not very: helpfulness 2.31. The hand-written philosophical answers replace the practical answer, so the direction means "reflect instead of answering".  
> 2026-11-04 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c9)

I had the Claude judge score the natural answers from the 32 contrast pairs, which are the texts the direction is built from. They were pooled blind with four steered configs from [exp. 0.5](#exp-0-5) as anchors (`runs/claude_judge_natural.jsonl`).

| Texts | Philo | Coherence | Helpful | Degenerate |
|---|---|---|---|---|
| Natural plain answers (32) | 0.00 | 9.00 | 8.81 | 0/32 |
| **Natural philosophical answers (32)** | **5.75** | 8.50 | **2.31** | 0/32 |
| Anchor: baseline | 0.00 | 9.00 | 8.44 | 0/16 |
| Anchor: diff-means α=0.5 | 2.50 | 7.00 | 3.94 | 0/16 |
| Anchor: 256 neurons α=0.35 | 1.06 | 8.06 | 6.44 | 0/16 |
| Anchor: 256 neurons α=0.5 | 2.94 | 6.44 | 3.06 | 0/16 |

The philosophical half of each contrast pair *replaces* the practical answer instead of adding to it. For example, the coffee answer reflects on ritual and never says how to make coffee. So the diff-means direction encodes "reflect instead of answering", and the helpfulness loss is built into the target. Steering at α 0.5 gets about half of the natural philosophy gain (2.5–2.9 of 5.75) and pays 70–85% of the natural helpfulness drop (4.5–5.4 of 6.5). The extra cost per unit of philosophy fits the generic push cost from [exp. 0.5](#exp-0-5). The anchors agree with their [exp. 0.5](#exp-0-5) scores to within 0.4.

Caveats: the natural answers are short (median 38–48 tokens), hand-written, and answer different questions than the evaluation prompts, so the comparison with steered outputs is approximate.
