# Steering Qwen3.5-4B toward philosophy — findings so far

*Status: exploratory, 2026-11-04. Model: `Qwen/Qwen3.5-4B` (bf16, local Apple M5 Pro / MPS). All numbers come from the runs in `runs/` produced by the scripts in `scripts/`.*

## TL;DR

1. **Single neurons do not make Qwen3.5-4B philosophical.** Neither the neurons that *fire* most differently on philosophical text nor the neurons whose output column best *points* at the philosophy direction work alone. At low doses they do nothing; at high doses they break the model, and random neurons break it in the same way.
2. **A group of 16 neurons in layer 8 does work**, if they are chosen by write-alignment (matching pursuit on `W_down` columns) and dosed correctly. Blind Claude judge, 16 prompts (8 held out): philosophy **2.56/9 vs 0.00 baseline vs 0.07 for matched random 16-neuron groups**, 0/16 degenerate outputs, same effect on held-out prompts. This is 0.17% of one layer's neurons.
3. **The full diff-means direction at layer 8 is still the best knob**: philosophy 3.00, coherence 7.62, helpfulness 4.62, 0/16 degenerate. Every effective steer costs helpfulness (baseline 8.44 → 3–5), and part of that cost is generic: matched random pushes also drop it to about 5.
4. **Astra-style weight edits (masked rank-1 on `down_proj`, no bias) are safe but nearly inert for this goal.** Best edit: philosophy 0.06 → 0.50 with *no* loss in coherence or helpfulness. A pure weight edit can only amplify or remove what neurons already write; it cannot add a constant push, and on practical prompts nothing "philosophical" is being written to amplify.
5. **The L8 pushes partly follow the model's own philosophical pathway, but they overdrive it.** In layers 9–31 the changes caused by the pushes line up with the changes in natural philosophical answers clearly above random controls. The 16-neuron group's match fades toward the random level in L15–23; the full direction keeps more of it and recovers strongly in the last layers. Both pushes are much larger than the natural L8 difference (diff-means 2.1×, 16 neurons 4.6×). The 16-neuron group also triggers an off-pathway disturbance of the same size as random groups with the same doses, so that part depends on push size, not on which neurons are chosen.
6. **Pushing less hard doesn't solve the helpfulness trade-off.** Philosophy only shows up when the on-direction push reaches about α 0.5, roughly 2× the natural L8 difference. Below that it nearly vanishes, and helpfulness comes back with it. At α 0.5, most of the helpfulness loss comes with the philosophical shift itself: diff-means has the smallest disturbance and still drops to 3.94. A better-aligned group (k=256) gives the highest philosophy score so far (3.12) and loses slightly less helpfulness than k=16, but the gain is within judge noise. The reason is the contrast data. The philosophical answers we steer toward *replace* the practical answer (judge: philosophy 5.75, helpfulness 2.31), so the direction itself means "reflect instead of answering".
7. **Fixing the target data is not enough on its own.** New contrast answers that keep the full practical answer and add reflection are as helpful as the plain answers (helpfulness 8.6). But their direction is 84–92% the old one, and steering along it sits on the same philosophy-for-helpfulness line. A constant push at every token can't express "answer first, then reflect" (Results §7).
8. **Asking beats steering.** A system prompt asking for an answer with brief philosophical reflections woven in scores philosophy 3.44, coherence 8.81 and helpfulness 7.06, better than any steering setting. Adding a light push (α 0.35) on top gives the highest philosophy so far (4.62), but helpfulness drops at the usual steering rate. A direction built from the model's own prompted answers breaks the model at the strengths we tried (Results §8).
9. **When we push matters less than expected.** Steering only the first 24 tokens or only the prompt gives the same result as steering everything: the start sets a mode that persists. A push that arrives mid-answer adds little philosophy and breaks coherence (Results §9).
10. **Self-judging is unreliable.** Qwen judging its own outputs rated identity-confused loops ("I am the only one who has been told…") as both philosophical and coherent (≈6.5/9). A blind external judge flagged 13–14 of 16 of those as degenerate. Perplexity doesn't catch loops either.
11. **A tiny trained update gives a prompt-free "philosophy Qwen".** Training a LoRA so the model answers *without* the prompt the way it does *with* the woven prompt matches the prompt's balance on held-out prompts: philosophy 2.25–2.50, helpfulness 6.2–6.6, against the prompt's 2.94 / 6.44. A rank-1 update to one matrix (layer 8's `down_proj`, 11,776 parameters) is as good as a 9M-parameter adapter. Its push is token-dependent, mostly on the prompt and the first answer tokens, smaller than any steering push that worked, and nearly orthogonal to the diff-means direction (Results §10).

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

### 4. Downstream pathway check

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

### 5. Less overdrive: dose and group-size sweep

Follow-up to §4. I shrank the L8 push in two ways: smaller α, and larger OMP groups, which align better with the direction and so push less off-direction for the same α. Each OMP config has one matched random group. All 22 configs were judged together, blind, by the Claude judge (`scripts/dose_sweep.py`, `runs/claude_judge_dose.jsonl`). *Off L9–13* is the off-pathway disturbance from §4, averaged over L9–13, in units of the natural response.

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

### 6. Helpfulness ceiling: the target itself is unhelpful

I had the Claude judge score the natural answers from the 32 contrast pairs, which are the texts the direction is built from. They were pooled blind with four steered configs from §5 as anchors (`runs/claude_judge_natural.jsonl`).

| Texts | Philo | Coherence | Helpful | Degenerate |
|---|---|---|---|---|
| Natural plain answers (32) | 0.00 | 9.00 | 8.81 | 0/32 |
| **Natural philosophical answers (32)** | **5.75** | 8.50 | **2.31** | 0/32 |
| Anchor: baseline | 0.00 | 9.00 | 8.44 | 0/16 |
| Anchor: diff-means α=0.5 | 2.50 | 7.00 | 3.94 | 0/16 |
| Anchor: 256 neurons α=0.35 | 1.06 | 8.06 | 6.44 | 0/16 |
| Anchor: 256 neurons α=0.5 | 2.94 | 6.44 | 3.06 | 0/16 |

The philosophical half of each contrast pair *replaces* the practical answer instead of adding to it. For example, the coffee answer reflects on ritual and never says how to make coffee. So the diff-means direction encodes "reflect instead of answering", and the helpfulness loss is built into the target. Steering at α 0.5 gets about half of the natural philosophy gain (2.5–2.9 of 5.75) and pays 70–85% of the natural helpfulness drop (4.5–5.4 of 6.5). The extra cost per unit of philosophy fits the generic push cost from §5. The anchors agree with their §5 scores to within 0.4.

Caveats: the natural answers are short (median 38–48 tokens), hand-written, and answer different questions than the evaluation prompts, so the comparison with steered outputs is approximate.

### 7. "Answer + reflect" contrast data (experiment A)

Branch `exp-answer-reflect`. `data/contrast_pairs_v2.json` keeps the 32 plain answers and adds two philosophical versions that keep **all** the practical content:
- **appended**: the plain answer word for word, then one or two reflective sentences.
- **woven**: the same facts and steps, with the reflection worked into the answer.

The original pairs are untouched. Directions are per-layer diff-means against the plain answers. Everything was judged in one blind pool, including the new target texts themselves (`scripts/exp_answer_reflect.py`, `runs/answer_reflect/`).

The new directions are mostly the old one. At L8, cos with the original direction is 0.92 for appended and 0.84 for woven, and 0.86 between the two new ones. The natural L8 difference is smaller: 0.095·p50 for appended and 0.065·p50 for woven, against 0.24 for the original.

| Config | Philo | Coherence | Helpful | Degenerate |
|---|---|---|---|---|
| Baseline | 0.00 | 9.00 | 8.56 | 0/16 |
| Original direction α=0.5 | 2.81 | 7.25 | 4.06 | 0/16 |
| Appended α=0.5 | 1.94 | 7.50 | 5.25 | 0/16 |
| Appended α=0.75 | 3.25 | 5.75 | 1.81 | 2/16 |
| Appended α=1.0 | 2.56 | 3.06 | 0.19 | 12/16 |
| Woven α=0.5 | 1.50 | 8.19 | 6.12 | 0/16 |
| Woven α=0.75 | 3.31 | 6.12 | 2.38 | 3/16 |
| Woven α=1.0 | 2.69 | 3.50 | 0.69 | 13/16 |
| Random unit direction α=1.0 | 0.06 | 7.94 | 6.56 | 0/16 |
| *Target texts: appended (32)* | 2.94 | 9.00 | 8.56 | 0/32 |
| *Target texts: woven (32)* | 1.56 | 9.00 | 8.62 | 0/32 |

What this says:
- **The new targets are what we wanted.** The texts themselves are as helpful as the plain answers (8.56–8.62) and moderately philosophical (2.94 appended, 1.56 woven).
- **Steering toward them still hits the same trade-off.** Every new config lies on roughly the same line as the original direction: about 0.6 philosophy gained per point of helpfulness lost. None comes near its own target (appended α=0.75 scores philosophy 3.25 but helpfulness 1.81, while the target text has 8.56). Above α 0.75 the outputs degenerate.
- **So the trade-off comes from the method, not only the data.** A constant push at every token puts the model in "reflecting" mode from the first word. The target texts are *answer first, then reflect*, which is a change over time that one fixed vector can't express. Their direction is also 92% the old one, so the difference between "reflect after answering" and "reflect instead" is mostly not in the mean. That points to steering only part of the answer (experiment C).

### 8. Prompting baseline and a self-generated direction (experiment B)

Branch `exp-selfgen` (`scripts/exp_selfgen.py`, `runs/selfgen/`). Two questions. How well does simply *asking* work? And does a direction built from the model's own answers steer better than one built from our hand-written texts?

- **Prompts** (system message, no steering): *answer-then-reflect* asks for the full answer followed by one or two reflective sentences; *reflect-first* opens with one reflective sentence and then answers; *woven* asks for an answer with brief reflections woven in.
- **Self-generated direction**: the model answered the 32 contrast questions twice, greedily, with and without the answer-then-reflect prompt. Both answers were fed back *without* the prompt, and per-layer diff-means gave the direction.

All 13 configs were judged in one blind pool.

| Config | Philo | Coherence | Helpful | Degenerate |
|---|---|---|---|---|
| Baseline | 0.00 | 9.00 | 9.00 | 0/16 |
| Prompt: answer-then-reflect | 0.50 | 8.69 | 8.31 | 0/16 |
| Prompt: answer-then-reflect, 200 tokens | 1.50 | 8.50 | 8.38 | 0/16 |
| Prompt: reflect-first | 2.31 | 8.69 | 8.12 | 0/16 |
| **Prompt: woven** | **3.44** | **8.81** | **7.06** | 0/16 |
| Prompt reflect-first + original direction α=0.35 | 3.56 | 7.81 | 5.69 | 0/16 |
| **Prompt woven + original direction α=0.35** | **4.62** | 7.94 | 4.56 | 0/16 |
| Self-generated direction L8 α=0.5 | 2.00 | 3.94 | 1.75 | 10/16 |
| Self-generated direction L8 α=0.75 / 1.0 | 2.94 / 3.00 | 2.25 / 2.31 | 0.00 / 0.06 | 16/16 |
| Self-generated direction L12 α=0.5 | 3.25 | 5.25 | 2.62 | 8/16 |
| Self-generated direction L12 α=0.75 / 1.0 | 2.56 / 3.12 | 2.69 / 2.25 | 0.00 | 14–16/16 |

What this says:
- **Asking beats steering.** The woven prompt reaches philosophy 3.44 while keeping coherence (8.81) and most of the helpfulness (7.06). No steering config so far comes close to that combination: the best steers reach philosophy 3 only at helpfulness 3–4. The reflect-first prompt keeps helpfulness at 8.12 with philosophy 2.31.
- **Where the reflection goes matters, because answers are cut at 120 tokens.** With answer-then-reflect, the model writes a long answer and the reflection usually falls after the cut (philosophy 0.50; 1.50 with 200 tokens). Prompts that put the reflection at the start or inside the answer avoid this. The same cut also penalises any "reflect at the end" approach, so experiment C uses 160-token answers.
- **Prompt plus steering adds up, at the usual price.** A light push (α 0.35) on top of the woven prompt gives the highest philosophy so far (4.62, with 0 degenerate outputs), but helpfulness drops from 7.06 to 4.56. That is about 0.5 philosophy per point of helpfulness, the same slope as steering alone.
- **The self-generated direction is a poor steering vector.** It is only 0.43 aligned with the original direction at L8 (0.50 with appended), and its natural size is small (0.073·p50 at L8). So α 0.5 is about 7× the natural difference, and the outputs break: 8–16 of 16 are degenerate at every setting tried. Smaller α (0.1–0.25) was not tested.
- **Judge noise, again.** The first 8 configs were judged twice, once alone and once in this pool. Coherent configs (baseline, answer-then-reflect prompt) moved by at most 0.13. Degenerate configs moved much more, by up to 1.0 in philosophy and 0.9 in coherence, because the judge scores broken text inconsistently.

### 9. Steering only part of the answer (experiment C)

Branch `exp-token-window` (`scripts/exp_token_window.py`, `runs/token_window/`). Experiment A suggested that a constant push can't express "answer first, then reflect". Here the original L8 direction is applied only during part of the generation, using `Steerable.steer_window`:
- **late N**: generated tokens N and later only.
- **early N**: the prompt plus the first N generated tokens.
- **prompt-only**: the prompt tokens only. The push then reaches the answer only through attention to the prompt.

A full window reproduces `steer()` exactly (checked in the script). Answers are 160 tokens long so a late window has room. The baseline, full steering and the woven prompt from experiment B were regenerated at that length in the same blind pool.

| Config | Philo | Coherence | Helpful | Degenerate |
|---|---|---|---|---|
| Baseline | 0.00 | 9.00 | 8.94 | 0/16 |
| Full steering α=0.5 | 3.75 | 7.50 | 4.44 | 0/16 |
| Late 40 α=0.5 / 0.75 / 1.0 | 0.69 / 2.00 / 2.38 | 7.81 / 5.12 / 3.00 | 7.31 / 4.69 / 2.81 | 1 / 4 / 13 of 16 |
| Late 80 α=0.75 / 1.0 | 1.00 / 1.69 | 6.81 / 3.62 | 6.75 / 4.38 | 0 / 13 of 16 |
| Late 40, appended direction α=0.75 | 2.00 | 5.75 | 5.12 | 3/16 |
| Early 24 α=0.75 | 3.69 | 7.56 | 3.88 | 0/16 |
| Prompt-only α=1.0 / 1.5 | 3.31 / 3.12 | 7.75 / 5.69 | 4.50 / 1.50 | 1 / 5 of 16 |
| Prompt woven (no push) | 3.56 | 8.56 | 6.94 | 0/16 |

What this says:
- **A late push doesn't add a reflection; it breaks the answer.** Pushes that start mid-answer give less philosophy for the same helpfulness lost, and many more broken outputs (13/16 at α 1.0). The steered tokens clash with the unsteered text the model has already written. Instead of closing with a reflection, the answer drifts into vague metaphysics ("to grasp the nature of being, one must see that everything is always already known to be otherwise").
- **The start decides the mode.** Steering only the first 24 tokens, or only the prompt, gives about the same result as steering every token: philosophy 3.3–3.7, helpfulness about 4. Once the answer starts in a reflective register it stays there, and when only the prompt is pushed, attention carries the effect into the whole answer.
- **So the trade-off is not about *when* we push.** Windowing moves the same trade-off around, or makes it worse; it never reaches the prompt's balance. The woven prompt is still best (philosophy 3.56, helpfulness 6.94), and full steering looks better at 160 tokens (3.75) than at 120 (2.8–3.0) only because the answer is longer.

### 10. Distilling the prompt into the weights (experiment D)

Branch `exp-distill` (`scripts/exp_distill.py`, `scripts/distill_probe.py`, `runs/distill/`). The woven prompt beats every steering setting, but it isn't a "philosophy Qwen": take the prompt away and the behaviour goes with it. Context distillation (LITERATURE.md H7) trains small low-rank weight updates (LoRA, `W + scale·B·A`, B starts at zero, original weights frozen) so the model answers *without* the prompt the way it answers *with* it.
- **Targets**: the model's own greedy answers, with the woven prompt, to 128 practical questions (`data/train_questions.json`, 96 new ones, plus the 32 contrast questions), cut at the last full sentence. None is an evaluation prompt (asserted in the script).
- **Loss**: next-token cross-entropy on the target answer, given the question with no system prompt.
- **Variants**: `mlp-r8`, rank 8 on gate/up/down of all 32 MLPs (9.0M parameters, 0.2% of the model); `mixers-r8`, rank 8 on the token mixers only (attention q/k/v/o in the 8 full-attention layers, in/out projections of the 24 DeltaNet layers; 4.9M); `down8-r1`, rank 1 on layer 8's `down_proj` only (11,776 parameters). 2 epochs (3 for `down8-r1`), AdamW, one sequence at a time with 4-step gradient accumulation and activation checkpointing. A batch of 4 with full activations needed more than the 24 GB of unified memory and crashed; this setup trains in about 3 minutes.
- **Evaluation**: the usual 16 prompts (8 held out), no system prompt, 120 tokens, blind Claude judge in one pool with the baseline and the prompted model.

| Config | Philo | Coherence | Helpful | Degenerate | Held-out philo |
|---|---|---|---|---|---|
| Baseline | 0.00 | 9.00 | 9.00 | 0/16 | 0.00 |
| Prompt woven (no LoRA) | 2.94 | 8.12 | 6.44 | 0/16 | 2.88 |
| LoRA `mlp-r8`, no prompt | 2.38 | 7.94 | 6.25 | 0/16 | 2.62 |
| LoRA `mixers-r8`, no prompt | 2.50 | 7.88 | 6.19 | 0/16 | 2.25 |
| LoRA `down8-r1`, no prompt | 2.25 | 8.12 | 6.62 | 1/16 | 2.25 |

Final training loss (mean of the last 3 updates, starting from 1.35): `mlp-r8` 0.17, `mixers-r8` 0.28, `down8-r1` 0.57.

What this says:
- **The prompt can be moved into the weights.** All three adapters write in the prompted style with no prompt ("To prepare scrambled eggs is to engage in a small, daily alchemy…") and land close to the prompt on all three scores, held-out prompts included. This is the first intervention with no prompt that keeps the prompt's balance. For comparison, steering at helpfulness 6–6.6 gave philosophy of only about 1.5 (Results §5, §7). The one "degenerate" flag is a fluent answer about changing a light bulb, so it is probably judge noise.
- **The ceiling is the prompt.** Distillation copies the prompt's trade-off; it doesn't improve on it (philosophy 2.3–2.5 vs 2.94). The prompt scored 3.44 in the experiment B pool and 2.94 here, which is the judge's noise between pools.
- **One rank-1 update at L8 is enough.** 11,776 parameters at the same site as all our steering reach the same balance as the 9M-parameter adapter. So the problem was never the place (L8) or the size of the change, but its *form*. The LoRA adds `scale·B·(A·a)`, a push along one fixed direction B whose size depends on the token through A·a. A steering vector is the same push with a constant size.
- **And it pushes in a different direction.** B has cosine 0.13 with the diff-means direction (0.09–0.10 with the appended, woven and self-generated ones). Gradient descent found a route to the prompted behaviour that the contrast data didn't point to.
- **It pushes gently, mostly at the start.** Measured on the 16 evaluation prompts (`runs/distill/probe.json`), its push averages 0.21·p50 on prompt tokens and 0.19·p50 on the first 10 answer tokens, but only 0.12·p50 over the whole answer (0.08 on the baseline's answer text). That is *below* the natural L8 difference (0.24·p50) and 2.5–6× smaller than the α 0.5 steering push, which is the same at every token. It matches experiment C: the start sets the mode, and a small push in the right direction at the right tokens is enough. A constant push big enough to change the mode is too big everywhere else.
- **The token mixers alone are as good as the MLPs.** `mixers-r8` (attention + DeltaNet, no MLP) matches `mlp-r8`. Where the change is made matters little at this scale; what it is trained to do matters most. This doesn't support or refute the attention hypothesis on its own (experiment G tests it directly).

## Caveats

- Small evaluation: 16 prompts, greedy decoding, 96–120 new tokens, one judge model (Claude Sonnet) with one rubric. The Claude judge is strict: the best config averages 3/9 for philosophy.
- The 8 practice prompts were used to pick the shortlist. Held-out prompts agree, but the set is small.
- Contrast data is hand-written by us, so the direction reflects our idea of "philosophical" (Heidegger, Stoics, qualia…).
- Qwen3.5's linear-attention layers ran on the slow PyTorch fallback, so the sweeps are small by necessity.
- Pathway check: the natural change compares *different texts* (philosophical vs plain words), so it includes token-identity effects that no push on plain text can reproduce. These matter most in late layers; the random controls set the floor. The bootstrap intervals in `runs/pathway.json` are percentile intervals over pairs. For cosines and norm ratios they are biased low (resampling adds noise to the natural vector), so compare conditions within the same layer rather than reading them as absolute bounds.
- Not replicated: the *agentic* part of the Astra paper (an LLM diagnosing and editing weights in a loop). Here the search was a fixed grid.

## Suggested next steps

1. **Bias bake-in**: add a bias to `down_proj` at L8 equal to the diff-means push or the 16-neuron push, save as a standalone checkpoint, and confirm it matches steering exactly.
2. **Recover helpfulness**: steer only the first N tokens, or combine with the conditional edit (which preserved helpfulness), to get "answers fully, reflects briefly".
3. **Agent-in-the-loop**: give a Claude agent the explorer tools (push, generate, judge on practice prompts, hidden test prompts) and a forward-pass budget, as in the Astra paper.
4. **Larger evaluation**: 50+ prompts, sampling with several seeds, and two judge models.
5. ~~**Downstream pathway check**~~: done, see Results §4. Follow-ups it suggests:
   - ~~**Less overdrive**~~: done, see Results §5. The disturbance shrinks with push size, but philosophy has a threshold near α 0.5, and most of the helpfulness cost comes with the philosophical shift itself. Open: map α 0.35–0.5 more finely for k=256 and diff-means, and confirm the k=256 α 0.35 middle point on a larger prompt set.
   - **Mid-layer re-injection**: the 16-neuron group loses the natural pathway in L15–23. Add a second, smaller push there, using the natural change at L15–18 as the target. This is the cheapest version of step 6.
   - ~~**Helpfulness ceiling**~~: done, see Results §6. The target answers are unhelpful by construction (helpfulness 2.31).
   - ~~**"Answer + reflect" contrast data**~~: done, see Results §7. The targets are helpful, but steering along their direction keeps the same trade-off.
   - ~~**Distil the prompt**~~: done, see Results §10. A LoRA (even rank 1 at L8) reproduces the woven prompt without it. Open: distil a *better* teacher (prompt + light push, or a stronger prompt), since distillation can only copy the prompt's balance; and turn the rank-1 update into a standalone checkpoint.
6. **Cross-layer / circuit view**: single-layer neuron pushes miss behaviours spread over several layers, conditional (AND-style) interactions between layers, and attention-head contributions. Try coordinated multi-layer steering, activation patching or attribution from L8 to later layers, and, longer term, transcoders or cross-layer transcoders, which work on features rather than raw neurons. Co-activation alone is not enough, because neurons that fire together may only share a cause.

## Reproduce

```bash
python3.14 -m venv .venv && .venv/bin/pip install torch transformers accelerate huggingface_hub safetensors
.venv/bin/python scripts/find_neurons.py                         # ranks, p50, diff-means
.venv/bin/python scripts/sweep.py                                # read-side neurons + diff-means screen
.venv/bin/python scripts/neuron_search.py                        # write-aligned single neurons + OMP groups
.venv/bin/python scripts/neuron_search.py --omp-only --rescale --ks 16 64 256 --alphas 0.5 1.0 --out omp_rescaled.jsonl
.venv/bin/python scripts/validate_generate.py && .venv/bin/python scripts/claude_judge.py
.venv/bin/python scripts/pathway_check.py                        # downstream pathway check (forward passes only, ~1 min)
.venv/bin/python scripts/dose_sweep.py --stage pathway           # dose/group-size sweep: disturbance per config (~3 min)
.venv/bin/python scripts/dose_sweep.py --stage generate && .venv/bin/python scripts/claude_judge.py --input dose_sweep.jsonl --output claude_judge_dose.jsonl
.venv/bin/python scripts/natural_ceiling.py                       # helpfulness ceiling: contrast-pair answers + 4 dose-sweep anchors
.venv/bin/python scripts/claude_judge.py --input natural_ceiling.jsonl --output claude_judge_natural.jsonl
.venv/bin/python scripts/exp_answer_reflect.py && .venv/bin/python scripts/claude_judge.py --input answer_reflect/gen.jsonl --output answer_reflect/judge.jsonl
.venv/bin/python scripts/exp_selfgen.py && .venv/bin/python scripts/claude_judge.py --input selfgen/gen.jsonl --output selfgen/judge.jsonl
.venv/bin/python scripts/exp_token_window.py && .venv/bin/python scripts/claude_judge.py --input token_window/gen.jsonl --output token_window/judge.jsonl
.venv/bin/python scripts/exp_distill.py && .venv/bin/python scripts/claude_judge.py --input distill/gen.jsonl --output distill/judge.jsonl
.venv/bin/python scripts/distill_probe.py                         # size and direction of the rank-1 LoRA's push
.venv/bin/python scripts/astra_edit.py && .venv/bin/python scripts/claude_judge.py --input astra.jsonl --output claude_judge_astra.jsonl
```
