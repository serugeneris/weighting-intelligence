# Steering Qwen3.5-4B toward philosophy — findings so far

*Status: exploratory, 2026-11-04 (experiments D–K added 2026-11-05). Model: `Qwen/Qwen3.5-4B` (bf16, local Apple M5 Pro / MPS). All numbers come from the runs in `runs/` produced by the scripts in `scripts/`.*

## TL;DR

1. **Single neurons do not make Qwen3.5-4B philosophical.** Neither the neurons that *fire* most differently on philosophical text nor the neurons whose output column best *points* at the philosophy direction work alone. At low doses they do nothing; at high doses they break the model, and random neurons break it in the same way.
2. **A group of 16 neurons in layer 8 does work**, if they are chosen by write-alignment (matching pursuit on `W_down` columns) and dosed correctly. Blind Claude judge, 16 prompts (8 held out): philosophy **2.56/9 vs 0.00 baseline vs 0.07 for matched random 16-neuron groups**, 0/16 degenerate outputs, same effect on held-out prompts. This is 0.17% of one layer's neurons.
3. **The full diff-means direction at layer 8 is still the best knob**: philosophy 3.00, coherence 7.62, helpfulness 4.62, 0/16 degenerate. Every effective steer costs helpfulness (baseline 8.44 → 3–5), and part of that cost is generic: matched random pushes also drop it to about 5.
4. **Astra-style weight edits (masked rank-1 on `down_proj`, no bias) are safe but nearly inert for this goal.** Best edit: philosophy 0.06 → 0.50 with *no* loss in coherence or helpfulness. A pure weight edit can only amplify or remove what neurons already write; it cannot add a constant push, and on practical prompts nothing "philosophical" is being written to amplify. *Refined in Results §18: a trained rank-1 edit with a dense read side does add the behaviour (TL;DR 19).*
5. **The L8 pushes partly follow the model's own philosophical pathway, but they overdrive it.** In layers 9–31 the changes caused by the pushes line up with the changes in natural philosophical answers clearly above random controls. The 16-neuron group's match fades toward the random level in L15–23; the full direction keeps more of it and recovers strongly in the last layers. Both pushes are much larger than the natural L8 difference (diff-means 2.1×, 16 neurons 4.6×). The 16-neuron group also triggers an off-pathway disturbance of the same size as random groups with the same doses, so that part depends on push size, not on which neurons are chosen.
6. **Pushing less hard doesn't solve the helpfulness trade-off.** Philosophy only shows up when the on-direction push reaches about α 0.5, roughly 2× the natural L8 difference. Below that it nearly vanishes, and helpfulness comes back with it. At α 0.5, most of the helpfulness loss comes with the philosophical shift itself: diff-means has the smallest disturbance and still drops to 3.94. A better-aligned group (k=256) gives the highest philosophy score so far (3.12) and loses slightly less helpfulness than k=16, but the gain is within judge noise. The reason is the contrast data. The philosophical answers we steer toward *replace* the practical answer (judge: philosophy 5.75, helpfulness 2.31), so the direction itself means "reflect instead of answering".
7. **Fixing the target data is not enough on its own.** New contrast answers that keep the full practical answer and add reflection are as helpful as the plain answers (helpfulness 8.6). But their direction is 84–92% the old one, and steering along it sits on the same philosophy-for-helpfulness line. A constant push at every token can't express "answer first, then reflect" (Results §7).
8. **Asking beats steering.** A system prompt asking for an answer with brief philosophical reflections woven in scores philosophy 3.44, coherence 8.81 and helpfulness 7.06, better than any steering setting. Adding a light push (α 0.35) on top gives the highest philosophy so far (4.62), but helpfulness drops at the usual steering rate. A direction built from the model's own prompted answers breaks the model at the strengths we tried (Results §8).
9. **When we push matters less than expected.** Steering only the first 24 tokens or only the prompt gives the same result as steering everything: the start sets a mode that persists. A push that arrives mid-answer adds little philosophy and breaks coherence (Results §9).
10. **Self-judging is unreliable.** Qwen judging its own outputs rated identity-confused loops ("I am the only one who has been told…") as both philosophical and coherent (≈6.5/9). A blind external judge flagged 13–14 of 16 of those as degenerate. Perplexity doesn't catch loops either.
11. **A tiny trained update gives a prompt-free "philosophy Qwen".** Training a LoRA so the model answers *without* the prompt the way it does *with* the woven prompt matches the prompt's balance on held-out prompts: philosophy 2.25–2.50, helpfulness 6.2–6.6, against the prompt's 2.94 / 6.44. A rank-1 update to one matrix (layer 8's `down_proj`, 11,776 parameters) is as good as a 9M-parameter adapter. Its push is token-dependent, mostly on the prompt and the first answer tokens, smaller than any steering push that worked, and nearly orthogonal to the diff-means direction (Results §10).
12. **Where and how widely we push doesn't change the trade-off.** Spreading the push over 3–11 layers, or recreating the natural difference layer by layer, stays on the same 0.5–0.8 philosophy-per-helpfulness-point line. Only the middle layers (L8–L16) respond to a modest push. Recreating the natural trajectory up to L24 gives the most philosophical prompt-free output so far (4.75) but copies the hand-written targets' unhelpfulness (Results §11).
13. **The prompt doesn't act through a decision at its end.** Pasting the prompted model's state at the last 16 prompt positions (every layer) into an unprompted run recovers under half of the prompt's effect; one position, one layer or a mean "prompt vector" recovers nothing. The answer keeps reading the system prompt while it writes (Results §12).
14. **Attention is a hot spot, but DeltaNet carries as much.** Undoing the system prompt's influence in all 8 full-attention layers removes 83% of its effect; in all 24 DeltaNet layers, 93%. Three heads in layers 11 and 15 remove 38%. A steering push on the prompt reaches the answer through the DeltaNet recurrent state, not the attention cache (Results §13).
15. **Changing the push's geometry doesn't help.** Keeping the residual's length (norm-preserving rotation, or +10%) scores the same as a plain push. Projecting the "drop the answer" component out of the direction leaves no philosophy at all: in mean-difference data, "be philosophical" and "stop answering" are the same direction (Results §14).
16. **A better teacher doesn't make a better student here.** A prompt search and best-of-4 selection (judge-picked answers) gave students that copy their teacher's average balance, not its best answers. All prompts and students share one frontier (philosophy + helpfulness ≈ 9.1–9.4), about twice as efficient as steering. The choice of teacher only moves the student along it (Results §15).
17. **Confirmed on 48 unseen prompts with two judges.** Students match their prompts, and at equal philosophy the rank-1 LoRA keeps about 1.7 more helpfulness points than steering (about 5 standard errors) under both Claude Sonnet and Claude Opus. The two judges agree per item (r 0.85–0.90); Opus is about 0.7 stricter on helpfulness (Results §16).
18. **Steering the prompt's hot spots doesn't help.** Adding the prompt's average effect at the three key heads, or at attention L11/L15 or DeltaNet L16–18, does almost nothing: what those sites carry depends on the token, and the average cancels out. Summing the average over all 32 mixers works, but it lands on the same trade-off as ordinary steering (Results §17).
19. **The prompt-free philosophy Qwen is a single rank-1 weight edit.** Baking the rank-1 LoRA into layer 8's `down_proj` (no bias, no hooks) scores the same as the adapter (2.62 / 8.44 / 6.75). Hand-built rank-1 edits failed because they read from a few neurons that don't fire on practical prompts. The trained edit reads a dense mix of all 9,216 neurons that fires at the start of every answer (Results §18).
20. **The trained update's direction is a much better steering vector.** A constant push along the LoRA's write direction B (cosine 0.13 with diff-means) beats diff-means at every strength: philosophy 3.94 / helpfulness 5.12 at 0.6·p50, and 5.12 philosophy at 0.8·p50 with no broken outputs. Amplifying the LoRA itself (×1.5–2) gives philosophy 3.4 at helpfulness about 6 (on 48 new prompts this lands on, not past, the prompt's frontier; TL;DR 22). At its own size, the LoRA needs both its gate and its direction (Results §19).
21. **Preference training (DPO) on judged pairs nudged a student up on both scores at once** (16 prompts: philosophy 2.19 → 2.50, helpfulness 7.75 → 7.94; ×1.5 gain: 3.00 / 7.12). This is the first intervention to improve both scores, but the gain is about the size of the judge's noise (Results §20; checked on 48 prompts in §21).
22. **Confirmed on 48 unseen prompts, greedy and sampled.** The learned direction B beats diff-means by +2.06 in philosophy + helpfulness (paired, about 7 standard errors). The amplified rank-1 LoRA moves *along* the prompt's frontier (×2: philosophy 3.5, helpfulness 5.6), not past it. Only the DPO student shows evidence of beating the frontier: at ×1.5 it matches the woven prompt's philosophy with +0.44 ± 0.22 helpfulness, which is suggestive rather than settled (Results §21).

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
- **The start decides the mode.** Steering only the first 24 tokens, or only the prompt, gives about the same result as steering every token: philosophy 3.3–3.7, helpfulness about 4. Once the answer starts in a reflective register it stays there, and when only the prompt is pushed, the token mixers carry the effect into the whole answer. (This originally said "attention"; experiment G found the route is mostly the DeltaNet state, see §13.)
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

### 11. Several layers, other layers (experiment E)

Branch `exp-multilayer` (`scripts/exp_multilayer.py`, `runs/multilayer/`). The pathway check (§4) found the L8 push 2–5× bigger than the natural L8 difference, and losing the natural route in the middle layers. Two ways to push more gently and more naturally:
- **spread**: the same total push split over several layers, each along its own diff-means direction, `Σ_L (α/n)·p50_L·unit(dm_L)`.
- **natural path ×g up to L K**: push every layer up to K by g times the *increment* of the natural difference, `g·(dm_L − dm_{L−1})`. The residual stream is a running sum, so at g = 1 the steered offset after each layer equals the natural philosophical-minus-plain difference at that layer, before the model reacts to it.
- **layer sweep** (LITERATURE.md H5, "abstraction lives in the middle"): one layer at a time, L4–L24, at α 0.5 and at 2× that layer's own natural difference.

All 25 configs were judged blind in one pool (16 prompts, 8 held out, 120 tokens).

| Config | Philo | Coherence | Helpful | Degenerate |
|---|---|---|---|---|
| Baseline | 0.00 | 9.00 | 8.81 | 0/16 |
| Single L8 α 0.5 (the usual) | 2.38 | 7.50 | 4.62 | 0/16 |
| Spread L8–16 (3 layers) α 0.5 / 1.0 | 2.31 / 3.12 | 7.50 / 3.44 | 5.44 / 0.75 | 0 / 10 of 16 |
| Spread L4–20 (5 layers) α 0.5 / 1.0 | 1.12 / 4.06 | 8.38 / 5.00 | 7.44 / 1.62 | 0 / 5 of 16 |
| Spread L4–24 every 2 (11 layers) α 0.5 / 1.0 | 0.50 / 4.19 | 8.44 / 5.69 | 7.38 / 2.19 | 0 / 1 of 16 |
| Natural path ×1 up to L12 / L24 | 1.19 / 4.75 | 8.38 / 6.12 | 7.25 / 1.62 | 0 / 2 of 16 |
| Natural path ×2 up to L12 / L24 | 2.25 / 1.19 | 2.31 / 0.94 | 0.56 / 0.00 | 14 / 16 of 16 |
| Natural path ×3 up to L12 / L24 | 0.38 / 0.25 | 0.06 / 0.00 | 0.00 / 0.00 | 16 / 16 of 16 |
| Single L4 α 0.5 / 2×natural | 0.12 / 0.00 | 8.75 / 9.00 | 8.06 / 8.81 | 0 / 0 of 16 |
| Single L12 α 0.5 / 2×natural | 2.12 / 2.06 | 5.94 / 5.00 | 4.31 / 3.50 | 0 / 7 of 16 |
| Single L16 α 0.5 / 2×natural | 2.88 / 3.81 | 7.31 / 5.56 | 5.06 / 2.44 | 0 / 0 of 16 |
| Single L20 α 0.5 / 2×natural | 0.38 / 2.06 | 8.56 / 5.88 | 7.94 / 4.44 | 0 / 0 of 16 |
| Single L24 α 0.5 / 2×natural | 0.06 / 0.38 | 8.81 / 6.44 | 8.31 / 6.31 | 0 / 1 of 16 |

Single L8 at 2× its natural difference (0.48·p50) is nearly the same push as α 0.5 and scores the same (2.06 / 7.50 / 5.00).

What this says:
- **Spreading the push doesn't escape the trade-off.** Every config that adds philosophy lies on the same line as single-layer steering: about 0.5–0.8 philosophy gained per helpfulness point lost (single L8: 0.57; spread L8–16 α 0.5: 0.69; spread L4–20 α 1.0: 0.56). Gentler per-layer pushes mostly just do less.
- **Recreating the natural trajectory recreates the target texts, including their unhelpfulness.** Natural path ×1 up to L24 gives the highest philosophy of any prompt-free intervention so far (4.75), at helpfulness 1.62. That is close to how the judge scores the hand-written philosophical answers themselves (5.75 / 2.31, §6). The push does what it was built to do: it moves the model toward texts that *replace* the answer. Doubling it breaks the model.
- **The middle layers are where a push works (H5, partly supported).** At α 0.5, only L8–L16 respond (philosophy 2.1–2.9). L4 does nothing even at 2× its natural difference; L20 and L24 need 2× natural to show anything, and then lose coherence. L16 at α 0.5 is slightly better than L8 (philosophy 2.88 vs 2.38, helpfulness 5.06 vs 4.62), but that is within judge noise.
- **So the place and spread of a constant push matter less than its form.** No layer or combination reaches the prompt's balance. The rank-1 LoRA of §10 sits at the same L8 site with a smaller push; what it adds is a token-dependent size.

### 12. Is the prompt's effect stored at the end of the prompt? (experiment F)

Branch `exp-prompt-transplant` (`scripts/exp_prompt_transplant.py`, `runs/prompt_transplant/`). Two literature leads (LITERATURE.md H1, H8) say a prompt mostly acts through a few positions where the model "decides" what kind of answer to write. With the chat template, the last 9 prompt tokens (`<|im_end|>`, `\n`, `<|im_start|>`, `assistant`, `\n`, `<think>`, `\n\n`, `</think>`, `\n\n`) are the same for every prompt, so they are a natural decision point. Experiment C also found that pushing only the prompt tokens is enough.
- **Transplant**: run the model *with* the woven prompt, record the residual stream at the last k prompt positions, and paste it into the run *without* the prompt at the same positions, at every layer or at one layer. The answer is then generated with no system prompt. A question here is 19–25 tokens, so the last 16 positions cover the whole question as well as the header.
- **Prompt vector**: the mean (prompted − unprompted) residual at the last k positions of one layer, computed on the 32 contrast questions (never evaluation prompts) and added at those positions with no prompt. Its size at L8 is 0.28·p50 (k = 9) and 0.37·p50 (k = 1).

| Config | Philo | Coherence | Helpful | Degenerate |
|---|---|---|---|---|
| Baseline | 0.06 | 9.00 | 8.44 | 0/16 |
| Prompt woven | 2.94 | 8.44 | 6.81 | 0/16 |
| Transplant last 1, all layers | 0.19 | 8.88 | 8.12 | 0/16 |
| Transplant last 9, all layers | 1.12 | 8.69 | 7.81 | 0/16 |
| Transplant last 16, all layers | 1.38 | 8.62 | 7.69 | 0/16 |
| Transplant last 9, L8 only / L16 only | 0.00 / 0.12 | 8.81 / 8.88 | 8.31 / 8.50 | 0/16 |
| Prompt vector last 9, L8 ×1 / ×2 | 0.00 / 0.00 | 8.94 / 9.00 | 8.88 / 8.56 | 0/16 |
| Prompt vector last 9, L16 ×1 / ×2 | 0.06 / 0.75 | 8.88 / 8.75 | 8.38 / 8.12 | 0/16 |
| Prompt vector last 1, L8 ×2 | 0.00 | 8.94 | 8.38 | 0/16 |

What this says:
- **The prompt's effect is not stored at the end of the prompt.** Even pasting the full prompted state of the last 16 positions (header and question) at every layer recovers under half of the prompt's philosophy (1.38 of 2.94). The last position alone recovers almost nothing. While it writes, the answer keeps reading the system prompt itself, through attention to its tokens and through the DeltaNet state that summarises them. Neither of those exists in the unprompted run.
- **A single layer or a mean offset does nothing.** Transplanting the last 9 positions at one layer, or adding the average prompt offset there (even ×2), leaves the answer plain. The prompt's information at those positions is spread over layers and is specific to each question, not one shared offset.
- **Why prompt-only steering still worked (§9).** That push was large (α 1.0), went into *every* prompt position, and pointed along a direction that already means "philosophical text". The prompt's own end-state is smaller and less specific. So H1/H8 in their strong form ("the decision sits in a few tokens") are not supported here: the prompt acts over the whole answer, more like a reference the answer keeps returning to.

### 13. Which route carries the philosophical mode: attention or DeltaNet? (experiment G)

Branch `exp-routes` (`scripts/exp_routes.py`, `runs/routes/`). The user's original lead, and LITERATURE.md H3: philosophy relates ideas, so the lever might be attention, the part of the model that relates tokens. Qwen3.5-4B moves information between tokens in two ways. 8 layers (3, 7, …, 31) use full attention (16 heads each). The other 24 use Gated DeltaNet, a linear-attention layer that carries a running recurrent state instead of attending to each past token. Experiment C's prompt-only result can't tell these apart, so two direct tests were run.

**Part 1: mediation of the woven prompt** (forward passes only). For 16 contrast questions, we take the prompted model's woven answer A_P and the unprompted answer A_U, and measure the preference `pref = mean log-prob per token of A_P − of A_U`. The prompt raises it by E = 1.08 nats per token on average (0.66–1.33, positive for every question). In the prompted run, we then restore one token mixer's output, at every position after the system block, to its value in the unprompted run. The share of E that disappears is that mixer's share of the prompt's route.

| Restored (prompted run → unprompted values) | Share of the prompt's effect removed |
|---|---|
| All 8 full-attention layers | 83% |
| All 24 DeltaNet layers | 93% |
| One full-attention layer, mean (sum of the 8) | 12% (97%) |
| One DeltaNet layer, mean (sum of the 24) | 6% (153%) |
| Largest single layers | L11 attention 32%, L18 DeltaNet 29%, L15 attention 24%, L16 DeltaNet 20%, L7 attention 19% |
| Largest single heads | L15 head 8 19%, L15 head 2 11%, L11 head 0 8% (top 3: 38%; top 10: 66%) |
| Heads by layer (sum of the 16) | L3 9%, L7 11%, L11 19%, L15 34%, L19 9%, L23–31 ≈ 0 |

**Part 2: routes of prompt-only steering.** The L8 push (α 1.5) is applied to the prompt only, and the prompt's cache is built twice, with and without the push. We then generate, with no push, from a mixed cache: only the full-attention keys/values from the steered run, only the DeltaNet states, both, or neither. Judged blind (16 prompts).

| Cache taken from the steered run | Philo | Coherence | Helpful | Degenerate |
|---|---|---|---|---|
| Nothing (= baseline) | 0.00 | 9.00 | 9.00 | 0/16 |
| Full-attention keys/values only | 0.00 | 8.94 | 8.88 | 0/16 |
| DeltaNet states only | 1.19 | 8.56 | 6.94 | 0/16 |
| Both | 2.88 | 7.19 | 3.12 | 1/16 |

Sanity check: a rebuilt, unsteered cache reproduces plain generation exactly on 6/16 prompts. On the rest it drifts after 23–500 characters, because a split prefill rounds slightly differently in bf16. The text stays plain (philosophy 0.00).

What this says:
- **Full attention is a hot spot, not the whole story.** Per layer, a full-attention layer carries twice the prompt's effect of a DeltaNet layer (12% vs 6%), and a handful of heads in the middle layers carries a large share: three heads (L15.8, L15.2, L11.0) remove 38%, like the "few style heads" in the persona-steering literature. The late attention layers (23–31) carry nothing.
- **But both routes are needed, in series.** Restoring all attention layers removes 83% and restoring all DeltaNet layers removes 93%, so neither route alone carries the prompt; they hand information to each other. The busiest region is L11–L18, where attention layers 11 and 15 sit next to DeltaNet layers 16–18.
- **For our steering push, the route is the DeltaNet state, not attention.** A push on the prompt reaches the answer through the DeltaNet recurrent state: alone it gives 41% of the effect, with helpfulness kept at 6.94. The attention keys/values alone give nothing, though they add the rest when combined with the state. Results §9 had credited attention; this corrects it.
- **Verdict on the attention hypothesis (H3): partly supported for prompts, not supported for steering.** A few full-attention heads matter disproportionately to how a system prompt shapes the answer, so head-level interventions at L11/L15 are a reasonable next target. But the 24 DeltaNet layers, which attention-centred papers on full-attention models don't have, carry at least as much. For this hybrid model, "attention is the lever" is too narrow.

### 14. A different push geometry: orthogonal and norm-preserving (experiment H)

Branch `exp-geometry` (`scripts/exp_geometry.py`, `runs/geometry/`). Two ideas to make the L8 push cheaper, both at the same site and size as before.
- **Remove "drop the answer" from the direction.** The original contrast pairs compare a plain answer with a philosophical text that *replaces* it; experiment A's "appended" pairs add the reflection *after* the same answer. So `drop = dm_original − dm_appended` approximates "stop answering", and a direction with `drop` projected out might keep the reflection and lose less of the answer. Measured first, at L8: the appended direction has cosine 0.92 with the original and 40% of its size, and `drop` has cosine 0.97 with the original. All three are nearly the same direction, and what remains after projecting `drop` out has cosine 0.23 with the original (0.15 for the woven direction). Because `drop` lies in the plane of the original and appended directions, "original ⊥ drop" and "appended ⊥ drop" are the same vector. The two runs produced identical text, which gives a free measure of judge noise (≤ 0.3 on every score).
- **Norm-preserving pushes** (LITERATURE.md H4: Selective Steering, COAST, the angle/norm paper). Adding a vector also lengthens the residual, which later layers have never seen. Here the residual after L8 turns toward the direction by the same angle as the plain push but keeps its length (`h' = (h + v)·|h| / |h + v|`), or grows 10% (the "mild norm increase" reported best for refusal).

| Config | Philo | Coherence | Helpful | Degenerate |
|---|---|---|---|---|
| Baseline | 0.00 | 9.00 | 8.75 | 0/16 |
| Add α 0.5 / 0.75 (the usual push) | 2.88 / 4.00 | 7.50 / 5.12 | 4.38 / 1.25 | 0 / 6 of 16 |
| Norm-preserving α 0.5 / 0.75 / 1.0 / 1.5 | 2.75 / 3.50 / 2.19 / 1.19 | 7.56 / 4.94 / 2.00 / 0.50 | 4.62 / 1.31 / 0.06 / 0.00 | 0 / 3 / 15 / 16 of 16 |
| Norm +10% α 0.75 / 1.0 | 3.81 / 2.00 | 4.88 / 2.50 | 1.31 / 0.19 | 4 / 14 of 16 |
| Original (= appended) ⊥ drop α 0.5 | 0.00 | 9.00 | 8.56 | 0/16 |
| Original (= appended) ⊥ drop α 1.0 | 0.44 | 4.75 | 1.69 | 1/16 |
| Woven ⊥ drop α 0.5 / 1.0 | 0.06 / 0.00 | 8.94 / 4.00 | 8.44 / 2.88 | 0 / 4 of 16 |

What this says:
- **Keeping the residual's length changes nothing.** Norm-preserving and norm +10% pushes score the same as the plain push at α 0.5 and 0.75, within judge noise, and break the same way above that. The helpfulness cost isn't caused by the residual getting longer. It comes with the direction itself, as §5–§6 suggested.
- **There is no philosophy left once "drop the answer" is removed.** Pushes along the remaining directions add no philosophy at α 0.5. At α 1.0 they break the answer (helpfulness 1.7–2.9) and still add none: outputs start mid-sentence ("it's about consistency and attention to detail.") or in lowercase fragments. In this data, "be philosophical" and "stop answering" are the same direction, so a mean-difference vector can't separate them.
- **So the geometry isn't the problem; the constant direction is.** This rules out the cheapest fixes for the trade-off. The approaches that kept helpfulness were a prompt and a trained, token-dependent update (§8, §10).

### 15. Distilling a better teacher (experiment J)

Branch `exp-teacher` (`scripts/exp_teacher.py`, `runs/teacher/`). Experiment D's student copied the prompt's balance but couldn't beat it. Two ways to raise the teacher, using only training questions:
- **Prompt search.** Six system prompts on 16 training questions (blind judge): the two best from experiment B and four new ones that stress practical detail (`CANDIDATES` in the script). The winner by mean philosophy + helpfulness was *reflect-first* (2.06 / 7.69; woven 2.94 / 5.94, frame 2.19 / 7.31, practical-philosopher 1.25 / 7.88, named-idea 0.19 / 8.38, practical-first 0.31 / 8.12).
- **Best-of-4 (rejection sampling).** Four sampled answers (temperature 0.8) with the reflect-first prompt for each of the 128 training questions, 512 judged blind. For each question we kept the answer with the highest philosophy + helpfulness (non-degenerate, coherence ≥ 7). The kept answers average 2.50 / 7.98 / coherence 8.52, against 2.12–2.23 / 7.34–7.51 for single samples.
- **Students**, trained exactly as in experiment D: rank-1 L8 on the reflect-first greedy answers, and rank-1 L8 and rank-8 MLP on the best-of-4 answers. They were evaluated without a prompt on the usual 16 prompts, in one pool with both prompts and experiment D's woven student (reloaded).

| Config | Philo | Coherence | Helpful | Degenerate | Philo + help |
|---|---|---|---|---|---|
| Baseline | 0.00 | 9.00 | 8.94 | 0/16 | 8.94 |
| Prompt woven | 2.81 | 8.19 | 6.38 | 0/16 | 9.19 |
| Prompt reflect-first | 1.88 | 8.19 | 7.56 | 0/16 | 9.44 |
| Student of woven (exp D, rank 1 L8) | 2.56 | 7.94 | 6.75 | 1/16 | 9.31 |
| Student of reflect-first (rank 1 L8) | 1.44 | 8.19 | 7.62 | 0/16 | 9.06 |
| Student of best-of-4 (rank 1 L8) | 1.38 | 8.31 | 7.56 | 0/16 | 8.94 |
| Student of best-of-4 (rank 8 MLP) | 1.69 | 8.19 | 7.56 | 0/16 | 9.25 |

What this says:
- **A student copies its teacher's average, not its best answers.** The best-of-4 students score like the plain reflect-first student and its prompt, not like the selected targets (2.50 / 7.98). The selection gain was small: about 0.35 philosophy and 0.55 helpfulness over a random sample, the same size as the judge's noise (§5, §14). So selection mostly kept answers the judge happened to score high, and that isn't something a student can learn.
- **Prompts and their students share one frontier.** Every prompt and student here has philosophy + helpfulness of about 9.1–9.4. Moving along it trades about one helpfulness point per philosophy point. Steering trades about two (0.5–0.8 philosophy per point, §5–§14). Choosing a teacher picks a point on this frontier: woven is more philosophical, reflect-first more helpful. Neither moves the frontier.
- **The experiment D result replicates.** Reloaded in a new pool, the woven student scores 2.56 / 6.75 (2.25 / 6.62 before), matching its prompt again.
- **To go beyond the frontier you need a stronger signal than 4 noisy samples.** Options: many more samples per question, pairwise preference training (DPO) instead of imitation, or a stronger teacher model. None of these were run here.

### 16. Confirmation on 48 new prompts with two judges (experiment I)

Branch `exp-confirm` (`scripts/exp_confirm.py`, `runs/confirm/`). Every earlier comparison used the same 16 prompts, one judge model and greedy decoding. Here the main candidates were re-run on `data/confirm_prompts.json`: 48 new practical prompts that no experiment had seen (not in the evaluation, held-out or training sets; asserted in the script). Greedy decoding, 120 tokens, one blind pool, scored separately by Claude Sonnet and Claude Opus. ± is the standard error over the 48 prompts (Sonnet).

| Config | Philo (Sonnet) | Help (Sonnet) | Coherence (Sonnet) | Philo (Opus) | Help (Opus) | Degenerate (S / O) |
|---|---|---|---|---|---|---|
| Baseline | 0.06 ± 0.04 | 8.69 ± 0.12 | 8.92 | 0.02 | 8.23 | 1 / 0 of 48 |
| Prompt woven | 2.94 ± 0.16 | 6.19 ± 0.21 | 8.15 | 2.67 | 5.25 | 0 / 0 |
| Prompt reflect-first | 1.88 ± 0.10 | 7.75 ± 0.10 | 8.46 | 1.79 | 7.75 | 0 / 0 |
| Prompt woven + push α 0.35 | 3.46 ± 0.14 | 4.02 ± 0.20 | 7.08 | 3.19 | 2.85 | 0 / 0 |
| Steering, diff-means L8 α 0.5 | 2.67 ± 0.14 | 4.10 ± 0.25 | 6.69 | 2.42 | 2.98 | 1 / 1 |
| LoRA rank 1 L8, woven student (D) | 2.75 ± 0.15 | 5.81 ± 0.25 | 8.10 | 2.67 | 4.69 | 0 / 0 |
| LoRA rank 8 MLP, woven student (D) | 2.62 ± 0.17 | 6.00 ± 0.17 | 8.08 | 2.65 | 5.00 | 0 / 1 |
| LoRA rank 1 L8, reflect-first student (J) | 1.62 ± 0.11 | 7.79 ± 0.11 | 8.50 | 1.56 | 7.46 | 0 / 0 |
| LoRA rank 8 MLP, best-of-4 student (J) | 1.85 ± 0.11 | 7.60 ± 0.13 | 8.35 | 1.67 | 7.52 | 0 / 0 |

Judge agreement, per item over all 432 outputs: Pearson 0.85 (philosophy), 0.79 (coherence), 0.90 (helpfulness). Opus gives 0.69 less helpfulness on average, mostly on the more philosophical answers, and 0.14 less philosophy.

What this says:
- **The main results hold on new prompts and with a second judge.** The ordering is the same under both judges. Each student lands next to its own prompt (woven student 2.75 / 5.81 vs woven prompt 2.94 / 6.19; reflect-first student 1.62 / 7.79 vs 1.88 / 7.75), all within about two standard errors.
- **Students beat steering clearly.** At the same philosophy as diff-means α 0.5 (2.67), the rank-1 woven student keeps 1.7 more helpfulness points under either judge (5.81 vs 4.10 Sonnet, 4.69 vs 2.98 Opus) and much more coherence (8.10 vs 6.69). With 48 prompts this is about 5 standard errors.
- **The rank-1 adapter is as good as the 9M-parameter one** (2.75 / 5.81 vs 2.62 / 6.00), confirming §10.
- **The most philosophical usable option is still prompt + push** (3.46), at a steering-sized helpfulness cost (4.02). Nothing tested escapes the trade-off; the prompt and its students are the best frontier found.
- **The second judge is harsher on helpfulness for reflective answers.** That widens the gap between the reflective configs and the baseline, but it doesn't reorder anything.

### 17. Steering at the prompt's hot spots (experiment K)

Branch `exp-heads` (`scripts/exp_heads.py`, `runs/heads/`). Experiment G found where the woven prompt's effect travels: three attention heads (L15.8, L15.2, L11.0) carry 38% of it, attention layers 11 and 15 the most of any layer, and DeltaNet layers 16–18 right after them. The head-steering literature (LITERATURE.md H3) reports that steering a few heads beats residual steering on the trait-vs-coherence trade-off. So we built steering vectors from the prompt itself, at those sites. For the 32 contrast questions, the prompted model's woven answer is fed after the prompted and after the unprompted context. At each site we take the mean (prompted − unprompted) output over the answer tokens: per head, its 256-number slice of `o_proj`'s input; per layer, the token mixer's output. During generation with no prompt we add g × that mean at every position.

The means are small. The top heads write 0.016–0.029·p50 into the residual, and the mixers at L11–L18 write 0.04–0.06·p50.

| Config (round 1) | Philo | Coherence | Helpful | Degenerate |
|---|---|---|---|---|
| Baseline | 0.00 | 9.00 | 8.75 | 0/16 |
| Top-3 heads ×1 / ×2 / ×4 / ×8 | 0.00 / 0.12 / 0.06 / 0.50 | 9.00 / 9.00 / 9.00 / 8.69 | 8.75 / 8.44 / 8.94 / 8.00 | 0/16 |
| Attention L11 + L15, all heads ×1 / ×2 / ×4 | 0.06 / 0.06 / 0.06 | 9.00 | 8.88 / 8.62 / 8.75 | 0/16 |
| DeltaNet L16–18 ×1 / ×2 / ×4 | 0.06 / 0.06 / 0.69 | 9.00 / 8.94 / 8.62 | 8.81 / 8.69 / 8.12 | 0/16 |
| All 32 token mixers ×0.5 / ×1 / ×2 | 0.06 / 0.25 / 2.25 | 9.00 / 8.81 / 7.25 | 8.62 / 8.38 / 6.06 | 0 / 0 / 1 of 16 |

Only the all-mixer setting responded, so a second pool mapped it next to the woven prompt and the usual L8 push:

| Config (round 2) | Philo | Coherence | Helpful | Degenerate |
|---|---|---|---|---|
| Baseline | 0.00 | 9.00 | 8.94 | 0/16 |
| Prompt woven | 2.69 | 8.25 | 6.50 | 0/16 |
| Diff-means L8 α 0.5 | 2.88 | 7.31 | 4.88 | 0/16 |
| All 32 token mixers ×1.5 / ×2 / ×2.5 / ×3 | 1.12 / 2.50 / 3.06 / 3.94 | 8.56 / 7.50 / 6.19 / 5.31 | 7.94 / 6.00 / 4.25 / 2.88 | 0 / 0 / 0 / 1 of 16 |

What this says:
- **A hot spot for the prompt is not a good place for a constant push.** The prompt's effect at the top heads is large in mediation (§13) but tiny on average: what the head adds depends on the token, and the mean cancels out. Adding that mean, even ×8, barely moves the answer. Mediation finds where the information flows, not a vector to inject.
- **Spreading the prompt's mean effect over all 32 mixers works, but not better than steering.** At ×2 it reaches 2.50 / 6.00, the same in both pools, and sits about 0.6 helpfulness points above the diff-means line at equal philosophy. That is within judge noise on 16 prompts. At ×2.5–3 it falls back onto the same line (3.06 / 4.25, 3.94 / 2.88). Like experiment E, a constant push spread over many layers stays on the steering trade-off.
- **H3 for steering: not supported.** Head-level steering at the sites that matter most for the prompt does not beat the residual push in this model, at least with mean (prompt-derived) vectors. The token-dependent route (the LoRA of §10) remains the only prompt-free intervention that reaches the prompt's frontier.

### 18. A prompt-free philosophy Qwen in one weight edit (baking experiment D)

Branch `exp-bake` (`scripts/bake_lora.py`, `runs/bake/`). The rank-1 LoRA of §10 adds `scale·B·(A·a)` to layer 8's `down_proj` output, which is the same as editing that weight once: `W_down,8 ← W_down,8 + scale·B·Aᵀ`. That is a rank-1, bias-free edit, the same *form* as the Astra-style edits of §3, which were nearly inert. The script merges it into the stored weight (its size: 10% of ‖W_down,8‖) and regenerates the 16 evaluation prompts with no prompt and no adapter. `--out DIR` saves the full model.

| Config | Philo | Coherence | Helpful | Degenerate |
|---|---|---|---|---|
| Baseline | 0.00 | 9.00 | 8.75 | 0/16 |
| Rank-1 LoRA (adapter) | 2.81 | 8.19 | 6.81 | 1/16 |
| Same update baked into `W_down,8` | 2.62 | 8.44 | 6.75 | 0/16 |

5 of 16 generations are byte-identical to the adapter's. The rest drift after 0–470 characters, because the merged weight is rounded to bf16 while the adapter computed its correction in float32. The judged scores match within noise.

What this says:
- **A standalone, prompt-free philosophical Qwen3.5-4B is one rank-1 edit to one matrix.** No prompt, no hooks, no bias.
- **This refines TL;DR 4.** The hand-built Astra-style edits were inert because their read side was a few hand-picked neurons (masked `v`) that don't fire on practical prompts, so there was nothing to amplify. The trained edit reads a *dense* combination `A` of all 9,216 neurons, which fires on any prompt and on the opening of any answer (0.2·p50, §10), and writes along a direction `B` that diff-means never pointed to. A bias was never needed. A weight edit can add a behaviour if its read side is chosen for it, and gradient descent found one where hand-construction didn't.

### 19. Why the rank-1 update works: gate vs direction (experiment L)

Branch `exp-gate` (`scripts/exp_gate.py`, `runs/gate/`). The rank-1 LoRA adds `g_t · unit(B)` at layer 8, where `g_t = scale·|B|·(A·a_t)` is a token-dependent size (the *gate*, from the read side A) and B is a fixed *direction*. Steering uses a constant size along diff-means. Swapping the two parts tests LITERATURE.md H2 (gated or conditional steering) in a form that applies to every prompt: is it the gate or the direction that matters? B has cosine 0.13 with diff-means.

| Config (round 1) | Philo | Coherence | Helpful | Degenerate |
|---|---|---|---|---|
| Baseline | 0.06 | 9.00 | 8.69 | 0/16 |
| LoRA (gate A, direction B) | 2.62 | 8.06 | 6.88 | 1/16 |
| Constant along B, 0.12 / 0.2 / 0.4·p50 | 0.00 / 0.12 / 1.44 | 8.94 / 9.00 / 8.44 | 8.75 / 8.88 / 7.31 | 0/16 |
| Gate A, diff-means direction ×1 / ×2 | 0.19 / 1.12 | 9.00 / 7.12 | 8.81 / 5.69 | 0/16 |
| Gate A, random direction ×1 / ×2 | 0.06 / 0.06 | 8.94 / 9.00 | 8.69 / 8.50 | 0/16 |
| Gate A with its sign flipped, direction B | 0.25 | 8.94 | 8.75 | 0/16 |

0.12 and 0.2·p50 are the LoRA's mean push on answer and on prompt tokens (§10). Round 2 mapped the strongest leads in one pool:

| Config (round 2) | Philo | Coherence | Helpful | Degenerate | Philo + help |
|---|---|---|---|---|---|
| Baseline | 0.00 | 9.00 | 9.00 | 0/16 | 9.00 |
| Diff-means L8 α 0.5 | 2.94 | 6.94 | 4.06 | 0/16 | 7.00 |
| LoRA ×1 / ×1.5 / ×2 | 2.69 / 3.44 / 3.38 | 8.25 / 8.06 / 8.19 | 6.44 / 5.75 / 6.06 | 1 / 0 / 0 of 16 | 9.13 / 9.19 / 9.44 |
| Constant along B, 0.4·p50 | 1.56 | 8.56 | 7.06 | 0/16 | 8.62 |
| Constant along B, 0.5·p50 | 2.69 | 7.94 | 6.06 | 0/16 | 8.75 |
| Constant along B, 0.6·p50 | 3.94 | 7.81 | 5.12 | 0/16 | 9.06 |
| Constant along B, 0.8·p50 | 5.12 | 7.25 | 3.31 | 0/16 | 8.43 |

What this says:
- **Both parts are needed at the LoRA's size.** The gate with the wrong direction (diff-means or random) does nothing at ×1. The right direction at a constant size equal to the LoRA's average does nothing either. Flipping the gate's sign removes the effect, so this is a specific push, not a generic disturbance.
- **But the learned direction B is the bigger discovery.** A *constant* push along B, the ordinary steering recipe with only the direction swapped, beats diff-means at every strength. At 0.6·p50 it gives philosophy 3.94 with coherence 7.81, helpfulness 5.12 and no broken outputs, against diff-means α 0.5 at 2.94 / 6.94 / 4.06. At 0.8·p50 it reaches 5.12, the highest philosophy without broken outputs in the project. Its philosophy + helpfulness (8.4–9.1) sits on or near the prompt's frontier (§15), which no diff-means push reached.
- **Amplifying the LoRA goes past the prompt's point.** ×1.5 and ×2 raise philosophy to 3.4 and keep coherence above 8 and helpfulness near 6 (philosophy + helpfulness 9.2–9.4). In this pool that beats the woven prompt's own balance (2.69 / 6.50 in the §17 pool), so the student can be pushed beyond its teacher by turning up its gain.
- **Why diff-means was a poor direction.** Diff-means measures how philosophical *texts* differ from plain ones, which mixes in "replace the answer" (§6, §14). B was trained to make *this model's answers* look like its own prompted answers, which only needs what the model reads as "write reflectively". A direction learned from outcomes beats one averaged from data.
- Caveat: one pool of 16 prompts. On 48 unseen prompts (§21), B is confirmed, but the amplified LoRA sits on the prompt's frontier rather than past it.

### 20. Preference training on judged pairs (experiment M)

Branch `exp-dpo` (`scripts/exp_dpo.py`, `runs/dpo/`). Experiment J found that imitation copies the teacher's *average*: training on the best of 4 answers didn't help. Direct Preference Optimization (DPO; Rafailov et al., 2023) trains on *pairs* instead. It raises the probability of the preferred answer relative to the rejected one, compared with a frozen reference, so it learns what separates better answers from worse ones.
- **Pairs**: from experiment J's 512 judged samples (4 per training question, reflect-first prompt). Every ordered pair for the same question where the chosen answer is non-degenerate, at least as good on philosophy *and* helpfulness, and better by ≥ 2 in the sum: 175 pairs over 68 questions.
- **Policy**: J's rank-8 MLP best-of-4 student, LoRA trainable. **Reference**: the same student, frozen. The context is the question with no system prompt. β = 0.1, lr 5e-5, 2 epochs, 4 pairs per update.
- Training reward accuracy (share of pairs where the policy prefers the chosen answer more than the reference does) rose to 1.00 in the second epoch, so the policy fits the 175 pairs completely.

| Config (one pool, 16 prompts) | Philo | Coherence | Helpful | Degenerate | Philo + help |
|---|---|---|---|---|---|
| Baseline | 0.00 | 9.00 | 9.00 | 0/16 | 9.00 |
| Prompt woven | 3.12 | 8.06 | 6.75 | 0/16 | 9.87 |
| Starting student (J, best-of-4, rank 8 MLP) | 2.19 | 8.12 | 7.75 | 0/16 | 9.94 |
| DPO student | 2.50 | 8.38 | 7.94 | 0/16 | 10.44 |
| DPO student, LoRA ×1.5 | 3.00 | 8.00 | 7.12 | 0/16 | 10.12 |

What this says:
- **DPO moved the student up on both scores at once** (+0.31 philosophy, +0.19 helpfulness), the first intervention to improve both. At ×1.5 it nearly matches the woven prompt's philosophy (3.00 vs 3.12) with 0.4 more helpfulness. The answers now open with one reflective sentence and then give the practical content ("The true purpose of technology is not merely to manipulate matter, but … A refrigerator operates on the fundamental thermodynamic …").
- **But the evidence is thin.** The gains are about the size of the judge's noise on 16 prompts. The starting student also scored higher in this pool than in J's (2.19 / 7.75 vs 1.69 / 7.56), and the policy fitted its 175 pairs completely. The 48-prompt check (§21) is the real test.

### 21. Confirming the new candidates, greedy and sampled (experiment N)

Branch `exp-robust` (`scripts/exp_robust.py`, `runs/robust/`). Experiments L and M found new candidates on 16 prompts. Here they were re-run on the 48 unseen prompts of §16 with greedy decoding. The main configs were also run with sampled decoding (temperature 0.7, top-p 0.9, two seeds), since every earlier number was greedy. One blind pool of 1,104 outputs, judged by Claude Sonnet. ± is the standard error over the 48 prompts.

| Config (greedy) | Philo | Coherence | Helpful | Degenerate | Philo + help |
|---|---|---|---|---|---|
| Baseline | 0.04 | 8.94 | 8.65 | 1/48 | 8.69 ± 0.09 |
| Prompt woven | 2.83 | 8.12 | 6.38 | 0/48 | 9.21 ± 0.18 |
| Diff-means L8 α 0.5 | 2.48 | 6.67 | 4.27 | 0/48 | 6.75 ± 0.23 |
| Constant along B, 0.6·p50 | 3.69 | 7.33 | 5.12 | 0/48 | 8.81 ± 0.22 |
| Constant along B, 0.8·p50 | 4.25 | 6.52 | 3.69 | 2/48 | 7.94 ± 0.27 |
| Rank-1 LoRA ×1 / ×1.5 / ×2 | 2.83 / 3.02 / 3.50 | 8.25 / 8.06 / 7.88 | 5.69 / 6.04 / 5.58 | 0/48 | 8.52 / 9.06 / 9.08 |
| DPO starting student (J) | 1.85 | 8.27 | 7.54 | 1/48 | 9.40 ± 0.14 |
| DPO student | 2.33 | 8.23 | 7.35 | 0/48 | 9.69 ± 0.14 |
| DPO student ×1.5 | 2.90 | 7.67 | 6.81 | 0/48 | 9.71 ± 0.16 |

| Config (sampled, seed 0 / seed 1) | Philo | Helpful | Philo + help |
|---|---|---|---|
| Baseline | 0.00 / 0.02 | 8.67 / 8.33 | 8.67 / 8.35 |
| Prompt woven | 3.04 / 3.17 | 5.96 / 5.98 | 9.00 / 9.15 |
| Diff-means L8 α 0.5 | 2.67 / 2.62 | 3.52 / 3.90 | 6.19 / 6.52 |
| Constant along B, 0.6·p50 | 3.73 / 3.23 | 4.94 / 5.15 | 8.67 / 8.38 |
| Rank-1 LoRA ×1.5 | 3.17 / 3.17 | 5.44 / 5.79 | 8.60 / 8.96 |
| DPO student | 2.65 / 2.52 | 7.54 / 7.33 | 10.19 / 9.85 |

Paired differences (same 48 prompts, greedy):

| Comparison | Philo | Helpful | Philo + help |
|---|---|---|---|
| Constant along B 0.6 − diff-means α 0.5 | +1.21 ± 0.18 | +0.85 ± 0.28 | +2.06 ± 0.28 |
| DPO student − its starting student | +0.48 ± 0.14 | −0.19 ± 0.14 | +0.29 ± 0.17 |
| DPO student ×1.5 − prompt woven | +0.06 ± 0.19 | +0.44 ± 0.22 | +0.50 ± 0.22 |
| DPO student ×1.5 − rank-1 LoRA ×1.5 | −0.12 ± 0.17 | +0.77 ± 0.20 | +0.65 ± 0.21 |
| Rank-1 LoRA ×1.5 − prompt woven | +0.19 ± 0.18 | −0.33 ± 0.19 | −0.15 ± 0.19 |
| Rank-1 LoRA ×2 − prompt woven | +0.67 ± 0.20 | −0.79 ± 0.24 | −0.12 ± 0.24 |

What this says:
- **The learned direction B is confirmed as a far better steering vector than diff-means.** Paired over 48 new prompts, it gains 1.2 philosophy *and* 0.85 helpfulness (+2.06 in the sum, about 7 standard errors), and the gap holds in both sampled seeds. As a plain constant push it reaches philosophy 3.7 with helpfulness 5.1 and no broken outputs.
- **Amplifying the rank-1 LoRA slides along the prompt's frontier, it doesn't go past it.** ×1.5 matches the woven prompt (sum −0.15 ± 0.19), and ×2 trades 0.8 helpfulness for 0.7 philosophy. That still gives philosophy 3.5 at helpfulness 5.6, coherence 7.9, with no prompt. §19's "past the prompt" came from one 16-prompt pool and didn't hold up.
- **DPO is the only method with evidence of beating the frontier, and it is modest.** It mostly raised philosophy (+0.48 ± 0.14), and at ×1.5 it matches the woven prompt's philosophy with 0.44 ± 0.22 more helpfulness (sum +0.50 ± 0.22, about 2.3 standard errors). It clearly beats the amplified imitation student (+0.77 ± 0.20 helpfulness at equal philosophy). This is one judge and one training run, so it is suggestive, not settled.
- **Sampling doesn't change the picture.** Every ordering holds under temperature 0.7. Sampling slightly raises philosophy for prompts and students, and lowers helpfulness and coherence for diff-means steering.

## Caveats

- Small evaluation: 16 prompts, greedy decoding, 96–120 new tokens, one judge model (Claude Sonnet) with one rubric. The Claude judge is strict: the best config averages 3/9 for philosophy. The main results were re-checked on 48 new prompts with a second judge (Claude Opus) in §16. Sampled decoding (two seeds) was checked in §21 and changes no ordering. Both judges are Claude models: no non-Claude judge was available in this setup (no other provider's credentials, and the model's own self-judging is unreliable, see TL;DR 10).
- The 8 practice prompts were used to pick the shortlist. Held-out prompts agree, but the set is small.
- Contrast data is hand-written by us, so the direction reflects our idea of "philosophical" (Heidegger, Stoics, qualia…).
- Qwen3.5's linear-attention layers ran on the slow PyTorch fallback, so the sweeps are small by necessity.
- Pathway check: the natural change compares *different texts* (philosophical vs plain words), so it includes token-identity effects that no push on plain text can reproduce. These matter most in late layers; the random controls set the floor. The bootstrap intervals in `runs/pathway.json` are percentile intervals over pairs. For cosines and norm ratios they are biased low (resampling adds noise to the natural vector), so compare conditions within the same layer rather than reading them as absolute bounds.
- Not replicated: the *agentic* part of the Astra paper (an LLM diagnosing and editing weights in a loop). Here the search was a fixed grid.

## Suggested next steps

1. ~~**Bias bake-in**~~: superseded by Results §18. The rank-1 LoRA, merged into `W_down,8`, is a standalone checkpoint with no bias that beats steering (`scripts/bake_lora.py --out DIR`).
2. **Recover helpfulness**: steer only the first N tokens, or combine with the conditional edit (which preserved helpfulness), to get "answers fully, reflects briefly".
3. **Agent-in-the-loop**: give a Claude agent the explorer tools (push, generate, judge on practice prompts, hidden test prompts) and a forward-pass budget, as in the Astra paper.
4. ~~**Larger evaluation**~~: partly done, see Results §16 (48 new prompts, two judge models). Sampled decoding: done in §21. Open: a non-Claude judge.
5. ~~**Downstream pathway check**~~: done, see Results §4. Follow-ups it suggests:
   - ~~**Less overdrive**~~: done, see Results §5. The disturbance shrinks with push size, but philosophy has a threshold near α 0.5, and most of the helpfulness cost comes with the philosophical shift itself. Open: map α 0.35–0.5 more finely for k=256 and diff-means, and confirm the k=256 α 0.35 middle point on a larger prompt set.
   - ~~**Mid-layer re-injection**~~: done as multi-layer steering, see Results §11. Spreading the push, or recreating the natural path layer by layer, stays on the same trade-off line.
   - ~~**Helpfulness ceiling**~~: done, see Results §6. The target answers are unhelpful by construction (helpfulness 2.31).
   - ~~**"Answer + reflect" contrast data**~~: done, see Results §7. The targets are helpful, but steering along their direction keeps the same trade-off.
   - ~~**Distil the prompt**~~: done, see Results §10. A LoRA (even rank 1 at L8) reproduces the woven prompt without it. Turned into a standalone checkpoint in Results §18.
   - ~~**Distil a better teacher**~~: done, see Results §15. Prompt search and best-of-4 selection don't move the frontier; students copy their teacher's average. DPO on judged pairs: see Results §20–21. Open: many more samples per question, or a stronger teacher model.
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
.venv/bin/python scripts/exp_multilayer.py && .venv/bin/python scripts/claude_judge.py --input multilayer/gen.jsonl --output multilayer/judge.jsonl
.venv/bin/python scripts/exp_prompt_transplant.py && .venv/bin/python scripts/claude_judge.py --input prompt_transplant/gen.jsonl --output prompt_transplant/judge.jsonl
.venv/bin/python scripts/exp_routes.py && .venv/bin/python scripts/claude_judge.py --input routes/gen.jsonl --output routes/judge.jsonl
.venv/bin/python scripts/exp_geometry.py && .venv/bin/python scripts/claude_judge.py --input geometry/gen.jsonl --output geometry/judge.jsonl
.venv/bin/python scripts/exp_teacher.py --stage prompts && .venv/bin/python scripts/claude_judge.py --input teacher/prompts_gen.jsonl --output teacher/prompts_judge.jsonl
.venv/bin/python scripts/exp_teacher.py --stage samples && .venv/bin/python scripts/claude_judge.py --input teacher/samples_gen.jsonl --output teacher/samples_judge.jsonl
.venv/bin/python scripts/exp_teacher.py --stage train && .venv/bin/python scripts/claude_judge.py --input teacher/gen.jsonl --output teacher/judge.jsonl
.venv/bin/python scripts/exp_confirm.py --teacher-loras down8-r1:down8-r1_prompt.pt mlp-r8:mlp-r8_best-of-4.pt
.venv/bin/python scripts/claude_judge.py --input confirm/gen.jsonl --output confirm/judge.jsonl && .venv/bin/python scripts/claude_judge.py --input confirm/gen.jsonl --output confirm/judge_opus.jsonl --model opus
.venv/bin/python scripts/exp_heads.py && .venv/bin/python scripts/claude_judge.py --input heads/gen.jsonl --output heads/judge.jsonl
.venv/bin/python scripts/exp_heads.py --round 2 && .venv/bin/python scripts/claude_judge.py --input heads/gen_round2.jsonl --output heads/judge_round2.jsonl
.venv/bin/python scripts/bake_lora.py                            # merge the rank-1 LoRA into W_down at L8 (add --out DIR to save the model)
.venv/bin/python scripts/exp_gate.py && .venv/bin/python scripts/claude_judge.py --input gate/gen.jsonl --output gate/judge.jsonl
.venv/bin/python scripts/exp_gate.py --round 2 && .venv/bin/python scripts/claude_judge.py --input gate/gen_round2.jsonl --output gate/judge_round2.jsonl
.venv/bin/python scripts/exp_dpo.py && .venv/bin/python scripts/claude_judge.py --input dpo/gen.jsonl --output dpo/judge.jsonl
.venv/bin/python scripts/exp_robust.py --dpo && .venv/bin/python scripts/claude_judge.py --input robust/gen.jsonl --output robust/judge.jsonl
.venv/bin/python scripts/distill_probe.py                         # size and direction of the rank-1 LoRA's push
.venv/bin/python scripts/astra_edit.py && .venv/bin/python scripts/claude_judge.py --input astra.jsonl --output claude_judge_astra.jsonl
```
