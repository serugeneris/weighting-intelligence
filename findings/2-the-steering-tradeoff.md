# Part 2 · Can any change to the push remove the helpfulness cost?

[← Findings overview](../FINDINGS.md) · [Explainer, part 2](https://www.contrasentido.ar/weighting-intelligence/explainer.html#p2)

**Short answer.** No. Better target data, a partial push, other layers, several layers, a different geometry, the prompt's hot spots and sparse-autoencoder features all land on the same line of philosophy gained per helpfulness lost. A plain system prompt beats all of them.

| ID | Question | Result |
|---|---|---|
| [A](#exp-a) | Do targets that answer first and then reflect give a better direction? | No. The new targets are helpful (8.6), but their direction is 84–92% the old one and steering along it has the same trade-off. |
| [B](#exp-b) | Does simply asking beat steering? | Yes. The woven system prompt scores philosophy 3.44 with helpfulness 7.06, better than any steering setting. A direction built from the model's own prompted answers breaks the model. |
| [C](#exp-c) | Does it matter when we push? | Little. Pushing only the first 24 tokens, or only the prompt, equals pushing everywhere. A push that starts mid-answer breaks coherence. |
| [E](#exp-e) | Does spreading the push over layers, or choosing another layer, escape the trade-off? | No. Every variant stays on the same 0.5–0.8 philosophy per helpfulness point. Only layers 8–16 respond to a modest push. |
| [H](#exp-h) | Does a different push geometry cost less? | No. Norm-preserving pushes score the same, and projecting out the "drop the answer" component leaves no philosophy at all. |
| [K](#exp-k) | Does steering work better at the heads and layers the prompt uses? | No. Mean vectors at the key heads or layers do almost nothing; summed over all 32 mixers they land on the usual trade-off. |
| [T](#exp-t) | Do SAE features steer with fewer side effects? | No. The features are readable, but the hand-picked content features score 3.25 philosophy at 3.12 helpfulness: the diff-means trade-off, well below direction B (4.31 / 5.06). |

## Experiments

<a id="exp-a"></a>

### A · "Answer + reflect" contrast data

> **Question.** Do targets that answer first and then reflect give a better direction?  
> **Result.** No. The new targets are helpful (8.6), but their direction is 84–92% the old one and steering along it has the same trade-off.  
> 2026-11-04 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c10)

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

<a id="exp-b"></a>

### B · Prompting baseline and a self-generated direction

> **Question.** Does simply asking beat steering?  
> **Result.** Yes. The woven system prompt scores philosophy 3.44 with helpfulness 7.06, better than any steering setting. A direction built from the model's own prompted answers breaks the model.  
> 2026-11-04 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c11)

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

<a id="exp-c"></a>

### C · Steering only part of the answer

> **Question.** Does it matter when we push?  
> **Result.** Little. Pushing only the first 24 tokens, or only the prompt, equals pushing everywhere. A push that starts mid-answer breaks coherence.  
> 2026-11-04 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c12)

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
- **The start decides the mode.** Steering only the first 24 tokens, or only the prompt, gives about the same result as steering every token: philosophy 3.3–3.7, helpfulness about 4. Once the answer starts in a reflective register it stays there, and when only the prompt is pushed, the token mixers carry the effect into the whole answer. (This originally said "attention"; experiment G found the route is mostly the DeltaNet state, see [exp. G](3-how-the-prompt-works.md#exp-g).)
- **So the trade-off is not about *when* we push.** Windowing moves the same trade-off around, or makes it worse; it never reaches the prompt's balance. The woven prompt is still best (philosophy 3.56, helpfulness 6.94), and full steering looks better at 160 tokens (3.75) than at 120 (2.8–3.0) only because the answer is longer.

<a id="exp-e"></a>

### E · Several layers, other layers

> **Question.** Does spreading the push over layers, or choosing another layer, escape the trade-off?  
> **Result.** No. Every variant stays on the same 0.5–0.8 philosophy per helpfulness point. Only layers 8–16 respond to a modest push.  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c14)

Branch `exp-multilayer` (`scripts/exp_multilayer.py`, `runs/multilayer/`). The pathway check ([exp. 0.4](1-pushing-the-model.md#exp-0-4)) found the L8 push 2–5× bigger than the natural L8 difference, and losing the natural route in the middle layers. Two ways to push more gently and more naturally:
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
- **Recreating the natural trajectory recreates the target texts, including their unhelpfulness.** Natural path ×1 up to L24 gives the highest philosophy of any prompt-free intervention so far (4.75), at helpfulness 1.62. That is close to how the judge scores the hand-written philosophical answers themselves (5.75 / 2.31, [exp. 0.6](1-pushing-the-model.md#exp-0-6)). The push does what it was built to do: it moves the model toward texts that *replace* the answer. Doubling it breaks the model.
- **The middle layers are where a push works (H5, partly supported).** At α 0.5, only L8–L16 respond (philosophy 2.1–2.9). L4 does nothing even at 2× its natural difference; L20 and L24 need 2× natural to show anything, and then lose coherence. L16 at α 0.5 is slightly better than L8 (philosophy 2.88 vs 2.38, helpfulness 5.06 vs 4.62), but that is within judge noise.
- **So the place and spread of a constant push matter less than its form.** No layer or combination reaches the prompt's balance. The rank-1 LoRA of [exp. D](4-into-the-weights.md#exp-d) sits at the same L8 site with a smaller push; what it adds is a token-dependent size.

<a id="exp-h"></a>

### H · A different push geometry: orthogonal and norm-preserving

> **Question.** Does a different push geometry cost less?  
> **Result.** No. Norm-preserving pushes score the same, and projecting out the "drop the answer" component leaves no philosophy at all.  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c17)

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
- **Keeping the residual's length changes nothing.** Norm-preserving and norm +10% pushes score the same as the plain push at α 0.5 and 0.75, within judge noise, and break the same way above that. The helpfulness cost isn't caused by the residual getting longer. It comes with the direction itself, as [exp. 0.5](1-pushing-the-model.md#exp-0-5)–[0.6](1-pushing-the-model.md#exp-0-6) suggested.
- **There is no philosophy left once "drop the answer" is removed.** Pushes along the remaining directions add no philosophy at α 0.5. At α 1.0 they break the answer (helpfulness 1.7–2.9) and still add none: outputs start mid-sentence ("it's about consistency and attention to detail.") or in lowercase fragments. In this data, "be philosophical" and "stop answering" are the same direction, so a mean-difference vector can't separate them.
- **So the geometry isn't the problem; the constant direction is.** This rules out the cheapest fixes for the trade-off. The approaches that kept helpfulness were a prompt and a trained, token-dependent update ([exp. B](#exp-b), [exp. D](4-into-the-weights.md#exp-d)).

<a id="exp-k"></a>

### K · Steering at the prompt's hot spots

> **Question.** Does steering work better at the heads and layers the prompt uses?  
> **Result.** No. Mean vectors at the key heads or layers do almost nothing; summed over all 32 mixers they land on the usual trade-off.  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c20)

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
- **A hot spot for the prompt is not a good place for a constant push.** The prompt's effect at the top heads is large in mediation ([exp. G](3-how-the-prompt-works.md#exp-g)) but tiny on average: what the head adds depends on the token, and the mean cancels out. Adding that mean, even ×8, barely moves the answer. Mediation finds where the information flows, not a vector to inject.
- **Spreading the prompt's mean effect over all 32 mixers works, but not better than steering.** At ×2 it reaches 2.50 / 6.00, the same in both pools, and sits about 0.6 helpfulness points above the diff-means line at equal philosophy. That is within judge noise on 16 prompts. At ×2.5–3 it falls back onto the same line (3.06 / 4.25, 3.94 / 2.88). Like experiment E, a constant push spread over many layers stays on the steering trade-off.
- **H3 for steering: not supported.** Head-level steering at the sites that matter most for the prompt does not beat the residual push in this model, at least with mean (prompt-derived) vectors. The token-dependent route (the LoRA of [exp. D](4-into-the-weights.md#exp-d)) remains the only prompt-free intervention that reaches the prompt's frontier.

<a id="exp-t"></a>

### T · Sparse-autoencoder features at layer 8

> **Question.** Do SAE features steer with fewer side effects?  
> **Result.** No. The features are readable, but the hand-picked content features score 3.25 philosophy at 3.12 helpfulness: the diff-means trade-off, well below direction B (4.31 / 5.06).  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c30)

LITERATURE.md H6 asks whether learned features steer more cleanly than an averaged direction. No public SAE exists for Qwen3.5-4B (Qwen publishes residual SAEs for the 2B, 9B and 27B models, not 4B), so we trained one. A TopK SAE (8,192 features, k = 32, decoder untied, inputs centred and scaled) was fit on 300,000 layer-8 residual tokens from 7,264 texts this project had already generated (every `runs/**/*gen*.jsonl`: plain, prompted, steered and broken answers with their chat prompts; tokens above 4× the median norm dropped). After 3,000 Adam steps it leaves 16.3% of the variance unexplained, with 5,100 of 8,192 features alive (the rest never fire; we did no dead-feature resampling). Features were ranked by mean activation on the 32 contrast pairs' philosophical answers minus the plain ones (answer tokens).

**What the top features are** (`runs/sae/inspect.json`, top-activating contexts over 1,500 texts):

| Feature | Fires on | Top contexts | Kind |
|---|---|---|---|
| 275 (rank 1) | 15.6% of tokens | ungrammatical tokens: "clothes you need **wear**", "it is that I am not what I **is**" | disfluency |
| 6074, 6257 | 1.7%, 9.7% | broken pseudo-profound loops from earlier steering runs: "it remains true that **there** is no way" | rambling |
| 3424 | 2.6% | "useful fictions **rather** than sharp boundaries", "or **merely** sorts things" | hedge / contrast |
| 3565 | 5.4% | Claude-style codas: "What else only gets **attention** once it breaks", "a **reminder** that" | reflective coda |
| 2031 | 2.0% | "a metaphor for the nature of change and **identity**", "refiguring the **self**" | self / identity |
| 7238 | 2.1% | "a profound lesson **in how**", "a profound meditation **on**" | "profound X of" template |

The ranking picks up artefacts as readily as content, because the hand-written philosophical contrast answers differ from the plain ones in fluency too. The top feature's direction is almost orthogonal to diff-means (cosine 0.02) and to B (0.01); the weighted top-8 combination has cosine 0.33 with diff-means and 0.07 with B.

**Steering** (16 evaluation prompts, push added to layer 8's MLP output at every token, size a·p50 along the unit decoder direction; blind Sonnet judge, two pools):

| Config | Philosophy | Coherence | Helpfulness | Degenerate |
|---|---|---|---|---|
| *Pool 1* baseline | 0.00 | 9.00 | 8.94 | 0/16 |
| diff-means α 0.5 | 3.19 | 7.00 | 3.69 | 0/16 |
| learned direction B 0.6·p50 | 4.50 | 8.00 | 4.88 | 0/16 |
| top-1 feature (275) 0.5 / 0.8·p50 | 0.00 / 0.00 | 8.38 / 6.88 | 8.25 / 6.44 | 0 / 1 |
| top-8 combination 0.5 / 0.8·p50 | 0.00 / 0.00 | 8.38 / 4.25 | 8.31 / 4.12 | 0 / 3 |
| *Pool 2 (post hoc)* learned direction B 0.6·p50 | 4.31 | 8.00 | 5.06 | 0/16 |
| content features 3424+3565+2031+7238, 0.5·p50 | 0.00 | 9.00 | 8.75 | 0/16 |
| content features, 0.8·p50 | 3.25 | 6.19 | 3.12 | 0/16 |
| content features, 1.2·p50 | 3.12 | 4.81 | 0.62 | 4/16 |
| feature 7238 alone, 0.8 / 1.2·p50 | 1.12 / 2.31 | 7.00 / 3.12 | 4.94 / 1.19 | 0 / 12 |

The ranked features produce no philosophy at all: pushed hard, they only degrade grammar ("stir with a spoon till they mixed well"), as a disfluency feature should. The content features, chosen by hand after reading their contexts (so post hoc), do produce philosophy, but they replace the answer rather than adding to it ("The question 'how to make scrambled eggs' is not about a single method but about how we think"). They sit on the same philosophy-for-helpfulness line as diff-means ([exp. 0.5](1-pushing-the-model.md#exp-0-5)), with the same threshold near 0.5–0.8·p50. The learned direction B stays ahead by about 3 points of philosophy + helpfulness in the same pool.

**What it means.** At this scale an SAE is a good microscope and a poor lever. Its features name what philosophical text *contains* (codas, hedges, "profound lesson in"), token by token, but adding one of those features everywhere says "write that content now", which crowds out the answer. B was trained on the model's own answers to write a mode that the rest of the network turns into "answer, then reflect"; no single readable feature does that. This agrees with AxBench (2501.17148), which found SAE steering below prompting and below trained low-rank interventions. Caveats: one SAE size, one layer, no dead-feature revival, 16 prompts, one judge; feature selection by output score (2505.20063) rather than by activation difference was not tried.
