# Part 4 · Can the prompt's behaviour be moved into the weights?

[← Findings overview](../FINDINGS.md) · [Explainer, part 4](https://www.contrasentido.ar/weighting-intelligence/explainer.html#p4)

**Short answer.** Yes. A trained update, down to one rank-1 edit of one matrix, reproduces the prompt without the prompt and keeps about 1.7 more helpfulness points than steering at equal philosophy. Students copy their teacher's average, so a better Qwen teacher only moves them along one frontier.

| ID | Question | Result |
|---|---|---|
| [D](#exp-d) | Can the prompt's behaviour be trained into the weights? | Yes. A LoRA matches the woven prompt's balance without the prompt, and a rank-1 update at layer 8 (11,776 parameters) does as well as a 9M-parameter adapter. |
| [J](#exp-j) | Does a better Qwen teacher (prompt search, best-of-4) make a better student? | No. Students copy their teacher's average, and every prompt and student sits on one frontier (philosophy + helpfulness ≈ 9.1–9.4). |
| [I](#exp-i) | Do the results hold on unseen prompts and under a second judge? | Yes. Students match their prompts, and at equal philosophy the rank-1 LoRA keeps about 1.7 more helpfulness points than steering. Sonnet and Opus agree per item (r 0.85–0.90). |
| [D2](#exp-d2) | Can the distilled student be a single bias-free weight edit? | Yes. The rank-1 LoRA merged into layer 8's `down_proj` scores the same as the adapter (philosophy 2.62, helpfulness 6.75). It works because it reads a dense mix of all 9,216 neurons. |
| [L](#exp-l) | Is it the learned gate (when) or the learned direction (where)? | Mostly the direction. A constant push along the learned direction B beats diff-means at every strength. At its own size the update needs both gate and direction. |
| [M](#exp-m) | Can preference training (DPO) raise philosophy and helpfulness together? | A small gain on both (2.19 → 2.50 and 7.75 → 7.94), about the size of the judge's noise. |
| [N](#exp-n) | Do direction B, the amplified LoRA and the DPO student hold on 48 new prompts? | B beats diff-means by +2.06 in philosophy + helpfulness. The amplified LoRA moves along the prompt's frontier, not past it. The DPO student gains +0.44 ± 0.22 helpfulness, which is suggestive only. |

## Experiments

<a id="exp-d"></a>

### D · Distilling the prompt into the weights

> **Question.** Can the prompt's behaviour be trained into the weights?  
> **Result.** Yes. A LoRA matches the woven prompt's balance without the prompt, and a rank-1 update at layer 8 (11,776 parameters) does as well as a 9M-parameter adapter.  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c13)

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
- **The prompt can be moved into the weights.** All three adapters write in the prompted style with no prompt ("To prepare scrambled eggs is to engage in a small, daily alchemy…") and land close to the prompt on all three scores, held-out prompts included. This is the first intervention with no prompt that keeps the prompt's balance. For comparison, steering at helpfulness 6–6.6 gave philosophy of only about 1.5 ([exp. 0.5](1-pushing-the-model.md#exp-0-5), [exp. A](2-the-steering-tradeoff.md#exp-a)). The one "degenerate" flag is a fluent answer about changing a light bulb, so it is probably judge noise.
- **The ceiling is the prompt.** Distillation copies the prompt's trade-off; it doesn't improve on it (philosophy 2.3–2.5 vs 2.94). The prompt scored 3.44 in the experiment B pool and 2.94 here, which is the judge's noise between pools.
- **One rank-1 update at L8 is enough.** 11,776 parameters at the same site as all our steering reach the same balance as the 9M-parameter adapter. So the problem was never the place (L8) or the size of the change, but its *form*. The LoRA adds `scale·B·(A·a)`, a push along one fixed direction B whose size depends on the token through A·a. A steering vector is the same push with a constant size.
- **And it pushes in a different direction.** B has cosine 0.13 with the diff-means direction (0.09–0.10 with the appended, woven and self-generated ones). Gradient descent found a route to the prompted behaviour that the contrast data didn't point to.
- **It pushes gently, mostly at the start.** Measured on the 16 evaluation prompts (`runs/distill/probe.json`), its push averages 0.21·p50 on prompt tokens and 0.19·p50 on the first 10 answer tokens, but only 0.12·p50 over the whole answer (0.08 on the baseline's answer text). That is *below* the natural L8 difference (0.24·p50) and 2.5–6× smaller than the α 0.5 steering push, which is the same at every token. It matches experiment C: the start sets the mode, and a small push in the right direction at the right tokens is enough. A constant push big enough to change the mode is too big everywhere else.
- **The token mixers alone are as good as the MLPs.** `mixers-r8` (attention + DeltaNet, no MLP) matches `mlp-r8`. Where the change is made matters little at this scale; what it is trained to do matters most. This doesn't support or refute the attention hypothesis on its own (experiment G tests it directly).

<a id="exp-j"></a>

### J · Distilling a better teacher

> **Question.** Does a better Qwen teacher (prompt search, best-of-4) make a better student?  
> **Result.** No. Students copy their teacher's average, and every prompt and student sits on one frontier (philosophy + helpfulness ≈ 9.1–9.4).  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c18)

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
- **A student copies its teacher's average, not its best answers.** The best-of-4 students score like the plain reflect-first student and its prompt, not like the selected targets (2.50 / 7.98). The selection gain was small: about 0.35 philosophy and 0.55 helpfulness over a random sample, the same size as the judge's noise ([exp. 0.5](1-pushing-the-model.md#exp-0-5), [exp. H](2-the-steering-tradeoff.md#exp-h)). So selection mostly kept answers the judge happened to score high, and that isn't something a student can learn.
- **Prompts and their students share one frontier.** Every prompt and student here has philosophy + helpfulness of about 9.1–9.4. Moving along it trades about one helpfulness point per philosophy point. Steering trades about two (0.5–0.8 philosophy per point, [exp. 0.5](1-pushing-the-model.md#exp-0-5)–[H](2-the-steering-tradeoff.md#exp-h)). Choosing a teacher picks a point on this frontier: woven is more philosophical, reflect-first more helpful. Neither moves the frontier.
- **The experiment D result replicates.** Reloaded in a new pool, the woven student scores 2.56 / 6.75 (2.25 / 6.62 before), matching its prompt again.
- **To go beyond the frontier you need a stronger signal than 4 noisy samples.** Options: many more samples per question, pairwise preference training (DPO) instead of imitation, or a stronger teacher model. None of these were run here.

<a id="exp-i"></a>

### I · Confirmation on 48 new prompts with two judges

> **Question.** Do the results hold on unseen prompts and under a second judge?  
> **Result.** Yes. Students match their prompts, and at equal philosophy the rank-1 LoRA keeps about 1.7 more helpfulness points than steering. Sonnet and Opus agree per item (r 0.85–0.90).  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c19)

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
- **The rank-1 adapter is as good as the 9M-parameter one** (2.75 / 5.81 vs 2.62 / 6.00), confirming [exp. D](#exp-d).
- **The most philosophical usable option is still prompt + push** (3.46), at a steering-sized helpfulness cost (4.02). Nothing tested escapes the trade-off; the prompt and its students are the best frontier found.
- **The second judge is harsher on helpfulness for reflective answers.** That widens the gap between the reflective configs and the baseline, but it doesn't reorder anything.

<a id="exp-d2"></a>

### D2 · A prompt-free philosophy Qwen in one weight edit

> **Question.** Can the distilled student be a single bias-free weight edit?  
> **Result.** Yes. The rank-1 LoRA merged into layer 8's `down_proj` scores the same as the adapter (philosophy 2.62, helpfulness 6.75). It works because it reads a dense mix of all 9,216 neurons.  
> **Status.** Refines 0.3.  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c21)

Branch `exp-bake` (`scripts/bake_lora.py`, `runs/bake/`). The rank-1 LoRA of [exp. D](#exp-d) adds `scale·B·(A·a)` to layer 8's `down_proj` output, which is the same as editing that weight once: `W_down,8 ← W_down,8 + scale·B·Aᵀ`. That is a rank-1, bias-free edit, the same *form* as the Astra-style edits of [exp. 0.3](1-pushing-the-model.md#exp-0-3), which were nearly inert. The script merges it into the stored weight (its size: 10% of ‖W_down,8‖) and regenerates the 16 evaluation prompts with no prompt and no adapter. `--out DIR` saves the full model.

| Config | Philo | Coherence | Helpful | Degenerate |
|---|---|---|---|---|
| Baseline | 0.00 | 9.00 | 8.75 | 0/16 |
| Rank-1 LoRA (adapter) | 2.81 | 8.19 | 6.81 | 1/16 |
| Same update baked into `W_down,8` | 2.62 | 8.44 | 6.75 | 0/16 |

5 of 16 generations are byte-identical to the adapter's. The rest drift after 0–470 characters, because the merged weight is rounded to bf16 while the adapter computed its correction in float32. The judged scores match within noise.

What this says:
- **A standalone, prompt-free philosophical Qwen3.5-4B is one rank-1 edit to one matrix.** No prompt, no hooks, no bias.
- **This refines [exp. 0.3](1-pushing-the-model.md#exp-0-3).** The hand-built Astra-style edits were inert because their read side was a few hand-picked neurons (masked `v`) that don't fire on practical prompts, so there was nothing to amplify. The trained edit reads a *dense* combination `A` of all 9,216 neurons, which fires on any prompt and on the opening of any answer (0.2·p50, [exp. D](#exp-d)), and writes along a direction `B` that diff-means never pointed to. A bias was never needed. A weight edit can add a behaviour if its read side is chosen for it, and gradient descent found one where hand-construction didn't.

<a id="exp-l"></a>

### L · Why the rank-1 update works: gate vs direction

> **Question.** Is it the learned gate (when) or the learned direction (where)?  
> **Result.** Mostly the direction. A constant push along the learned direction B beats diff-means at every strength. At its own size the update needs both gate and direction.  
> **Status.** On 48 new prompts the amplified LoRA lands on the prompt's frontier, not past it (N).  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c22)

Branch `exp-gate` (`scripts/exp_gate.py`, `runs/gate/`). The rank-1 LoRA adds `g_t · unit(B)` at layer 8, where `g_t = scale·|B|·(A·a_t)` is a token-dependent size (the *gate*, from the read side A) and B is a fixed *direction*. Steering uses a constant size along diff-means. Swapping the two parts tests LITERATURE.md H2 (gated or conditional steering) in a form that applies to every prompt: is it the gate or the direction that matters? B has cosine 0.13 with diff-means.

| Config (round 1) | Philo | Coherence | Helpful | Degenerate |
|---|---|---|---|---|
| Baseline | 0.06 | 9.00 | 8.69 | 0/16 |
| LoRA (gate A, direction B) | 2.62 | 8.06 | 6.88 | 1/16 |
| Constant along B, 0.12 / 0.2 / 0.4·p50 | 0.00 / 0.12 / 1.44 | 8.94 / 9.00 / 8.44 | 8.75 / 8.88 / 7.31 | 0/16 |
| Gate A, diff-means direction ×1 / ×2 | 0.19 / 1.12 | 9.00 / 7.12 | 8.81 / 5.69 | 0/16 |
| Gate A, random direction ×1 / ×2 | 0.06 / 0.06 | 8.94 / 9.00 | 8.69 / 8.50 | 0/16 |
| Gate A with its sign flipped, direction B | 0.25 | 8.94 | 8.75 | 0/16 |

0.12 and 0.2·p50 are the LoRA's mean push on answer and on prompt tokens ([exp. D](#exp-d)). Round 2 mapped the strongest leads in one pool:

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
- **But the learned direction B is the bigger discovery.** A *constant* push along B, the ordinary steering recipe with only the direction swapped, beats diff-means at every strength. At 0.6·p50 it gives philosophy 3.94 with coherence 7.81, helpfulness 5.12 and no broken outputs, against diff-means α 0.5 at 2.94 / 6.94 / 4.06. At 0.8·p50 it reaches 5.12, the highest philosophy without broken outputs in the project. Its philosophy + helpfulness (8.4–9.1) sits on or near the prompt's frontier ([exp. J](#exp-j)), which no diff-means push reached.
- **Amplifying the LoRA goes past the prompt's point.** ×1.5 and ×2 raise philosophy to 3.4 and keep coherence above 8 and helpfulness near 6 (philosophy + helpfulness 9.2–9.4). In this pool that beats the woven prompt's own balance (2.69 / 6.50 in the [exp. K](2-the-steering-tradeoff.md#exp-k) pool), so the student can be pushed beyond its teacher by turning up its gain.
- **Why diff-means was a poor direction.** Diff-means measures how philosophical *texts* differ from plain ones, which mixes in "replace the answer" ([exp. 0.6](1-pushing-the-model.md#exp-0-6), [exp. H](2-the-steering-tradeoff.md#exp-h)). B was trained to make *this model's answers* look like its own prompted answers, which only needs what the model reads as "write reflectively". A direction learned from outcomes beats one averaged from data.
- Caveat: one pool of 16 prompts. On 48 unseen prompts ([exp. N](#exp-n)), B is confirmed, but the amplified LoRA sits on the prompt's frontier rather than past it.

<a id="exp-m"></a>

### M · Preference training on judged pairs

> **Question.** Can preference training (DPO) raise philosophy and helpfulness together?  
> **Result.** A small gain on both (2.19 → 2.50 and 7.75 → 7.94), about the size of the judge's noise.  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c23)

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
- **But the evidence is thin.** The gains are about the size of the judge's noise on 16 prompts. The starting student also scored higher in this pool than in J's (2.19 / 7.75 vs 1.69 / 7.56), and the policy fitted its 175 pairs completely. The 48-prompt check ([exp. N](#exp-n)) is the real test.

<a id="exp-n"></a>

### N · Confirming the new candidates, greedy and sampled

> **Question.** Do direction B, the amplified LoRA and the DPO student hold on 48 new prompts?  
> **Result.** B beats diff-means by +2.06 in philosophy + helpfulness. The amplified LoRA moves along the prompt's frontier, not past it. The DPO student gains +0.44 ± 0.22 helpfulness, which is suggestive only.  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c24)

Branch `exp-robust` (`scripts/exp_robust.py`, `runs/robust/`). Experiments L and M found new candidates on 16 prompts. Here they were re-run on the 48 unseen prompts of [exp. I](#exp-i) with greedy decoding. The main configs were also run with sampled decoding (temperature 0.7, top-p 0.9, two seeds), since every earlier number was greedy. One blind pool of 1,104 outputs, judged by Claude Sonnet. ± is the standard error over the 48 prompts.

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
- **Amplifying the rank-1 LoRA slides along the prompt's frontier, it doesn't go past it.** ×1.5 matches the woven prompt (sum −0.15 ± 0.19), and ×2 trades 0.8 helpfulness for 0.7 philosophy. That still gives philosophy 3.5 at helpfulness 5.6, coherence 7.9, with no prompt. [exp. L](#exp-l)'s "past the prompt" came from one 16-prompt pool and didn't hold up.
- **DPO is the only method with evidence of beating the frontier, and it is modest.** It mostly raised philosophy (+0.48 ± 0.14), and at ×1.5 it matches the woven prompt's philosophy with 0.44 ± 0.22 more helpfulness (sum +0.50 ± 0.22, about 2.3 standard errors). It clearly beats the amplified imitation student (+0.77 ± 0.20 helpfulness at equal philosophy). This is one judge and one training run, so it is suggestive, not settled.
- **Sampling doesn't change the picture.** Every ordering holds under temperature 0.7. Sampling slightly raises philosophy for prompts and students, and lowers helpfulness and coherence for diff-means steering.
