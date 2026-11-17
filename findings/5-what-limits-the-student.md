# Part 5 · What limits the student: teacher, data, adapter or base model?

[← Findings overview](../FINDINGS.md) · [Explainer, part 5](https://www.contrasentido.ar/weighting-intelligence/explainer.html#p5)

**Short answer.** A stronger teacher model (Claude) is the best lever found. More data, a more philosophical teacher, a larger adapter, preference training, stacking and reranking do not raise the frontier further. A larger base model does, with shrinking returns. The student learns reflection tied to the task, and the habit ignores a request for brevity unless it is retrained to follow one.

| ID | Question | Result |
|---|---|---|
| [O](#exp-o) | Does a stronger teacher model move the frontier? | Yes; this is the best lever found. A rank-8 LoRA trained on 128 Claude answers, at 1.5× gain, matches the woven prompt's philosophy with about 2 more helpfulness points under both judges. |
| [P](#exp-p) | Does three times more teacher data help? | It makes the plain student more philosophical (+0.7) at no helpfulness cost, but all settings stay near the same frontier (sum about 10). |
| [Q](#exp-q) | Does a teacher with more substantive philosophy help? | No. The student copies the name-dropping: +0.25 philosophy, about −0.7 helpfulness and coherence. |
| [R](#exp-r) | Does a 4× larger adapter (rank 32) help? | No. It fits the teacher better and generalises no better. |
| [S](#exp-s) | Does DPO stack on the Claude-taught student? | No gain under either judge. The student's sampled answers are mostly plain, so the pairs had little to teach. |
| [W](#exp-w) | Does the student learn a free-floating style or reflection tied to the task? | Reflection tied to the task. The practical part alone teaches no philosophy; mismatched reflections cost −0.8 philosophy, −0.8 helpfulness and −2.2 coherence. |
| [X](#exp-x) | Does the student hold up outside the test distribution? | Yes over two turns and on checkable questions (16/16 correct), but it keeps reflecting under a system prompt that asks for brief, literal answers. |
| [Y](#exp-y) | Do a prompt on top of the student, or best-of-4 selection, get past the frontier? | No. The stack lands on the same line, and best-of-4 does not beat the student's greedy answer (+0.08 ± 0.19). |
| [V](#exp-v) | Does the size of the base model matter? | Yes, with shrinking returns. The 4B student beats the 2B by +1.2 to +1.7 in philosophy + helpfulness; the 9B beats the 4B by +0.8 to +1.0 at equal philosophy (both 4-bit). |
| [AH](#exp-ah) | Can a short retraining make the habit follow instructions? | Yes in a pilot. The mixed-instruction student followed 8/8 held-out requests in each of three conditions; the original student 3/8, 0/8 and 5/8. |

## Experiments

<a id="exp-o"></a>

### O · A stronger teacher: Claude-written answers

> **Question.** Does a stronger teacher model move the frontier?  
> **Result.** Yes; this is the best lever found. A rank-8 LoRA trained on 128 Claude answers, at 1.5× gain, matches the woven prompt's philosophy with about 2 more helpfulness points under both judges.  
> **Status.** The helpfulness lead was measured at 120 tokens; AE shows much of the practical-coverage gap closes with a longer budget.  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c25)

Branch `exp-claude-teacher` (`scripts/exp_claude_teacher.py`, `runs/claude_teacher/`). Every student so far copied a Qwen teacher, whose prompted answers sit on a frontier of philosophy + helpfulness ≈ 9.2. A stronger model can write answers that are both fully helpful and genuinely reflective. Claude Sonnet, through the local `claude -p` CLI, wrote one answer to each of the same 128 training questions. The instruction: fully helpful and concrete, one or two brief philosophical reflections woven in where they fit, never replacing the practical content, at most 110 words (90 on average). LoRAs were trained on these answers exactly as in experiment D, with no system prompt (final loss: rank-8 MLP 0.93, rank-1 L8 1.65, from 2.17). Evaluation used the 48 confirmation prompts in one pool, with the baseline, the woven prompt and experiment M's DPO student ×1.5, scored by Sonnet and by Opus. The Claude-written answers themselves were judged in the same pool, on their training questions, as the teacher's level.

| Config | Philo (S / O) | Coherence (S / O) | Helpful (S / O) | Degenerate (S / O) |
|---|---|---|---|---|
| Baseline | 0.02 / 0.02 | 8.81 / 8.71 | 7.96 / 7.62 | 1 / 0 of 48 |
| Prompt woven | 2.96 / 2.96 | 8.08 / 8.02 | 5.52 / 4.62 | 0 / 0 |
| DPO student ×1.5 (M) | 3.06 / 2.83 | 7.29 / 7.54 | 6.17 / 5.67 | 0 / 0 |
| Claude-taught, rank 1 L8 | 1.71 / 1.85 | 8.27 / 7.85 | 7.94 / 7.04 | 0 / 0 |
| Claude-taught, rank 8 MLP | 1.38 / 1.42 | 8.31 / 8.08 | 8.00 / 7.38 | 0 / 0 |
| **Claude-taught, rank 8 MLP ×1.5** | **2.73 / 2.96** | 7.77 / 7.29 | **7.46 / 6.73** | 0 / 0 |
| Claude-written targets (128 training questions) | 2.49 / 2.66 | 8.94 / 8.95 | 8.82 / 8.87 | 0 / 0 of 128 |

Paired differences over the 48 prompts:

| Comparison | Philo (S / O) | Helpful (S / O) | Philo + help (S / O) |
|---|---|---|---|
| Claude-taught ×1.5 − prompt woven | −0.23 ± 0.19 / +0.00 ± 0.20 | +1.94 ± 0.27 / +2.10 ± 0.34 | +1.71 ± 0.25 / +2.10 ± 0.32 |
| Claude-taught ×1.5 − DPO student ×1.5 | −0.33 ± 0.18 / +0.12 ± 0.18 | +1.29 ± 0.26 / +1.06 ± 0.26 | +0.96 ± 0.30 / +1.19 ± 0.30 |
| DPO student ×1.5 − prompt woven | +0.10 ± 0.19 / −0.12 ± 0.19 | +0.65 ± 0.22 / +1.04 ± 0.25 | +0.75 ± 0.26 / +0.92 ± 0.30 |

What this says:
- **A stronger teacher moves the frontier.** At ×1.5 gain, the Claude-taught student matches the woven prompt's philosophy and keeps about 2 more helpfulness points, under both judges (+1.7 to +2.1 in the sum, 6–7 standard errors). It also beats the DPO student by about one helpfulness point. This is the best prompt-free philosophical Qwen found: a rank-8 MLP adapter (9M parameters) that can be merged into the weights like [exp. D2](4-into-the-weights.md#exp-d2).
- **The student is still far below its teacher.** The Claude-written answers score about 2.5–2.7 philosophy with 8.8 helpfulness, a sum of about 11.3. The student needs ×1.5 gain to reach their philosophy, and then gives up helpfulness. A 4B model trained on 128 examples copies the register more easily than the content. More data or more epochs might close more of the gap.
- **DPO's gain over the prompt replicates** (+0.75 / +0.92 in the sum), confirming [exp. N](4-into-the-weights.md#exp-n) with a second judge.
- **Caveat: everything here is Claude.** The teacher and both judges are Claude models, so a preference for Claude's own style can't be ruled out. The baseline also scored lower on helpfulness in this pool (7.96 vs 8.65 in [exp. N](4-into-the-weights.md#exp-n)), so compare configs within the pool, not across pools.

<a id="exp-p"></a>

### P · More teacher data

> **Question.** Does three times more teacher data help?  
> **Result.** It makes the plain student more philosophical (+0.7) at no helpfulness cost, but all settings stay near the same frontier (sum about 10).  
> **Status.** The "gain breaks the model" reading was an artefact; see Correction 1.  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c26)

Branch `exp-more-teacher` (`scripts/exp_more_teacher.py`, `runs/more_teacher/`, `data/train_questions_extra.json`). Experiment O's student learned from 128 Claude-written answers and needed 1.5× gain to reach the prompt's philosophy. Here Claude wrote 256 new everyday questions in 16 topic areas, each checked against every existing prompt set (exact match, case-insensitive), and answered them with the same instruction as O. A rank-8 MLP LoRA was trained on all 384 answers exactly as before (final loss 0.96). It was evaluated on the 48 confirmation prompts against O's 128-answer student, the baseline and the woven prompt, scored by Sonnet and Opus. Pool 1 tested gains ×1 and ×1.5. Because ×1.5 broke the new student, pool 2 tested ×1.0, ×1.2 and ×1.3.

| Config | Pool | Philo (S / O) | Coherence (S / O) | Helpful (S / O) | Degenerate (S / O) |
|---|---|---|---|---|---|
| Baseline | 1 | 0.04 / 0.02 | 8.92 / 8.77 | 8.38 / 7.71 | 0 / 0 of 48 |
| Prompt woven | 1 | 3.25 / 2.77 | 8.33 / 8.04 | 5.96 / 4.79 | 0 / 0 |
| 128-answer student ×1 / ×1.5 | 1 | 1.40 / 1.44, 2.96 / 2.81 | 8.19 / 8.06, 7.62 / 7.33 | 7.88 / 7.33, 7.40 / 6.58 | 0 / 0 |
| 384-answer student ×1 | 1 | 2.04 / 2.23 | 8.40 / 8.27 | 8.02 / 7.52 | 0 / 0 |
| 384-answer student ×1.5 | 1 | 2.75 / 2.81 | 5.88 / 6.02 | 6.50 / 6.00 | **28 / 28** |
| Baseline | 2 | 0.04 / 0.02 | 8.94 / 8.81 | 8.27 / 7.69 | 1 / 0 |
| Prompt woven | 2 | 3.25 / 3.00 | 8.42 / 8.04 | 5.96 / 5.02 | 0 / 0 |
| 128-answer student ×1.5 | 2 | 2.77 / 3.02 | 7.67 / 7.33 | 7.29 / 6.77 | 0 / 0 |
| 384-answer student ×1.0 / ×1.2 / ×1.3 | 2 | 2.08 / 2.42, 2.44 / 2.71, 2.52 / 2.73 | 8.42 / 8.23, 8.25 / 7.92, 7.52 / 7.33 | 8.02 / 7.54, 7.58 / 6.90, 7.33 / 6.73 | 0 / 0, 0 / 0, 8 / 8 |

Paired, pool 1: 384-answer − 128-answer student, both ×1: philosophy +0.65 ± 0.18 / +0.79 ± 0.18, helpfulness +0.15 / +0.19, sum +0.79 ± 0.22 / +0.98 ± 0.27 (Sonnet / Opus).

What this says:
- **More data makes the plain student more philosophical at no cost.** With 3× the answers, the unamplified student gains about 0.7 philosophy and keeps its helpfulness (8.0 Sonnet). It is the closest any student has come to its teacher at native gain (Claude's answers: 2.5 / 8.8).
- *Correction ([Correction 1](6-can-we-trust-the-scores.md#correction-1)): the broken counts below are a generation-harness artefact (answers that ran on past the end of the turn). Re-judged after cutting at the turn end, ×1.3 and ×1.5 break 0/48 and stay on the same frontier; gain is not fragile.*
- **But it doesn't raise the frontier, and it makes gain fragile.** Every Claude-taught setting that doesn't break sits at about the same philosophy + helpfulness: about 10.0 (Sonnet) and 9.6–10.0 (Opus), against the woven prompt's 9.2 / 8.0. Gain just slides along that line. The better-trained adapter breaks earlier: 8 of 48 at ×1.3 and 28 of 48 at ×1.5, against 0 at ×1.5 for the 128-answer one. A stronger adapter is a bigger push, and over-amplifying it breaks the model the same way an oversized steering push does ([exp. 0.5](1-pushing-the-model.md#exp-0-5)).
- **Practical choice.** For the most philosophy, use the 128-answer student at ×1.5 (2.8–3.0 / 6.8–7.3). For the most helpfulness, use the 384-answer student at ×1.0–1.2 (2.1–2.7 / 6.9–8.0, no broken outputs at any setting up to ×1.2). Both are prompt-free and beat the woven prompt's balance under both judges.

<a id="exp-q"></a>

### Q · A more philosophical teacher

> **Question.** Does a teacher with more substantive philosophy help?  
> **Result.** No. The student copies the name-dropping: +0.25 philosophy, about −0.7 helpfulness and coherence.  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c27)

Branch `exp-deeper-teacher` (`scripts/exp_deeper_teacher.py`, `runs/deeper_teacher/`). Experiment P's teacher answers were helpfulness-heavy (philosophy about 2.3–2.5, helpfulness 8.8), and students inherit their teacher's balance. Claude re-answered the same 384 training questions with an instruction asking for substantive philosophy: two or three reflections, at least one naming a specific idea or thinker tied precisely to the task, and optionally a closing open question, while keeping every practical detail (at most 120 words). A rank-8 MLP LoRA was trained on them exactly as in P (final loss 1.19, against P's 0.96). It was compared with P's student, same questions but the plainer teacher, on the 48 confirmation prompts. 64 randomly chosen teacher answers of each style were judged in the same pool.

| Config | Philo (S / O) | Coherence (S / O) | Helpful (S / O) | Degenerate (S / O) |
|---|---|---|---|---|
| Baseline | 0.04 / 0.02 | 8.88 / 8.81 | 8.17 / 7.58 | 0 / 0 of 48 |
| Prompt woven | 3.23 / 2.67 | 8.02 / 8.06 | 5.42 / 4.71 | 0 / 1 |
| Plain-teacher student (P), ×1 | 2.04 / 2.15 | 8.40 / 8.29 | 8.10 / 7.44 | 0 / 0 |
| Deeper-teacher student ×1 | 2.27 / 2.42 | 7.65 / 7.42 | 7.50 / 6.69 | 0 / 0 |
| Deeper-teacher student ×1.2 | 2.54 / 2.69 | 7.33 / 7.27 | 7.40 / 6.83 | 0 / 0 |
| Teacher answers, plain (64) | 2.25 / 2.47 | 8.88 / 8.95 | 8.89 / 8.81 | 0 / 0 of 64 |
| Teacher answers, deeper (64) | 3.17 / 3.66 | 8.66 / 8.77 | 8.53 / 8.72 | 0 / 0 of 64 |

Paired (Sonnet / Opus): deeper-teacher student ×1 − plain-teacher student ×1: philosophy +0.23 ± 0.19 / +0.27 ± 0.17, helpfulness −0.60 ± 0.13 / −0.75 ± 0.21, sum −0.38 ± 0.16 / −0.48 ± 0.26. At ×1.2: philosophy +0.50 / +0.54, helpfulness −0.71 / −0.60, sum −0.21 / −0.06. Against the woven prompt, the deeper-teacher student ×1.2 still gains +1.29 ± 0.24 / +2.15 ± 0.31 in the sum.

What this says:
- **The better teacher didn't make a better student.** The deeper answers really are better: about +1 philosophy at almost the same helpfulness (sum 11.7–12.4 against 11.1–11.3). But the student gained only +0.25 philosophy and lost 0.6–0.75 helpfulness and 0.75 coherence, slightly worse overall than learning from the plainer teacher.
- **The student copies the surface of philosophy, not its substance.** Its answers drop names in the teacher's pattern ("Aristotle said we should act as friends would…") and sometimes garble the practical part. In one answer about a neighbour's barking dog, it asks the neighbour to tell the dog to settle. Denser teacher answers are harder to fit (higher final loss), and what doesn't fit is the content.
- **So at this scale the bottleneck has moved from the teacher to the student.** A 4B model with a 9M-parameter adapter and 384 examples learns the register easily but not precise, task-specific philosophical connections. Closing that gap would need a larger adapter or model, more data, or training signals that reward correctness, like the DPO of [exp. M](4-into-the-weights.md#exp-m) on top of distillation. None of these were run.

<a id="exp-r"></a>

### R · A larger student adapter

> **Question.** Does a 4× larger adapter (rank 32) help?  
> **Result.** No. It fits the teacher better and generalises no better.  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c28)

Branch `exp-capacity` (`scripts/exp_capacity.py`, `runs/capacity/`). Experiment Q suggested the student was the bottleneck: the deeper teacher's answers fit worse (final loss 1.19 vs 0.96) and the student copied their surface. Here the adapter is 4× larger, rank 32 on all MLPs (36M parameters, same recipe). It was trained on each teacher's 384 answers and compared with the rank-8 students on the 48 confirmation prompts, scored by Sonnet and Opus. Final training loss: plain 0.62 (rank 8: 0.96), deeper 0.77 (rank 8: 1.19).

| Config | Philo (S / O) | Coherence (S / O) | Helpful (S / O) | Degenerate (S / O) |
|---|---|---|---|---|
| Baseline | 0.02 / 0.04 | 8.90 / 8.69 | 8.29 / 7.69 | 1 / 0 of 48 |
| Prompt woven | 3.31 / 2.88 | 8.23 / 8.06 | 5.83 / 4.71 | 0 / 1 |
| Rank 8, plain teacher | 2.17 / 2.23 | 8.42 / 8.25 | 8.17 / 7.46 | 0 / 0 |
| Rank 32, plain teacher | 2.42 / 2.27 | 8.38 / 7.92 | 8.12 / 6.77 | 0 / 0 |
| Rank 8, deeper teacher | 2.23 / 2.48 | 7.52 / 7.23 | 7.54 / 6.48 | 0 / 0 |
| Rank 32, deeper teacher | 2.10 / 2.23 | 7.10 / 7.15 | 7.23 / 6.75 | 3 / 3 |

Paired (Sonnet / Opus), philosophy + helpfulness: rank 32 − rank 8, plain teacher: +0.21 ± 0.19 / −0.65 ± 0.26. Rank 32 − rank 8, deeper teacher: −0.44 ± 0.26 / +0.02 ± 0.23.

What this says:
- **More adapter capacity doesn't help.** Four times the parameters fit the training answers much better (lower loss) but don't generalize better. On the plainer teacher the two judges disagree in sign (+0.21 Sonnet, −0.65 Opus). On the deeper teacher, rank 32 is no better and adds a few broken outputs. The extra capacity goes into memorizing the 384 answers.
- **So the limit is the base model, the amount of data, or both, not the adapter.** Getting from the register to real, task-specific philosophical reasoning seems to need a bigger base model or much more data, beyond what a single machine and these runs covered.
- **The practical recommendation stands.** A rank-8 adapter trained on the plainer Claude answers is the best balance found: about 2.1–2.4 philosophy at 8.1–8.2 helpfulness (Sonnet) with no prompt, or ×1.5 gain on the 128-answer version for about 2.8–3.0 philosophy.

<a id="exp-s"></a>

### S · Preference training on top of the Claude-taught student

> **Question.** Does DPO stack on the Claude-taught student?  
> **Result.** No gain under either judge. The student's sampled answers are mostly plain, so the pairs had little to teach.  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c29)

Branch `exp-dpo-claude` (`scripts/exp_dpo_claude.py`, `runs/dpo_claude/`). This combines the two levers that moved the frontier: a stronger teacher ([exp. O](#exp-o)–[P](#exp-p)) and preference pairs ([exp. M](4-into-the-weights.md#exp-m)). Experiment P's rank-8 student (384 Claude-written answers) answered 192 training questions 4 times each with no prompt (temperature 0.8, 160 tokens). All 768 answers were judged blind, and dominance pairs were built with the same filter as [exp. M](4-into-the-weights.md#exp-m): 264 pairs over 107 questions. DPO was run from that student exactly as in [exp. M](4-into-the-weights.md#exp-m), with the same student frozen as the reference; training reward accuracy reached 1.00. Evaluation used the 48 confirmation prompts, scored by Sonnet and Opus.

The sampled answers used for the pairs were much less philosophical than the student's greedy answers: philosophy 0.57–0.66, helpfulness 7.8–8.0, with 12–22 of 192 degenerate per sample.

| Config | Philo (S / O) | Coherence (S / O) | Helpful (S / O) | Degenerate (S / O) |
|---|---|---|---|---|
| Baseline | 0.04 / 0.02 | 8.90 / 8.73 | 8.23 / 7.73 | 1 / 0 of 48 |
| Prompt woven | 3.04 / 3.02 | 8.00 / 8.00 | 5.81 / 4.85 | 0 / 0 |
| Claude-taught student (start) | 2.10 / 2.42 | 8.29 / 8.12 | 8.08 / 7.54 | 0 / 0 |
| DPO on it | 2.04 / 2.19 | 8.42 / 8.23 | 8.04 / 7.62 | 0 / 0 |
| DPO on it, ×1.2 | 2.08 / 2.42 | 8.19 / 7.88 | 7.77 / 6.98 | 0 / 1 |

Paired against the starting student (Sonnet / Opus), philosophy + helpfulness: DPO −0.10 ± 0.17 / −0.15 ± 0.21; DPO ×1.2 −0.33 ± 0.18 / −0.56 ± 0.19.

What this says:
- **DPO didn't improve the best student.** Neither judge shows a gain, and the ×1.2 version only loses helpfulness. The two levers don't stack here.
- **The pairs taught the wrong lesson.** Sampled at temperature 0.8, this student's answers are mostly plain (philosophy about 0.6), and 6–11% were flagged as broken. *Correction ([Correction 1](6-can-we-trust-the-scores.md#correction-1)): 71 of those 72 are answers that ran on past the end of the turn, a harness artefact; only 1 of 768 is genuinely broken, and only 11 of the 264 pairs had a run-on as the rejected answer. The conclusion stands: the samples were too plain.* So the pairs mostly separated broken or flat answers from ordinary ones, which greedy decoding already avoids. In [exp. M](4-into-the-weights.md#exp-m) the starting student's samples were reflective, and the pairs could pick out *better* reflection. Preference training needs samples that already vary along the dimension you want to improve.
- **The Claude-taught student's style is fragile under sampling, at least at 160 tokens and temperature 0.8.** The rank-1 woven student held up under temperature 0.7 ([exp. N](4-into-the-weights.md#exp-n)). This one wasn't re-checked at 0.7 on the evaluation prompts, which is the caveat on its [exp. O](#exp-o)–[P](#exp-p) results.

<a id="exp-w"></a>

### W · Style or content? Ablating the teacher's reflections

> **Question.** Does the student learn a free-floating style or reflection tied to the task?  
> **Result.** Reflection tied to the task. The practical part alone teaches no philosophy; mismatched reflections cost −0.8 philosophy, −0.8 helpfulness and −2.2 coherence.  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c33)

The question from the scouting pass: when the student learns from Claude's answers, is it learning *reflection about the task* or just *a reflective register* (any coda will do)? And does Claude's practical content help on its own? Claude (Sonnet) split each of the 384 teacher answers of experiment P into its practical part (reflective sentences removed, nothing else changed) and the removed reflection, verbatim. 381 answers had both parts, and the two parts cover 99.8% of the original characters (`runs/style/split.json`). Three rank-8 MLP students were trained exactly as in P (`scripts/exp_style.py`):

- **practical only:** just the practical part (66 words on average);
- **own reflection:** the practical part plus its own reflection, rebuilt by the same pipeline (89 words; the matched control);
- **mismatched reflection:** the practical part plus the reflection written for a different question (a fixed derangement; 89 words).

One blind pool on the 48 confirmation prompts, with the baseline, the woven prompt and P's original student. Outputs were cut at the end of the turn ([Correction 1](6-can-we-trust-the-scores.md#correction-1)) and judged by Sonnet and Opus.

| Config | Philosophy (S / O) | Helpfulness (S / O) | Coherence (S / O) | Degenerate (S / O) | Final train loss |
|---|---|---|---|---|---|
| Baseline | 0.02 / 0.04 | 8.44 / 7.79 | 8.92 / 8.81 | 1 / 1 | — |
| Prompt woven | 3.19 / 3.10 | 5.77 / 4.98 | 8.08 / 8.08 | 0 / 0 | — |
| Original answers (P's student) ×1.0 | 2.27 / 2.25 | 8.08 / 7.46 | 8.21 / 8.10 | 0 / 0 | 1.00 |
| Practical only ×1.0 / ×1.2 | 0.04 / 0.04, 0.04 / 0.02 | 8.35 / 7.42, 7.92 / 6.96 | 8.69 / 8.21, 8.31 / 7.96 | 1 / 1, 0 / 0 | 1.04 |
| Own reflection ×1.0 / ×1.2 | 2.33 / 2.29, 2.46 / 2.46 | 8.10 / 7.23, 7.81 / 6.73 | 7.96 / 7.81, 7.88 / 7.50 | 0 / 0, 1 / 1 | 1.09 |
| Mismatched reflection ×1.0 / ×1.2 | 1.50 / 1.60, 1.83 / 1.65 | 7.27 / 6.44, 6.92 / 6.06 | **5.81 / 5.54**, 5.50 / 5.31 | 6 / 3, 8 / 0 | 1.42 |

Paired over the 48 prompts (Sonnet / Opus, ×1.0):

| Contrast | Philosophy | Helpfulness | Coherence |
|---|---|---|---|
| own − mismatched | +0.83 ± 0.14 / +0.69 ± 0.15 | +0.83 ± 0.15 / +0.79 ± 0.19 | +2.15 ± 0.21 / +2.27 ± 0.19 |
| own − practical only | +2.29 ± 0.11 / +2.25 ± 0.11 | −0.25 ± 0.18 / −0.19 ± 0.24 | −0.73 ± 0.13 / −0.40 ± 0.15 |
| practical only − baseline | +0.02 ± 0.02 / 0.00 ± 0.03 | −0.08 ± 0.15 / −0.38 ± 0.23 | −0.23 ± 0.09 / −0.60 ± 0.15 |
| own − original student | +0.06 ± 0.13 / +0.04 ± 0.13 | +0.02 ± 0.14 / −0.23 ± 0.21 | −0.25 ± 0.12 / −0.29 ± 0.14 |

What this says:
- **The rebuilt control works.** "Own reflection" matches P's original student within noise, so the split-and-rebuild pipeline doesn't distort anything.
- **All the philosophy comes from the reflections, and it costs almost no helpfulness.** The practical part alone teaches nothing a judge can see: no philosophy, and no helpfulness gain over the baseline, even though Claude's full answers are rated about 8.8 for helpfulness. Adding the reflections back adds 2.3 philosophy for 0.2–0.25 helpfulness (within about 1 SE) and 0.4–0.7 coherence. The student's "+2 helpfulness over the prompt" ([exp. O](#exp-o)) is the baseline's helpfulness kept, not new helpfulness learned from Claude.
- **The student learns which reflection fits which task.** With mismatched reflections, it can't predict the coda from the question, so it learns a few generic codas and pastes them anywhere. In 28 of 48 answers it is the same remark about budgets ("A budget is a way of making your values visible…") after questions about thunder or first-aid kits, although only 3 of the 384 training reflections mention budgets. The judges score these non-sequiturs as incoherent (5.8 against 8.0). The matched students never do this (0/48), and their codas are about the question: "…it is really the heated air, and this reminds us that what we perceive directly are effects, not causes." So the behaviour isn't a register that any coda would satisfy. The topic binding in the teacher data is what the student generalizes, and it can't be faked with shuffled data.
- **Implication for data.** The quality lever is the reflection, and specifically its fit to the task. A cheaper teacher dataset (a plain practical answer plus one task-specific reflective sentence) should work as well as a full rewrite.

<a id="exp-x"></a>

### X · Off the beaten path: two turns, a competing system prompt, checkable questions

> **Question.** Does the student hold up outside the test distribution?  
> **Result.** Yes over two turns and on checkable questions (16/16 correct), but it keeps reflecting under a system prompt that asks for brief, literal answers.  
> **Status.** The judge here did not see the system prompt (AF). A retrained student can switch the habit off in a pilot (AH). Three of the 16 two-turn openers repeat a confirmation, training or contrast prompt ([AK](6-can-we-trust-the-scores.md#exp-ak)).  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c34)

Every earlier evaluation is a single-turn everyday request. Here the two best prompt-free students (the 128-answer student at ×1.5, [exp. O](#exp-o), and the 384-answer student at ×1.0, [exp. P](#exp-p)), the woven prompt and the baseline face three situations they weren't built for (`data/ood_prompts.json`, 16 items each, 160 new tokens):

- **two turns:** an everyday request, the config's own answer, then a practical follow-up ("My apartment has no balcony. Can I still do it indoors?");
- **terse system prompt:** the first 16 confirmation prompts under "Answer as briefly and literally as possible. Give only the requested information, with no commentary." The students get only this prompt; for the woven config it is appended to the woven prompt;
- **checkable:** 16 arithmetic, fact and code questions with a regex-checked answer.

Judged blind by Sonnet and Opus in one pool, outputs cut at the turn end ([Correction 1](6-can-we-trust-the-scores.md#correction-1); none needed cutting).

| Config | Turn 1: P / H | Turn 2: P / H | Terse system: P / H, words | Checkable: P / H, correct |
|---|---|---|---|---|
| Baseline | 0.00 / 8.19 | 0.06 / 8.00 | 0.00 / 6.94, 28 | 0.00 / 8.62, 13/16 |
| Prompt woven | 4.19 / 5.69 | 3.56 / 6.56 | **0.44** / 7.50, 68 | 3.06 / 7.00, 13/16 |
| 128-answer student ×1.5 | 2.75 / 7.56 | 3.19 / 7.25 | **2.31** / 7.25, 71 | 3.31 / 7.44, 16/16 |
| 384-answer student ×1.0 | 2.12 / 8.19 | 2.38 / 7.94 | **1.81** / 7.75, 63 | 2.38 / 8.69, 16/16 |

(Sonnet shown; Opus agrees within 0.5 on philosophy and is 0.2–1.1 lower on helpfulness; no answer was judged degenerate by either.)

What this says:
- **Two turns: the register is stable.** The students' philosophy doesn't decay or pile up in the follow-up; it rises slightly, and helpfulness holds. The woven prompt keeps more philosophy (3.6) but fades from turn 1. Turn-2 codas stay on topic ("Keep the mix moist and covered. It's a modest act of care: you're returning nutrients…").
- **Checkable questions: correct, but reflective everywhere.** Both students answered all 16 correctly. The baseline and the prompt missed 3 each, not by being wrong but by deriving at length (LaTeX, "Method 1: long division") and hitting the token limit before the result. The students also add a reflection to "What is 17 multiplied by 23?", so their philosophy on facts and arithmetic (2.4–3.3) is as high as on everyday requests. The judges didn't penalize it (helpfulness 7.4–8.7), but it is the habit generalizing to places nobody asked for it.
- **A competing system prompt: the students don't switch off.** Asked for brief, literal answers, the woven model drops its philosophy almost entirely (0.44), because the later instruction wins. The students keep 1.8–2.3 philosophy and write 63–71 words where the baseline writes 28. The judges rated the students *more* helpful than the terse baseline (7.3–7.8 against 6.9), but this rubric rewards completeness, not following the operator's instruction. By the instruction's own standard, the students partly ignore it.
- **Trade-off.** A trained-in register is robust (it survives turns and odd question types) for the same reason it is hard to control: it doesn't depend on any instruction in the context. To deploy one, keep a way to turn it off: the LoRA gain (×0 = off, [exp. D2](4-into-the-weights.md#exp-d2)–[L](4-into-the-weights.md#exp-l)) or an instruction-conditioned training set (answers that drop the reflection when asked to be terse), which was not tried.

<a id="exp-y"></a>

### Y · Stacking levers and best-of-4 headroom

> **Question.** Do a prompt on top of the student, or best-of-4 selection, get past the frontier?  
> **Result.** No. The stack lands on the same line, and best-of-4 does not beat the student's greedy answer (+0.08 ± 0.19).  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c35)

Two inference-only questions about the 384-answer student ([exp. P](#exp-p)), suggested by the scouting pass. (1) **Stacking:** does the student plus a system prompt go past the frontier, or is it the same push twice? (2) **Headroom:** how much better is the best of the student's own samples than its usual answer? That bounds what a perfect reranker or preference signal could extract. It also explains [exp. S](#exp-s), where DPO on the student's samples found nothing.

Method (`scripts/exp_stack.py`, 48 confirmation prompts, generation now stopping at the turn end, [Correction 1](6-can-we-trust-the-scores.md#correction-1)). Greedy configs: baseline, woven prompt, a light prompt ("Answer the user's request fully and practically. Where it fits naturally, add one brief reflective thought."), the student, the student plus the woven prompt, and the student plus the light prompt. Headroom: the student and, for comparison, the woven prompt each sampled 4 answers per prompt (T 0.7, top-p 0.9); Opus scored all 384 samples and picked, per prompt, the non-degenerate answer with the best philosophy + helpfulness. The picked answer and an unselected draw (sample 0) went into the blind pool with the greedy configs, scored by Sonnet and Opus. Opus picked, so its scores of the picked answers are optimistic; Sonnet's are the cleaner test.

| Config | Philosophy (S / O) | Helpfulness (S / O) | Sum (S / O) | Sum − woven (S / O) |
|---|---|---|---|---|
| Baseline | 0.04 / 0.04 | 8.40 / 7.67 | 8.44 / 7.71 | −1.02 ± 0.19 / +0.29 ± 0.23 |
| Woven prompt | 3.29 / 2.73 | 6.17 / 4.69 | 9.46 / 7.42 | — |
| Light prompt | 0.08 / 0.06 | 8.50 / 7.77 | 8.58 / 7.83 | −0.88 ± 0.21 / +0.42 ± 0.21 |
| Student (greedy) | 2.12 / 2.08 | 8.00 / 7.56 | 10.12 / 9.65 | +0.67 ± 0.26 / +2.23 ± 0.28 |
| Student + woven prompt | 2.75 / 2.85 | 7.50 / 6.77 | 10.25 / 9.62 | +0.79 ± 0.26 / +2.21 ± 0.31 |
| Student + light prompt | 1.94 / 2.02 | 8.06 / 7.00 | 10.00 / 9.02 | +0.54 ± 0.21 / +1.60 ± 0.28 |
| Student, one sample (T 0.7) | 2.15 / 1.98 | 7.71 / 6.85 | 9.85 / 8.83 | +0.40 ± 0.27 / +1.42 ± 0.34 |
| Student, best of 4 (Opus-picked) | 2.27 / 2.21 | 7.94 / 7.35 | 10.21 / 9.56 | +0.75 ± 0.21 / +2.15 ± 0.30 |
| Woven, one sample (T 0.7) | 3.71 / 2.94 | 5.56 / 4.38 | 9.27 / 7.31 | −0.19 ± 0.24 / −0.10 ± 0.20 |
| Woven, best of 4 (Opus-picked) | 3.56 / 3.04 | 5.83 / 4.85 | 9.40 / 7.90 | −0.06 ± 0.26 / +0.48 ± 0.26 |

Paired contrasts, philosophy + helpfulness (Sonnet / Opus): student best-of-4 − one sample +0.35 ± 0.18 / +0.73 ± 0.24; student best-of-4 − greedy student +0.08 ± 0.19 / −0.08 ± 0.21; student + woven − student +0.12 ± 0.19 / −0.02 ± 0.22; student + light − student −0.12 ± 0.19 / −0.62 ± 0.26.

What this says:
- **Stacking the prompt is the same push again.** The woven prompt on the student adds 0.6–0.8 philosophy and removes 0.5–0.8 helpfulness: about the same sum, further along the line. The light prompt is too weak to matter: alone it adds no philosophy (0.08), and on the student it changes nothing under Sonnet and costs helpfulness under Opus.
- **There's little headroom in the student's samples.** Sampling costs 0.3–0.8 (one sample vs greedy, Sonnet / Opus) and picking the best of four wins it back, but no more: the best-of-4 answer is as good as the greedy one. So the student's distribution doesn't hold answers that are clearly beyond its frontier, and a perfect reranker over 4 samples would add nothing. That is why DPO on its samples ([exp. S](#exp-s)) had nothing to learn from.
- **Judge noise inflates selection.** By its own scores of the samples, Opus's picks were +1.12 better than the average sample (10.08 vs 8.96). Re-scored in the blind pool, the same answers are +0.73 better than an unselected draw by Opus and +0.35 by Sonnet. Much of what a judge "selects" is its own scoring noise, a caution for any best-of-n or judge-filtered training data ([exp. J](4-into-the-weights.md#exp-j), [exp. M](4-into-the-weights.md#exp-m)).

<a id="exp-v"></a>

### V · A smaller and a larger base model

> **Question.** Does the size of the base model matter?  
> **Result.** Yes, with shrinking returns. The 4B student beats the 2B by +1.2 to +1.7 in philosophy + helpfulness; the 9B beats the 4B by +0.8 to +1.0 at equal philosophy (both 4-bit).  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c36)

[exp. Q](#exp-q)–[R](#exp-r) pointed at the 4B base model as the remaining limit. The direct test is to run the same recipe on a larger sibling, and, as a cheaper second point, on a smaller one. Both checkpoints were downloaded from Hugging Face: Qwen/Qwen3.5-2B (4.6 GB) and Qwen/Qwen3.5-9B (19.3 GB). `scripts/exp_scale.py --model 2B|4B|9B` runs, for each model, its own baseline, its own woven prompt, and a rank-8 MLP LoRA on all layers trained on experiment P's 384 Claude answers with the same recipe (4B reuses P's adapter), evaluated at ×1.0 and ×1.2 on the 48 confirmation prompts. Everything is in one blind pool, cut at the turn end ([Correction 1](6-can-we-trust-the-scores.md#correction-1); none needed cutting), and judged by Sonnet and Opus.

**9B didn't fit.** Loaded in bf16 on MPS (18 GB resident, `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`), it didn't finish the first batch of 8 baseline answers in 30 minutes: swap full, CPU about 6%, i.e. paging. The 4B model generates all 48 in about 1.5 minutes. MPS has no 8-bit kernels, so there was no lower-precision fallback in this stack; the run was stopped (`runs/scale/9B_train_error.txt`). The comparison below is therefore 2B against 4B.

| Config | Philosophy (S / O) | Helpfulness (S / O) | Coherence (S / O) | Sum (S / O) |
|---|---|---|---|---|
| 2B baseline | 0.02 / 0.00 | 7.96 / 7.15 | 8.65 / 8.27 | 7.98 / 7.15 |
| 2B woven prompt | 2.50 / 2.21 | 5.67 / 4.85 | 7.81 / 7.56 | 8.17 / 7.06 |
| 2B student ×1.0 | 2.27 / 2.31 | 6.50 / 5.77 | 7.12 / 6.96 | 8.77 / 8.08 |
| 2B student ×1.2 | 2.29 / 2.42 | 6.10 / 5.15 | 6.62 / 6.42 | 8.40 / 7.56 |
| 4B baseline | 0.04 / 0.02 | 8.31 / 7.71 | 8.92 / 8.77 | 8.35 / 7.73 |
| 4B woven prompt | 3.04 / 3.02 | 5.85 / 4.90 | 7.96 / 8.02 | 8.90 / 7.92 |
| 4B student ×1.0 | 2.19 / 2.27 | 7.81 / 7.46 | 8.08 / 8.21 | 10.00 / 9.73 |
| 4B student ×1.2 | 2.46 / 2.48 | 7.46 / 7.02 | 7.98 / 7.85 | 9.92 / 9.50 |

Paired over the 48 prompts (Sonnet / Opus):

| Contrast | Philosophy + helpfulness |
|---|---|
| student ×1.0 − own woven prompt, 2B | +0.60 ± 0.28 / +1.02 ± 0.32 |
| student ×1.0 − own woven prompt, 4B | +1.10 ± 0.30 / +1.81 ± 0.27 |
| 4B student − 2B student (×1.0) | +1.23 ± 0.26 / +1.65 ± 0.30 |
| helpfulness lost against own baseline, 2B / 4B student | −1.46 ± 0.24 / −0.50 ± 0.20 (S); −1.38 ± 0.27 / −0.25 ± 0.22 (O) |

The 2B adapter (4.7M parameters) ends training at loss 1.35, against 0.99 for the 4B one on the same answers.

What this says:
- **Both sizes learn the philosophy equally.** The 2B student reaches the same philosophy as the 4B one (2.3), and as its own prompt. Picking up "reflect on this task" isn't what needs scale.
- **The smaller model pays for it in helpfulness and coherence.** The 2B student loses about 1.4 helpfulness and 1.3–1.5 coherence against its own baseline; the 4B student loses 0.25–0.5 and 0.6–0.8. The 4B student is +1.2 to +1.7 better overall on the same prompts, and its lead over its own prompt is roughly twice the 2B's. The 2B model also fits the teacher's answers worse.
- **So the frontier moves with model size**, and by more than any intervention since [exp. O](#exp-o). This supports the suspicion of [exp. Q](#exp-q)–[R](#exp-r) that the 4B base model is what limits the student. Part 2 below extends it to 9B.

**Part 2: 9B through MLX, 4-bit** (`scripts/exp_scale_mlx.py`, separate `.venv-mlx` with `mlx-lm` 0.32.2). Apple's MLX runs 4-bit quantized models natively, so the official mlx-community checkpoints Qwen3.5-9B-4bit (5.95 GB) and, as the in-stack reference, Qwen3.5-4B-4bit (3.03 GB) got the identical recipe: a QLoRA with a frozen 4-bit base. It used the same 384 answers and the same formatting (`chat(q)` + answer + `<|im_end|>`, loss on answer tokens), a rank-8 LoRA at scale 2 on gate/up/down of every layer (9.04M parameters on 4B, exactly the PyTorch count; 12.58M on 9B), and Adam 2e-4 for 2 epochs at batch 1 × 4 accumulation, with activation checkpointing. Evaluation was greedy, repetition penalty 1.1, 120 tokens, stopping at `<|im_end|>` (no run-ons), in one blind pool scored by Sonnet and Opus. MLX's Gated-DeltaNet Metal kernel has no backward pass, so training uses the model's pure-ops path (`model.train()`) and generation the kernel. One difference from PyTorch: MLX initializes LoRA A uniformly (a 0.58× smaller spread), so training starts slower (4B loss 1.81 at step 96 vs 1.53 in PyTorch). Final losses (mean of the last 16 updates): 4B 0.82, 9B 0.70; training took 32 and 41 minutes.

| Config (4-bit MLX) | Philosophy (S / O) | Helpfulness (S / O) | Coherence (S / O) | Sum (S / O) |
|---|---|---|---|---|
| 4B baseline | 0.02 / 0.02 | 8.23 / 7.71 | 8.96 / 8.79 | 8.25 / 7.73 |
| 4B woven prompt | 3.38 / 3.17 | 5.29 / 4.50 | 8.04 / 8.00 | 8.67 / 7.67 |
| 4B student ×1.0 | 2.25 / 2.27 | 7.31 / 6.35 | 8.02 / 7.58 | 9.56 / 8.62 |
| 4B student ×1.2 | 2.40 / 2.46 | 6.85 / 5.58 | 7.38 / 6.85 | 9.25 / 8.04 |
| 9B baseline | 0.02 / 0.04 | 8.31 / 7.90 | 8.98 / 8.88 | 8.33 / 7.94 |
| 9B woven prompt | 3.08 / 3.02 | 6.15 / 4.88 | 8.40 / 8.17 | 9.23 / 7.90 |
| 9B student ×1.0 | 1.98 / 2.10 | 7.69 / 7.06 | 8.27 / 7.98 | 9.67 / 9.17 |
| 9B student ×1.2 | 2.42 / 2.29 | 7.65 / 6.73 | 8.12 / 7.69 | 10.06 / 9.02 |

Paired over the 48 prompts (Sonnet / Opus), philosophy + helpfulness:

| Contrast | Sonnet | Opus |
|---|---|---|
| 9B student − 4B student, ×1.0 | +0.10 ± 0.28 (philosophy −0.27, helpfulness +0.38) | +0.54 ± 0.28 (−0.17, +0.71) |
| 9B student − 4B student, ×1.2 (equal philosophy, about 2.4) | **+0.81 ± 0.27** | **+0.98 ± 0.29** |
| student ×1.0 / ×1.2 − own woven prompt, 4B | +0.90 ± 0.32 / +0.58 ± 0.30 | +0.96 ± 0.35 / +0.38 ± 0.37 |
| student ×1.0 / ×1.2 − own woven prompt, 9B | +0.44 ± 0.24 / +0.83 ± 0.24 | +1.27 ± 0.25 / +1.12 ± 0.29 |
| 9B woven − 4B woven | +0.56 ± 0.25 | +0.23 ± 0.25 |
| helpfulness lost against own baseline, 4B / 9B student ×1.0 | −0.92 ± 0.19 / −0.62 ± 0.19 | −1.35 ± 0.21 / −0.83 ± 0.22 |

What part 2 says:
- **9B helps, the same way 4B helped over 2B, but less.** At equal philosophy the 9B student is +0.8 to +1.0 better than the 4B, all of it in kept helpfulness and coherence. Amplified to ×1.2, it holds its coherence (8.1 vs 7.4) where the 4B starts to fray. The step is smaller than 2B→4B (+1.2 to +1.7 at ×1.0).
- **The prompt scales too.** Under Sonnet, the 9B follows the woven prompt better (+0.56 over the 4B's), so the student's lead over its *own* prompt doesn't clearly grow (Sonnet mixed, Opus +0.3). A bigger model raises both ways of getting the behaviour.
- **Quantization costs something.** The 4-bit 4B student loses more helpfulness against its baseline (0.9–1.35) than the bf16 PyTorch one (0.25–0.5, part 1), probably from the 4-bit base and the different LoRA initialization. That is why every 9B comparison here is made against the 4-bit 4B in the same pool, never across stacks.
- **Overall:** scale is a real lever with diminishing steps per doubling (2B→4B large, 4B→9B moderate). Larger models, or full-precision training of the 9B on a bigger machine, would say whether the steps keep shrinking.

<a id="exp-ah"></a>
<a id="conditional-reflection"></a>

### AH · Can the student switch reflection off and back on?

> **Question.** Can a short retraining make the habit follow instructions?  
> **Result.** Yes in a pilot. The mixed-instruction student followed 8/8 held-out requests in each of three conditions; the original student 3/8, 0/8 and 5/8.  
> **Status.** Addresses the open problem from X.  
> 2026-11-14 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c45)

Experiment X found that the student kept adding reflections under a terse instruction. We tested whether a short continuation run could teach the existing adapter to follow an explicit preference while keeping the practical result.

**Method** (`scripts/exp_conditional_reflection.py`, `data/conditional_reflection.json`). Warm-start the 384-answer rank-8 student. Use 20 authored tasks under three training instructions, with the same practical target in all three. Append one task-specific sentence when reflection is requested. Keep the practical target alone when reflection is optional or suppressed. Train on 60 examples for two epochs and compare greedy continuations from base, the original student, and the mixed-instruction student. Evaluate eight held-out questions with paraphrased system instructions. Exact-result scoring uses typed JSON matching, Python syntax-tree matching, and literal text matching for the remaining tasks. The requested condition requires the correct first line followed by additional text.

| Configuration | Requested, correct first line plus text | Optional, exact result | Suppressed, exact result |
| --- | ---: | ---: | ---: |
| Base | 6/8 | 2/8 | 7/8 |
| Original rank-8 student | 3/8 | 0/8 | 5/8 |
| Mixed-instruction student | 8/8 | 8/8 | 8/8 |

Four three-turn conversations per configuration asked for reflection, then asked for the same result alone, then requested reflection on a task in another format. These instructions appeared in the user turns, while training placed them in the system message.

| Configuration | Request | Suppress | Request again |
| --- | ---: | ---: | ---: |
| Base | 2/4 | 3/4 | 3/4 |
| Original rank-8 student | 2/4 | 3/4 | 3/4 |
| Mixed-instruction student | 4/4 | 4/4 | 4/4 |

What this says:

- **A small continuation run improved format control on these tasks.** When asked for a JSON status by itself, the mixed student returned just the object. When asked to add a thought after the object, it gave a separate sentence on why an explicit status helps another system act on the record.
- **The mix also followed a change of instruction across turns.** Each mixed-student conversation returned to a correct first line plus text after the suppression turn. The test measures format and the presence of extra text, with no independent rating of philosophical quality.
- **The result is local to a small pilot.** Held-out questions share task families with training. The run uses one training seed and one starting adapter. A matched continuation control and broader tasks are needed to separate instruction conditioning from extra practice with these output formats.

[Single-turn replies](../data/results/conditional-reflection.jsonl) · [Three-turn replies](../data/results/conditional-reflection-switches.jsonl) · [Training loss](../data/results/conditional-reflection-loss.json)
