# Pilot AL: results (2026-11-20)

Draft write-up for comparison with the author's own evaluations. Not yet in FINDINGS.md or `findings/7-register-versus-skill.md`. Saved outputs: [outputs/](outputs/) (`summary.md`, `summary.json`, `scores.jsonl.gz`).

> **Question.** On tasks built from real philosophical texts and keyed without a grader, does skill grow with model size, and do the register levers leave it at base level?
> **Result.** Yes from 2B to 4B, not from 4B to 9B. On the single-passage validity task, the 4B tells an intact argument from its corrupted version on 82% of 87 items, against 64% for the 2B (z 3.0). Prompts and students stay at base; constant pushes lose 16–22 points (z −2.9 to −3.8). Blind Claude judges agree with 151 of 152 keys; the author agrees with 19 of 20.

## What was measured

Ten configurations, all forced choice with AI's scorer unchanged ("Answer:" prefill plus the bold path, both letter orders):

- **Scale ladder:** base 2B (bf16), base 4B (bf16), base 4B (4-bit MLX), base 9B (4-bit MLX).
- **Register grid, 4B bf16:** woven prompt, rank-1 student ×1, Claude-taught 384 ×1.0, diff-means α 0.5, B 0.6·p50, random 0.6·p50 (seed 0).

Three tasks over 8 public-domain sources (Descartes, Hobbes, Leibniz–Clarke, Hume, Berkeley, Anselm with Gaunilo, Vitoria, Galileo):

| Task | Units | What the model sees |
|---|---|---|
| `corrupt` (paired, as planned) | 22 arguments | Two passages; which one's conclusion follows? |
| `verdict` (single passage, added) | 87 (argument, corruption) units | One passage; does its conclusion follow? Original, paraphrase, and 1 + 3 corrupted versions per argument. |
| `objreply` (as planned) | 20 pairs | Objection → which of four replies from the same exchange, and the reverse; original and paraphrased wording. |

## Why the paired task was replaced

The paired `corrupt` format measured letter preference. The 4B bf16 picked A on 89% of keyed pairs and the 9B picked B on 81%; only 14–34% of pairs were answered correctly in both orders. `verdict` asks about one passage at a time. Its main score, **P(follows | original) > P(follows | corrupted)** with each averaged over both letter orders, cannot be moved by a constant letter or yes/no preference. Three extra corruptions per argument (mixed types) raised the units from 22 to 88; one was dropped after the blind judge showed it still followed, leaving 87.

## Human-proxy baseline (blind Claude judges)

Two fresh Claude instances answered blind packets, reading only the packet file and never the keys or data:

- **20-item baseline packet** (10 `corrupt`, 10 `objreply`): 20/20.
- **132 single passages** (every original, paraphrase and corrupted text, shuffled, unlabelled): 131/132. The one miss was a bad item (`descartes-bodies-exist`, corrupted2), now dropped.

The plan's gate (≥ 85% agreement with the keys) passes with Claude as the grader.

**Author's own grading (MA, 2026-11-21), same 20-item packet:** 19/20 = 95% (objreply 10/10, corrupt 9/10), so the gate passes with a human grader too ([baseline/answers-MA.json](baseline/answers-MA.json)). The author and the Claude judge disagree on one item, q11 (`berkeley-one-sensation`, original vs corrupted). The author picked the corrupted version. That corruption replaces the premise "one simple, uniform sensation, and not two distinct sensations" with "a sensation that grows stronger as the hand comes closer". In the original, the conclusion that heat and pain are one idea rests on that premise together with their simultaneity; without it, only simultaneity is left. The original's step is contested, though (Hylas's own objection), so a reader can fairly judge that neither version strictly follows. Mark this item for the item review rather than treating the answer as a grader error.

## Results

### Single-passage validity (`verdict`, 87 units)

| config | orig > corr | para > corr | gap orig (pts) | gap para (pts) |
|---|---|---|---|---|
| 2B base | 64.4 ± 5.2 | 67.8 ± 5.0 | 1.9 ± 0.6 | 2.7 ± 0.8 |
| 4B base (bf16) | 81.6 ± 4.2 | 81.6 ± 4.2 | 12.2 ± 1.7 | 14.9 ± 1.8 |
| 4B base (4-bit) | 79.3 ± 4.4 | 75.9 ± 4.6 | 10.7 ± 1.6 | 13.3 ± 1.7 |
| 9B base (4-bit) | 80.5 ± 4.3 | 65.5 ± 5.1 | 12.1 ± 1.6 | 5.9 ± 1.9 |
| woven | 80.5 ± 4.3 | 74.7 ± 4.7 | 11.8 ± 1.8 | 12.8 ± 1.8 |
| rank-1 ×1 | 85.1 ± 3.8 | 79.3 ± 4.4 | 10.7 ± 1.5 | 12.4 ± 1.5 |
| Claude-taught 384 ×1.0 | 81.6 ± 4.2 | 69.0 ± 5.0 | 15.8 ± 1.9 | 13.0 ± 2.1 |
| diff-means 0.5 | 65.5 ± 5.1 | 73.6 ± 4.8 | 3.4 ± 0.8 | 4.5 ± 0.9 |
| B 0.6 | 65.5 ± 5.1 | 65.5 ± 5.1 | 2.3 ± 0.6 | 2.3 ± 0.6 |
| random 0.6 | 59.8 ± 5.3 | 79.3 ± 4.4 | 3.6 ± 1.2 | 10.8 ± 1.3 |

"orig > corr" is the share of units where the model gives the intact original a higher P(follows) than the corrupted version (chance 50%). "gap" is the mean difference in P(follows), in points.

### Objection–reply matching (`objreply`, 20 pairs)

| config | all | original wording | paraphrased |
|---|---|---|---|
| 2B base | 76.2 | 77.5 | 75.0 |
| 4B base (bf16) | 91.2 | 92.5 | 90.0 |
| 4B base (4-bit) | 91.2 | 92.5 | 90.0 |
| 9B base (4-bit) | 92.5 | 95.0 | 90.0 |
| woven / rank-1 / Claude 384 | 91.2 / 91.2 / 90.0 | | |
| diff-means / B / random | 67.5 / 65.0 / 75.0 | | |

### Predictions

- **P1, scale. Holds from 2B to 4B, fails from 4B to 9B.** 2B → 4B (bf16): discrimination +17 (z 3.0), gap +10 points (z 6.8); objreply +15 (z 2.1). 4B → 9B (both 4-bit): discrimination +1 (z 0.2), gap on originals +1 (z 0.9), gap on paraphrases −7 (z −3.4); objreply +1. The blind judges score 99–100%, so the tasks are not at their own ceiling: the models stop improving well below it. 2B → 9B across stacks: +16 (z 3.0).
- **P2, register is not skill. Holds.** Against 4B base, the woven prompt (−1, z −0.4), the rank-1 student (+3.5, z 1.8) and the Claude-taught student (0) keep discrimination at base, and keep objreply at 90–91%. The constant pushes lose 16 (diff-means, z −2.9), 16 (B, z −3.0) and 22 (random, z −3.8) points of discrimination, and their gap falls from 12 to 2–4 points (z −5.9 to −6.8); on objreply they lose 16–26 points (z −2.9 to −3.7). This repeats AI's split on items the project did not write and with keys that need no grader.
- **P3, memorisation. Holds at 4B; possibly not at 9B.** 4B: P(follows) for the original and the paraphrase differ by −2.5 points (z −1.0); discrimination is 81.6% with either as the intact version; objreply is within 2.5 points across wordings. 9B: originals get +6 points more "follows" than paraphrases (z 2.0), and discrimination is 80.5% with the original against 65.5% with the paraphrase. That is the pattern familiarity with the classic text would produce. In the paired format the 4-bit models also chose the original over its paraphrase more often (4B z 2.8, 9B z 4.0).

### Other checks

- **Ceiling and floor:** no `verdict` unit is at ceiling or floor for every model (0 of 87); 1 of 20 `objreply` pairs is.
- **By corruption type (4B base, discrimination):** overreach 96%, quantifier 83%, negation 79%, drop-premise 75%, conditional reversal 58% (n = 6).
- **Yes-bias:** every configuration leans towards "follows". The order-averaged absolute accuracy is 53–72%, and the woven prompt and Claude-taught student shift it most (−7 and −17 points, z −3.1 and −5.2) while leaving discrimination at base. That shift is a change in threshold, not in skill.

## Caveats

- The 9B is 4-bit and the 2B bf16; the 4B → 9B comparison is within the 4-bit stack, and 2B → 9B crosses stacks.
- Claude wrote every corruption and paraphrase, and Claude instances served as the baseline graders. The author's independent check covers only the 20-item packet (95%), not the 132 single passages that `verdict` scores.
- The `verdict` task and the extra corruptions were added after the first run, in response to the letter-bias problem. The predictions were set before any run, but the choice of main metric was not.
- `objreply` is near ceiling at 4B: the blind judge noted that wrong options can often be ruled out by keyword matching.
- 87 units come from 22 arguments, so units within an argument share the original and paraphrase. Treating the 22 arguments as the units (mean discrimination per argument) leaves the conclusions unchanged: 2B → 4B +17 (z 2.9); 4B → 9B +1 (z 0.1); woven −1 (z −0.4), rank-1 +3 (z 1.8), Claude 384 −1 (z −0.1); diff-means −17 (z −2.4), B −16 (z −3.8), random −22 (z −3.0).
