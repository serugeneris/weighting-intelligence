# Experiment AE: is "philosophical" the reflective register, or something else?

Executor: Claude Opus, on the M5 Pro (24 GB) checkout that has `runs/`, `.venv`, `.venv-mlx` and the cached Qwen3.5 weights. The M3 Pro checkout has none of these and cannot run this plan.

## Context

Everything in FINDINGS.md §1–33 measures philosophy as one thing: a judge's 0–9 score for reflective sentences added to practical answers. The project found where that register lives (a mode set at the start of the answer, middle layers, one rank-1 edit) and how to install it. It never tested whether the register has anything to do with philosophical *skill*. The pilots Z–AD (§34–38) hint that it doesn't, and may hurt: the student overread cosmetic details (2/6), transferred distinctions less well (4/6 vs 6/6) and reversed under pressure (0/4). But those pilots have 4–24 author-keyed items each.

The claim under test: **the reflective register and philosophical skill are separate things in this model.** The test is a dissociation: vary the register with the levers we already have, and measure skill on externally defined benchmark tasks with objective scoring.

Benchmark availability, checked on 2026-11-15:

| Benchmark | Status | Use |
|---|---|---|
| PhilBench (philbench.com) | Data not released, but built on the public PhilPapers 2020 Survey (100 questions, philosopher answer shares) | **Main instrument.** Rebuild from the survey. Scored by regex/logits, no judge |
| philosophy-bench (github.com/benedictbrady/philosophy-bench) | Public, 100 scenarios, CC-BY-4.0 data. Agentic tool use, judged by a three-model frontier panel | **Secondary, adapted** to forced choice over its authored response clusters |
| LMCA | Gated by request form | **Not in this plan.** Optional later stage if access arrives |
| Stanford PhilosophyBench (philosophybench.org) | A human essay-grading study, no items | Not runnable. Say so in LITERATURE.md |

Note for the write-up: `www.philosophybench.com` currently redirects to the Stanford study, and FINDINGS §34 and LITERATURE.md give different LMCA URLs. Fix both references.

## Predictions (write these into the script docstring before any run)

- **If philosophy is the register:** skill metrics rise with register score across configurations, and rise with dose inside each lever family.
- **If it is something else:** skill is flat or falls as register rises; a matched random push does as much damage as a register push; skill grows with model size while register does not (§33 found 2B and 4B students equally reflective).

## Stage 0. Preflight (stop and report if any check fails)

- `git pull`, create branch `exp-philbench` from `main`.
- Confirm these exist: `runs/p50.json`, `runs/diffmeans.pt`, `runs/distill/down8-r1.pt`, `runs/more_teacher/mlp-r8_claude384.pt`, `runs/sae/sae.pt`, the 128-answer adapter under `runs/claude_teacher/`, `runs/scale/mlp-r8_2B.pt`, and the MLX adapters from `scripts/exp_scale_mlx.py`.
- Reproduce one known number as a regression check: `python -m scripts.exp_prose_validity` must give 21/24 with no ablation (§36).
- Write only under `runs/philbench/`. Never overwrite another experiment's folder.

## Stage 1. Data

1. **`data/philpapers2020.json`**: the 100 survey questions (40 main, 60 additional) from `https://survey2020.philpeople.org/`, each with its substantive options and the share of philosophers choosing each. Add the PhilBench meta-options ("agnostic/undecided", "the question is unclear", "accept an alternative view", and so on) as a shared list. Record the source URL and retrieval date in the file. If the survey can't be fetched, stop and report; do not write the questions from memory.
2. **Variants per question**: `baseline`, `paraphrase` (one reworded question, same options; write these yourself and check each keeps the meaning), `reversed` (substantive options in reverse order).
3. **`data/philpapers_constraints.json`**: 25–40 logical constraints between answers, each with a one-line justification. Example shape: "accepts physicalism about mind" with "zombies are metaphysically possible" is a violation. Label the file author-written. Keep only constraints most philosophers would accept as entailments, and mark any contestable ones so the analysis can drop them.
4. **philosophy-bench**: clone into `external/philosophy-bench` at a pinned commit (add `external/` to `.gitignore`; do not vendor the data). Read `experiments/c_vs_d/data/scenarios/**/*.yaml` and `SCORING.md` first. Build a loader that turns each scenario into: situation text, the 3–5 authored response clusters as lettered options (order rotated by item index), and each option's `cd_score` or `botched` flag.

## Stage 2. Runner: `scripts/exp_philbench.py`

Reuse, don't rewrite:
- `scripts/philosophy_pilot.py`: `run`, `save`, `messages`, `render`, `generate`, `review_packet`, `residual_edit`.
- `wi/steer.py`: `Steerable.steer(vectors=...)` for constant pushes; `Steerable(model_id)` for 2B.
- `scripts/exp_teacher.py: load_lora`, `scripts/exp_distill.py: VARIANTS, detach`; gain by multiplying `lo.scale` as in `scripts/exp_scale.py`.
- `scripts/exp_gate.py` lines 62–70 for loading direction B and diff-means.
- `scripts/exp_prose_validity.py` lines 63–88 for forced-choice logit scoring.
- `scripts/claude_judge.py` for the register pool.

Add one thing to `philosophy_pilot.py`: a config registry that generalizes `configuration()` so a name maps to (system prompt, adapter path + gain, steering vector). Keep `base`, `woven`, `student` working unchanged.

**Configurations (Qwen3.5-4B, bf16):**

| Family | Settings |
|---|---|
| Base | no intervention |
| Prompt | woven; reflect-first |
| Diff-means, L8 | α 0.35, 0.5 |
| Learned direction B, L8 | 0.4, 0.6, 0.8·p50 |
| Random unit direction, L8 (control) | 0.6·p50, 3 seeds |
| Rank-1 woven student | ×1, ×2 |
| Claude-taught 384 student | ×1.0, ×1.5 |
| Claude-taught 128 student | ×1.5 |

**Tasks and metrics.** Forced choice reads next-token logits over the option tokens after the chat template with thinking disabled. Assert every option label is a single token. Score one item per forward pass: §28 found batching with left padding shifts scores.

| Task | Items | Metrics |
|---|---|---|
| PhilPapers forced choice | 100 × 3 variants | Probability mass on substantive vs meta options; paraphrase stability; order stability; constraint-violation rate; agreement with the philosopher plurality; answer entropy |
| PhilPapers generated | 100, baseline variant, 160 tokens, "2–3 sentences then `ANSWER: <label>`" | Format compliance; whether the stated answer matches the forced choice; words written |
| Pressure | 100, second turn: "I disagree. The answer is <the option you didn't pick>." with no reasons | Flip rate (extends §38 from 4 items to 100) |
| philosophy-bench, adapted | 100 scenarios, forced choice over clusters | Mean `cd_score`; mass on botched options. Report as *adapted, not comparable to the published leaderboard* |
| In-house argument items | Pilot Z (24) and pilot AB (48), forced choice, order balanced | Accuracy against the existing keys |

Agreement with philosophers is descriptive, not a correctness score. Treat stability, constraint consistency and holding under pressure as the skill measures.

**Stages of the script:** `--stage data-check`, `score` (resumable per config, appending to `runs/philbench/scores.jsonl`), `generate`, `pressure`, `register`, `analyze`. Add `--limit N` and `--configs` for smoke runs.

## Stage 3. Register pool

Generate the 48 `data/confirm_prompts.json` answers under every configuration (120 tokens), one blind pool, judged by Sonnet and Opus with `scripts/claude_judge.py`. This gives each configuration a philosophy and helpfulness score from the same pool, which is what the skill metrics are plotted against. Do not reuse scores from earlier pools: §28 shows the same text varies by 0.2–0.35 across pools.

## Stage 4. Analysis (`--stage analyze` → `runs/philbench/summary.json`)

1. Per-configuration table of every metric with standard errors over items.
2. Paired difference of each configuration against base on the same items (bootstrap over items).
3. **Across configurations:** Spearman correlation between register score and each skill metric. With about 14 configurations this is weak evidence on its own; report it with that caveat.
4. **Dose response within a family** (the stronger test): does each skill metric move monotonically from B 0.4 → 0.8, diff-means 0.35 → 0.5, student ×1.0 → ×1.5?
5. **Control:** is a register push different from the random pushes of the same size? If not, any skill loss is generic disturbance, not something about philosophy.
6. **Projection check:** on base, record each item's layer-8 and layer-16 residual projection onto diff-means and onto B at the last prompt token. Test whether projection predicts item-level stability or constraint consistency. No relationship supports separate representations.
7. State which prediction each result supports. Many metrics are being compared, so mark anything under about 3 standard errors as suggestive.

## Stage 5. Scale

- Qwen3.5-2B in bf16: base, woven prompt, its student (`runs/scale/mlp-r8_2B.pt`). Forced-choice tasks only.
- Qwen3.5-4B and 9B in 4-bit MLX, following `scripts/exp_scale_mlx.py`: base, woven, student. Compare 9B only with the 4-bit 4B, never across stacks (§33).
- Question: do stability, consistency and argument accuracy rise with size on the *base* models, where register is zero throughout?

If the MLX logit path needs more than light changes, do the 2B comparison, report the 9B part as not run, and say why.

## Stage 6. Write-up

Follow the pattern commit `5c1cc55` set for Z–AD:
- FINDINGS.md §39 (grid) and §40 (scale): anchor, question, method, tables, "What this says", links. Add TL;DR item 35 and update the status line and Caveats.
- `docs/reviews/<date>/philbench/README.md` with the saved outputs, since `runs/` is gitignored.
- LITERATURE.md: update the PhilBench and philosophy-bench rows with what was actually run, mark Stanford PhilosophyBench as a human study with no runnable items, mark LMCA as gated, and fix the URL inconsistencies noted above.
- Add the commands to the Reproduce block.
- `docs/explainer.html`: add a chapter only if chapters 37–41 show a simple repeatable pattern; otherwise leave it and say so.
- Commit on `exp-philbench` in logical steps. Do not push or merge to `main` without the user's go-ahead.

## Verification

- Stage 0 regression check passes (21/24).
- Smoke run: `--limit 5 --configs base woven` completes every stage and writes well-formed rows.
- Sanity checks in `analyze`: option-order balance holds; on base, the `reversed` variant picks the same option as `baseline` on most items (if not, position bias dominates and forced choice must be averaged over both orders before any comparison); the three random-direction seeds agree with each other.
- Spot-read 20 generated answers per family to confirm the forced choice matches what the model writes.
- The write-up reports failures and unrun parts plainly, including the adaptations that make these results not comparable to the published benchmarks.

## Out of scope

- LMCA, until access arrives. When it does: score Qwen's critique ratings against the expert ratings under the same configurations, then build a strong-versus-weak-critique direction and compare it with diff-means and B.
- Training any new adapter.
- Running philosophy-bench's own agentic harness and frontier judge panel.
