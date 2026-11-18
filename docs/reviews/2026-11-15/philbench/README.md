# Experiment AI · Is the reflective register philosophical skill?

15 November 2026

[Findings, exp. AI](../../../../findings/7-register-versus-skill.md#philbench-grid) · [Findings, exp. AI2](../../../../findings/7-register-versus-skill.md#philbench-scale) · [Interactive explainer](../../../explainer.html#c46) · [Plan](../../../plans/experiment-ae-philbench.md)

The plan was written as "Experiment AE". By the time it ran, `main` already used AE–AH, so it was renamed Experiment AI (write-ups AI and AI2).

## Reproduce

Needs the cached `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-2B` checkpoints, `mlx-community/Qwen3.5-{4B,9B}-4bit`, and these saved artifacts: `runs/p50.json`, `runs/diffmeans.pt`, `runs/distill/down8-r1.pt`, `runs/more_teacher/mlp-r8_claude384.pt`, `runs/claude_teacher/mlp-r8_claude.pt` and `runs/scale/mlp-r8_2B.pt`. Run from the repository root.

```sh
python3 scripts/build_philpapers.py data/sources/philpapers2020_results_all.json
git clone https://github.com/benedictbrady/philosophy-bench external/philosophy-bench
git -C external/philosophy-bench checkout 012a5913b272094dcaf76bec17e544b16d3ea5d5
.venv/bin/python -m scripts.exp_philbench --stage data-check score pressure register generate
.venv/bin/python scripts/claude_judge.py --input philbench/register_gen.jsonl --output philbench/register_judge.jsonl
.venv/bin/python scripts/claude_judge.py --model opus --input philbench/register_gen.jsonl --output philbench/register_judge_opus.jsonl
.venv/bin/python -m scripts.exp_philbench --model 2B --stage score pressure
.venv-mlx/bin/python scripts/exp_philbench_mlx.py --model 4B
.venv-mlx/bin/python scripts/exp_philbench_mlx.py --model 9B
.venv/bin/python -m scripts.exp_philbench --stage analyze
```

Run one model at a time on a 24 GB machine. Running the scoring pass alongside generation pushed the machine into 16 GB of swap and stalled generation for about an hour. The rerun of the generate stage is resumable per configuration.

## How the data was built

- **PhilPapers 2020 Survey.** The default view of the results page (all respondents, all regions, skipped and insufficiently familiar answers excluded), read on 2026-11-15. The page sits behind a Cloudflare browser check that a normal browser clears without a CAPTCHA, so the table was read in a browser and saved as `data/sources/philpapers2020_results_all.json`. The two accept/reject grids ("Philosophical methods", "Other minds") are kept in the data but left out of forced choice, leaving 98 items. The five meta-options are the survey's own non-substantive answers.
- **Paraphrases and constraints** are author-written. Each paraphrase keeps its question's meaning and options. Nine of the 37 constraints have a live defended position on the other side and are marked contestable; the results are reported with and without them.
- **philosophy-bench** is cloned at a pinned commit and not vendored. Each scenario becomes text: the operator instructions, the environment state as YAML and the first user turn, followed by its authored response clusters as lettered options, rotated by item index.

## Measurement notes

- **Prefill.** The first scoring pass read the first token of the assistant turn. Prose-first configurations put almost none of that token's mass on the letters (median: reflect-first 0.5%, diff-means 0.5 39%, B 0.8 39%, Claude-taught 384 ×1.5 47%), so their renormalized letter distribution was noise. All reported scores prefill `Answer:` and add the bold path, P(" A") + P(" **") · P("A" | "Answer: **"). The first pass is kept in `scores_noprefill.jsonl.gz`. Its `letter_mass` measures how often a configuration answers at once.
- **Position bias.** On base, the reversed variant gives the same option as the original on 64/98 questions. Constraint and plurality metrics therefore average each question's probabilities over both orders. The argument items are scored in both A/B orders; `summary.json → sanity` lists accuracy by order.
- **Pressure.** The flip rate is at its ceiling everywhere (98/98). The drop in p(own answer) depends on starting confidence, so the reported measure is the mass on the user's option after the disagreement.
- **Generated answers.** The stored `match` field in `generate.jsonl` was computed against the first, unprefilled scoring. `summary.json` recomputes the match against the prefilled forced choice and splits answers that ended naturally from those that hit the 160-token cap.

## Review

24 generated answers were read in full across the families: one question under eight configurations, plus spot checks. The stated `ANSWER:` letter was tallied for all 1,568. That is fewer than the plan's 20 answers per family. The answers read as coherent and on topic. B 0.8 usually ran out of tokens while weighing options in prose. Written answers chose meta options ("a combination of views") much more often than the forced choice did.

## Files

- [summary.md](summary.md), [summary.json](summary.json): every metric with standard errors, paired differences against base, Spearman correlations against the register, dose response, random control, projection check, scale and sanity checks
- [scores.jsonl.gz](scores.jsonl.gz): prefilled forced-choice probabilities for every configuration and item (4B grid, 2B, 4-bit 4B and 9B), with base layer-8/16 projections
- [scores_noprefill.jsonl.gz](scores_noprefill.jsonl.gz): the first, unprefilled pass (4B grid)
- [pressure.jsonl](pressure.jsonl): second-turn probabilities after bare disagreement
- [generate.jsonl](generate.jsonl): 160-token answers with stop records
- [register_gen.jsonl](register_gen.jsonl), [register_judge.jsonl](register_judge.jsonl), [register_judge_opus.jsonl](register_judge_opus.jsonl): the register pool and its Sonnet and Opus scores

## Not run

- The 4-bit MLX students of [exp. V](../../../../findings/5-what-limits-the-student.md#exp-v) were trained in memory and never saved, so [exp. AI2](../../../../findings/7-register-versus-skill.md#exp-ai2) has base and woven only for 4B and 9B. Retraining was out of scope.
- LMCA: gated behind a request form.
- philosophy-bench's own agentic harness and frontier judge panel.
