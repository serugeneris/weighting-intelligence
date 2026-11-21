# Pilot: a philosophical-skill measure built from real texts

> Not yet run. ID **AL** confirmed free (AK is the last ID on `main`).
>
> **Status, 2026-11-21.** Run and written up as [exp. AL](../../findings/7-register-versus-skill.md#exp-al). Stage 2 passed: blind Claude graders 151/152 and the author 19/20 on the baseline packet. The main measure became a single-passage validity task (`verdict`) after the paired format measured letter preference. The Stage 1 item review is still open.
>
> **Status, 2026-11-20.** Stage 0 passed: the AI scorer reproduces 83/96 on 4B base validity (all 96 picks identical), and the rank-1 (D2) and Claude-taught 384 (P) adapters, 2B bf16 and 4B/9B 4-bit MLX all load. Stage 1 is drafted and awaits the author's check: 22 corrupted arguments (132 items) and 20 objection-reply pairs from four exchanges (80 items), loaders in `scripts/philbench_items.py`, tests in `tests/test_real_text_items.py`, run scripts `scripts/exp_real_text.py` and `scripts/exp_real_text_mlx.py` (smoke-tested only). Sources changed at the author's request: Mill dropped; Anselm with Gaunilo (medieval), Vitoria (Spanish) and Galileo (Italian) added. Review sheet and blind baseline packet: [`docs/reviews/real-text-skill/`](../reviews/real-text-skill/README.md).

Executor: Claude, on the M5 Pro checkout that has `runs/`, `.venv`, `.venv-mlx` and the cached Qwen3.5 weights (2B, 4B, 9B 4-bit MLX) and the trained adapters from D, P and O. A checkout without these cannot run stages 2–3.

## Context

[Exp. AI](../../findings/7-register-versus-skill.md#exp-ai) showed that every lever that raises the reflective register leaves judge-free skill at base level, and [AI2](../../findings/7-register-versus-skill.md#exp-ai2) that skill grows with model size. The weak point is the skill measure itself:

- The validity, objection and revision items were **written by the project**, and the revision and transfer cases (workshops, donor lists, rosters) are closer to practical ethics than to philosophy.
- The free-text labels depend on a rubric: the human grader in [AK](../../findings/6-can-we-trust-the-scores.md#exp-ak) disagreed with both models on "asks a targeted question" (κ 0.32–0.39).
- The PhilPapers measures in AI are consistency (same answer under paraphrase and reordering), which is necessary for skill but is not skill.

This pilot builds two tasks whose **content comes from real philosophical texts** and whose **keys come from the text or from how the item was built**, so no grader is needed. It reuses AI's forced-choice scorer unchanged.

## Predictions (write these into the script docstring before any run)

- **P1, does the measure measure skill.** Accuracy rises 2B → 4B → 9B, and the human graders score at or above the 9B. **If P1 fails, stop and report**: the tasks measure something other than skill, and P2 cannot be read.
- **P2, the main claim.** Prompts and trained students score like 4B base; constant pushes (diff-means, B, random) score lower.
- **P3, memorization.** Accuracy is the same whether the correct option is the original passage or its paraphrase, and the no-key original-vs-paraphrase pairs sit near 50/50. If the original wins, the model is using familiarity with the text.

## Sources (pilot: public domain only)

| Source | Edition | Used for |
|---|---|---|
| Descartes, *Meditations* with *Objections and Replies* (Hobbes' Third, Arnauld's Fourth) | Haldane & Ross, 1911 | Task 2, task 1 |
| Leibniz–Clarke correspondence (Clarke answers by numbered paragraph) | A pre-1928 English edition | Task 2 |
| Hume, *Dialogues Concerning Natural Religion* | Original English | Task 1 |
| Mill, *Utilitarianism* | Original English | Task 1 |
| Berkeley, *Three Dialogues between Hylas and Philonous* | Original English | Task 1, task 2 |

Store each source text under `data/sources/` with its edition and URL in a header. Record, for every item, the source, edition and location (book, section, paragraph).

**Phase 2 (not in this pilot):** papers posted after the models' release, from open-access sources, licences checked; publish item IDs and locations rather than text.

## Task 1: corrupted arguments (`corrupt`, ~30 passages)

From each passage take a short argument (3–6 sentences) whose conclusion follows from what comes before. Make two versions:

- **Corrupted:** change exactly one sentence so the conclusion no longer follows. Use a mix of types and record the type per item: quantifier swap (all ↔ some), dropping a premise the conclusion needs, reversing a conditional (affirming the consequent), negation flip, conclusion that claims more than the premises support.
- **Paraphrase:** same meaning, different wording, same length within ±15%.

Three pair types, each asked as "In which passage does the conclusion follow from what comes before?":

| Pair | Key | Measures |
|---|---|---|
| Original vs corrupted | Original | Raw accuracy |
| Paraphrase vs corrupted | Paraphrase | Accuracy when familiarity cannot help (both sides edited) |
| Original vs paraphrase | None (~50/50 expected) | Preference for the memorized text |

About 90 pairs, each in both A/B orders. The paraphrase-vs-corrupted pair also controls for the Claude-taught student recognising Claude's editing style, since Claude edited both sides.

## Task 2: objection–reply matching (`objreply`, ~40 items)

- **Objection → reply:** one objection, four replies: the real one and three other replies **from the same exchange** (same author, period and style), so only content separates them. Prefer distractors about the same meditation or section, to avoid matching by topic.
- **Reply → objection:** the same in reverse.
- **Memorization control:** a paraphrased version of each item (objection and all replies paraphrased).

About 80 items, the key's position rotated across A–D.

## Stage 0. Preflight (stop and report if any check fails)

1. `main` is checked out and up to date; the next free experiment ID is confirmed.
2. AI's scorer still reproduces: rerun `scripts/exp_philbench.py` stage `score` on 4B base for the `validity` task and match AI's 83/96.
3. The adapters for the rank-1 student (D2), the Claude-taught 384 student (P) load, and the 2B, 4B and 9B 4-bit MLX models load.

## Stage 1. Build the items

1. Extract candidate passages and exchanges from the sources. Claude drafts corruptions and paraphrases.
2. **The author checks every item**: the original argument's conclusion does follow, the corruption breaks exactly that, the paraphrase keeps the meaning, and task 2's real reply is unambiguous.
3. Save as `data/pilot-real-corruption.json` and `data/pilot-real-objreply.json`, each item with source, edition, location, corruption type (task 1) and key.
4. Add `corrupt` and `objreply` loaders to the `TASKS` registry in [`scripts/philbench_items.py`](../../scripts/philbench_items.py), producing items in the same shape as `validity` and `critique`.
5. Smoke test in `tests/` that loads both files and checks keys, both orders and rotated positions, with no model download.

## Stage 2. Human baseline

The author and one philosophy colleague answer a 20-item subset (10 per task, no configuration outputs shown). Report agreement with the keys per grader. **Gate:** at least 85% agreement; otherwise fix or drop the disputed items and say so.

## Stage 3. Runs

Same scoring as AI: `forced_choice` with the `Answer:` prefill, the bold path, one item per forward pass, no padding, both orders.

| Group | Configurations |
|---|---|
| Scale ladder (P1) | Base 2B (bf16), base 4B (bf16), base 4B (4-bit MLX), base 9B (4-bit MLX) |
| Register grid (P2), 4B bf16 | Woven prompt; rank-1 student ×1; Claude-taught 384 ×1.0; diff-means α 0.5; B 0.6·p50; random 0.6·p50 (seed 0) |

As in AI2, compare the 9B only with the 4-bit 4B.

## Stage 4. Analysis and report

Per configuration and task: accuracy per pair type, preference for the original in the no-key pairs, accuracy per corruption type, and paired z against base (as in AI2).

**The pilot works if:**

- Human agreement with the keys ≥ 85%.
- The 2B → 9B gap is about 15 points or more, z > 2 (with the 4-bit stack caveat).
- Fewer than a third of items are at ceiling or floor for every model.

Write up in `findings/7-register-versus-skill.md` with the usual question/result block, and a review folder under `docs/reviews/<date>/real-text-skill/` with saved outputs. Update FINDINGS.md (index row, verdict table, caveats).

## Risks

- **Memorized classics dominate** (original ≫ paraphrase): bring phase 2's recent papers forward.
- **4B at ceiling on task 2**, because same-exchange distractors still differ by topic: choose distractors from the same section.
- **Corruptions too easy** (an obvious "not"): mix types and report per type.
- **Claude wrote the edits:** the paraphrase-vs-corrupted pairs control for style detection; report them separately.

## What comes after

If P1 holds, this is the skill measure for the next step: train an adapter to move it (for example, distilled from the 9B on these task formats, or on strong versus weak critiques), then read the direction that adapter adds against base, and compare it with direction B. Skill and register directions coming out near-orthogonal would be the mechanism behind "register is not skill"; no adapter of this size moving skill would support "skill comes with the base model".
