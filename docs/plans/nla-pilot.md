# Pilot AM: reading the 9B's layer-21 state with a natural-language autoencoder

> Not yet run. ID **AM** confirmed free (AL is the last ID on `main`, 7476237).
>
> **Status, 2026-11-23 (later).** Stage 0 passed. Round-trip FVE **0.799** (95% CI 0.779–0.816) on 128 held-out rows with the 4-bit AV and bf16 AR in MLX (card: 0.786). The smoke test on our chat format parses and stays in English, and the read positions need no change. Stages 1–2 not started. Details in [Stage 0 results](#stage-0-results-2026-11-23).
>
> **Status, 2026-11-23.** Plan written.

Executor: Claude, on the M5 Pro checkout with `.venv-mlx`, the cached `Qwen/Qwen3.5-9B` (bf16) and `mlx-community/Qwen3.5-9B-4bit`, the AL items (`data/pilot-real-corruption*.json`) and the 48 confirmation prompts from exp. I (`data/confirm_prompts.json`; `data/heldout_prompts.json` has only 8).

## Context

### The method

Anthropic's [natural language autoencoders](https://transformer-circuits.pub/2026/nla/index.html) (NLAs; Transformer Circuits, 2026-05-07; code at [kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders)) turn one residual-stream activation into text and check the text by turning it back:

- **Activation verbalizer (AV):** a copy of the target model. It gets a fixed prompt with a marker token, the activation is injected at that marker, and it writes an explanation.
- **Activation reconstructor (AR):** the target model cut after the read layer, plus an affine head. It reads the explanation and predicts the activation.
- **Training:** an SFT warm start on Claude-written summaries of text prefixes, then GRPO on the AV with reward −‖h − AR(z)‖², with the AR regressed on the same rollouts. The paper reports 0.6–0.8 FVE after RL.
- **Limitations the paper names:** confabulation, where specifics can be false but on-theme; "excessive expressivity", meaning the AV is a full LM and can infer beyond the activation; a few hundred tokens per activation; and a degenerate solution where the AV copies the context.
- **Case studies** include evaluation awareness, user traits inferable from context and the Opus 4.6 language-switching bug. They also include a causal test: edit an explanation, pass both versions through the AR, and steer with the difference. It worked about 50% of the time.

### Which of our models have an NLA

Searched on 2026-11-23: the Hugging Face API for `nla`, `nla-qwen`, `qwen3.5 nla` and similar, plus the released collection [`kitft/nla-models`](https://huggingface.co/collections/kitft/nla-models).

| Model | Official (Anthropic/kitft) | Community | Usable here? |
|---|---|---|---|
| Qwen3.5-2B | none | `adamjb111/nld27-*-qwen35-2b-from-nla-qwen36-27b-*` are 2B students that read **Qwen3.6-27B** layer-42 activations, not the 2B's own | **No** |
| Qwen3.5-4B | none | [`Solshine/qwen3.5-4b-nla-L23-av-priordev-relabel-v2`](https://huggingface.co/Solshine/qwen3.5-4b-nla-L23-av-priordev-relabel-v2): AV only, and its card says "This adapter does not produce usable descriptions" (exact-document top-1 3/294, chance) | **No** |
| Qwen3.5-9B | none | [`therem/qwen3.5-9b-nla-L21-{av,ar}-{sft,rl}`](https://huggingface.co/therem/qwen3.5-9b-nla-L21-av-rl): full AV + AR pair, layer 21 of 32, SFT then GRPO (400 steps). Round-trip FVE **0.786** with the co-trained AR and 0.699 with the frozen SFT AR, on 512 held-out rows | **Yes, with caveats** |

The official releases cover Llama-3.3-70B, Gemma-3-27B, Gemma-3-12B and Qwen2.5-7B only. None of them is a model we test.

**Caveats on the 9B pair. Each one shapes the design below.**

1. **It is a community port.** It was trained with EasyNLA's Qwen3.5 port, not Anthropic's code, and is not peer-reviewed. Its card gives no licence; the base model is Apache-2.0.
2. **Its own card warns that FVE is mostly verbatim tail recovery.** Stripping quoted spans from gold explanations drops AR FVE from 0.752 to −0.04, and quoting the raw last 80 characters gets 0.58. **Every measure below therefore has a "tail-only" control and a "quotes stripped" condition.**
3. **Training distribution.** It was trained on pretraining-style document prefixes, last token, from `ceselder/easynla-dsv4-warmstart-opus5`. Our inputs are chat transcripts, which is out of distribution.
4. **Injection.** The vector is added, norm-matched, at the output of decoder layer 1 at marker `㈜` (id 158983, neighbours 29/510). If the injection is silently skipped, the AV writes Chinese, so check every output for CJK.
5. **Size.** AV merged bf16 is 17.9 GB and AR (22 blocks + head) 11.6 GB. Neither AV nor target fits in PyTorch on this 24 GB Mac (memory: 9B bf16 pages). Run both in MLX: AV quantized to 4-bit (about 5 GB), AR in bf16 (11.6 GB), one at a time. Target activations come from the cached bf16 9B **cut after block 21** (about 11 GB), not from the 4-bit model the NLA never saw.

### Why this matters for the project

Part 7 found that **register is not skill.** Raising the reflective register leaves judge-free skill at base level (AI, AL), and skill grows with size. The 9B is our most skilled model: AL corrupt-verdict about 83% against 82% for 4B. Every skill measure so far reads the model's **output**. An NLA lets us ask what the model represents **before** it answers:

- When it misjudges a corrupted argument, has it represented the flaw and failed to report it (a readout failure)? Or is the flaw absent at layer 21 (a representation failure)?
- Does a constant push replace the representation of the user's task, as the 0.6 reading "reflect instead of answering" suggests? Does the woven prompt keep it?

## Questions and predictions (copy into the script docstring before any run)

- **Q1, validity in the state (main).** At the pre-answer position of an AL single-passage `verdict` item, a blind judge that reads **only the NLA explanation** predicts the model's own verdict above chance.
  - **P1a.** Agreement with the model's verdict exceeds the **tail-only** control: the judge sees the last 80 characters of the context in place of the explanation. It also exceeds the **shuffled-activation** control: the explanation of another item's activation.
  - **P1b, faithfulness to the model rather than the task.** On the items the 9B gets **wrong**, explanation-predicted verdicts agree more often with the model's (wrong) verdict than with the key. If they agree with the key, the AV is doing the task itself from quoted text ("excessive expressivity"), and Q1 cannot be read as the model's belief.
  - **P1c.** A logistic probe on the same activations (item-disjoint 5-fold CV, predicting the model's verdict) sets the ceiling for what is linearly present. The NLA is expected to sit below the probe. It is interesting only where it adds a *reason*, such as naming the swapped quantifier.
- **Q2, does a push erase the task.** At the first assistant position of the 48 confirmation prompts (`data/confirm_prompts.json`), under base, woven prompt and a 9B diff-means push:
  - **P2a.** A blind judge identifies which of 4 user requests the explanation came from (the true one and 3 random distractors). It does so at the same rate for base and woven prompt, and less often under the push.
  - **P2b.** "Reflective / philosophical" content appears more often under prompt and push than base. This measures the NLA's sensitivity, not a finding.
- **Q3 (optional, run only if Q2's P2b holds).** **NLA-derived direction.** Write paired explanations (plain plan vs "answer, then connect it to a philosophical idea"), map both through the AR and use the mean difference as a steering vector at layer 21. Compare it with diff-means at layer 21 on the philosophy + helpfulness frontier (16 eval prompts, Sonnet judge, same rubric). Prior from E: in the 4B only layers 8–16 respond to modest pushes, and layer 21 of 32 may be too late for *any* direction. The layer-21 diff-means arm is the control for that.

**Stop rules.** If Stage 0 fails, stop and report. If P1a fails (no gain over tail-only), report Q1 as "the NLA does not read validity from chat-formatted items at layer 21" and do not interpret P1b.

## Stages

### Stage 0: does the NLA work on this machine (gate)

1. **Ask the author, then download** `therem/qwen3.5-9b-nla-L21-av-rl` (17.9 GB) and `therem/qwen3.5-9b-nla-L21-ar-rl` (11.6 GB) into the HF cache. Record the commit hashes.
2. **Convert the AV** with `mlx_lm.convert -q --q-bits 4`. If the round trip in step 4 falls short, retry at 8-bit (about 10 GB).
3. **Write `wi/nla.py`:**
   - `extract(texts, layer=21)`: load bf16 Qwen3.5-9B with only blocks 0–21 in MLX, and return the block-21 output at the chosen token.
   - `verbalize(vecs)`: wrap AV decoder layer 1 so that `h[p] += ‖h[p]‖ · v/‖v‖` at the marker, found by id and neighbours. Use the `nla_meta.yaml` actor prompt as a user chat message, and sample at T=1 and greedy, max 256 tokens.
   - `reconstruct(texts)`: use the critic prompt, take the last-token state, normalize to ‖·‖ = 64 and apply `value_head`.
   - Read every template and id from `nla_meta.yaml`, not from code.
4. **Reproduce the card.** Re-extract layer-21 activations for 128 rows of the SFT held-out split, `test.parquet` of the dataset, which is document-disjoint. Then run AV (greedy) → AR. **Pass: FVE ≥ 0.70 with the RL AR** (card: 0.786 on 512 rows; this allows for 4-bit AV and MLX numerics). Also report:
   - no-injection FVE, expected near the tail-only 0.58;
   - CJK rate, expected ≤ 1/128.
5. **Smoke on our format.** Run 8 AL items and 8 held-out prompts and read every explanation. Check that they parse as `<explanation>` and are not Chinese, and note how much is verbatim chat template (`<|im_start|>assistant`). If the AV mostly narrates the chat template, move the read position one token back and re-check.
6. **Tests:** `tests/test_nla.py` covers marker location, the injection hook firing exactly once per sequence, and the AR output shape. Keep tests model-free, as the existing ones are.

### Stage 0 results (2026-11-23)

Code: [`wi/nla.py`](../../wi/nla.py) (MLX port of EasyNLA's conventions), [`scripts/exp_nla.py`](../../scripts/exp_nla.py) (phases `convert`, `acts`, `explain`, `score`, `smoke-acts`, `smoke-explain`), [`tests/test_nla.py`](../../tests/test_nla.py). Outputs in `runs/nla/`.

**Checkpoints:** `therem/qwen3.5-9b-nla-L21-av-rl` at commit `c277b9e0`, `-ar-rl` at `3914c005`. Held-out rows: the first 128 of `corpus/av_val.parquet` from `ceselder/easynla-dsv4-warmstart-opus5` at commit `d934ad27`.

**Port checks:**
- Target. MLX block-21 outputs match a PyTorch forward of the same 22 blocks on 4 rows (80–2,130 tokens): cosine 0.9998–1.0000, same norms.
- AV. The 4-bit conversion gives 4.50 bits per weight and 4.7 GB, at about 45 tok/s (3.6 s per explanation).
- Injection. The hook fires exactly once per prompt with injection and never without (160/160).
- **Prompt fix.** The AV prompt must use the chat template with **thinking off**. After the default `<think>\n` generation prompt, the AV gives the opening `<` log-prob −16.6 and writes untagged prose. The checkpoint's reference generations all open with `<explanation>`.

**Round trip (128 rows, greedy AV):**

| Condition | FVE |
|---|---|
| AV → AR | **0.799** [0.779, 0.816] |
| AV → AR, spans of ≥5 tokens copied from the source removed | 0.515 |
| Claude-written training explanation → AR | 0.700 |
| Last 80 characters of the source → AR | 0.236 |
| AV without injection → AR (32 rows; same rows with injection: 0.761) | −0.795 |
| AV explanations shuffled across rows → AR | −0.951 |

Other measures:
- **Quoting.** Only 1.9% of explanation tokens are copied from the source, yet removing those spans costs 0.28 FVE.
- **Tail baseline.** At 0.236, the tail alone is far below the card's 0.58. The card measured that baseline with the SFT AR, which reads raw text better.
- **Format.** Mean length 176 tokens; 0/128 truncated, 0/128 unparsed.
- **Chinese words.** 5/128 explanations slip a few Chinese words into English text (card: 1/128, from a Chinese source). The hook fired in all five, so this is likely a 4-bit cost, not a skipped injection.

**Smoke test (24 activations, every explanation read):**
- **Held-out prompts (assistant position).** The explanations name the task's domain and the planned register: "That's a great question!", "It depends—", a numbered plan. **Specifics drift:** bicycle gears became an internal-combustion engine, weather vs climate became "mass vs climate", and the 15% tip on $48 became "12 ÷ 3.5". For P2a, 4-way matching against random distractors looks feasible. Same-domain distractors would fail.
- **AL verdict, passage position.** The explanations recover the topic and form (a syllogism, motion through a vacuum, Hume's design argument) with invented wording. Both intact and corrupted versions get called "erroneous", so there is no visible discrimination by eye.
- **AL verdict, pre-answer position.** The explanations are generic and template-like ("Chinese/English logic exam … Mendelian … ozone"), the passage content is lost, but each states an expected letter. On the 8 prompts the explanation's letter matched the 9B's own argmax (from `runs/real_text/scores.jsonl`) on 6 of the 7 non-tied prompts, but the key on only 4 of 8. The 9B leans B on most of these items, and so do the explanations. This is the P1b pattern, on far too few items to read. Note that a constant-B judge would score the same here; Stage 1 must compare against a majority-letter baseline.

**Decisions for Stage 1:**
- Keep both read positions.
- Add a **majority-letter baseline** next to tail-only and shuffled.
- Treat the 5/128 Chinese-word rate as a 4-bit cost. Retry the AV at 8-bit only if Stage 1's real-vs-shuffled gap is small.

### Stage 1: Q1, validity at the pre-answer position

- Items: the 87 AL `verdict` units, intact and corrupted versions each (174 prompts), formatted exactly as in `scripts/exp_real_text_mlx.py`, with the `Answer:` prefill.
- Positions: (a) the last passage token and (b) the last token of the prefill. (b) is the main measure; (a) asks whether the flaw is there before the question is asked.
- Per prompt: 1 greedy + 3 sampled explanations (T=1). The paper uses claims that recur across samples and adjacent tokens as a confabulation filter, so a claim counts only if it appears in ≥2 of 4 samples.
- Controls per prompt: tail-only text (last 80 characters); the shuffled activation of a random other item; and the explanation with quoted spans (≥5-token overlap with the context) removed. Also a majority-letter baseline (the judge always says the model's most common verdict), since the 9B leans B on many items (Stage 0 smoke test).
- Judge: blind `claude -p` (Sonnet). It sees one explanation and the question "Does this description suggest the passage's conclusion follows from its premises? Answer valid / invalid / no signal." It never sees the passage.
- The model's own verdict comes from AL's existing 9B-4bit scores (`runs/real_text/scores.jsonl`). **Re-score the bf16 truncated-plus-head path on these items first**: if bf16 and 4-bit disagree on more than 5% of items, use the bf16 verdicts.
- Probe (P1c): logistic regression on the 4096-d activations, item-disjoint CV, with the label being the model's verdict and, separately, the key.
- Cost: 174 × 2 positions × 4 samples ≈ 1,400 explanations. At about 180 tokens each on the 4-bit AV, that is roughly 3–4 h. Run it in the background, alone, never alongside another model job.

### Stage 1 results (2026-11-23)

**Setup as run.** 218 AL verdict prompts: the 22 originals and 87 corrupted units, both answer orders, with paraphrases excluded. Three prompts where the 9B's two letters tie are dropped, leaving n = 215. The 9B's verdict is the argmax of the 4-bit base model's letter probabilities in `runs/real_text/scores.jsonl`, and it is correct on 56% of these prompts. Each activation got 4 explanations (1 greedy, 3 at T=1), 1,744 in all: 0 unparsed, and the hook fired once on each. The blind Sonnet judge read 3,924 items. Script: `scripts/exp_nla_validity.py`. Results: `runs/nla/s1_results.json`.

**Agreement with the 9B's verdict** (all 215 prompts; "unclear" counts as a miss; 95% bootstrap CI):

| Readout | Pre-answer (letter) | Passage end (follows / not) |
|---|---|---|
| Linear probe, 5-fold CV grouped by argument (P1c ceiling) | 0.833 | 0.777 |
| Explanations, majority of 4 | **0.623** [0.558, 0.688] | **0.674** [0.609, 0.735] |
| Explanations, greedy only | 0.651 | 0.493 (74 unclear) |
| Explanations with quoted spans removed | 0.628 | 0.647 |
| Explanation of another prompt's activation (shuffled) | 0.479 | 0.409 |
| Always the 9B's most common verdict | 0.535 (B) | 0.544 (follows) |
| Last 80 characters of the context | 0.000 (always unclear) | 0.079 (195 unclear) |

**Agreement with the key** on decided items: 0.475 at the pre-answer position and 0.516 at the passage end, which is chance.

**On the 94 prompts where the 9B is wrong** (decided items, explanation agrees with the model : with the key):

| Readout | Pre-answer | Passage end |
|---|---|---|
| Explanations | **63 : 25** | **66 : 20** |
| Shuffled | 43 : 45 | 44 : 44 |

**Reading against the predictions:**
- **P1a passes at both positions.** Explanations beat the tail, shuffled and majority controls. At the pre-answer position the margin over the majority is small: the CI's lower end, 0.558, sits just above 0.535. Removing quoted spans changes almost nothing, so the signal does not come from copied text.
- **P1b passes.** When the 9B is wrong, the explanations side with its wrong verdict about 2.5 to 3 times as often as with the key. Shuffled explanations keep the same overall lean toward B and toward "follows", yet split 44:44, so the gap is not that lean. The explanations read the model's verdict, not the truth.
- **P1c: the NLA sits well below the probe**, at 0.62–0.67 against 0.78–0.83. It is also below the probe on decided items (0.66 / 0.76). **No reasons were found.** In sampled passage-end explanations of corrupted items, the passage's form and topic are reconstructed with invented quotes. None of them names the swapped quantifier or the broken step; the verdict comes across only as a lean.
- **The verdict is present before the question.** At the passage end, both the probe and the explanations recover "will the model say follows" well above the majority baseline. Neither recovers the truth: the key-follows probe scores 0.789, against 0.798 for always answering "does not follow".

**Caveats:**
- Activations come from the bf16 target, while the verdicts come from 4-bit runs. Cosine between the two is 0.968 on average (min 0.869), so a few verdicts may not match the state being read.
- The original items repeat in both orders and across their corruptions, so the 215 prompts are not independent. The CIs above resample prompts, not arguments, and are too narrow. The probe CV is grouped by argument.
- One judge model, one prompt; no human check of the judge's labels yet.

### Stage 2: Q2, task retention under register levers

- Prompts: the 48 confirmation prompts (`data/confirm_prompts.json`), at the first assistant position (after `<|im_start|>assistant\n`).
- Configurations, all on bf16 9B blocks 0–21:
  - base;
  - woven system prompt (the text from B);
  - diff-means push. Rebuild it at 9B from `data/contrast_pairs.json` at the layer matching 4B layer 8 (9B layer 8, same depth fraction) and apply it at α = 0.5, with α defined as in `scripts/sweep.py`: α × the median residual norm at that layer × the unit direction.
  - Also a random direction of equal norm as a push control.
- Samples and controls as in Stage 1.
- Judges: (i) 4-way request matching (P2a); (ii) "does this mention reflection, philosophy, meaning, or abstract ideas?" (P2b).
- Cost: 48 × 4 configurations × 4 samples ≈ 770 explanations, about 2 h.

### Stage 3 (optional): Q3, steering with an AR-derived direction

Only if P2b holds. Write 16 paired explanations from the style of the Stage 2 base explanations, not invented from scratch, so they stay in the AR's distribution. Steer the 4-bit 9B at layer 21 with the AR difference and with layer-21 diff-means at matched norms (α 0.25, 0.5, 1 in the `sweep.py` sense). Generate on the 16 eval prompts at 120 tokens, cut at `<|im_end|>`, and judge with `scripts/claude_judge.py`.

## Outputs

- `scripts/exp_nla.py` (`--stage 0|1|2|3`) → `runs/nla/` (`stage0.json`, `explanations.jsonl`, `judge.jsonl`, `probe.json`).
- Write-up as exp. AM in `findings/7-register-versus-skill.md` (Q1 and Q2), plus the usual FINDINGS row, README row, explainer chapter and TOC entry.
- `LITERATURE.md`: add the NLA paper and the therem checkpoints.

## Risks

- **The AV reads the tail, not the state.** This is the main risk, and the tail-only and quote-stripped controls are there for it. If the controls equal the real condition, that is the result.
- **Chat format is out of distribution.** The Stage 0 smoke test catches gross failure. A subtle degradation would appear as P1a failing.
- **The judge infers validity from quoted premises.** It never sees the passage, but quotes can carry the argument. Quote stripping is the check.
- **Memory.** Never hold AV and target at once. The 4-bit AV is about 5 GB and the bf16 blocks 0–21 about 11 GB; load them sequentially and cache activations to disk (`runs/nla/acts_*.npy`).

## Out of scope: NLAs for 2B and 4B

No usable NLA exists for either. Training one means an SFT warm start on the same Opus-written corpus (therem: 300k rows, about 13 h on one A800 at 9B) plus GRPO (about 17 h on 4×A100), which does not fit this Mac. If Stage 1 shows a signal at 9B, a rented-GPU run of EasyNLA's Qwen3.5 port on the 4B would let us compare the base 4B with the Claude-taught student **in their own activations**. The 9B students from V were not saved, so student-versus-base comparisons are not possible in this pilot.
