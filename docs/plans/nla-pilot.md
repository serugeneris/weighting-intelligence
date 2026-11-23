# Pilot AM: reading the 9B's layer-21 state with a natural-language autoencoder

> Not yet run. ID **AM** confirmed free (AL is the last ID on `main`, 7476237).
>
> **Status, 2026-11-23.** Plan only. Nothing downloaded, no code written. Stage 0 needs ~29.5 GB of downloads (below); ask the author before starting it.

Executor: Claude, on the M5 Pro checkout with `.venv-mlx`, the cached `Qwen/Qwen3.5-9B` (bf16) and `mlx-community/Qwen3.5-9B-4bit`, the AL items (`data/pilot-real-corruption*.json`) and the 48 held-out prompts (`data/heldout_prompts.json`).

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
- **Q2, does a push erase the task.** At the first assistant position of the 48 held-out prompts, under base, woven prompt and a 9B diff-means push:
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

### Stage 1: Q1, validity at the pre-answer position

- Items: the 87 AL `verdict` units, intact and corrupted versions each (174 prompts), formatted exactly as in `scripts/exp_real_text_mlx.py`, with the `Answer:` prefill.
- Positions: (a) the last passage token and (b) the last token of the prefill. (b) is the main measure; (a) asks whether the flaw is there before the question is asked.
- Per prompt: 1 greedy + 3 sampled explanations (T=1). The paper uses claims that recur across samples and adjacent tokens as a confabulation filter, so a claim counts only if it appears in ≥2 of 4 samples.
- Controls per prompt: tail-only text (last 80 characters); the shuffled activation of a random other item; and the explanation with quoted spans (≥5-token overlap with the context) removed.
- Judge: blind `claude -p` (Sonnet). It sees one explanation and the question "Does this description suggest the passage's conclusion follows from its premises? Answer valid / invalid / no signal." It never sees the passage.
- The model's own verdict comes from AL's existing 9B-4bit scores (`runs/real_text/scores.jsonl`). **Re-score the bf16 truncated-plus-head path on these items first**: if bf16 and 4-bit disagree on more than 5% of items, use the bf16 verdicts.
- Probe (P1c): logistic regression on the 4096-d activations, item-disjoint CV, with the label being the model's verdict and, separately, the key.
- Cost: 174 × 2 positions × 4 samples ≈ 1,400 explanations. At about 180 tokens each on the 4-bit AV, that is roughly 3–4 h. Run it in the background, alone, never alongside another model job.

### Stage 2: Q2, task retention under register levers

- Prompts: the 48 held-out prompts, at the first assistant position (after `<|im_start|>assistant\n`).
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
