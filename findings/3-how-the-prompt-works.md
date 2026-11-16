# Part 3 · How does a prompt do what steering cannot?

[← Findings overview](../FINDINGS.md) · [Explainer, part 3](https://www.contrasentido.ar/weighting-intelligence/explainer.html#p3)

**Short answer.** It is not a decision stored at the end of the prompt. The answer keeps reading the system prompt while it is written, through attention and the DeltaNet state alike, and what flows depends on the token. An average vector cannot reproduce that.

| ID | Question | Result |
|---|---|---|
| [F](#exp-f) | Does the prompt act through a decision made at its last positions? | No. The last 16 prompt positions carry under half of its effect; one position, one layer or a mean "prompt vector" carries nothing. |
| [G](#exp-g) | Is attention the route the prompt uses? | Both routes carry it. Undoing the prompt in the 8 attention layers removes 83% of its effect, in the 24 DeltaNet layers 93%; three heads carry 38%. A steering push travels through the DeltaNet state. |

## Experiments

<a id="exp-f"></a>

### F · Is the prompt's effect stored at the end of the prompt?

> **Question.** Does the prompt act through a decision made at its last positions?  
> **Result.** No. The last 16 prompt positions carry under half of its effect; one position, one layer or a mean "prompt vector" carries nothing.  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c15)

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
- **Why prompt-only steering still worked ([exp. C](2-the-steering-tradeoff.md#exp-c)).** That push was large (α 1.0), went into *every* prompt position, and pointed along a direction that already means "philosophical text". The prompt's own end-state is smaller and less specific. So H1/H8 in their strong form ("the decision sits in a few tokens") are not supported here: the prompt acts over the whole answer, more like a reference the answer keeps returning to.

<a id="exp-g"></a>

### G · Which route carries the philosophical mode: attention or DeltaNet?

> **Question.** Is attention the route the prompt uses?  
> **Result.** Both routes carry it. Undoing the prompt in the 8 attention layers removes 83% of its effect, in the 24 DeltaNet layers 93%; three heads carry 38%. A steering push travels through the DeltaNet state.  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c16)

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
- **For our steering push, the route is the DeltaNet state, not attention.** A push on the prompt reaches the answer through the DeltaNet recurrent state: alone it gives 41% of the effect, with helpfulness kept at 6.94. The attention keys/values alone give nothing, though they add the rest when combined with the state. [exp. C](2-the-steering-tradeoff.md#exp-c) had credited attention; this corrects it.
- **Verdict on the attention hypothesis (H3): partly supported for prompts, not supported for steering.** A few full-attention heads matter disproportionately to how a system prompt shapes the answer, so head-level interventions at L11/L15 are a reasonable next target. But the 24 DeltaNet layers, which attention-centred papers on full-attention models don't have, carry at least as much. For this hybrid model, "attention is the lever" is too narrow.
