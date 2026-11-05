# Literature review: steering an LLM toward philosophy

*Compiled 2026-11-05 by a research subagent (Claude Sonnet, with Claude Haiku readers) for this project. Most papers were read at abstract level only: the abstract page, a fetch summary or a search snippet. Numbers are the authors' own claims and were not checked. Rows marked "abs" or "snippet" were not read in full. Treat everything here as leads to test, not as facts.*

**Bottom line.** No paper was found on steering toward philosophical reasoning as such. What follows is adjacent work: persona, style, moral and abstract-concept steering, and prompt distillation. No steering paper was found that studies a hybrid DeltaNet model like Qwen3.5-4B, and none measures helpfulness at 4B under style steering.

## Hypotheses and how we test them

| # | Hypothesis | Evidence for | Evidence against | Test in this repo |
|---|---|---|---|---|
| H1 | "Philosophical" is a response-mode decision made at the prompt or first tokens | Our experiment C (prompt-only = full steering). Dong et al. 2502.06258: prompt representations encode attributes of the future answer | Planning signals also peak at the end of answers; nothing tests style specifically | Experiment F (prompt transplant, last-token steering). **Not supported in its strong form**: the last 16 positions carry under half of the prompt's effect (FINDINGS §12) |
| H2 | Gated steering (only when a condition is detected) avoids the cost | CAST 2409.05907; moral vectors + probe gate 2601.10307 report near-baseline utility | Our cost appears on prompts where we *want* the shift, so a topic gate can't remove it | Not run: every evaluation prompt here should get the shift |
| H3 | Attention is the lever ("philosophy relates ideas") | Izawa et al. 2603.13249: 3 heads carry persona/style; head steering wins the trait-vs-coherence Pareto in most conditions (Qwen2.5-7B, Llama-8B). Attention-level gated steering 2605.10664. ITI 2306.03341. KV-cache steering 2507.08799 | Exceptions for safety personas; residual-stream methods also work; all head evidence is on full-attention 7–8B models, while 24 of Qwen3.5-4B's 32 layers are DeltaNet | Experiment G (attention vs DeltaNet). **Partly supported for prompts** (3 heads at L11/L15 carry 38%), **not for steering** (the push travels through the DeltaNet state); DeltaNet carries as much as attention (FINDINGS §13) |
| H4 | A different push geometry (rotation, mild norm change) costs less | COAST 2605.01167, Selective Steering 2601.19375, angle-norm 2606.06735 | All measured on refusal, not style; our cost looks semantic | Not run yet |
| H5 | The layer matters ("abstraction lives in the middle") | Concept-subspace paper 2602.07794 (title unverified); style vectors at mid-late layers (2402.01618, 2503.05213) | 2405.17345 (Gemma-2B): early and late layers beat the middle; we never swept | Experiment E (layer sweep). **Partly supported**: only L8–L16 respond to a modest push, but no layer escapes the trade-off (FINDINGS §11) |
| H6 | Persona/register SAE features steer with fewer side effects | 2505.20063 (output-scored features) | AxBench 2501.17148: SAEs not competitive, prompting best; no public SAEs for Qwen3.5-4B | Not run |
| H7 | A learned low-rank intervention (ReFT/LoRA) trained on prompted answers beats diff-means | ReFT 2404.03592; AxBench: ReFT-r1 competitive at 2B; prompt internalization 2411.15927 | May just learn "reflect instead of answer" | Experiment D (LoRA distillation). **Supported**: all three adapters, even rank 1 at L8, match the prompt's balance without the prompt (FINDINGS §10) |
| H8 | Prompting acts strongly on a few tokens; steering can mimic it with per-token coefficients | Heyman & Vandeputte 2605.03907 (abs) | 2604.09839 (abs): some steered states can't be reached by any prompt | Experiment F. **Not supported**: a mean prompt offset at the last 1–9 positions does nothing (FINDINGS §12) |

**On our prompt-only result (experiment C).** It does *not* tell the attention hypothesis apart from the alternatives. A push on prompt positions reaches the answer through full attention or through the DeltaNet recurrent state, which is how every residual change propagates, and H1 predicts the same thing. Experiment G separates the two routes.

## Papers consulted

| Title (year) | Link | Method | Takeaway |
|---|---|---|---|
| Inference-Time Intervention, Li et al. (2023) | arxiv.org/abs/2306.03341 | probe-selected heads shifted along a truth direction | Selected heads beat all-head steering; no style test |
| Programming Refusal with CAST, Lee et al. (2024) | arxiv.org/abs/2409.05907 | condition vector gates a behaviour vector | Gate on prompt-pass cosine |
| ReFT, Wu et al. (2024) | arxiv.org/abs/2404.03592 | low-rank intervention at prompt positions | Parameter-efficient; prompt positions are a standard site |
| Persona Vectors, Chen et al. (2025) | arxiv.org/abs/2507.21509 | residual diff-means at one layer | Residual steering works; no head/MLP comparison |
| Steering at the Source (style modulation heads), Izawa et al. (2026) | arxiv.org/abs/2603.13249 | head-level persona steering | Best evidence for heads; exceptions on safety personas |
| Prompt-Activation Duality (GCAD) (2026) | arxiv.org/html/2605.10664v1 | gated attention delta from a system prompt | Attention-level steering reduces multi-turn drift (7–8B) |
| COAST, Nguyen et al. (2026) | arxiv.org/html/2605.01167 | collateral-damage-minimising steering | Beats ActAdd at equal attack success; refusal only |
| Selective Steering, Dang & Ngo (2026) | arxiv.org/abs/2601.19375 | norm-preserving rotation on selected layers | Capability retained; refusal only |
| Geometric account of steering, Aparin & Gaintseva (2026) | arxiv.org/pdf/2606.06735 | angle/norm decomposition | Mild norm increase beats strict preservation |
| KV Cache Steering (2025) | arxiv.org/html/2507.08799v2 | one-time K/V shift after prefill | Works from 360M to 8B on reasoning-format tasks |
| AxBench, Wu et al. (2025) | arxiv.org/abs/2501.17148 | steering/detection benchmark | Prompting best; SAEs weak |
| Steer Like the LLM, Heyman & Vandeputte (2026) | arxiv.org/abs/2605.03907 | per-token coefficients mimicking prompting (abs) | Prompting acts on a few tokens |
| Subliminal Learning Is Steering Vector Distillation (2026) | arxiv.org/abs/2606.00995 | system prompt approximated by a vector (abs) | — |
| Steered LLM Activations are Non-Surjective (2026) | arxiv.org/abs/2604.09839 | theory (abs) | Some steered states are unreachable by prompts |
| Steer Model beyond Assistant (2026) | arxiv.org/abs/2601.06403 | contrastive decoding on a system prompt (abs) | Decoding-time control of prompt strength |
| Limitations of Steering in Alignment (2025) | arxiv.org/abs/2505.01162 | hook interventions (abs) | Steering weak in complex contexts |
| Generative Prompt Internalization, Shin et al. (2024/25) | arxiv.org/abs/2411.15927 | prompt distillation (abs) | Closest published match to experiment D |
| Exploring and steering the moral compass, Tlaie (2024) | arxiv.org/abs/2405.17345 | similarity-based steering, Gemma-2B | Early and late layers beat the middle |
| The Straight and Narrow, Hu et al. (2026) | arxiv.org/abs/2601.10307 | moral vectors + probe gate, Llama-8B | Near-baseline utility |
| Steering Geometry (2026) | arxiv.org/abs/2609.06289 | geometry of steering methods | Includes Qwen3.5-4B CAA (geometry only) |
| Steering LLMs to Evaluate and Amplify Creativity, Olson et al. (2024) | arxiv.org/abs/2412.06060 | diff-means at L8, Llama3-8B | Layer chosen qualitatively |
| Style Vectors for Steering, Konen et al. (2024) | arxiv.org/abs/2402.01618 | activation style vectors, layers 18–20 | Large strengths cause repetition |
| StyleVector, Zhang et al. (2025) | arxiv.org/abs/2503.05213 | user-style diff-means | Mid-late layers |
| Emergent Response Planning, Dong et al. (2025) | arxiv.org/pdf/2502.06258 (snippet) | probes on prompt representations | Prompt encodes response attributes |
| Taxonomy of Representation Engineering (2025) | arxiv.org/pdf/2502.19649 (snippet) | survey | First-token-only steering viable |
| Mitigating Content Effects on Reasoning via Steering (2025) | arxiv.org/abs/2505.12189 (snippet) | steering for logical validity | Closest to philosophical argument, but formal |
| SAEs Are Good for Steering if You Select the Right Features (2025) | arxiv.org/html/2505.20063v2 (snippet) | output-score feature selection | Output features steer more coherently |
| Imagining and building wise machines, Johnson et al. (2024) | arxiv.org/abs/2411.02478 (snippet) | position paper | Wisdom as metacognition; no recipe |
| Creating a Large Language Model of a Philosopher, Schwitzgebel et al. (2023) | arxiv 2302.01339 (snippet) | GPT-3 fine-tuned on Dennett | Fine-tuning gives a convincing philosophical style |
| Socratic tutoring (EULER; Favero et al. 2409.05511; KELE) (2024–25) | arxiv.org/pdf/2409.05511 (snippet) | SFT/DPO for Socratic tutors | DPO on judged pairs is a usable recipe |
| Concept-subspace paper 2602.07794 (title unverified); Gated DeltaNet 2412.06464 | snippets | concept subspace; architecture | Not used beyond what is stated above |
