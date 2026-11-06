# weighting-intelligence

Exploratory experiments on steering **Qwen3.5-4B** toward more philosophical answers. We tried two ways to do it: pushing individual MLP neurons, adding a "philosophy direction" at inference time, and editing the weights directly (Astra-style).

- **[Interactive explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html)**: animated walkthrough of the core ideas:
  - what a neuron reads and writes, and how the residual stream carries it forward
  - what the diff-means direction is, and why we compare with cosine
  - how matching pursuit picks the neurons, and why weight edits can't add a behavior
- **[FINDINGS.md](FINDINGS.md)**: the full write-up, covering method, results, caveats and next steps.

## Results in short

Scores come from a blind Claude judge, 0–9, on 16 prompts, 8 of them held out.

- **Single neurons** don't work. At doses high enough to matter they break the model, the same way random neurons do.
- **16 neurons chosen by write-alignment** at layer 8 score philosophy 2.56, against 0.00 for the baseline and 0.07 for matched random groups. None of the outputs are degenerate.
- **The full diff-means direction** at layer 8 is still the best knob, with philosophy 3.00. Every steer that worked also cost helpfulness.
- **Bias-free rank-1 weight edits** are safe but nearly inert, reaching philosophy 0.50 at most.

## Layout

| Path | What |
|---|---|
| `wi/steer.py` | Toolkit: load the model, calibrate layer scales, capture neuron activations, push neurons/vectors, generate, self-judge |
| `scripts/find_neurons.py` | Read-side neuron ranking and per-layer diff-means directions |
| `scripts/sweep.py`, `scripts/analyze.py` | Screening sweeps and their analysis |
| `scripts/neuron_search.py` | Write-side search: cosine with `W_down` columns, matching-pursuit groups |
| `scripts/validate_generate.py`, `scripts/claude_judge.py` | Held-out validation with matched random controls and a blind Claude judge |
| `scripts/pathway_check.py` | Downstream pathway check: do L8 pushes recreate the natural philosophical changes in later layers? |
| `scripts/dose_sweep.py` | Smaller pushes and larger neuron groups: disturbance, then generation for the blind judge |
| `scripts/natural_ceiling.py` | Helpfulness ceiling: judge the natural contrast-pair answers alongside steered anchors |
| `scripts/exp_answer_reflect.py` | Experiment A: directions from "answer + reflect" contrast data (`data/contrast_pairs_v2.json`) |
| `scripts/exp_selfgen.py` | Experiment B: prompting baseline, prompt + steering, and a direction from the model's own prompted answers |
| `scripts/exp_token_window.py` | Experiment C: steering only part of the answer (late, early, prompt-only windows) |
| `scripts/exp_distill.py`, `scripts/distill_probe.py` | Experiment D: distil the woven prompt into LoRA weights (MLP, token mixers, rank-1 at L8); probe the rank-1 push |
| `scripts/exp_multilayer.py` | Experiment E: push spread over several layers, the natural layer-by-layer path, and a single-layer sweep L4–L24 |
| `scripts/exp_prompt_transplant.py` | Experiment F: paste the prompted model's state at the last prompt positions into an unprompted run; a mean "prompt vector" |
| `scripts/exp_routes.py` | Experiment G: attention vs DeltaNet. Mediation of the system prompt by layer and head; routes of prompt-only steering through the cache |
| `scripts/exp_geometry.py` | Experiment H: norm-preserving pushes and directions with the "drop the answer" component projected out |
| `scripts/exp_teacher.py` | Experiment J: distil a better teacher (prompt search on training questions, judge-picked best-of-4 answers) |
| `scripts/exp_confirm.py` | Experiment I: the main candidates on 48 unseen prompts, judged by Claude Sonnet and Claude Opus |
| `scripts/exp_heads.py` | Experiment K: prompt-derived mean vectors added at the hot spots from experiment G (heads, attention and DeltaNet layers, all mixers) |
| `scripts/bake_lora.py` | Merge the rank-1 LoRA into layer 8's `down_proj`: a standalone prompt-free philosophy Qwen (optionally saved with `--out`) |
| `scripts/exp_gate.py` | Experiment L: the rank-1 LoRA split into gate (read side A) and direction (write side B); constant pushes along B |
| `scripts/exp_dpo.py` | Experiment M: preference training (DPO) on judged answer pairs, starting from experiment J's student |
| `scripts/exp_robust.py` | Experiment N: the new candidates (learned direction B, amplified LoRA, DPO) on 48 unseen prompts, greedy and sampled |
| `scripts/exp_claude_teacher.py` | Experiment O: a stronger teacher. Distil Claude-written answers into a LoRA; judged by Sonnet and Opus on 48 prompts |
| `scripts/exp_more_teacher.py` | Experiment P: 3× more Claude-written answers (`data/train_questions_extra.json`), and a gain sweep |
| `scripts/exp_deeper_teacher.py` | Experiment Q: a more philosophical teacher (Claude answers naming specific ideas) on the same 384 questions |
| `scripts/exp_capacity.py` | Experiment R: a 4× larger adapter (rank 32) on both Claude teachers |
| `scripts/exp_dpo_claude.py` | Experiment S: DPO on judged samples of the Claude-taught student |
| `scripts/exp_sae.py` | Experiment T: a TopK sparse autoencoder at layer 8, its top features, and steering with them |
| `scripts/exp_qwen_judge.py` | Experiment U: Qwen3.5-4B as a non-Claude judge on the six 48-prompt pools |
| `scripts/exp_style.py` | Experiment W: students trained on the teacher's practical part only, with its own reflection, or with a mismatched reflection |
| `scripts/exp_ood.py` | Experiment X: two-turn conversations, a competing terse system prompt, and checkable questions (`data/ood_prompts.json`) |
| `scripts/exp_stack.py` | Experiment Y: the student plus a system prompt, and best-of-4 headroom with an Opus picker |
| `scripts/exp_scale.py` | Experiment V: the same Claude-teacher recipe on Qwen3.5-2B, 4B (and 9B, which doesn't fit 24 GB) |
| `scripts/exp_scale_mlx.py` | Experiment V, part 2: Qwen3.5-4B and 9B as 4-bit MLX models, same recipe (QLoRA); run with `.venv-mlx` |
| `scripts/cut_turns.py` | Cuts saved answers at the end of the assistant's turn (the run-on artefact of FINDINGS §29) |
| `data/train_questions.json`, `data/confirm_prompts.json` | 96 training questions for distillation; 48 new held-out prompts for the final confirmation |
| `LITERATURE.md` | Literature review: related papers and the hypotheses H1–H8 they suggest, mapped to experiments D–H |
| `scripts/astra_edit.py` | Astra-style masked rank-1 `down_proj` edits |
| `data/` | Contrast pairs and evaluation prompts |
| `docs/explainer.html` | The interactive explainer, served by GitHub Pages |

Reproduction commands are at the end of [FINDINGS.md](FINDINGS.md#reproduce). Run outputs go to `runs/`, which is not tracked in git.
