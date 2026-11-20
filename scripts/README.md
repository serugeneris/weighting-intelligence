# Scripts

[← README](../README.md) · [Findings](../FINDINGS.md)

Run outputs go to `runs/`, which is not tracked in git. Experiment IDs match the index in [FINDINGS.md](../FINDINGS.md#experiment-index).

## Toolkit and the starting study

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
| `scripts/bake_lora.py` | Merge the rank-1 LoRA into layer 8's `down_proj`: a standalone prompt-free philosophy Qwen (optionally saved with `--out`) |
| `scripts/cut_turns.py` | Cuts saved answers at the end of the assistant's turn (the run-on artefact of [Correction 1](../findings/6-can-we-trust-the-scores.md#correction-1)) |
| `scripts/astra_edit.py` | Astra-style masked rank-1 `down_proj` edits |

## One script per experiment

| Path | What |
|---|---|
| `scripts/exp_answer_reflect.py` | Experiment A: directions from "answer + reflect" contrast data (`data/contrast_pairs_v2.json`) |
| `scripts/exp_selfgen.py` | Experiment B: prompting baseline, prompt + steering, and a direction from the model's own prompted answers |
| `scripts/exp_token_window.py` | Experiment C: steering only part of the answer (late, early, prompt-only windows) |
| `scripts/exp_distill.py`, `scripts/distill_probe.py` | Experiment D: distil the woven prompt into LoRA weights (MLP, token mixers, rank-1 at L8); probe the rank-1 push |
| `scripts/exp_multilayer.py` | Experiment E: push spread over several layers, the natural layer-by-layer path, and a single-layer sweep L4–L24 |
| `scripts/exp_prompt_transplant.py` | Experiment F: paste the prompted model's state at the last prompt positions into an unprompted run; a mean "prompt vector" |
| `scripts/exp_routes.py` | Experiment G: attention vs DeltaNet. Mediation of the system prompt by layer and head; routes of prompt-only steering through the cache |
| `scripts/exp_geometry.py` | Experiment H: norm-preserving pushes and directions with the "drop the answer" component projected out |
| `scripts/exp_confirm.py` | Experiment I: the main candidates on 48 unseen prompts, judged by Claude Sonnet and Claude Opus |
| `scripts/exp_teacher.py` | Experiment J: distil a better teacher (prompt search on training questions, judge-picked best-of-4 answers) |
| `scripts/exp_heads.py` | Experiment K: prompt-derived mean vectors added at the hot spots from experiment G (heads, attention and DeltaNet layers, all mixers) |
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
| `scripts/exp_scale.py` | Experiment V: the same Claude-teacher recipe on Qwen3.5-2B, 4B (and 9B, which doesn't fit 24 GB) |
| `scripts/exp_scale_mlx.py` | Experiment V, part 2: Qwen3.5-4B and 9B as 4-bit MLX models, same recipe (QLoRA); run with `.venv-mlx` |
| `scripts/exp_style.py` | Experiment W: students trained on the teacher's practical part only, with its own reflection, or with a mismatched reflection |
| `scripts/exp_ood.py` | Experiment X: two-turn conversations, a competing terse system prompt, and checkable questions (`data/ood_prompts.json`) |
| `scripts/exp_stack.py` | Experiment Y: the student plus a system prompt, and best-of-4 headroom with an Opus picker |
| `scripts/exp_argument_critique.py` | Experiment Z: Compare objection selection on 16 original and eight harder items |
| `scripts/exp_persona_drift.py` | Experiment AA: Track assistant-role projections across practical and AI self-awareness conversations |
| `scripts/exp_prose_validity.py` | Experiment AB: Cross prose style and logical validity, then ablate layer-8 SAE features |
| `scripts/exp_revision_transfer.py` | Experiment AC: Test relevant and irrelevant updates and transfer a distinction to a new case |
| `scripts/exp_ethics_pressure.py` | Experiment AD: Compare ethical decisions under new evidence and unsupported user pressure |
| `scripts/exp_answer_budget.py`, `scripts/score_answer_budget.py` | Experiment AE: the same tasks at 120, 256 and 512 new tokens, scored against a practical checklist |
| `scripts/exp_judge_context.py` | Experiment AF: a judge with the full chat versus the last user message only |
| `scripts/exp_premise_sensitivity.py` | Experiment AG: pairs of cases that change one decisive fact or one irrelevant detail |
| `scripts/exp_conditional_reflection.py` | Experiment AH: a continuation run that teaches the student to add or drop its reflection on instruction |
| `scripts/exp_philbench.py`, `scripts/philbench_items.py`, `scripts/exp_philbench_analyze.py`, `scripts/build_philpapers.py` | Experiment AI: register versus judge-free skill (PhilPapers stability and constraints, objections, validity, adapted philosophy-bench) |
| `scripts/exp_philbench_mlx.py` | Experiment AI2: the same tasks on 4-bit MLX builds of Qwen3.5-4B and 9B |
| (no script) | Experiment AK: a regrade of saved responses from Z, AC and AD by a second model, and of AC by a human; the grades are in [`docs/reviews/2026-11-16/codex-regrade/`](../docs/reviews/2026-11-16/codex-regrade/README.md) and [`docs/reviews/human-regrade/`](../docs/reviews/human-regrade/README.md) |
| `scripts/human_regrade.py` | Builds the blind packet and grading page for a human regrade of AC, and scores an exported grades file against the Claude and Codex labels of AK; see [`docs/reviews/human-regrade/`](../docs/reviews/human-regrade/README.md) |
| `scripts/philosophy_pilot.py` | Shared inference, review packets and residual hooks for experiments Z–AI |

## Reproduce

```bash
python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt    # torch transformers accelerate huggingface_hub safetensors PyYAML
.venv/bin/pip install pytest && .venv/bin/python -m pytest -q             # smoke tests: no model download, CPU only
.venv/bin/python scripts/find_neurons.py                         # ranks, p50, diff-means
.venv/bin/python scripts/sweep.py                                # read-side neurons + diff-means screen
.venv/bin/python scripts/neuron_search.py                        # write-aligned single neurons + OMP groups
.venv/bin/python scripts/neuron_search.py --omp-only --rescale --ks 16 64 256 --alphas 0.5 1.0 --out omp_rescaled.jsonl
.venv/bin/python scripts/validate_generate.py && .venv/bin/python scripts/claude_judge.py
.venv/bin/python scripts/pathway_check.py                        # downstream pathway check (forward passes only, ~1 min)
.venv/bin/python scripts/dose_sweep.py --stage pathway           # dose/group-size sweep: disturbance per config (~3 min)
.venv/bin/python scripts/dose_sweep.py --stage generate && .venv/bin/python scripts/claude_judge.py --input dose_sweep.jsonl --output claude_judge_dose.jsonl
.venv/bin/python scripts/natural_ceiling.py                       # helpfulness ceiling: contrast-pair answers + 4 dose-sweep anchors
.venv/bin/python scripts/claude_judge.py --input natural_ceiling.jsonl --output claude_judge_natural.jsonl
.venv/bin/python scripts/exp_answer_reflect.py && .venv/bin/python scripts/claude_judge.py --input answer_reflect/gen.jsonl --output answer_reflect/judge.jsonl
.venv/bin/python scripts/exp_selfgen.py && .venv/bin/python scripts/claude_judge.py --input selfgen/gen.jsonl --output selfgen/judge.jsonl
.venv/bin/python scripts/exp_token_window.py && .venv/bin/python scripts/claude_judge.py --input token_window/gen.jsonl --output token_window/judge.jsonl
.venv/bin/python scripts/exp_distill.py && .venv/bin/python scripts/claude_judge.py --input distill/gen.jsonl --output distill/judge.jsonl
.venv/bin/python scripts/exp_multilayer.py && .venv/bin/python scripts/claude_judge.py --input multilayer/gen.jsonl --output multilayer/judge.jsonl
.venv/bin/python scripts/exp_prompt_transplant.py && .venv/bin/python scripts/claude_judge.py --input prompt_transplant/gen.jsonl --output prompt_transplant/judge.jsonl
.venv/bin/python scripts/exp_routes.py && .venv/bin/python scripts/claude_judge.py --input routes/gen.jsonl --output routes/judge.jsonl
.venv/bin/python scripts/exp_geometry.py && .venv/bin/python scripts/claude_judge.py --input geometry/gen.jsonl --output geometry/judge.jsonl
.venv/bin/python scripts/exp_teacher.py --stage prompts && .venv/bin/python scripts/claude_judge.py --input teacher/prompts_gen.jsonl --output teacher/prompts_judge.jsonl
.venv/bin/python scripts/exp_teacher.py --stage samples && .venv/bin/python scripts/claude_judge.py --input teacher/samples_gen.jsonl --output teacher/samples_judge.jsonl
.venv/bin/python scripts/exp_teacher.py --stage train && .venv/bin/python scripts/claude_judge.py --input teacher/gen.jsonl --output teacher/judge.jsonl
.venv/bin/python scripts/exp_confirm.py --teacher-loras down8-r1:down8-r1_prompt.pt mlp-r8:mlp-r8_best-of-4.pt
.venv/bin/python scripts/claude_judge.py --input confirm/gen.jsonl --output confirm/judge.jsonl && .venv/bin/python scripts/claude_judge.py --input confirm/gen.jsonl --output confirm/judge_opus.jsonl --model opus
.venv/bin/python scripts/exp_heads.py && .venv/bin/python scripts/claude_judge.py --input heads/gen.jsonl --output heads/judge.jsonl
.venv/bin/python scripts/exp_heads.py --round 2 && .venv/bin/python scripts/claude_judge.py --input heads/gen_round2.jsonl --output heads/judge_round2.jsonl
.venv/bin/python scripts/bake_lora.py                            # merge the rank-1 LoRA into W_down at L8 (add --out DIR to save the model)
.venv/bin/python scripts/exp_gate.py && .venv/bin/python scripts/claude_judge.py --input gate/gen.jsonl --output gate/judge.jsonl
.venv/bin/python scripts/exp_gate.py --round 2 && .venv/bin/python scripts/claude_judge.py --input gate/gen_round2.jsonl --output gate/judge_round2.jsonl
.venv/bin/python scripts/exp_dpo.py && .venv/bin/python scripts/claude_judge.py --input dpo/gen.jsonl --output dpo/judge.jsonl
.venv/bin/python scripts/exp_robust.py --dpo && .venv/bin/python scripts/claude_judge.py --input robust/gen.jsonl --output robust/judge.jsonl
.venv/bin/python scripts/exp_claude_teacher.py --stage targets && .venv/bin/python scripts/exp_claude_teacher.py --stage train
.venv/bin/python scripts/claude_judge.py --input claude_teacher/gen.jsonl --output claude_teacher/judge.jsonl && .venv/bin/python scripts/claude_judge.py --input claude_teacher/gen.jsonl --output claude_teacher/judge_opus.jsonl --model opus
.venv/bin/python scripts/exp_more_teacher.py --stage data && .venv/bin/python scripts/exp_more_teacher.py --stage train && .venv/bin/python scripts/exp_more_teacher.py --stage gains
for f in gen gen_gains; do .venv/bin/python scripts/claude_judge.py --input more_teacher/$f.jsonl --output more_teacher/${f/gen/judge}.jsonl; .venv/bin/python scripts/claude_judge.py --input more_teacher/$f.jsonl --output more_teacher/${f/gen/judge}_opus.jsonl --model opus; done
.venv/bin/python scripts/exp_deeper_teacher.py --stage targets && .venv/bin/python scripts/exp_deeper_teacher.py --stage train
.venv/bin/python scripts/claude_judge.py --input deeper_teacher/gen.jsonl --output deeper_teacher/judge.jsonl && .venv/bin/python scripts/claude_judge.py --input deeper_teacher/gen.jsonl --output deeper_teacher/judge_opus.jsonl --model opus
.venv/bin/python scripts/exp_capacity.py && .venv/bin/python scripts/claude_judge.py --input capacity/gen.jsonl --output capacity/judge.jsonl && .venv/bin/python scripts/claude_judge.py --input capacity/gen.jsonl --output capacity/judge_opus.jsonl --model opus
.venv/bin/python scripts/exp_dpo_claude.py --stage samples && .venv/bin/python scripts/claude_judge.py --input dpo_claude/samples_gen.jsonl --output dpo_claude/samples_judge.jsonl
.venv/bin/python scripts/exp_dpo_claude.py --stage train && .venv/bin/python scripts/claude_judge.py --input dpo_claude/gen.jsonl --output dpo_claude/judge.jsonl && .venv/bin/python scripts/claude_judge.py --input dpo_claude/gen.jsonl --output dpo_claude/judge_opus.jsonl --model opus
.venv/bin/python scripts/exp_sae.py --stage acts && .venv/bin/python scripts/exp_sae.py --stage train && .venv/bin/python scripts/exp_sae.py --stage steer
.venv/bin/python scripts/claude_judge.py --input sae/gen.jsonl --output sae/judge.jsonl && .venv/bin/python scripts/exp_sae.py --stage inspect
.venv/bin/python scripts/exp_sae.py --stage steer2 && .venv/bin/python scripts/claude_judge.py --input sae/gen_round2.jsonl --output sae/judge_round2.jsonl
.venv/bin/python scripts/exp_qwen_judge.py --stage score && .venv/bin/python scripts/exp_qwen_judge.py --stage analyze
.venv/bin/python scripts/cut_turns.py runs/more_teacher/gen.jsonl && .venv/bin/python scripts/cut_turns.py runs/more_teacher/gen_gains.jsonl   # then judge gen_cut / gen_gains_cut (Sonnet + Opus)
.venv/bin/python scripts/exp_style.py --stage split && .venv/bin/python scripts/exp_style.py --stage train && .venv/bin/python scripts/cut_turns.py runs/style/gen.jsonl
.venv/bin/python scripts/claude_judge.py --input style/gen_cut.jsonl --output style/judge_cut.jsonl && .venv/bin/python scripts/claude_judge.py --input style/gen_cut.jsonl --output style/judge_cut_opus.jsonl --model opus
.venv/bin/python scripts/exp_ood.py && .venv/bin/python scripts/cut_turns.py runs/ood/gen.jsonl
.venv/bin/python scripts/claude_judge.py --input ood/gen_cut.jsonl --output ood/judge.jsonl && .venv/bin/python scripts/claude_judge.py --input ood/gen_cut.jsonl --output ood/judge_opus.jsonl --model opus
.venv/bin/python scripts/exp_stack.py --stage gen && .venv/bin/python scripts/claude_judge.py --input stack/samples_gen.jsonl --output stack/samples_judge_opus.jsonl --model opus
.venv/bin/python scripts/exp_stack.py --stage select && .venv/bin/python scripts/claude_judge.py --input stack/gen.jsonl --output stack/judge.jsonl && .venv/bin/python scripts/claude_judge.py --input stack/gen.jsonl --output stack/judge_opus.jsonl --model opus
.venv/bin/python scripts/exp_scale.py --model 2B && .venv/bin/python scripts/exp_scale.py --model 4B   # --model 9B needs more than 24 GB
.venv/bin/python scripts/cut_turns.py runs/scale/gen.jsonl && .venv/bin/python scripts/claude_judge.py --input scale/gen_cut.jsonl --output scale/judge.jsonl && .venv/bin/python scripts/claude_judge.py --input scale/gen_cut.jsonl --output scale/judge_opus.jsonl --model opus
.venv-mlx/bin/python scripts/exp_scale_mlx.py --model 4B && .venv-mlx/bin/python scripts/exp_scale_mlx.py --model 9B   # python3.14 -m venv .venv-mlx && .venv-mlx/bin/pip install mlx-lm
.venv/bin/python scripts/cut_turns.py runs/scale_mlx/gen.jsonl && .venv/bin/python scripts/claude_judge.py --input scale_mlx/gen_cut.jsonl --output scale_mlx/judge.jsonl && .venv/bin/python scripts/claude_judge.py --input scale_mlx/gen_cut.jsonl --output scale_mlx/judge_opus.jsonl --model opus
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.exp_argument_critique              # experiment Z, [exp. Z](../findings/7-register-versus-skill.md#exp-z)
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.exp_persona_drift                  # experiment AA, [exp. AA](../findings/7-register-versus-skill.md#exp-aa)
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.exp_prose_validity                 # experiment AB, [exp. AB](../findings/7-register-versus-skill.md#exp-ab)
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.exp_revision_transfer              # experiment AC, [exp. AC](../findings/7-register-versus-skill.md#exp-ac)
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.exp_ethics_pressure                # experiment AD, [exp. AD](../findings/7-register-versus-skill.md#exp-ad)
python3 scripts/build_philpapers.py data/sources/philpapers2020_results_all.json   # experiment AI data (survey read 2026-11-15)
git clone https://github.com/benedictbrady/philosophy-bench external/philosophy-bench && git -C external/philosophy-bench checkout 012a5913
.venv/bin/python -m scripts.exp_philbench --stage data-check score pressure register generate   # experiment AI, [exp. AI](../findings/7-register-versus-skill.md#exp-ai)
.venv/bin/python scripts/claude_judge.py --input philbench/register_gen.jsonl --output philbench/register_judge.jsonl && .venv/bin/python scripts/claude_judge.py --model opus --input philbench/register_gen.jsonl --output philbench/register_judge_opus.jsonl
.venv/bin/python -m scripts.exp_philbench --model 2B --stage score pressure                       # [exp. AI2](../findings/7-register-versus-skill.md#exp-ai2)
.venv-mlx/bin/python scripts/exp_philbench_mlx.py --model 4B && .venv-mlx/bin/python scripts/exp_philbench_mlx.py --model 9B
.venv/bin/python -m scripts.exp_philbench --stage analyze                                         # -> runs/philbench/summary.{json,md}
.venv/bin/python scripts/distill_probe.py                         # size and direction of the rank-1 LoRA's push
.venv/bin/python scripts/astra_edit.py && .venv/bin/python scripts/claude_judge.py --input astra.jsonl --output claude_judge_astra.jsonl
```
