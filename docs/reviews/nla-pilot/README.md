# Pilot AM: saved outputs

Write-up: [findings part 7, exp. AM](../../../findings/7-register-versus-skill.md#exp-am). Plan, predictions and stage-by-stage results: [docs/plans/nla-pilot.md](../../plans/nla-pilot.md).

NLA: `therem/qwen3.5-9b-nla-L21-av-rl` (verbalizer, run in 4 bits) and `therem/qwen3.5-9b-nla-L21-ar-rl` (reconstructor, bf16), reading block 21 of `Qwen/Qwen3.5-9B`. Activations are not saved; `scripts/exp_nla*.py` regenerates them.

| File | Stage | Contents |
|---|---|---|
| `outputs/stage0.json` | 0 | Gate FVE and controls on 128 held-out corpus rows |
| `outputs/stage0_expl.jsonl.gz` | 0 | AV explanations, with and without injection |
| `outputs/smoke_expl.jsonl` | 0 | Smoke test on our chat format (24 activations) |
| `outputs/s1_rows.json.gz` | 1 | The 436 read positions: prompt, labels (9B verdict, key), context |
| `outputs/s1_probe.json`, `s1_acts4_cos.json` | 1 | Linear-probe accuracies; bf16 vs 4-bit activation cosine |
| `outputs/s1_expl.jsonl.gz`, `s1_judge.jsonl.gz` | 1 | 1,744 explanations; 3,924 blind Sonnet judgments |
| `outputs/s1_results.json` | 1 | Agreement with model and key per readout and control |
| `outputs/s2_rows.json.gz`, `s2_expl.jsonl.gz`, `s2_judge.jsonl.gz` | 2 | 192 activations (48 prompts × 4 configs), 768 explanations, 2,304 judgments |
| `outputs/s2_gen.jsonl` | 2 | Greedy 4-bit 9B answers under each configuration, 8 prompts |
| `outputs/s2_results.json` | 2 | Request matching, reflective rate, layer-21 cosine to base |
| `outputs/s3_pairs.json` | 3 | 16 plain explanations and their Sonnet rewrites |
| `outputs/s3_gen.jsonl`, `s3_judge.jsonl.gz`, `s3_results.json` | 3 | Steered answers (9 arms × 16 prompts), rubric scores, geometry |

Judge keys in `s1_judge` are `row|seed|condition`; in `s2_judge`, `row|seed|condition|task`. Seed 0 is greedy, seeds 1–3 are sampled at T = 1.
