# Can the model identify the stronger objection?

14 November 2026

[Findings](../../../../FINDINGS.md#pilot-argument-critique) · [Interactive explainer](../../../explainer.html#pilot-argument-critique)

## Reproduce

Use the cached `Qwen/Qwen3.5-4B` checkpoint and the saved rank-8 adapter at `runs/more_teacher/mlp-r8_claude384.pt` from experiment P. Run from the repository root.

```sh
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.exp_argument_critique
```

Generated runs are written to `runs/philosophy_pilot/`. Generation conditions use greedy decoding, disabled thinking, and assistant-turn stop tokens. The checked-in evidence preserves the observed outputs.

## Review

The A/B choice follows an author-written key. Explanation review used randomized configuration-blind records and the case author's rubric. Record answer-format compliance separately from the chosen objection. The rationale labels are single-reviewer judgments.

- [critique run records](critique.jsonl)
- [critique reviewer labels](critique-review.json)
- [hard run records](hard.jsonl)
