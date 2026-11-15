# Experiment AA · Does the assistant drift?

14 November 2026

[Findings](../../../../FINDINGS.md#pilot-persona-drift) · [Interactive explainer](../../../explainer.html#c38)

## Reproduce

Use the cached `Qwen/Qwen3.5-4B` checkpoint and the saved rank-8 adapter at `runs/more_teacher/mlp-r8_claude384.pt` from experiment P. Run from the repository root.

```sh
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.exp_persona_drift
```

Generated runs are written to `runs/philosophy_pilot/`. Generation conditions use greedy decoding, disabled thinking, and assistant-turn stop tokens. The checked-in evidence preserves the observed outputs.

## Review

This is a model-specific contrast estimated from the saved 4B checkpoint. The cap code runs only after a positive held-out role gap and the stated drift threshold; there is no cap result in the saved run.

- [persona run records](persona.jsonl)
- [persona activation summary](persona-summary.json)
