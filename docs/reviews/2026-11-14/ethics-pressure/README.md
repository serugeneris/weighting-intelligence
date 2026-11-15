# Experiment AD · Will it hold its position under pressure?

14 November 2026

[Findings](../../../../FINDINGS.md#pilot-ethics-pressure) · [Interactive explainer](../../../explainer.html#c41)

## Reproduce

Use the cached `Qwen/Qwen3.5-4B` checkpoint and the saved rank-8 adapter at `runs/more_teacher/mlp-r8_claude384.pt` from experiment P. Run from the repository root.

```sh
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.exp_ethics_pressure
```

Generated runs are written to `runs/philosophy_pilot/`. Generation conditions use greedy decoding, disabled thinking, and assistant-turn stop tokens. The checked-in evidence preserves the observed outputs.

## Review

The configuration-blind review asks whether the proposed decision stays stable under pressure and reflects a material update when facts change. It does not assign a preferred ethical theory. The four cases are author-written and descriptive.

- [pressure run records](pressure.jsonl)
- [pressure reviewer labels](pressure-review.json)
