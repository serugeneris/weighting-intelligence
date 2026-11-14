# Theatrical wording and logical validity

14 November 2026

[Findings](../../../../FINDINGS.md#pilot-prose-validity) · [Interactive explainer](../../../explainer.html#pilot-prose-validity)

## Reproduce

Use the cached `Qwen/Qwen3.5-4B` checkpoint and the saved layer-8 SAE at `runs/sae/sae.pt` from experiment T. Run from the repository root.

```sh
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.exp_prose_validity
```

Generated runs are written to `runs/philosophy_pilot/`. Generation conditions use greedy decoding, disabled thinking, and assistant-turn stop tokens. The checked-in evidence preserves the observed outputs.

## Review

Evidence retains all 48 variants and the 24 held-out predictions under each condition. Ranking used the other six argument families. This is a forced-choice inference test over one pair of wording templates.

- [sae run records](sae.jsonl)
- [sae activation summary](sae-summary.json)
