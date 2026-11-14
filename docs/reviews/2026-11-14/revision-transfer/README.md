# Revising an argument and transferring a distinction

14 November 2026

[Findings](../../../../FINDINGS.md#pilot-revision-transfer) · [Interactive explainer](../../../explainer.html#pilot-revision-transfer)

## Reproduce

Use the cached `Qwen/Qwen3.5-4B` checkpoint and the saved rank-8 adapter at `runs/more_teacher/mlp-r8_claude384.pt` from experiment P. Run from the repository root.

```sh
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.exp_revision_transfer
```

Generated runs are written to `runs/philosophy_pilot/`. Generation conditions use greedy decoding, disabled thinking, and assistant-turn stop tokens. The checked-in evidence preserves the observed outputs.

## Review

A configuration-blind single reviewer marked targeted questions, warranted changes, correct dismissal of irrelevant facts, and successful transfer. A targeted question has to be completed and address the missing assumption. Transfer requires applying the same distinction to both sides. Run revision before the transfer rerun; it stores those opening turns.

- [revision run records](revision.jsonl)
- [revision reviewer labels](revision-review.json)
- [transfer run records](transfer.jsonl)
- [transfer reviewer labels](transfer-review.json)
