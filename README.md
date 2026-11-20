# weighting-intelligence

Exploratory experiments on making **Qwen3.5-4B** answer practical questions in a more philosophical way, and on what that change is worth. We tried three kinds of intervention: pushes at inference time (single neurons, neuron groups, a "philosophy direction"), system prompts, and trained weight updates distilled from a teacher. Then we tested whether the result is philosophical skill or only a way of talking.

## What we found

1. **Pushing works, at a price.** A direction at layer 8, or 16 well-chosen neurons, makes answers reflective and much less helpful. No change to the push removes that cost.
2. **Asking works better.** A system prompt beats every steering setting.
3. **The prompt can be trained into the weights.** One rank-1 edit of one matrix reproduces it without the prompt.
4. **A stronger teacher gives the best student; then the base model is the limit.** A student distilled from Claude-written answers matches the prompt's philosophy with more helpfulness. Part of that lead comes from finishing early within a short token budget.
5. **The reflective register is not philosophical skill.** None of these levers improves skill measured without a judge. Model size does.

Scores come from blind Claude judges on a 0–9 scale. This is exploratory work on one model family with small prompt sets; [FINDINGS.md](FINDINGS.md#caveats) lists the caveats.

## Where to read what

| If you want | Read |
|---|---|
| The hypotheses, their verdicts and what still stands | [FINDINGS.md](FINDINGS.md) |
| One experiment in full: method, tables, interpretation | [`findings/`](findings/), one file per question |
| The ideas explained with animations, no background needed | [Interactive explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html) |
| The related papers and the hypotheses H1–H8 they suggested | [LITERATURE.md](LITERATURE.md) |
| Which script ran which experiment, and how to reproduce | [`scripts/README.md`](scripts/README.md) |

The work is organised around seven questions. Each has a file in `findings/` and a part in the explainer.

| # | Question | Experiments |
|---|---|---|
| 1 | [Can we push the model into a philosophical register?](findings/1-pushing-the-model.md) | 0.1–0.6 |
| 2 | [Can any change to the push remove the helpfulness cost?](findings/2-the-steering-tradeoff.md) | A, B, C, E, H, K, T |
| 3 | [How does a prompt do what steering cannot?](findings/3-how-the-prompt-works.md) | F, G |
| 4 | [Can the prompt's behaviour be moved into the weights?](findings/4-into-the-weights.md) | D, J, I, D2, L, M, N |
| 5 | [What limits the student: teacher, data, adapter or base model?](findings/5-what-limits-the-student.md) | O, P, Q, R, S, W, X, Y, V, AH |
| 6 | [Can we trust the measurements?](findings/6-can-we-trust-the-scores.md) | U, Correction 1, AE, AF, AK |
| 7 | [Is the reflective register philosophical skill?](findings/7-register-versus-skill.md) | Z, AA, AB, AC, AD, AG, AI, AI2 |

## Repository layout

| Path | What |
|---|---|
| `FINDINGS.md`, `findings/` | Overview and the full write-ups |
| `LITERATURE.md` | Literature review and hypotheses H1–H8 |
| `wi/steer.py` | Toolkit: load the model, calibrate layer scales, push neurons and vectors, generate, self-judge |
| `scripts/` | One script per experiment; see [`scripts/README.md`](scripts/README.md) |
| `data/` | Contrast pairs, training questions, evaluation prompts, and saved results for experiments AE–AH |
| `docs/explainer.html` | The interactive explainer, served by GitHub Pages |
| `docs/reviews/` | Saved outputs and review notes for experiments Z–AD, AI and AK, and the human regrade of AC |
| `docs/plans/` | The plan written before experiment AI |
| `tests/` | Smoke tests for the toolkit and the pure-Python scripts; they need no model download |

Run outputs go to `runs/`, which is not tracked in git. Dependencies are listed in `requirements.txt`; `python -m pytest -q` runs the tests.
