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
| `scripts/astra_edit.py` | Astra-style masked rank-1 `down_proj` edits |
| `data/` | Contrast pairs and evaluation prompts |
| `docs/explainer.html` | The interactive explainer, served by GitHub Pages |

Reproduction commands are at the end of [FINDINGS.md](FINDINGS.md#reproduce). Run outputs go to `runs/`, which is not tracked in git.
