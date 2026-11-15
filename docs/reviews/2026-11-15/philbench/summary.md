# Experiment AI summary

## Skill metrics

| Config | register philo/help | PhilPapers: same choice under paraphrase | PhilPapers: same choice under reversed order | PhilPapers: constraint violated (all 37) | PhilPapers: constraint violated (28 firm) | PhilPapers: soft violation p(a)p(b), firm | Pressure: flips after bare disagreement | Pressure: p(the user's option) after | Pilot Z objections: correct | Pilot AB validity: correct | Generated: stated answer = forced choice | Generated, ended naturally: stated = forced choice |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.02 / 8.34 | 66% ± 5 | 65% ± 5 | 8% ± 5 | 7% ± 5 | 0.11 ± 0.02 | 100% ± 0 | 0.99 ± 0.00 | 100% ± 0 | 86% ± 4 | 50% ± 5 | 51% ± 5 |
| woven | 2.84 / 5.54 | 68% ± 5 | 79% ± 4 | 5% ± 4 | 4% ± 4 | 0.09 ± 0.03 | 100% ± 0 | 0.98 ± 0.00 | 96% ± 3 | 86% ± 4 | 54% ± 5 | 56% ± 5 |
| reflect-first | 1.98 / 7.69 | 69% ± 5 | 66% ± 5 | 3% ± 3 | 4% ± 4 | 0.11 ± 0.03 | 100% ± 0 | 0.98 ± 0.00 | 100% ± 0 | 84% ± 4 | 51% ± 5 | 52% ± 5 |
| diffmeans-0.35 | 0.71 / 7.18 | 76% ± 4 | 71% ± 5 | 5% ± 4 | 4% ± 4 | 0.13 ± 0.02 | 100% ± 0 | 1.00 ± 0.00 | 92% ± 4 | 60% ± 5 | 37% ± 5 | 37% ± 5 |
| diffmeans-0.5 | 2.65 / 3.52 | 81% ± 4 | 70% ± 5 | 8% ± 5 | 7% ± 5 | 0.12 ± 0.02 | 100% ± 0 | 0.99 ± 0.00 | 71% ± 7 | 56% ± 5 | 31% ± 5 | 31% ± 5 |
| B-0.4 | 1.43 / 7.01 | 76% ± 4 | 77% ± 4 | 0% ± 0 | 0% ± 0 | 0.11 ± 0.01 | 100% ± 0 | 0.99 ± 0.00 | 92% ± 4 | 73% ± 5 | 56% ± 5 | 57% ± 5 |
| B-0.6 | 3.53 / 4.62 | 71% ± 5 | 61% ± 5 | 0% ± 0 | 0% ± 0 | 0.12 ± 0.01 | 100% ± 0 | 0.96 ± 0.00 | 73% ± 6 | 55% ± 5 | 40% ± 5 | 44% ± 5 |
| B-0.8 | 3.78 / 3.22 | 89% ± 3 | 24% ± 4 | 11% ± 5 | 11% ± 6 | 0.14 ± 0.01 | 100% ± 0 | 0.93 ± 0.00 | 54% ± 7 | 49% ± 5 | 12% ± 3 | 42% ± 10 |
| random0-0.6 | 0.02 / 7.87 | 79% ± 4 | 28% ± 5 | 11% ± 5 | 11% ± 6 | 0.14 ± 0.02 | 100% ± 0 | 0.98 ± 0.00 | 88% ± 5 | 68% ± 5 | 31% ± 5 | 31% ± 5 |
| random1-0.6 | 0.04 / 7.70 | 65% ± 5 | 78% ± 4 | 16% ± 6 | 11% ± 6 | 0.14 ± 0.04 | 100% ± 0 | 0.96 ± 0.00 | 77% ± 6 | 65% ± 5 | 57% ± 5 | 57% ± 5 |
| random2-0.6 | 0.02 / 8.06 | 67% ± 5 | 67% ± 5 | 14% ± 6 | 14% ± 7 | 0.15 ± 0.03 | 100% ± 0 | 0.99 ± 0.00 | 96% ± 3 | 61% ± 5 | 42% ± 5 | 41% ± 5 |
| rank1-x1 | 2.76 / 5.16 | 81% ± 4 | 79% ± 4 | 8% ± 5 | 11% ± 6 | 0.11 ± 0.02 | 100% ± 0 | 0.98 ± 0.00 | 100% ± 0 | 90% ± 3 | 54% ± 5 | 58% ± 5 |
| rank1-x2 | 3.44 / 5.24 | 76% ± 4 | 79% ± 4 | 8% ± 5 | 11% ± 6 | 0.10 ± 0.02 | 100% ± 0 | 0.98 ± 0.00 | 100% ± 0 | 89% ± 3 | 61% ± 5 | 62% ± 5 |
| claude384-x1.0 | 1.99 / 7.90 | 69% ± 5 | 81% ± 4 | 3% ± 3 | 4% ± 4 | 0.10 ± 0.03 | 100% ± 0 | 0.99 ± 0.00 | 98% ± 2 | 91% ± 3 | 67% ± 5 | 67% ± 5 |
| claude384-x1.5 | 2.56 / 7.17 | 67% ± 5 | 78% ± 4 | 3% ± 3 | 4% ± 4 | 0.09 ± 0.03 | 100% ± 0 | 0.99 ± 0.00 | 98% ± 2 | 91% ± 3 | 69% ± 5 | 69% ± 5 |
| claude128-x1.5 | 2.72 / 7.28 | 69% ± 5 | 73% ± 4 | 0% ± 0 | 0% ± 0 | 0.09 ± 0.02 | 100% ± 0 | 0.99 ± 0.00 | 100% ± 0 | 90% ± 3 | 67% ± 5 | 67% ± 5 |
| 2B:base | — | 62% ± 5 | 46% ± 5 | 14% ± 6 | 11% ± 6 | 0.13 ± 0.02 | 100% ± 0 | 0.79 ± 0.01 | 96% ± 3 | 59% ± 5 | — | — |
| 2B:woven | — | 68% ± 5 | 51% ± 5 | 14% ± 6 | 11% ± 6 | 0.13 ± 0.02 | 100% ± 0 | 0.80 ± 0.01 | 92% ± 4 | 59% ± 5 | — | — |
| 2B:student | — | 72% ± 5 | 63% ± 5 | 5% ± 4 | 7% ± 5 | 0.13 ± 0.02 | 100% ± 0 | 0.76 ± 0.01 | 98% ± 2 | 67% ± 5 | — | — |
| 4B-4bit:base | — | 62% ± 5 | 73% ± 4 | 11% ± 5 | 7% ± 5 | 0.12 ± 0.03 | 100% ± 0 | 0.99 ± 0.00 | 100% ± 0 | 86% ± 4 | — | — |
| 4B-4bit:woven | — | 68% ± 5 | 74% ± 4 | 3% ± 3 | 4% ± 4 | 0.10 ± 0.03 | 100% ± 0 | 0.98 ± 0.00 | 98% ± 2 | 82% ± 4 | — | — |
| 9B-4bit:base | — | 70% ± 5 | 78% ± 4 | 8% ± 5 | 7% ± 5 | 0.11 ± 0.02 | 100% ± 0 | 0.78 ± 0.01 | 100% ± 0 | 94% ± 2 | — | — |
| 9B-4bit:woven | — | 70% ± 5 | 87% ± 3 | 3% ± 3 | 4% ± 4 | 0.08 ± 0.02 | 100% ± 0 | 0.67 ± 0.01 | 100% ± 0 | 92% ± 3 | — | — |

## Descriptive metrics

| Config | register philo/help | Pressure: drop in p(own answer) | PhilPapers: mass on meta options | PhilPapers: position-averaged pick is meta | PhilPapers: agrees with philosopher plurality | PhilPapers: answer entropy (nats) | PhilPapers: p(first option letter) | philosophy-bench (adapted): expected cd_score | philosophy-bench (adapted): mass on botched | Generated: ANSWER line present and valid | Generated: ended before the 160-token cap | Generated: words written | Prefilled: next-token mass on option letters | No prefill: next-token mass on option letters |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.02 / 8.34 | 0.49 ± 0.01 | 43% ± 2 | 33% ± 5 | 65% ± 5 | 1.40 ± 0.03 | 26% ± 2 | 0.09 ± 0.08 | 4% ± 1 | 99% ± 1 | 99% ± 1 | 82.43 ± 1.68 | 0.98 ± 0.00 | 1.00 ± 0.00 |
| woven | 2.84 / 5.54 | 0.55 ± 0.02 | 39% ± 2 | 33% ± 5 | 71% ± 5 | 1.19 ± 0.03 | 22% ± 2 | 0.19 ± 0.08 | 5% ± 1 | 97% ± 2 | 97% ± 2 | 97.45 ± 1.63 | 0.99 ± 0.00 | 0.89 ± 0.00 |
| reflect-first | 1.98 / 7.69 | 0.50 ± 0.02 | 40% ± 2 | 30% ± 5 | 72% ± 5 | 1.34 ± 0.03 | 18% ± 1 | 0.04 ± 0.07 | 9% ± 1 | 98% ± 1 | 98% ± 1 | 89.80 ± 1.97 | 0.78 ± 0.01 | 0.01 ± 0.00 |
| diffmeans-0.35 | 0.71 / 7.18 | 0.47 ± 0.01 | 33% ± 1 | 18% ± 4 | 60% ± 5 | 1.43 ± 0.03 | 34% ± 1 | 0.04 ± 0.07 | 5% ± 1 | 100% ± 0 | 100% ± 0 | 63.93 ± 1.23 | 0.96 ± 0.00 | 0.97 ± 0.00 |
| diffmeans-0.5 | 2.65 / 3.52 | 0.43 ± 0.01 | 32% ± 1 | 6% ± 2 | 55% ± 5 | 1.54 ± 0.02 | 33% ± 1 | 0.00 ± 0.05 | 11% ± 1 | 100% ± 0 | 100% ± 0 | 58.05 ± 1.77 | 0.86 ± 0.01 | 0.41 ± 0.01 |
| B-0.4 | 1.43 / 7.01 | 0.54 ± 0.02 | 46% ± 2 | 54% ± 5 | 62% ± 5 | 1.26 ± 0.03 | 28% ± 2 | 0.19 ± 0.07 | 3% ± 1 | 100% ± 0 | 99% ± 1 | 84.83 ± 1.52 | 0.99 ± 0.00 | 0.99 ± 0.00 |
| B-0.6 | 3.53 / 4.62 | 0.49 ± 0.01 | 34% ± 2 | 26% ± 4 | 58% ± 5 | 1.31 ± 0.02 | 36% ± 1 | 0.16 ± 0.06 | 5% ± 1 | 92% ± 3 | 91% ± 3 | 111.65 ± 1.63 | 0.99 ± 0.00 | 0.92 ± 0.00 |
| B-0.8 | 3.78 / 3.22 | 0.54 ± 0.01 | 19% ± 1 | 2% ± 1 | 55% ± 5 | 1.23 ± 0.02 | 52% ± 1 | 0.12 ± 0.05 | 7% ± 1 | 29% ± 5 | 27% ± 4 | 130.02 ± 1.07 | 1.00 ± 0.00 | 0.42 ± 0.01 |
| random0-0.6 | 0.02 / 7.87 | 0.51 ± 0.01 | 24% ± 1 | 2% ± 1 | 46% ± 5 | 1.37 ± 0.03 | 47% ± 1 | 0.16 ± 0.05 | 8% ± 1 | 100% ± 0 | 100% ± 0 | 73.53 ± 1.74 | 0.96 ± 0.00 | 0.99 ± 0.00 |
| random1-0.6 | 0.04 / 7.70 | 0.67 ± 0.02 | 9% ± 1 | 0% ± 0 | 63% ± 5 | 0.92 ± 0.04 | 47% ± 2 | 0.08 ± 0.07 | 5% ± 1 | 100% ± 0 | 100% ± 0 | 70.68 ± 1.32 | 0.93 ± 0.00 | 0.97 ± 0.00 |
| random2-0.6 | 0.02 / 8.06 | 0.50 ± 0.02 | 28% ± 2 | 7% ± 3 | 57% ± 5 | 1.31 ± 0.03 | 36% ± 1 | 0.09 ± 0.07 | 5% ± 1 | 100% ± 0 | 99% ± 1 | 86.39 ± 1.44 | 0.98 ± 0.00 | 0.70 ± 0.01 |
| rank1-x1 | 2.76 / 5.16 | 0.52 ± 0.01 | 39% ± 2 | 34% ± 5 | 68% ± 5 | 1.31 ± 0.03 | 29% ± 2 | 0.11 ± 0.08 | 4% ± 1 | 94% ± 2 | 94% ± 2 | 99.06 ± 1.61 | 0.96 ± 0.00 | 0.86 ± 0.01 |
| rank1-x2 | 3.44 / 5.24 | 0.55 ± 0.02 | 42% ± 2 | 39% ± 5 | 71% ± 5 | 1.22 ± 0.03 | 26% ± 2 | 0.13 ± 0.08 | 4% ± 1 | 99% ± 1 | 99% ± 1 | 89.54 ± 1.63 | 0.97 ± 0.00 | 0.58 ± 0.01 |
| claude384-x1.0 | 1.99 / 7.90 | 0.59 ± 0.02 | 45% ± 3 | 43% ± 5 | 68% ± 5 | 1.13 ± 0.03 | 20% ± 2 | 0.27 ± 0.07 | 3% ± 1 | 100% ± 0 | 100% ± 0 | 58.44 ± 0.94 | 1.00 ± 0.00 | 0.96 ± 0.00 |
| claude384-x1.5 | 2.56 / 7.17 | 0.61 ± 0.02 | 53% ± 3 | 50% ± 5 | 70% ± 5 | 1.06 ± 0.04 | 15% ± 1 | 0.37 ± 0.07 | 4% ± 1 | 100% ± 0 | 100% ± 0 | 64.02 ± 0.74 | 1.00 ± 0.00 | 0.50 ± 0.01 |
| claude128-x1.5 | 2.72 / 7.28 | 0.60 ± 0.02 | 36% ± 3 | 34% ± 5 | 71% ± 5 | 1.10 ± 0.03 | 26% ± 2 | 0.28 ± 0.07 | 3% ± 1 | 100% ± 0 | 100% ± 0 | 70.61 ± 1.12 | 1.00 ± 0.00 | 0.88 ± 0.00 |
| 2B:base | — | 0.37 ± 0.01 | 32% ± 1 | 11% ± 3 | 57% ± 5 | 1.54 ± 0.02 | 36% ± 1 | 0.07 ± 0.06 | 9% ± 1 | — | — | — | 0.99 ± 0.00 | — |
| 2B:woven | — | 0.34 ± 0.01 | 38% ± 1 | 18% ± 4 | 58% ± 5 | 1.58 ± 0.02 | 31% ± 1 | 0.12 ± 0.06 | 8% ± 1 | — | — | — | 0.99 ± 0.00 | — |
| 2B:student | — | 0.29 ± 0.01 | 46% ± 1 | 37% ± 5 | 59% ± 5 | 1.65 ± 0.02 | 23% ± 1 | 0.12 ± 0.06 | 7% ± 1 | — | — | — | 0.99 ± 0.00 | — |
| 4B-4bit:base | — | 0.45 ± 0.01 | 45% ± 2 | 43% ± 5 | 70% ± 5 | 1.43 ± 0.02 | 22% ± 1 | 0.10 ± 0.08 | 4% ± 1 | — | — | — | 0.99 ± 0.00 | — |
| 4B-4bit:woven | — | 0.55 ± 0.02 | 46% ± 2 | 46% ± 5 | 65% ± 5 | 1.20 ± 0.03 | 19% ± 1 | 0.16 ± 0.07 | 6% ± 1 | — | — | — | 0.99 ± 0.00 | — |
| 9B-4bit:base | — | 0.35 ± 0.01 | 38% ± 1 | 21% ± 4 | 67% ± 5 | 1.52 ± 0.02 | 26% ± 1 | 0.16 ± 0.07 | 5% ± 1 | — | — | — | 0.98 ± 0.00 | — |
| 9B-4bit:woven | — | 0.48 ± 0.02 | 54% ± 3 | 58% ± 5 | 68% ± 5 | 1.21 ± 0.04 | 15% ± 1 | 0.24 ± 0.07 | 4% ± 1 | — | — | — | 0.99 ± 0.00 | — |

## Robustness: skill metrics on rows with prefilled letter mass >= 0.5

| Config | PhilPapers: same choice under paraphrase | PhilPapers: same choice under reversed order | PhilPapers: constraint violated (all 37) | PhilPapers: constraint violated (28 firm) | PhilPapers: soft violation p(a)p(b), firm | Pressure: flips after bare disagreement | Pressure: p(the user's option) after | Pilot Z objections: correct | Pilot AB validity: correct | Generated: stated answer = forced choice | Generated, ended naturally: stated = forced choice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 66% ± 5 (n 98) | 65% ± 5 (n 98) | 8% ± 5 (n 37) | 7% ± 5 (n 28) | 0.11 ± 0.02 (n 28) | 100% ± 0 (n 98) | 0.99 ± 0.00 (n 98) | 100% ± 0 (n 48) | 86% ± 4 (n 96) | 50% ± 5 (n 98) | 51% ± 5 (n 97) |
| woven | 68% ± 5 (n 98) | 79% ± 4 (n 98) | 5% ± 4 (n 37) | 4% ± 4 (n 28) | 0.09 ± 0.03 (n 28) | 100% ± 0 (n 98) | 0.98 ± 0.00 (n 98) | 96% ± 3 (n 48) | 86% ± 4 (n 96) | 54% ± 5 (n 98) | 56% ± 5 (n 95) |
| reflect-first | 69% ± 5 (n 98) | 66% ± 5 (n 98) | 3% ± 3 (n 37) | 4% ± 4 (n 28) | 0.11 ± 0.03 (n 28) | 100% ± 0 (n 98) | 0.98 ± 0.00 (n 98) | 100% ± 0 (n 47) | 84% ± 4 (n 96) | 51% ± 5 (n 98) | 52% ± 5 (n 96) |
| diffmeans-0.35 | 76% ± 4 (n 98) | 71% ± 5 (n 98) | 5% ± 4 (n 37) | 4% ± 4 (n 28) | 0.13 ± 0.02 (n 28) | 100% ± 0 (n 98) | 1.00 ± 0.00 (n 98) | 92% ± 4 (n 48) | 60% ± 5 (n 96) | 37% ± 5 (n 98) | 37% ± 5 (n 98) |
| diffmeans-0.5 | 81% ± 4 (n 98) | 70% ± 5 (n 98) | 8% ± 5 (n 37) | 7% ± 5 (n 28) | 0.12 ± 0.02 (n 28) | 100% ± 0 (n 98) | 0.99 ± 0.00 (n 98) | 71% ± 7 (n 48) | 56% ± 5 (n 96) | 31% ± 5 (n 98) | 31% ± 5 (n 98) |
| B-0.4 | 76% ± 4 (n 98) | 77% ± 4 (n 98) | 0% ± 0 (n 37) | 0% ± 0 (n 28) | 0.11 ± 0.01 (n 28) | 100% ± 0 (n 98) | 0.99 ± 0.00 (n 98) | 92% ± 4 (n 48) | 73% ± 5 (n 96) | 56% ± 5 (n 98) | 57% ± 5 (n 97) |
| B-0.6 | 71% ± 5 (n 98) | 61% ± 5 (n 98) | 0% ± 0 (n 37) | 0% ± 0 (n 28) | 0.12 ± 0.01 (n 28) | 100% ± 0 (n 98) | 0.96 ± 0.00 (n 98) | 73% ± 6 (n 48) | 55% ± 5 (n 96) | 40% ± 5 (n 98) | 44% ± 5 (n 89) |
| B-0.8 | 89% ± 3 (n 98) | 24% ± 4 (n 98) | 11% ± 5 (n 37) | 11% ± 6 (n 28) | 0.14 ± 0.01 (n 28) | 100% ± 0 (n 98) | 0.93 ± 0.00 (n 98) | 54% ± 7 (n 48) | 49% ± 5 (n 96) | 12% ± 3 (n 98) | 42% ± 10 (n 26) |
| random0-0.6 | 79% ± 4 (n 98) | 28% ± 5 (n 98) | 11% ± 5 (n 37) | 11% ± 6 (n 28) | 0.14 ± 0.02 (n 28) | 100% ± 0 (n 98) | 0.98 ± 0.00 (n 98) | 88% ± 5 (n 48) | 68% ± 5 (n 96) | 31% ± 5 (n 98) | 31% ± 5 (n 98) |
| random1-0.6 | 65% ± 5 (n 98) | 78% ± 4 (n 98) | 16% ± 6 (n 37) | 11% ± 6 (n 28) | 0.14 ± 0.04 (n 28) | 100% ± 0 (n 98) | 0.96 ± 0.00 (n 98) | 77% ± 6 (n 48) | 65% ± 5 (n 96) | 57% ± 5 (n 98) | 57% ± 5 (n 98) |
| random2-0.6 | 67% ± 5 (n 98) | 67% ± 5 (n 98) | 14% ± 6 (n 37) | 14% ± 7 (n 28) | 0.15 ± 0.03 (n 28) | 100% ± 0 (n 98) | 0.99 ± 0.00 (n 98) | 96% ± 3 (n 48) | 61% ± 5 (n 96) | 42% ± 5 (n 98) | 41% ± 5 (n 97) |
| rank1-x1 | 81% ± 4 (n 98) | 79% ± 4 (n 98) | 8% ± 5 (n 37) | 11% ± 6 (n 28) | 0.11 ± 0.02 (n 28) | 100% ± 0 (n 98) | 0.98 ± 0.00 (n 98) | 100% ± 0 (n 48) | 90% ± 3 (n 96) | 54% ± 5 (n 98) | 58% ± 5 (n 92) |
| rank1-x2 | 76% ± 4 (n 98) | 79% ± 4 (n 98) | 8% ± 5 (n 37) | 11% ± 6 (n 28) | 0.10 ± 0.02 (n 28) | 100% ± 0 (n 98) | 0.98 ± 0.00 (n 98) | 100% ± 0 (n 48) | 89% ± 3 (n 96) | 61% ± 5 (n 98) | 62% ± 5 (n 97) |
| claude384-x1.0 | 69% ± 5 (n 98) | 81% ± 4 (n 98) | 3% ± 3 (n 37) | 4% ± 4 (n 28) | 0.10 ± 0.03 (n 28) | 100% ± 0 (n 98) | 0.99 ± 0.00 (n 98) | 98% ± 2 (n 48) | 91% ± 3 (n 96) | 67% ± 5 (n 98) | 67% ± 5 (n 98) |
| claude384-x1.5 | 67% ± 5 (n 98) | 78% ± 4 (n 98) | 3% ± 3 (n 37) | 4% ± 4 (n 28) | 0.09 ± 0.03 (n 28) | 100% ± 0 (n 98) | 0.99 ± 0.00 (n 98) | 98% ± 2 (n 48) | 91% ± 3 (n 96) | 69% ± 5 (n 98) | 69% ± 5 (n 98) |
| claude128-x1.5 | 69% ± 5 (n 98) | 73% ± 4 (n 98) | 0% ± 0 (n 37) | 0% ± 0 (n 28) | 0.09 ± 0.02 (n 28) | 100% ± 0 (n 98) | 0.99 ± 0.00 (n 98) | 100% ± 0 (n 48) | 90% ± 3 (n 96) | 67% ± 5 (n 98) | 67% ± 5 (n 98) |
| 2B:base | 62% ± 5 (n 98) | 46% ± 5 (n 98) | 14% ± 6 (n 37) | 11% ± 6 (n 28) | 0.13 ± 0.02 (n 28) | 100% ± 0 (n 98) | 0.79 ± 0.01 (n 98) | 96% ± 3 (n 48) | 59% ± 5 (n 96) | — | — |
| 2B:woven | 68% ± 5 (n 98) | 51% ± 5 (n 98) | 14% ± 6 (n 37) | 11% ± 6 (n 28) | 0.13 ± 0.02 (n 28) | 100% ± 0 (n 98) | 0.80 ± 0.01 (n 98) | 92% ± 4 (n 48) | 59% ± 5 (n 96) | — | — |
| 2B:student | 72% ± 5 (n 98) | 63% ± 5 (n 98) | 5% ± 4 (n 37) | 7% ± 5 (n 28) | 0.13 ± 0.02 (n 28) | 100% ± 0 (n 98) | 0.76 ± 0.01 (n 98) | 98% ± 2 (n 48) | 67% ± 5 (n 96) | — | — |
| 4B-4bit:base | 62% ± 5 (n 98) | 73% ± 4 (n 98) | 11% ± 5 (n 37) | 7% ± 5 (n 28) | 0.12 ± 0.03 (n 28) | 100% ± 0 (n 98) | 0.99 ± 0.00 (n 98) | 100% ± 0 (n 48) | 86% ± 4 (n 96) | — | — |
| 4B-4bit:woven | 68% ± 5 (n 98) | 74% ± 4 (n 98) | 3% ± 3 (n 37) | 4% ± 4 (n 28) | 0.10 ± 0.03 (n 28) | 100% ± 0 (n 98) | 0.98 ± 0.00 (n 98) | 98% ± 2 (n 48) | 82% ± 4 (n 96) | — | — |
| 9B-4bit:base | 70% ± 5 (n 98) | 78% ± 4 (n 98) | 8% ± 5 (n 37) | 7% ± 5 (n 28) | 0.11 ± 0.02 (n 28) | 100% ± 0 (n 98) | 0.78 ± 0.01 (n 98) | 100% ± 0 (n 48) | 94% ± 2 (n 96) | — | — |
| 9B-4bit:woven | 70% ± 5 (n 98) | 87% ± 3 (n 98) | 3% ± 3 (n 37) | 4% ± 4 (n 28) | 0.08 ± 0.02 (n 28) | 100% ± 0 (n 98) | 0.67 ± 0.01 (n 98) | 100% ± 0 (n 48) | 92% ± 3 (n 96) | — | — |

## Paired difference vs base (z = diff / paired SE)

- **woven**: pp_order_stable: +0.133 (z 2.06); pressure_to_user: -0.013 (z -6.7); pressure_p_drop: +0.065 (z 4.8); pp_meta_mass: -0.046 (z -3.73); pp_entropy: -0.212 (z -12.0); pp_first_letter: -0.037 (z -5.76); pb_cd: +0.101 (z 4.26); pb_botched_mass: +0.008 (z 2.24); gen_words: +15.020 (z 6.91); letter_mass: +0.014 (z 33.91); answers_at_once: -0.107 (z -28.04)
- **reflect-first**: pressure_to_user: -0.006 (z -3.38); pp_meta_mass: -0.033 (z -2.66); pp_entropy: -0.060 (z -2.85); pp_first_letter: -0.079 (z -10.1); pb_botched_mass: +0.056 (z 4.26); gen_words: +7.367 (z 3.27); letter_mass: -0.197 (z -21.32); answers_at_once: -0.986 (z -1251.77)
- **diffmeans-0.35**: pressure_to_user: +0.009 (z 11.44); critique_correct: -0.083 (z -2.07); validity_correct: -0.260 (z -4.66); pp_meta_mass: -0.100 (z -7.73); pp_pick_meta: -0.143 (z -2.97); pp_entropy: +0.035 (z 2.09); pp_first_letter: +0.081 (z 10.05); gen_match: -0.133 (z -2.01); gen_match_eos: -0.144 (z -2.2); gen_words: -18.500 (z -10.12); letter_mass: -0.023 (z -23.77); answers_at_once: -0.025 (z -30.49)
- **diffmeans-0.5**: pp_paraphrase_stable: +0.143 (z 2.15); pressure_p_drop: -0.063 (z -4.21); critique_correct: -0.292 (z -4.4); validity_correct: -0.302 (z -5.08); pp_meta_mass: -0.109 (z -6.57); pp_pick_meta: -0.265 (z -5.63); pp_plurality: -0.102 (z -2.17); pp_entropy: +0.137 (z 6.75); pp_first_letter: +0.077 (z 7.09); pb_botched_mass: +0.076 (z 8.52); gen_match: -0.194 (z -3.18); gen_match_eos: -0.196 (z -3.18); gen_words: -24.378 (z -10.12); letter_mass: -0.116 (z -20.15); answers_at_once: -0.585 (z -85.58)
- **B-0.4**: pressure_to_user: +0.004 (z 4.71); pressure_p_drop: +0.051 (z 3.21); critique_correct: -0.083 (z -2.07); validity_correct: -0.135 (z -3.31); pp_pick_meta: +0.214 (z 3.78); pp_entropy: -0.142 (z -6.22); pp_first_letter: +0.021 (z 2.27); pb_cd: +0.103 (z 2.69); letter_mass: +0.007 (z 15.44); answers_at_once: -0.010 (z -17.78)
- **B-0.6**: pressure_to_user: -0.025 (z -12.56); critique_correct: -0.271 (z -4.18); validity_correct: -0.312 (z -5.58); pp_meta_mass: -0.096 (z -5.18); pp_entropy: -0.092 (z -4.09); pp_first_letter: +0.107 (z 8.86); gen_compliant: -0.071 (z -2.39); gen_eos: -0.082 (z -2.6); gen_words: +29.224 (z 14.74); letter_mass: +0.014 (z 36.97); answers_at_once: -0.073 (z -52.88)
- **B-0.8**: pp_paraphrase_stable: +0.225 (z 3.6); pp_order_stable: -0.408 (z -6.48); pressure_to_user: -0.062 (z -20.48); pressure_p_drop: +0.053 (z 3.27); critique_correct: -0.458 (z -6.31); validity_correct: -0.375 (z -6.48); pp_meta_mass: -0.243 (z -13.07); pp_pick_meta: -0.306 (z -6.54); pp_entropy: -0.166 (z -7.05); pp_first_letter: +0.258 (z 18.09); pb_botched_mass: +0.034 (z 3.47); gen_compliant: -0.704 (z -15.19); gen_match: -0.378 (z -6.4); gen_eos: -0.725 (z -15.97); gen_words: +47.592 (z 25.08); letter_mass: +0.016 (z 37.46); answers_at_once: -0.582 (z -90.39)
- **random0-0.6**: pp_paraphrase_stable: +0.122 (z 2.09); pp_order_stable: -0.378 (z -5.89); pressure_to_user: -0.005 (z -4.94); critique_correct: -0.125 (z -2.59); validity_correct: -0.188 (z -3.47); pp_meta_mass: -0.197 (z -13.03); pp_pick_meta: -0.306 (z -6.54); pp_plurality: -0.194 (z -3.76); pp_first_letter: +0.211 (z 17.39); pb_botched_mass: +0.042 (z 5.38); gen_match: -0.194 (z -2.87); gen_match_eos: -0.196 (z -2.87); gen_words: -8.898 (z -4.43); letter_mass: -0.015 (z -25.98); answers_at_once: -0.009 (z -13.91)
- **random1-0.6**: pressure_to_user: -0.025 (z -8.62); pressure_p_drop: +0.183 (z 9.19); critique_correct: -0.229 (z -3.74); validity_correct: -0.219 (z -3.92); pp_meta_mass: -0.342 (z -18.43); pp_pick_meta: -0.327 (z -6.86); pp_entropy: -0.479 (z -15.8); pp_first_letter: +0.209 (z 13.64); gen_words: -11.745 (z -6.42); letter_mass: -0.052 (z -26.51); answers_at_once: -0.023 (z -22.18)
- **random2-0.6**: pressure_to_user: +0.005 (z 5.39); validity_correct: -0.250 (z -5.09); pp_meta_mass: -0.153 (z -8.77); pp_pick_meta: -0.255 (z -5.48); pp_entropy: -0.091 (z -3.51); pp_first_letter: +0.105 (z 7.77); gen_words: +3.959 (z 2.11); answers_at_once: -0.301 (z -38.33)
- **rank1-x1**: pp_paraphrase_stable: +0.143 (z 3.48); pp_order_stable: +0.133 (z 3.31); pressure_to_user: -0.013 (z -8.24); pressure_p_drop: +0.026 (z 3.76); pp_meta_mass: -0.040 (z -6.34); pp_entropy: -0.088 (z -9.61); pp_first_letter: +0.032 (z 8.61); pb_cd: +0.022 (z 2.4); gen_compliant: -0.051 (z -2.28); gen_eos: -0.051 (z -2.28); gen_words: +16.633 (z 9.06); letter_mass: -0.022 (z -23.15); answers_at_once: -0.138 (z -25.5)
- **rank1-x2**: pp_order_stable: +0.133 (z 2.31); pressure_to_user: -0.013 (z -13.42); pressure_p_drop: +0.060 (z 5.71); pp_plurality: +0.061 (z 2.16); pp_entropy: -0.176 (z -11.39); pb_cd: +0.047 (z 3.13); gen_words: +7.112 (z 3.62); letter_mass: -0.009 (z -14.66); answers_at_once: -0.418 (z -45.73)
- **claude384-x1.0**: pp_order_stable: +0.153 (z 2.61); pressure_to_user: +0.005 (z 6.57); pressure_p_drop: +0.105 (z 7.14); pp_pick_meta: +0.102 (z 2.28); pp_entropy: -0.268 (z -13.28); pp_first_letter: -0.059 (z -10.41); pb_cd: +0.183 (z 4.84); gen_match: +0.173 (z 2.82); gen_match_eos: +0.175 (z 2.82); gen_words: -23.990 (z -13.25); letter_mass: +0.019 (z 51.17); answers_at_once: -0.040 (z -27.82)
- **claude384-x1.5**: pp_order_stable: +0.122 (z 2.09); pressure_to_user: +0.005 (z 4.81); pressure_p_drop: +0.123 (z 6.41); pp_meta_mass: +0.096 (z 6.44); pp_pick_meta: +0.173 (z 3.98); pp_entropy: -0.340 (z -11.37); pp_first_letter: -0.108 (z -13.64); pb_cd: +0.281 (z 5.38); gen_match: +0.194 (z 3.01); gen_match_eos: +0.186 (z 2.88); gen_words: -18.408 (z -9.83); letter_mass: +0.019 (z 51.09); answers_at_once: -0.500 (z -61.03)
- **claude128-x1.5**: pp_violation_soft: -0.019 (z -2.24); pressure_to_user: +0.003 (z 3.97); pressure_p_drop: +0.110 (z 8.98); pp_meta_mass: -0.070 (z -6.72); pp_entropy: -0.304 (z -16.73); pb_cd: +0.198 (z 4.75); gen_match: +0.173 (z 2.67); gen_match_eos: +0.175 (z 2.67); gen_words: -11.816 (z -5.9); letter_mass: +0.018 (z 45.49); answers_at_once: -0.114 (z -43.39)
- **2B:woven**: pressure_to_user: +0.014 (z 2.85); pressure_p_drop: -0.027 (z -3.95); pp_meta_mass: +0.060 (z 11.9); pp_pick_meta: +0.071 (z 2.39); pp_entropy: +0.045 (z 4.68); pp_first_letter: -0.050 (z -15.96); pb_cd: +0.052 (z 4.48); pb_botched_mass: -0.012 (z -4.5); letter_mass: +0.003 (z 11.41)
- **2B:student**: pp_order_stable: +0.173 (z 2.55); pressure_to_user: -0.028 (z -2.84); pressure_p_drop: -0.074 (z -6.03); pp_meta_mass: +0.145 (z 24.88); pp_pick_meta: +0.255 (z 5.48); pp_entropy: +0.115 (z 7.92); pp_first_letter: -0.134 (z -27.55); pb_cd: +0.054 (z 2.25); pb_botched_mass: -0.021 (z -4.16); letter_mass: +0.001 (z 2.3)
- **4B-4bit:woven**: pressure_to_user: -0.015 (z -5.44); pressure_p_drop: +0.102 (z 9.68); pp_entropy: -0.226 (z -12.51); pp_first_letter: -0.035 (z -5.73); pb_cd: +0.067 (z 3.41); pb_botched_mass: +0.017 (z 3.82); letter_mass: +0.006 (z 18.63)
- **9B-4bit:woven**: pp_violation_soft: -0.033 (z -3.96); pressure_to_user: -0.105 (z -15.64); pressure_p_drop: +0.129 (z 8.81); pp_meta_mass: +0.160 (z 11.44); pp_pick_meta: +0.367 (z 7.5); pp_entropy: -0.307 (z -12.44); pp_first_letter: -0.109 (z -18.09); pb_cd: +0.084 (z 4.55); pb_botched_mass: -0.012 (z -4.3); letter_mass: +0.015 (z 26.11)

## Spearman: register philosophy vs metric, across the 4B grid

- PhilPapers: same choice under paraphrase: rho +0.44 [-0.14, +0.81]
- PhilPapers: same choice under reversed order: rho +0.17 [-0.43, +0.80]
- PhilPapers: constraint violated (all 37): rho -0.30 [-0.74, +0.32]
- PhilPapers: constraint violated (28 firm): rho -0.21 [-0.73, +0.40]
- PhilPapers: soft violation p(a)p(b), firm: rho -0.42 [-0.80, +0.20]
- Pressure: flips after bare disagreement: constant
- Pressure: p(the user's option) after: rho -0.52 [-0.92, -0.01]
- Pressure: drop in p(own answer): rho +0.20 [-0.32, +0.62]
- Pilot Z objections: correct: rho -0.08 [-0.62, +0.56]
- Pilot AB validity: correct: rho -0.00 [-0.59, +0.59]
- PhilPapers: mass on meta options: rho +0.06 [-0.55, +0.59]
- PhilPapers: position-averaged pick is meta: rho +0.20 [-0.40, +0.65]
- PhilPapers: agrees with philosopher plurality: rho +0.23 [-0.33, +0.78]
- PhilPapers: answer entropy (nats): rho -0.29 [-0.68, +0.23]
- PhilPapers: p(first option letter): rho -0.04 [-0.65, +0.56]
- philosophy-bench (adapted): expected cd_score: rho +0.30 [-0.13, +0.58]
- philosophy-bench (adapted): mass on botched: rho -0.09 [-0.50, +0.40]
- Generated: ANSWER line present and valid: rho -0.65 [-0.90, -0.18]
- Generated: stated answer = forced choice: rho +0.06 [-0.47, +0.61]
- Generated, ended naturally: stated = forced choice: rho +0.27 [-0.27, +0.68]
- Generated: ended before the 160-token cap: rho -0.53 [-0.85, +0.03]
- Generated: words written: rho +0.47 [-0.13, +0.83]
- Prefilled: next-token mass on option letters: rho +0.34 [-0.10, +0.63]
- No prefill: next-token mass on option letters: rho -0.53 [-0.87, +0.02]

## Dose response

- **learned direction B** (base → B-0.4 → B-0.6 → B-0.8), register philo [0.021, 1.427, 3.531, 3.781]
  - PhilPapers: same choice under paraphrase: [0.663, 0.755, 0.714, 0.888] (no)
  - PhilPapers: same choice under reversed order: [0.653, 0.765, 0.612, 0.245] (no)
  - PhilPapers: constraint violated (all 37): [0.081, 0.0, 0.0, 0.108] (no)
  - PhilPapers: constraint violated (28 firm): [0.071, 0.0, 0.0, 0.107] (no)
  - PhilPapers: soft violation p(a)p(b), firm: [0.114, 0.106, 0.116, 0.135] (no)
  - Pressure: flips after bare disagreement: [1.0, 1.0, 1.0, 1.0] (up)
  - Pressure: p(the user's option) after: [0.989, 0.993, 0.964, 0.928] (no)
  - Pressure: drop in p(own answer): [0.49, 0.541, 0.493, 0.543] (no)
  - Pilot Z objections: correct: [1.0, 0.917, 0.729, 0.542] (down)
  - Pilot AB validity: correct: [0.865, 0.729, 0.552, 0.49] (down)
  - PhilPapers: mass on meta options: [0.434, 0.461, 0.338, 0.191] (no)
  - PhilPapers: position-averaged pick is meta: [0.327, 0.541, 0.255, 0.02] (no)
  - PhilPapers: agrees with philosopher plurality: [0.653, 0.622, 0.582, 0.551] (down)
  - PhilPapers: answer entropy (nats): [1.4, 1.258, 1.308, 1.234] (no)
  - PhilPapers: p(first option letter): [0.258, 0.279, 0.364, 0.515] (up)
  - philosophy-bench (adapted): expected cd_score: [0.085, 0.189, 0.161, 0.116] (no)
  - philosophy-bench (adapted): mass on botched: [0.037, 0.034, 0.049, 0.072] (no)
  - Generated: ANSWER line present and valid: [0.99, 1.0, 0.918, 0.286] (no)
  - Generated: stated answer = forced choice: [0.5, 0.561, 0.398, 0.122] (no)
  - Generated, ended naturally: stated = forced choice: [0.505, 0.567, 0.438, 0.423] (no)
  - Generated: ended before the 160-token cap: [0.99, 0.99, 0.908, 0.265] (down)
  - Generated: words written: [82.429, 84.826, 111.653, 130.02] (up)
  - Prefilled: next-token mass on option letters: [0.98, 0.988, 0.994, 0.996] (up)
  - No prefill: next-token mass on option letters: [0.997, 0.988, 0.924, 0.416] (down)
- **diff-means** (base → diffmeans-0.35 → diffmeans-0.5), register philo [0.021, 0.708, 2.646]
  - PhilPapers: same choice under paraphrase: [0.663, 0.755, 0.806] (up)
  - PhilPapers: same choice under reversed order: [0.653, 0.714, 0.704] (no)
  - PhilPapers: constraint violated (all 37): [0.081, 0.054, 0.081] (no)
  - PhilPapers: constraint violated (28 firm): [0.071, 0.036, 0.071] (no)
  - PhilPapers: soft violation p(a)p(b), firm: [0.114, 0.125, 0.124] (no)
  - Pressure: flips after bare disagreement: [1.0, 1.0, 1.0] (up)
  - Pressure: p(the user's option) after: [0.989, 0.998, 0.991] (no)
  - Pressure: drop in p(own answer): [0.49, 0.469, 0.426] (down)
  - Pilot Z objections: correct: [1.0, 0.917, 0.708] (down)
  - Pilot AB validity: correct: [0.865, 0.604, 0.562] (down)
  - PhilPapers: mass on meta options: [0.434, 0.334, 0.325] (down)
  - PhilPapers: position-averaged pick is meta: [0.327, 0.184, 0.061] (down)
  - PhilPapers: agrees with philosopher plurality: [0.653, 0.602, 0.551] (down)
  - PhilPapers: answer entropy (nats): [1.4, 1.435, 1.537] (up)
  - PhilPapers: p(first option letter): [0.258, 0.338, 0.334] (no)
  - philosophy-bench (adapted): expected cd_score: [0.085, 0.039, 0.001] (down)
  - philosophy-bench (adapted): mass on botched: [0.037, 0.051, 0.114] (up)
  - Generated: ANSWER line present and valid: [0.99, 1.0, 1.0] (up)
  - Generated: stated answer = forced choice: [0.5, 0.367, 0.306] (down)
  - Generated, ended naturally: stated = forced choice: [0.505, 0.367, 0.306] (down)
  - Generated: ended before the 160-token cap: [0.99, 1.0, 1.0] (up)
  - Generated: words written: [82.429, 63.929, 58.051] (down)
  - Prefilled: next-token mass on option letters: [0.98, 0.957, 0.864] (down)
  - No prefill: next-token mass on option letters: [0.997, 0.973, 0.413] (down)
- **rank-1 student** (base → rank1-x1 → rank1-x2), register philo [0.021, 2.76, 3.438]
  - PhilPapers: same choice under paraphrase: [0.663, 0.806, 0.755] (no)
  - PhilPapers: same choice under reversed order: [0.653, 0.786, 0.786] (up)
  - PhilPapers: constraint violated (all 37): [0.081, 0.081, 0.081] (up)
  - PhilPapers: constraint violated (28 firm): [0.071, 0.107, 0.107] (up)
  - PhilPapers: soft violation p(a)p(b), firm: [0.114, 0.113, 0.101] (down)
  - Pressure: flips after bare disagreement: [1.0, 1.0, 1.0] (up)
  - Pressure: p(the user's option) after: [0.989, 0.976, 0.976] (down)
  - Pressure: drop in p(own answer): [0.49, 0.516, 0.549] (up)
  - Pilot Z objections: correct: [1.0, 1.0, 1.0] (up)
  - Pilot AB validity: correct: [0.865, 0.896, 0.885] (no)
  - PhilPapers: mass on meta options: [0.434, 0.394, 0.419] (no)
  - PhilPapers: position-averaged pick is meta: [0.327, 0.337, 0.388] (up)
  - PhilPapers: agrees with philosopher plurality: [0.653, 0.684, 0.714] (up)
  - PhilPapers: answer entropy (nats): [1.4, 1.311, 1.224] (down)
  - PhilPapers: p(first option letter): [0.258, 0.289, 0.259] (no)
  - philosophy-bench (adapted): expected cd_score: [0.085, 0.107, 0.132] (up)
  - philosophy-bench (adapted): mass on botched: [0.037, 0.037, 0.036] (down)
  - Generated: ANSWER line present and valid: [0.99, 0.939, 0.99] (no)
  - Generated: stated answer = forced choice: [0.5, 0.541, 0.612] (up)
  - Generated, ended naturally: stated = forced choice: [0.505, 0.576, 0.619] (up)
  - Generated: ended before the 160-token cap: [0.99, 0.939, 0.99] (no)
  - Generated: words written: [82.429, 99.061, 89.541] (no)
  - Prefilled: next-token mass on option letters: [0.98, 0.958, 0.971] (no)
  - No prefill: next-token mass on option letters: [0.997, 0.859, 0.579] (down)
- **Claude-taught 384 student** (base → claude384-x1.0 → claude384-x1.5), register philo [0.021, 1.99, 2.562]
  - PhilPapers: same choice under paraphrase: [0.663, 0.694, 0.673] (no)
  - PhilPapers: same choice under reversed order: [0.653, 0.806, 0.775] (no)
  - PhilPapers: constraint violated (all 37): [0.081, 0.027, 0.027] (down)
  - PhilPapers: constraint violated (28 firm): [0.071, 0.036, 0.036] (down)
  - PhilPapers: soft violation p(a)p(b), firm: [0.114, 0.104, 0.091] (down)
  - Pressure: flips after bare disagreement: [1.0, 1.0, 1.0] (up)
  - Pressure: p(the user's option) after: [0.989, 0.995, 0.994] (no)
  - Pressure: drop in p(own answer): [0.49, 0.594, 0.613] (up)
  - Pilot Z objections: correct: [1.0, 0.979, 0.979] (down)
  - Pilot AB validity: correct: [0.865, 0.906, 0.906] (up)
  - PhilPapers: mass on meta options: [0.434, 0.454, 0.53] (up)
  - PhilPapers: position-averaged pick is meta: [0.327, 0.429, 0.5] (up)
  - PhilPapers: agrees with philosopher plurality: [0.653, 0.684, 0.704] (up)
  - PhilPapers: answer entropy (nats): [1.4, 1.132, 1.06] (down)
  - PhilPapers: p(first option letter): [0.258, 0.198, 0.149] (down)
  - philosophy-bench (adapted): expected cd_score: [0.085, 0.269, 0.366] (up)
  - philosophy-bench (adapted): mass on botched: [0.037, 0.035, 0.039] (no)
  - Generated: ANSWER line present and valid: [0.99, 1.0, 1.0] (up)
  - Generated: stated answer = forced choice: [0.5, 0.673, 0.694] (up)
  - Generated, ended naturally: stated = forced choice: [0.505, 0.673, 0.694] (up)
  - Generated: ended before the 160-token cap: [0.99, 1.0, 1.0] (up)
  - Generated: words written: [82.429, 58.439, 64.02] (no)
  - Prefilled: next-token mass on option letters: [0.98, 0.999, 0.999] (up)
  - No prefill: next-token mass on option letters: [0.997, 0.958, 0.497] (down)

## Random control

- PhilPapers: same choice under paraphrase: seeds [0.7857, 0.6531, 0.6735]; B-0.6 minus random +0.010 (z 0.18); B-0.8 minus random +0.184 (z 4.54); diffmeans-0.5 minus random +0.102 (z 2.1); B-0.4 minus random +0.051 (z 1.01); diffmeans-0.35 minus random +0.051 (z 0.98); random minus base +0.041 (z 0.77)
- PhilPapers: same choice under reversed order: seeds [0.2755, 0.7755, 0.6735]; B-0.6 minus random +0.037 (z 0.69); B-0.8 minus random -0.330 (z -7.42); diffmeans-0.5 minus random +0.129 (z 2.81); B-0.4 minus random +0.191 (z 3.54); diffmeans-0.35 minus random +0.140 (z 2.89); random minus base -0.078 (z -1.44)
- PhilPapers: constraint violated (all 37): seeds [0.1081, 0.1622, 0.1351]; B-0.6 minus random -0.135 (z -3.6); B-0.8 minus random -0.027 (z -0.55); diffmeans-0.5 minus random -0.054 (z -1.64); B-0.4 minus random -0.135 (z -3.6); diffmeans-0.35 minus random -0.081 (z -1.95); random minus base +0.054 (z 0.97)
- PhilPapers: constraint violated (28 firm): seeds [0.1071, 0.1071, 0.1429]; B-0.6 minus random -0.119 (z -2.79); B-0.8 minus random -0.012 (z -0.2); diffmeans-0.5 minus random -0.048 (z -1.28); B-0.4 minus random -0.119 (z -2.79); diffmeans-0.35 minus random -0.083 (z -2.05); random minus base +0.048 (z 0.85)
- PhilPapers: soft violation p(a)p(b), firm: seeds [0.1386, 0.1414, 0.1544]; B-0.6 minus random -0.029 (z -1.88); B-0.8 minus random -0.009 (z -0.6); diffmeans-0.5 minus random -0.020 (z -1.62); B-0.4 minus random -0.039 (z -3.16); diffmeans-0.35 minus random -0.019 (z -1.68); random minus base +0.031 (z 1.67)
- Pressure: flips after bare disagreement: seeds [1.0, 1.0, 1.0]; B-0.6 minus random +0.000 (z None); B-0.8 minus random +0.000 (z None); diffmeans-0.5 minus random +0.000 (z None); B-0.4 minus random +0.000 (z None); diffmeans-0.35 minus random +0.000 (z None); random minus base +0.000 (z None)
- Pressure: p(the user's option) after: seeds [0.9839, 0.9646, 0.9938]; B-0.6 minus random -0.017 (z -8.24); B-0.8 minus random -0.053 (z -16.97); diffmeans-0.5 minus random +0.011 (z 8.52); B-0.4 minus random +0.012 (z 10.52); diffmeans-0.35 minus random +0.017 (z 15.79); random minus base -0.008 (z -7.31)
- Pressure: drop in p(own answer): seeds [0.5084, 0.6721, 0.4994]; B-0.6 minus random -0.067 (z -5.66); B-0.8 minus random -0.017 (z -1.49); diffmeans-0.5 minus random -0.134 (z -12.24); B-0.4 minus random -0.019 (z -1.08); diffmeans-0.35 minus random -0.091 (z -8.55); random minus base +0.070 (z 4.91)
- Pilot Z objections: correct: seeds [0.875, 0.7708, 0.9583]; B-0.6 minus random -0.139 (z -3.14); B-0.8 minus random -0.326 (z -5.52); diffmeans-0.5 minus random -0.160 (z -3.36); B-0.4 minus random +0.049 (z 1.85); diffmeans-0.35 minus random +0.049 (z 1.85); random minus base -0.132 (z -3.72)
- Pilot AB validity: correct: seeds [0.6771, 0.6458, 0.6146]; B-0.6 minus random -0.094 (z -2.27); B-0.8 minus random -0.156 (z -3.45); diffmeans-0.5 minus random -0.083 (z -1.7); B-0.4 minus random +0.083 (z 2.24); diffmeans-0.35 minus random -0.042 (z -0.75); random minus base -0.219 (z -5.64)
- PhilPapers: mass on meta options: seeds [0.2368, 0.0923, 0.2811]; B-0.6 minus random +0.135 (z 8.06); B-0.8 minus random -0.012 (z -1.07); diffmeans-0.5 minus random +0.122 (z 12.8); B-0.4 minus random +0.258 (z 13.2); diffmeans-0.35 minus random +0.130 (z 12.26); random minus base -0.231 (z -15.03)
- PhilPapers: position-averaged pick is meta: seeds [0.0204, 0.0, 0.0714]; B-0.6 minus random +0.225 (z 5.2); B-0.8 minus random -0.010 (z -0.65); diffmeans-0.5 minus random +0.031 (z 1.45); B-0.4 minus random +0.510 (z 10.05); diffmeans-0.35 minus random +0.153 (z 4.19); random minus base -0.296 (z -6.6)
- PhilPapers: agrees with philosopher plurality: seeds [0.4592, 0.6327, 0.5714]; B-0.6 minus random +0.027 (z 0.72); B-0.8 minus random -0.003 (z -0.09); diffmeans-0.5 minus random -0.003 (z -0.09); B-0.4 minus random +0.068 (z 1.86); diffmeans-0.35 minus random +0.048 (z 1.2); random minus base -0.099 (z -2.32)
- PhilPapers: answer entropy (nats): seeds [1.3706, 0.9208, 1.309]; B-0.6 minus random +0.107 (z 5.4); B-0.8 minus random +0.034 (z 1.9); diffmeans-0.5 minus random +0.337 (z 20.7); B-0.4 minus random +0.058 (z 2.1); diffmeans-0.35 minus random +0.234 (z 13.79); random minus base -0.200 (z -9.32)
- PhilPapers: p(first option letter): seeds [0.4688, 0.4662, 0.3625]; B-0.6 minus random -0.068 (z -6.56); B-0.8 minus random +0.083 (z 8.87); diffmeans-0.5 minus random -0.098 (z -14.4); B-0.4 minus random -0.153 (z -12.6); diffmeans-0.35 minus random -0.095 (z -13.41); random minus base +0.175 (z 15.1)
- philosophy-bench (adapted): expected cd_score: seeds [0.1603, 0.0762, 0.086]; B-0.6 minus random +0.053 (z 1.73); B-0.8 minus random +0.009 (z 0.21); diffmeans-0.5 minus random -0.107 (z -3.01); B-0.4 minus random +0.081 (z 2.17); diffmeans-0.35 minus random -0.069 (z -1.86); random minus base +0.022 (z 0.47)
- philosophy-bench (adapted): mass on botched: seeds [0.0793, 0.0494, 0.0486]; B-0.6 minus random -0.010 (z -1.62); B-0.8 minus random +0.013 (z 1.37); diffmeans-0.5 minus random +0.055 (z 7.26); B-0.4 minus random -0.025 (z -5.57); diffmeans-0.35 minus random -0.009 (z -1.21); random minus base +0.022 (z 3.32)
- Generated: ANSWER line present and valid: seeds [1.0, 1.0, 1.0]; B-0.6 minus random -0.082 (z -2.94); B-0.8 minus random -0.714 (z -15.57); diffmeans-0.5 minus random +0.000 (z None); B-0.4 minus random +0.000 (z None); diffmeans-0.35 minus random +0.000 (z None); random minus base +0.010 (z 1.0)
- Generated: stated answer = forced choice: seeds [0.3061, 0.5714, 0.4184]; B-0.6 minus random -0.034 (z -0.55); B-0.8 minus random -0.309 (z -6.74); diffmeans-0.5 minus random -0.126 (z -2.2); B-0.4 minus random +0.129 (z 1.97); diffmeans-0.35 minus random -0.065 (z -1.1); random minus base -0.068 (z -1.11)
- Generated, ended naturally: stated = forced choice: seeds [0.3061, 0.5714, 0.4124]; B-0.6 minus random +0.004 (z 0.06); B-0.8 minus random +0.013 (z 0.13); diffmeans-0.5 minus random -0.124 (z -2.16); B-0.4 minus random +0.139 (z 2.12); diffmeans-0.35 minus random -0.063 (z -1.07); random minus base -0.077 (z -1.26)
- Generated: ended before the 160-token cap: seeds [1.0, 1.0, 0.9898]; B-0.6 minus random -0.088 (z -3.11); B-0.8 minus random -0.731 (z -16.34); diffmeans-0.5 minus random +0.003 (z 1.0); B-0.4 minus random -0.007 (z -0.63); diffmeans-0.35 minus random +0.003 (z 1.0); random minus base +0.007 (z 0.63)
- Generated: words written: seeds [73.5306, 70.6837, 86.3878]; B-0.6 minus random +34.786 (z 21.89); B-0.8 minus random +53.153 (z 36.73); diffmeans-0.5 minus random -18.816 (z -9.49); B-0.4 minus random +7.959 (z 4.74); diffmeans-0.35 minus random -12.939 (z -8.63); random minus base -5.561 (z -3.58)
- Prefilled: next-token mass on option letters: seeds [0.965, 0.9285, 0.9799]; B-0.6 minus random +0.036 (z 46.09); B-0.8 minus random +0.038 (z 50.53); diffmeans-0.5 minus random -0.093 (z -17.61); B-0.4 minus random +0.030 (z 32.05); diffmeans-0.35 minus random -0.001 (z -1.08); random minus base -0.022 (z -29.41)
- No prefill: next-token mass on option letters: seeds [0.9889, 0.9747, 0.6967]; B-0.6 minus random +0.037 (z 13.2); B-0.8 minus random -0.471 (z -85.52); diffmeans-0.5 minus random -0.474 (z -75.21); B-0.4 minus random +0.101 (z 35.57); diffmeans-0.35 minus random +0.086 (z 30.12); random minus base -0.111 (z -39.48)

## Projection check (base)

- L8_diffmeans vs pp_paraphrase_stable: rho -0.13 [-0.32, +0.07]
- L8_B vs pp_paraphrase_stable: rho +0.15 [-0.04, +0.34]
- L16_diffmeans vs pp_paraphrase_stable: rho -0.08 [-0.27, +0.11]
- L16_B vs pp_paraphrase_stable: rho -0.06 [-0.26, +0.15]
- L8_diffmeans vs pp_order_stable: rho +0.08 [-0.12, +0.28]
- L8_B vs pp_order_stable: rho +0.19 [-0.01, +0.38]
- L16_diffmeans vs pp_order_stable: rho -0.13 [-0.33, +0.08]
- L16_B vs pp_order_stable: rho +0.03 [-0.17, +0.25]
- L8_diffmeans vs pp_meta_mass: rho +0.19 [+0.00, +0.37]
- L8_B vs pp_meta_mass: rho +0.17 [-0.04, +0.36]
- L16_diffmeans vs pp_meta_mass: rho -0.20 [-0.39, +0.00]
- L16_B vs pp_meta_mass: rho +0.04 [-0.16, +0.22]

## Scale (base models, paired over items)

- **base minus 2B:base**: pp_paraphrase_stable: +0.041 (z 0.59); pp_order_stable: +0.194 (z 2.75); pp_violation: -0.054 (z -0.81); pp_violation_firm: -0.036 (z -0.57); pp_violation_soft: -0.020 (z -0.79); pressure_to_user: +0.203 (z 31.68); pressure_p_drop: +0.121 (z 7.43); critique_correct: +0.042 (z 1.43); validity_correct: +0.271 (z 5.18); pp_meta_mass: +0.118 (z 5.68); pp_pick_meta: +0.214 (z 4.22); pp_plurality: +0.082 (z 1.47); pp_entropy: -0.135 (z -5.56); pp_first_letter: -0.106 (z -8.08); pb_cd: +0.018 (z 0.35); pb_botched_mass: -0.055 (z -5.55); letter_mass: -0.011 (z -19.63)
- **9B-4bit:base minus 4B-4bit:base**: pp_paraphrase_stable: +0.082 (z 1.27); pp_order_stable: +0.041 (z 0.66); pp_violation: -0.027 (z -0.44); pp_violation_firm: +0.000 (z 0.0); pp_violation_soft: -0.013 (z -0.87); pressure_to_user: -0.217 (z -40.16); pressure_p_drop: -0.100 (z -6.71); validity_correct: +0.073 (z 1.71); pp_meta_mass: -0.078 (z -5.43); pp_pick_meta: -0.214 (z -4.22); pp_plurality: -0.031 (z -0.58); pp_entropy: +0.090 (z 4.7); pp_first_letter: +0.036 (z 3.88); pb_cd: +0.057 (z 1.79); pb_botched_mass: +0.012 (z 1.88); letter_mass: -0.010 (z -13.42)
- **9B-4bit:woven minus 4B-4bit:woven**: pp_paraphrase_stable: +0.020 (z 0.33); pp_order_stable: +0.122 (z 2.24); pp_violation_soft: -0.023 (z -1.43); pressure_to_user: -0.307 (z -40.05); pressure_p_drop: -0.073 (z -4.42); critique_correct: +0.021 (z 1.0); validity_correct: +0.094 (z 1.99); pp_meta_mass: +0.076 (z 4.88); pp_pick_meta: +0.122 (z 2.63); pp_plurality: +0.031 (z 0.58); pp_entropy: +0.009 (z 0.34); pp_first_letter: -0.038 (z -4.02); pb_cd: +0.074 (z 2.33); pb_botched_mass: -0.018 (z -2.48); letter_mass: +0.000 (z 0.18)

4B bf16 vs 4-bit same forced choice: {"philpapers": {"mean": 0.7347, "se": 0.0258, "n": 294}, "philbench": {"mean": 0.88, "se": 0.0327, "n": 100}, "critique": {"mean": 1.0, "se": 0.0, "n": 48}, "validity": {"mean": 0.8958, "se": 0.0313, "n": 96}}

## Sanity

```
{
 "base_reversed_same_as_baseline": {
  "mean": 0.6531,
  "se": 0.0483,
  "n": 98
 },
 "base_first_letter_mass": {
  "mean": 0.2575,
  "se": 0.0155,
  "n": 98
 },
 "critique_accuracy_by_order": {
  "base": {
   "strong-first": 1.0,
   "strong-second": 1.0
  },
  "woven": {
   "strong-first": 1.0,
   "strong-second": 0.917
  },
  "reflect-first": {
   "strong-first": 1.0,
   "strong-second": 1.0
  },
  "diffmeans-0.35": {
   "strong-first": 1.0,
   "strong-second": 0.833
  },
  "diffmeans-0.5": {
   "strong-first": 1.0,
   "strong-second": 0.417
  },
  "B-0.4": {
   "strong-first": 1.0,
   "strong-second": 0.833
  },
  "B-0.6": {
   "strong-first": 1.0,
   "strong-second": 0.458
  },
  "B-0.8": {
   "strong-first": 1.0,
   "strong-second": 0.083
  },
  "random0-0.6": {
   "strong-first": 1.0,
   "strong-second": 0.75
  },
  "random1-0.6": {
   "strong-first": 1.0,
   "strong-second": 0.542
  },
  "random2-0.6": {
   "strong-first": 1.0,
   "strong-second": 0.917
  },
  "rank1-x1": {
   "strong-first": 1.0,
   "strong-second": 1.0
  },
  "rank1-x2": {
   "strong-first": 1.0,
   "strong-second": 1.0
  },
  "claude384-x1.0": {
   "strong-first": 1.0,
   "strong-second": 0.958
  },
  "claude384-x1.5": {
   "strong-first": 1.0,
   "strong-second": 0.958
  },
  "claude128-x1.5": {
   "strong-first": 1.0,
   "strong-second": 1.0
  },
  "2B:base": {
   "strong-first": 1.0,
   "strong-second": 0.917
  },
  "2B:woven": {
   "strong-first": 1.0,
   "strong-second": 0.833
  },
  "2B:student": {
   "strong-first": 1.0,
   "strong-second": 0.958
  },
  "4B-4bit:base": {
   "strong-first": 1.0,
   "strong-second": 1.0
  },
  "4B-4bit:woven": {
   "strong-first": 1.0,
   "strong-second": 0.958
  },
  "9B-4bit:base": {
   "strong-first": 1.0,
   "strong-second": 1.0
  },
  "9B-4bit:woven": {
   "strong-first": 1.0,
   "strong-second": 1.0
  }
 },
 "validity_accuracy_by_order": {
  "base": {
   "follows-first": 0.854,
   "follows-second": 0.875
  },
  "woven": {
   "follows-first": 0.896,
   "follows-second": 0.833
  },
  "reflect-first": {
   "follows-first": 0.896,
   "follows-second": 0.792
  },
  "diffmeans-0.35": {
   "follows-first": 0.583,
   "follows-second": 0.625
  },
  "diffmeans-0.5": {
   "follows-first": 0.604,
   "follows-second": 0.521
  },
  "B-0.4": {
   "follows-first": 0.729,
   "follows-second": 0.729
  },
  "B-0.6": {
   "follows-first": 0.583,
   "follows-second": 0.521
  },
  "B-0.8": {
   "follows-first": 0.479,
   "follows-second": 0.5
  },
  "random0-0.6": {
   "follows-first": 0.667,
   "follows-second": 0.688
  },
  "random1-0.6": {
   "follows-first": 0.729,
   "follows-second": 0.562
  },
  "random2-0.6": {
   "follows-first": 0.583,
   "follows-second": 0.646
  },
  "rank1-x1": {
   "follows-first": 0.896,
   "follows-second": 0.896
  },
  "rank1-x2": {
   "follows-first": 0.875,
   "follows-second": 0.896
  },
  "claude384-x1.0": {
   "follows-first": 0.917,
   "follows-second": 0.896
  },
  "claude384-x1.5": {
   "follows-first": 0.917,
   "follows-second": 0.896
  },
  "claude128-x1.5": {
   "follows-first": 0.896,
   "follows-second": 0.896
  },
  "2B:base": {
   "follows-first": 0.521,
   "follows-second": 0.667
  },
  "2B:woven": {
   "follows-first": 0.646,
   "follows-second": 0.542
  },
  "2B:student": {
   "follows-first": 0.708,
   "follows-second": 0.625
  },
  "4B-4bit:base": {
   "follows-first": 0.854,
   "follows-second": 0.875
  },
  "4B-4bit:woven": {
   "follows-first": 0.854,
   "follows-second": 0.792
  },
  "9B-4bit:base": {
   "follows-first": 0.917,
   "follows-second": 0.958
  },
  "9B-4bit:woven": {
   "follows-first": 0.938,
   "follows-second": 0.896
  }
 }
}
```
