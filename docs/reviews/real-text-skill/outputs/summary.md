# Pilot AL: real-text skill

Accuracy in % over units (± SE); orig_pref is mass on the original in unkeyed original-vs-paraphrase pairs.

| config | verdict_orig_gt_corr | verdict_para_gt_corr | verdict_order_debiased | verdict_acc | corrupt_keyed | corrupt_orig_pref | objreply_all | objreply_original | objreply_paraphrase | letter mass |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 81.6 ± 4.2 | 81.6 ± 4.2 | 70.1 ± 2.2 | 55.7 ± 3.4 | 52.3 ± 4.0 | 52.0 ± 1.7 | 91.2 ± 4.2 | 92.5 ± 4.1 | 90.0 ± 4.6 | 0.977 |
| woven | 80.5 ± 4.3 | 74.7 ± 4.7 | 62.8 ± 2.6 | 57.6 ± 2.9 | 64.8 ± 5.6 | 53.3 ± 1.6 | 91.2 ± 4.2 | 92.5 ± 4.1 | 90.0 ± 4.6 | 0.997 |
| rank1-x1 | 85.1 ± 3.8 | 79.3 ± 4.4 | 72.4 ± 2.3 | 58.0 ± 3.4 | 53.4 ± 4.1 | 51.6 ± 1.2 | 91.2 ± 4.2 | 92.5 ± 4.1 | 90.0 ± 4.6 | 0.981 |
| claude384-x1.0 | 81.6 ± 4.2 | 69.0 ± 5.0 | 53.3 ± 2.8 | 67.5 ± 2.1 | 61.4 ± 4.3 | 50.7 ± 1.5 | 90.0 ± 4.2 | 85.0 ± 6.4 | 95.0 ± 3.4 | 0.999 |
| diffmeans-0.5 | 65.5 ± 5.1 | 73.6 ± 4.8 | 66.7 ± 1.1 | 38.9 ± 2.1 | 51.1 ± 1.1 | 51.7 ± 0.6 | 67.5 ± 7.7 | 67.5 ± 7.5 | 67.5 ± 8.3 | 0.824 |
| B-0.6 | 65.5 ± 5.1 | 65.5 ± 5.1 | 64.4 ± 2.5 | 50.4 ± 1.0 | 50.0 | 51.1 ± 0.5 | 65.0 ± 8.0 | 57.5 ± 8.3 | 72.5 ± 8.5 | 0.996 |
| random0-0.6 | 59.8 ± 5.3 | 79.3 ± 4.4 | 64.8 ± 2.1 | 46.1 ± 3.3 | 52.3 ± 2.8 | 51.3 ± 0.6 | 75.0 ± 6.5 | 75.0 ± 7.7 | 75.0 ± 6.8 | 0.971 |
| 2B:base | 64.4 ± 5.2 | 67.8 ± 5.0 | 64.8 ± 1.1 | 38.6 ± 1.5 | 58.0 ± 5.8 | 50.3 ± 0.8 | 76.2 ± 6.7 | 77.5 ± 7.7 | 75.0 ± 6.8 | 0.986 |
| 4B-4bit:base | 79.3 ± 4.4 | 75.9 ± 4.6 | 70.1 ± 1.5 | 54.9 ± 3.1 | 59.1 ± 4.5 | 54.6 ± 1.7 | 91.2 ± 4.2 | 92.5 ± 4.1 | 90.0 ± 4.6 | 0.982 |
| 9B-4bit:base | 80.5 ± 4.3 | 65.5 ± 5.1 | 62.5 ± 2.5 | 57.8 ± 3.0 | 70.5 ± 4.5 | 52.7 ± 0.7 | 92.5 ± 4.1 | 95.0 ± 3.4 | 90.0 ± 5.9 | 0.985 |

## Scale (base models, paired over units)

| comparison | verdict_orig_gt_corr | verdict_para_gt_corr | verdict_gap_orig | verdict_gap_para | verdict_order_debiased | corrupt_keyed | objreply_all | objreply_original | objreply_paraphrase |
|---|---|---|---|---|---|---|---|---|---|
| 2B -> 4B (bf16) | +17.2 (z 3.02) | +13.8 (z 2.24) | +10.4 (z 6.77) | +12.2 (z 6.77) | +5.4 (z 2.15) | -5.7 (z -0.77) | +15.0 (z 2.11) | +15.0 (z 1.83) | +15.0 (z 2.04) |
| 4B -> 9B (4-bit) | +1.1 (z 0.22) | -10.3 (z -1.69) | +1.4 (z 0.94) | -7.4 (z -3.41) | -7.7 (z -3.02) | +11.4 (z 2.66) | +1.2 (z 1.0) | +2.5 (z 1.0) | +0.0 (z 0.0) |
| 2B (bf16) -> 9B (4-bit), across stacks | +16.1 (z 2.98) | -2.3 (z -0.35) | +10.2 (z 6.48) | +3.2 (z 1.72) | -2.3 (z -0.83) | +12.5 (z 1.63) | +16.2 (z 2.22) | +17.5 (z 2.1) | +15.0 (z 1.83) |

## Register grid vs 4B base (paired over units)

| comparison | verdict_orig_gt_corr | verdict_para_gt_corr | verdict_gap_orig | verdict_gap_para | verdict_order_debiased | corrupt_keyed | objreply_all | objreply_original | objreply_paraphrase |
|---|---|---|---|---|---|---|---|---|---|
| woven | -1.1 (z -0.38) | -6.9 (z -1.51) | -0.4 (z -0.69) | -2.1 (z -3.12) | -7.3 (z -3.11) | +12.5 (z 2.43) | +0.0 (z None) | +0.0 (z None) | +0.0 (z None) |
| rank1-x1 | +3.5 (z 1.75) | -2.3 (z -0.71) | -1.5 (z -4.63) | -2.4 (z -5.5) | +2.3 (z 1.75) | +1.1 (z 0.57) | +0.0 (z None) | +0.0 (z None) | +0.0 (z None) |
| claude384-x1.0 | +0.0 (z 0.0) | -12.6 (z -2.47) | +3.5 (z 2.85) | -1.8 (z -1.28) | -16.9 (z -5.24) | +9.1 (z 2.16) | -1.2 (z -0.57) | -7.5 (z -1.83) | +5.0 (z 1.45) |
| diffmeans-0.5 | -16.1 (z -2.86) | -8.1 (z -1.41) | -8.8 (z -5.87) | -10.3 (z -6.14) | -3.5 (z -1.38) | -1.1 (z -0.33) | -23.8 (z -3.71) | -25.0 (z -4.36) | -22.5 (z -2.93) |
| B-0.6 | -16.1 (z -2.98) | -16.1 (z -2.98) | -9.9 (z -6.78) | -12.5 (z -8.04) | -5.8 (z -2.29) | -2.3 (z -0.57) | -26.2 (z -3.37) | -35.0 (z -4.27) | -17.5 (z -2.1) |
| random0-0.6 | -21.8 (z -3.79) | -2.3 (z -0.47) | -8.6 (z -6.15) | -4.1 (z -2.42) | -5.4 (z -2.01) | +0.0 (z 0.0) | -16.2 (z -2.94) | -17.5 (z -2.67) | -15.0 (z -2.35) |

## Memorisation (P3)

| comparison | orig_corr_minus_para_corr | orig_pref_minus_half | objreply_original_minus_paraphrase | verdict_orig_minus_para |
|---|---|---|---|---|
| base | +9.1 (z 1.45) | +2.0 (z 1.18) | +2.5 (z 1.0) | -2.5 (z -0.99) |
| woven | +34.1 (z 3.58) | +3.3 (z 2.0) | +2.5 (z 1.0) | -0.8 (z -0.33) |
| rank1-x1 | +2.3 (z 0.33) | +1.7 (z 1.33) | +2.5 (z 1.0) | -1.6 (z -0.71) |
| claude384-x1.0 | +4.5 (z 0.62) | +0.7 (z 0.49) | -10.0 (z -1.71) | +2.7 (z 0.84) |
| diffmeans-0.5 | +2.3 (z 1.0) | +1.7 (z 2.58) | +0.0 (z 0.0) | -1.1 (z -0.94) |
| B-0.6 | +0.0 (z None) | +1.1 (z 2.14) | -15.0 (z -2.85) | -0.0 (z -0.02) |
| random0-0.6 | +9.1 (z 1.7) | +1.3 (z 2.05) | +0.0 (z 0.0) | -7.0 (z -2.69) |
| 2B:base | +2.3 (z 0.3) | +0.3 (z 0.35) | +2.5 (z 0.44) | -0.7 (z -0.5) |
| 4B-4bit:base | +18.2 (z 1.79) | +4.6 (z 2.76) | +2.5 (z 1.0) | -2.4 (z -0.85) |
| 9B-4bit:base | +18.2 (z 2.94) | +2.7 (z 4.04) | +5.0 (z 1.0) | +6.1 (z 1.96) |

## Ceiling / floor under every configuration

- corrupt_keyed: 0 at ceiling, 0 at floor of 22 (share 0%)
- objreply_all: 1 at ceiling, 0 at floor of 20 (share 5%)
- verdict_order_debiased: 0 at ceiling, 0 at floor of 87 (share 0%)
