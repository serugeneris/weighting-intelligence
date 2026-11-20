# Human regrade of the revision and transfer pilot

Prepared 16 November 2026. Graded 19 November 2026 by one person, the repository's author.

[Experiment AK](../../../findings/6-can-we-trust-the-scores.md#exp-ak) compared Claude's original grades with a second model's (Codex). This folder holds the first human check on the 72 labels where grades vary: 18 revision records with three criteria each and 18 transfer records with one.

## Result

| Criterion | Human passes | Claude passes | Codex passes | Matches Claude (κ) | Matches Codex (κ) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Targeted question | 13/18 | 7/18 | 6/18 | 12/18 (0.39) | 11/18 (0.32) |
| Relevant fact | 15/18 | 13/18 | 14/18 | 16/18 (0.68) | 17/18 (0.82) |
| Irrelevant fact | 13/18 | 13/18 | 13/18 | 18/18 (1.00) | 18/18 (1.00) |
| Transfer | 16/18 | 16/18 | 15/18 | 16/18 (0.44) | 15/18 (0.31) |
| All 72 labels | | | | 62/72 | 61/72 |

Claude and Codex match each other on 69 of the same 72 labels. The human differs from both on 9 labels where they agree, 8 of them more leniently, six on targeted questions. Passes by configuration for the three criteria that moved, as human / Claude / Codex:

| Criterion | Base | Woven prompt | Student |
| --- | ---: | ---: | ---: |
| Targeted question | 6 / 3 / 3 | 1 / 0 / 0 | 6 / 4 / 3 |
| Relevant fact | 5 / 4 / 4 | 4 / 3 / 4 | 6 / 6 / 6 |
| Transfer | 6 / 6 / 5 | 5 / 6 / 6 | 5 / 4 / 4 |

The irrelevant-fact labels are identical across all three graders (base 6, woven prompt 5, student 2 of 6).

## Labels that differ

Twelve labels differ from at least one model. Case numbers are those of [`revision.jsonl`](../2026-11-14/revision-transfer/revision.jsonl), with the assumption each case tests.

| Packet id | Configuration | Case | Criterion | Human | Claude | Codex |
| --- | --- | --- | --- | --- | --- | --- |
| R01 | woven | 1, scope of the permission | Relevant fact | pass | fail | pass |
| R03 | base | 2, independence of evidence | Targeted question | pass | fail | fail |
| R09 | student | 4, improvement versus causal effect | Targeted question | pass | fail | fail |
| R10 | student | 1, scope of the permission | Targeted question | pass | fail | fail |
| R14 | student | 3, formal availability versus practical opportunity | Targeted question | pass | pass | fail |
| R15 | base | 4, improvement versus causal effect | Relevant fact | pass | fail | fail |
| R16 | woven | 2, independence of evidence | Targeted question | pass | fail | fail |
| R17 | base | 1, scope of the permission | Targeted question | pass | fail | fail |
| R18 | base | 5, agreement versus freedom to refuse | Targeted question | pass | fail | fail |
| T02 | base | 1, scope of the permission | Transfer | pass | pass | fail |
| T03 | student | 3, formal availability versus practical opportunity | Transfer | pass | fail | fail |
| T10 | woven | 3, formal availability versus practical opportunity | Transfer | fail | pass | pass |

The grader left two notes, both in [`grades-human-1.json`](grades-human-1.json). On R16 the first reply was cut off at the token cap, and the note says the question is incomplete but points the right way. The page's rule says to fail a cut-off question, so R16 is the one label graded against the page's own wording. Failing it gives 12/18 on targeted questions and changes no conclusion. On R04 the note says the reply assumes a perspective that lets it agree with the original argument from the start.

## Limits

One grader, 18 labels per criterion. The grader is the project's author, knows the hypotheses and the cases, and could probably tell the woven prompt apart: in the revision records every one of its first replies runs into the 96-token cap. The page paraphrased the criteria, so part of the difference from the models may come from wording rather than judgment. Claude's targeted-question grades carry no notes, so their reasons are unknown.

## Files

| File | What |
| --- | --- |
| `grade.html` | The grading page: `grade.template.html` with `packet.json` inlined |
| `packet.json` | The 36 records as the grader saw them, without configuration names |
| `key.json` | Packet id → set, configuration and case. It was kept from the grader until the export |
| `grades-human-1.json` | The grader's export, unchanged |
| `results.json` | Output of `score` on that export: matches, κ and passes per configuration |

## Scoring a new export

```sh
python3 scripts/human_regrade.py build                                                   # rebuilds packet.json, key.json and grade.html
python3 scripts/human_regrade.py score docs/reviews/human-regrade/grades-human-1.json     # or a new export
```

`score` joins an export to [`key.json`](key.json) and to the [paired grades of AK](../2026-11-16/codex-regrade/grades.jsonl), whose `original` field holds Claude's grades. It prints matches and Cohen's κ against Claude and Codex, and passes per configuration. Labels left empty are listed and left out of the counts. A second grader should not have written the cases, and should not open `key.json` or the findings before exporting. `tests/test_human_regrade.py` checks the numbers above against `grades-human-1.json`.

The criteria follow the wording used in the [original review](../2026-11-14/revision-transfer/README.md) and in AK.
