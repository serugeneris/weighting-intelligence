# Human regrade of the revision and transfer pilot

Prepared 16 November 2026. No human grades are recorded here yet.

[Experiment AK](../../../findings/6-can-we-trust-the-scores.md#exp-ak) compared the case author's grades with a second model's. This folder holds what a human grader needs to repeat that check on the 72 labels where grades vary: 18 revision records with three criteria each and 18 transfer records with one.

## For the grader

Open [`grade.html`](grade.html) in a browser. It shows the 36 records in a fixed shuffled order with the configuration names removed, explains the four criteria, and exports a `grades.json` file. Do not open `key.json`, the findings or the earlier grades before exporting.

The blinding has one known leak: in the revision records, every first reply from one configuration runs into its length limit, so a grader can learn to recognise that configuration.

## For the case author

```sh
python3 scripts/human_regrade.py build                      # rebuilds packet.json, key.json and grade.html
python3 scripts/human_regrade.py score path/to/grades.json  # agreement and Cohen's kappa against author and Codex
```

`score` joins the export to [`key.json`](key.json) and to the [paired grades of AK](../2026-11-16/codex-regrade/grades.jsonl). It prints, per criterion, how many labels match each earlier grader, Cohen's κ, and the passes per configuration under all three graders. Labels the grader left empty are listed and left out of the counts.

| File | What |
|---|---|
| `grade.html` | The grading page: `grade.template.html` with `packet.json` inlined |
| `packet.json` | The 36 records as the grader sees them, without configuration names |
| `key.json` | Packet id → set, configuration and case. Keep it away from the grader |

The criteria follow the wording used in the [original review](../2026-11-14/revision-transfer/README.md) and in AK. The grader should not be the person who wrote the cases.
