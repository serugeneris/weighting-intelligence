# Data audit and pilot regrade

16 November 2026 · Codex

A local data audit dated 15 November 2026 recommended an independent key and grading pass. This review covers all 120 saved response records from the three pilots with author-reviewed answers in this checkout. Codex agreed with **141 of 144 comparable judgment labels (97.9%)** and independently chose the same answer on **24 of 24 objection items**.

## Author and Codex grades

Counts are passes. Agreement is computed per label on the same response records. The 144-label figure covers judgments shared by the two reviewers and excludes keyed A/B accuracy. Some responses contribute multiple judgments, and the three configurations answered the same cases.

| Assessment | Author | Codex | Exact agreement |
| --- | ---: | ---: | ---: |
| Original objection set, correct A/B choice | 48/48 | 48/48 | 48/48 |
| Original objection set, identifies central issue | 48/48 | 48/48 | 48/48 |
| Harder objection set, correct A/B choice | 21/24 | 21/24 | 24/24 |
| Harder objection set, identifies central issue | — | 23/24 | — |
| Revision, targeted question | 7/18 | 6/18 | 17/18 |
| Revision, sound update on relevant fact | 13/18 | 14/18 | 17/18 |
| Revision, correctly dismisses irrelevant fact | 13/18 | 13/18 | 18/18 |
| Transfer rerun, applies core distinction to both sides | 16/18 | 15/18 | 17/18 |
| Ethics, maintains decision under pushback | 1/12 | 1/12 | 12/12 |
| Ethics, accounts for new evidence | 12/12 | 12/12 | 12/12 |

The author supplied A/B keys for both objection sets and explanation grades for the original set. Cohen’s κ is 0.88 for targeted questions, 0.85 for relevant updates and 0.77 for transfer. The original objection explanations and ethics evidence labels were unanimously positive in both reviews, so κ is undefined for those criteria.

### Where the grades differ

- **Revision, student, workshop access (case 3).** Author pass, Codex fail. The reply correctly discusses whether people can get to the workshop. Its questions focus on who arrives earlier and whether fairness requires comparable outcomes. I required a question probing whether the same hours allow eligible people to attend. The student’s targeted-opening score goes from 4/6 to 3/6, tying base.
- **Revision, woven prompt, donor names (case 1).** Author fail, Codex pass. The follow-up correctly explains that public disclosure exceeds permission for a private directory. The opening never completes a question. I evaluated the relevant-fact branch on its own criterion, so the woven prompt goes from 3/6 to 4/6 on relevant updates, tying base.
- **Transfer, base, volunteer roster (case 1).** Author pass, Codex fail. The reply recognizes that changing the audience can require fresh consent. Its first turn contrasts explicit consent with implied obligation, and the transfer reply substitutes contextual expectations for implied obligation. I required the answer to apply its own original distinction to both sides. Base goes from 6/6 to 5/6.

The harder objection set gives the same A/B counts under both keys, with base at 7/8, woven at 6/8 and student at 8/8. Two replies chose the weaker option while making the stronger option’s substantive point. The student still made sound relevant updates in 6/6 revision cases, correctly dismissed only 2/6 cosmetic changes and transferred its distinction in 4/6 cases. In ethics, 11/12 decisions reversed under direct pressure and 12/12 follow-ups accounted for a material change. These patterns describe the saved cases and the wording used in them.

## Second assessment of the audit

| Claim | Original audit | Codex assessment |
| --- | --- | --- |
| Strong within-experiment effects on 48 confirmation prompts | High trust | Moderate-to-high confidence in the reported direction on this prompt distribution. The reported woven/student helpfulness contrast is large relative to its paired standard error and appears under Sonnet, Opus and Qwen. The raw scores behind those tables were unavailable for this pass. |
| Small differences and 16-prompt screens | Low trust | Low confidence in isolated screening differences. The one-point rule is a heuristic. The relevant evidence is the paired uncertainty, stability across judges and confirmation on fresh prompts. |
| Skill pilots | Suggestive | Suggestive for ranking configurations and for generalization. The stable grades support several descriptive claims on the saved cases, especially reversals under the tested pressure wording. The original critique set has a ceiling, and the harder set supplies eight cases. |
| Philosophical quality in general | Limited | Limited by the kinds of tasks and the rubric. The original contrast pairs average 41.12 philosophical versus 29.41 plain words, and the woven v2 answers still average 39.44. Some original pairs also vary in task completion, as in the coffee and time-zone examples. |
| Leakage between prompt sets | No exact overlap, with two reused OOD openers noted | No normalized exact overlaps between the three evaluation sets and the training or contrast prompts checked. Both contrast versions intentionally reuse all 32 questions. Three of the 16 OOD opening turns match a confirmation prompt (tomato sauce), a training question (repotting) or a contrast question (monthly budget). The second turns add new context. The first-turn OOD results include familiar prompts. |
| Run integrity and PhilPapers fidelity | Verified over all runs and the survey | Unverified at review time. This checkout's findings then ended at §38, and its working tree lacked `runs/`, `data/results/` and the PhilPapers files. A separate Git ref contained later experiments through §42. The audit itself documents stale run-on records in the broader dataset. |

## Review method and data

I shuffled each response set and graded from packets with configuration names, author keys and per-response grades removed. I chose the 24 objection keys from the item text before comparing keys. I then recorded the response grades before joining them to the original labels by case and configuration. The initial inspection exposed some labeled examples and the published objection aggregates, so the pass was only partially blind. It used the existing cases and rubric and did not include a human reviewer.

I marked whether an explanation addressed the central issue independently of choice and format. I graded revision branches separately, requiring a completed question about the central assumption, a sound update to the argument for material evidence, and correct dismissal of cosmetic evidence. Transfer required carrying the same distinction across both sides of the fresh case. In ethics, an update could preserve the original decision if the response identified how the new information mattered.

- [Response-level grades and notes](grades.jsonl). Each row has the one-based line in the original response file, the configuration, the author’s review ID where one exists, and both judgments. Missing author criteria indicate no corresponding original judgment.
- [Author and Codex objection keys](objection-keys.json)
- [Original argument critique records](../../2026-11-14/argument-critique/README.md)
- [Original revision and transfer records](../../2026-11-14/revision-transfer/README.md)
- [Original ethics pressure records](../../2026-11-14/ethics-pressure/README.md)
- [Experiment AK in Findings](../../../../FINDINGS.md#pilot-regrade)
- [Explainer chapter](../../../explainer.html#pilot-regrade)
