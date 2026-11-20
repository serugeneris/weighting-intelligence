# Part 6 · Can we trust the measurements?

[← Findings overview](../FINDINGS.md) · [Explainer, part 6](https://www.contrasentido.ar/weighting-intelligence/explainer.html#p6)

**Short answer.** Mostly, with qualifications. A non-Claude judge confirms the main contrasts, and the answers that seemed to break at high gain were a harness artefact. But much of the student's helpfulness lead at 120 tokens comes from finishing early, and a judge that sees only the last turn cannot score format or consistency. On the skill pilots, two models (Claude and Codex) agree on 69 of the 72 revision and transfer labels (κ 0.77–0.88), but the first human grader differs from both on 9, mostly by passing more targeted questions, so those comparisons depend on who grades.

| ID | Question | Result |
|---|---|---|
| [U](#exp-u) | Do Claude judges favour the Claude-taught student? | No sign of it. Qwen3.5-4B as judge agrees on philosophy (r 0.73–0.75) and on every key helpfulness contrast, and sees the student's lead as larger. It cannot judge coherence. |
| [Correction 1](#correction-1) | Why did answers break at high LoRA gain? | They didn't break; they didn't stop. The wrong end token let 206 answers run on into an invented next turn. Cut at the turn end, 0/48 are broken. The frontier conclusions of P stand; the "fragile gain" claim does not. |
| [AE](#exp-ae) | Is the student's helpfulness lead an effect of the 120-token cap? | In good part. The student's lead over the woven prompt in practical coverage falls from 2.25 to 0.50 checklist items per task when the budget grows from 120 to 512 tokens. |
| [AF](#exp-af) | Can a judge shown only the last turn score format and consistency? | No. With the full chat it was right on 8/8 format pairs and 8/8 prior-reply pairs; with only the last user message, 1/8 and 0/8. |
| [AK](#exp-ak) | Do the pilot grades hold up under other graders? | Codex matched 24/24 objection keys and 141/144 of Claude's judgments. On the 72 revision and transfer labels the two models agree on 69 (κ 0.77–0.88); the first human grader matches them on 62 and 61 and passes more targeted questions (κ 0.32–0.39). |

## Experiments

<a id="exp-u"></a>

### U · A judge that isn't Claude

> **Question.** Do Claude judges favour the Claude-taught student?  
> **Result.** No sign of it. Qwen3.5-4B as judge agrees on philosophy (r 0.73–0.75) and on every key helpfulness contrast, and sees the student's lead as larger. It cannot judge coherence.  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c31)

From [exp. O](5-what-limits-the-student.md#exp-o) on, the teacher and both judges are Claude models, so the Claude-taught student's lead could partly be a judge preferring text in its own style (self-preference). We re-scored every answer of the six 48-prompt pools that already had Sonnet and Opus scores (`confirm`, `claude_teacher`, `more_teacher`, `deeper_teacher`, `capacity`, `dpo_claude`: 2,032 answers, including 256 Claude-written training answers) with the unsteered **Qwen3.5-4B** as judge: one criterion per query, with `claude_judge.py`'s rubric wording and the user's request, scored as the expected value of its next-digit distribution (0–9), each answer alone (no pool). Batched scoring with left padding shifted scores by up to 0.15 (the DeltaNet layers are padding-sensitive), so every item was scored one at a time. Qwen isn't a neutral judge either: it is the base of every student and wrote the baseline and prompted answers. The test is whether the *conclusions* survive a judge with the opposite bias.

**Agreement** (Pearson r per item, 2,032 answers):

| Criterion | Sonnet ~ Opus | Qwen ~ Sonnet | Qwen ~ Opus |
|---|---|---|---|
| philosophy | 0.87 | 0.73 | 0.75 |
| helpfulness | 0.85 | 0.69 | 0.70 |
| coherence | 0.77 | 0.33 | 0.30 |

Config rankings inside each pool (Spearman, Qwen ~ Sonnet): philosophy 0.70–0.97, helpfulness 0.30–0.88 (low where the configs are all Claude-taught students within a few tenths of each other), coherence 0.20–0.83.

**The key contrasts** (paired over the 48 prompts, config minus the woven prompt in its pool, mean ± SE):

| Config (pool) | Sonnet: philosophy / helpfulness | Opus | Qwen |
|---|---|---|---|
| 128-answer student ×1.5 (claude_teacher) | −0.23 ± 0.19 / **+1.94 ± 0.27** | 0.00 ± 0.20 / **+2.10 ± 0.34** | −0.07 ± 0.20 / **+2.93 ± 0.22** |
| 384-answer student ×1.0 (more_teacher) | −1.21 ± 0.23 / +2.06 ± 0.26 | −0.54 ± 0.20 / +2.73 ± 0.30 | −0.81 ± 0.25 / +3.55 ± 0.22 |
| 384-answer student ×1.5 (more_teacher) | −0.50 ± 0.21 / +0.54 ± 0.29 | +0.04 ± 0.19 / +1.21 ± 0.33 | +0.18 ± 0.21 / +1.54 ± 0.25 |
| DPO student ×1.5, exp M (claude_teacher) | +0.10 ± 0.19 / +0.65 ± 0.22 | −0.12 ± 0.19 / +1.04 ± 0.25 | −0.25 ± 0.16 / +0.29 ± 0.20 |
| woven prompt + push α 0.35 (confirm) | +0.52 ± 0.17 / −2.17 ± 0.22 | +0.52 ± 0.16 / −2.40 ± 0.22 | +0.39 ± 0.15 / −0.92 ± 0.15 |
| diff-means α 0.5 (confirm) | −0.27 ± 0.19 / −2.08 ± 0.27 | −0.25 ± 0.17 / −2.27 ± 0.29 | −0.17 ± 0.18 / −0.63 ± 0.22 |
| baseline (confirm) | −2.88 ± 0.16 / +2.50 ± 0.23 | −2.65 ± 0.16 / +2.98 ± 0.26 | −3.26 ± 0.16 / +1.52 ± 0.16 |

**Self-preference test** (mean scores by who wrote the text; Sonnet / Opus / Qwen):

| Text | Philosophy | Helpfulness | Claude − Qwen, helpfulness |
|---|---|---|---|
| Claude-written training answers (256) | 2.60 / 2.86 / 3.65 | 8.77 / 8.82 / 7.57 | +1.20 / +1.25 |
| Claude-taught students (816) | 2.17 / 2.29 / 3.05 | 7.71 / 7.01 / 6.90 | +0.81 / +0.11 |
| Qwen baseline (288) | 0.04 / 0.02 / 0.58 | 8.28 / 7.76 / 5.42 | +2.86 / +2.34 |
| Qwen with the woven prompt (288) | 3.12 / 2.83 / 3.84 | 5.79 / 4.82 / 3.91 | +1.88 / +0.91 |

If Claude favoured its own style, its lead over Qwen would be largest on Claude's own text. It is smallest there, apart from the students. Qwen is the harsher judge of plain Qwen answers (5.4 against Sonnet's 8.3). This isn't truncation at 120 tokens: Qwen scores cut-off and complete baseline answers alike (5.37 vs 5.62).

**Pool context.** 307 identical texts (the same greedy outputs, regenerated in different experiments) were judged in more than one pool. Claude's scores of the *same text* vary across pools by an SD of 0.24 (philosophy) and 0.35 (helpfulness) for Sonnet, 0.21 and 0.32 for Opus. That is the cost of pooled, relative scoring, and it is why every comparison in this file is made within one pool. Qwen scores items alone and has no such variation.

**What it means.** The central claim since [exp. O](5-what-limits-the-student.md#exp-o), that the Claude-taught student matches the prompt's philosophy at about 2 more helpfulness points, holds under a judge from a different model family, which sees the gap as even larger. Qwen compresses the helpfulness cost of steering (diff-means −0.63 instead of −2.1) because it rates all plain answers lower, so it is not a better judge, only an independent one. Its coherence scores are unusable, matching [exp. 0.2](1-pushing-the-model.md#exp-0-2). Remaining caveat: no judge from another provider; Qwen is a 4B model scoring by next-digit probability.

<a id="correction-1"></a>

### Correction 1 · Answers that didn't stop (a generation-harness artefact)

> **Question.** Why did answers break at high LoRA gain?  
> **Result.** They didn't break; they didn't stop. The wrong end token let 206 answers run on into an invented next turn. Cut at the turn end, 0/48 are broken. The frontier conclusions of P stand; the "fragile gain" claim does not.  
> 2026-11-05 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c32)

Experiment W ([exp. W](5-what-limits-the-student.md#exp-w)) flagged up to 34 of 48 answers of some students as degenerate. Reading them showed every flagged answer was a complete, sensible answer followed by an invented next turn: "…keep old accounts open.\nuser\nWhat is the best way to build…\nassistant\n<think>…". The cause is in `wi/steer.py`: `generate()` relied on the model config's end token, which for Qwen3.5-4B is `<|endoftext|>` (id 248044), while chat turns end with `<|im_end|>` (248046). The students *did* end their turns correctly; generation just didn't stop there. Any answer that finished before the token limit ran on, and the special tokens were dropped when decoding, which left the bare "user" / "assistant" labels.

Why it looked like breakage that grows with gain: amplified students write shorter answers, so more of them finish inside the 120-token budget and run on. The base model and the prompted model write long answers that hit the limit first, so they were almost never affected.

**Scope.** Every `runs/**/*gen*.jsonl` was scanned for a run-on turn (`\nuser\n` or `\nassistant\n`). Every run-on was judged degenerate wherever it was judged. Affected configs with more than one run-on:

| Run | Config | Run-on / outputs | Degenerate in the original judging |
|---|---|---|---|
| more_teacher ([exp. P](5-what-limits-the-student.md#exp-p)) | 384-answer student ×1.5 | 28 / 48 | 28 |
| more_teacher ([exp. P](5-what-limits-the-student.md#exp-p)) | 384-answer student ×1.3 | 8 / 48 | 8 |
| dpo_claude samples ([exp. S](5-what-limits-the-student.md#exp-s)) | 4 samples at T 0.8 | 71 / 768 | 71 (of 72) |
| capacity ([exp. R](5-what-limits-the-student.md#exp-r)) | rank 32, deeper teacher | 2 / 48 | 2 (of 3) |
| sae round 2 ([exp. T](2-the-steering-tradeoff.md#exp-t)) | content features 1.2·p50 | 2 / 16 | 2 (of 4) |
| style ([exp. W](5-what-limits-the-student.md#exp-w)) | practical only / mismatched, ×1.0 and ×1.2 | 7–34 / 48 | all |

Single run-ons in `bake`, `distill`, `gate`, `robust` and `teacher` (one of 16 or 48 each) don't change any conclusion.

**Fix.** Tokens before the end of a turn don't depend on when generation stops (greedy or sampled), so cutting a saved output at the first invented turn gives exactly what a correct harness would have returned. `scripts/cut_turns.py` does this, and `generate()` now stops at `<|im_end|>` as well as `<|endoftext|>`. The two [exp. P](5-what-limits-the-student.md#exp-p) pools were re-judged on cut outputs by Sonnet and Opus:

| Config (cut outputs) | Philosophy (S / O) | Helpfulness (S / O) | Coherence (S / O) | Degenerate | Sum − woven (S / O) |
|---|---|---|---|---|---|
| *Pool 1* 384-answer student ×1.0 | 2.08 / 2.15 | 8.19 / 7.42 | 8.40 / 8.19 | 0 / 0 | +1.40 ± 0.25 / +1.75 ± 0.32 |
| 384-answer student ×1.5 | 2.65 / 2.65 | 7.65 / 6.38 | 7.81 / 7.23 | **0 / 0** (was 28 / 28) | +1.42 ± 0.23 / +1.21 ± 0.32 |
| 128-answer student ×1.5 | 2.81 / 2.71 | 7.42 / 6.50 | 7.62 / 7.29 | 0 / 0 | +1.35 ± 0.27 / +1.40 ± 0.34 |
| *Pool 2* 384-answer student ×1.2 | 2.56 / 2.38 | 7.75 / 6.73 | 8.33 / 7.96 | 0 / 0 | +1.56 ± 0.25 / +1.58 ± 0.27 |
| 384-answer student ×1.3 | 2.67 / 2.46 | 7.65 / 6.90 | 8.17 / 7.83 | **0 / 0** (was 8 / 8) | +1.56 ± 0.25 / +1.83 ± 0.30 |
| woven prompt (pool 2) | 3.12 / 2.71 | 5.62 / 4.81 | 8.12 / 8.06 | 0 / 0 | — |

**What changes.** The 384-answer student doesn't break when amplified: from ×1.0 to ×1.5 it trades helpfulness for philosophy along the same line (sum about 10.3 Sonnet, 9.0–9.6 Opus), and ×1.3–1.5 is a usable "more philosophical" setting. So the [exp. P](5-what-limits-the-student.md#exp-p) claim that a better-trained adapter "makes gain fragile" is withdrawn. The frontier claim stands, and so does [exp. S](5-what-limits-the-student.md#exp-s)'s diagnosis that the sampled answers were too plain for DPO to work with. Earlier results whose configs wrote long answers (steering, prompts, the baseline) were essentially unaffected, because those answers hit the token limit before ending.

<a id="exp-ae"></a>
<a id="answer-budget"></a>

### AE · How much of the advantage comes from finishing early?

> **Question.** Is the student's helpfulness lead an effect of the 120-token cap?  
> **Result.** In good part. The student's lead over the woven prompt in practical coverage falls from 2.25 to 0.50 checklist items per task when the budget grows from 120 to 512 tokens.  
> **Status.** Qualifies the helpfulness comparisons of O, P and Y.  
> 2026-11-14 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c42)

The student was more helpful than the woven prompt in experiment Y at a cap of 120 new tokens. Experiment AC found different completion rates across configurations, but its rerun also changed the user's length request. A paired budget sweep tests whether more room lets the woven prompt recover useful steps.

**Method** (`scripts/exp_answer_budget.py`, `scripts/score_answer_budget.py`). Give 12 practical tasks to greedy Qwen3.5-4B base, woven, and the 384-answer rank-8 student at gain 1. Repeat each task unchanged at 120, 256, and 512 new tokens. In a separate 512-token condition, ask for at most 90 words. Record generated IDs and natural turn stops. Claude Sonnet scores three predefined practical criteria per task with configuration, system treatment, and budget hidden. Identical replies to the same request receive one shared rating. Checklist coverage counts useful content already present in a reply.

| Configuration | 120 tokens | 256 tokens | 512 tokens | 512 tokens, request ≤90 words |
| --- | ---: | ---: | ---: | ---: |
| Base | 1/36 | 14/36 | 24/36 | 28/36 |
| Woven prompt | 0/36 | 9/36 | 21/36 | 17/36 |
| Rank-8 student | 27/36 | 27/36 | 27/36 | 29/36 |

All 36 independent short generations matched the available initial token IDs from their 512-token counterparts. At 120 tokens, 9/12 student replies ended naturally. At 256 and 512, 12/12 did. Every unrestricted base and woven reply hit its cap, even at 512. With the 90-word instruction, all 36 replies ended naturally, and 10/12 base, 11/12 woven, and 11/12 student replies stayed within 90 whitespace-delimited words.

What this says:

- **The practical-coverage gap narrowed when woven had more room.** The student's paired lead over woven fell from 2.25 to 0.50 checklist items per task, a change of −1.75 items (paired SE 0.22). On the seed-storage prompt, woven was still introducing the subject at 120 tokens. Its 512-token answer covered drying and storage.
- **A word-limit instruction changed completion separately.** All configurations finished in that condition. Woven covered 17/36 criteria, while base and student covered 28/36 and 29/36. Completion and the amount of practical detail therefore need separate measures.
- **This is a practical-coverage pilot.** The unrestricted comparison still has asymmetric truncation at 512. Twelve authored tasks and one checklist judge leave general helpfulness and philosophical quality under matched completion open.

[Generated replies and stop records](../data/results/answer-budget.jsonl) · [Checklist scores](../data/results/answer-budget-coverage.json)

<a id="exp-af"></a>
<a id="judge-context"></a>

### AF · Can the judge see the task?

> **Question.** Can a judge shown only the last turn score format and consistency?  
> **Result.** No. With the full chat it was right on 8/8 format pairs and 8/8 prior-reply pairs; with only the last user message, 1/8 and 0/8.  
> **Status.** Qualifies X, whose judge saw no system prompt.  
> 2026-11-14 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c43)

Experiment X's judge saw the last user request and answer. Its records omitted the terse system instruction and the assistant's actual earlier reply. Could the evaluator still identify a format violation or a contradiction if the information needed to detect it was missing?

**Method** (`scripts/exp_judge_context.py`, `data/judge_context.json`). Present Claude Sonnet with 12 authored answer pairs. Four hinge on a system format instruction, four hinge on an earlier assistant reply, and four are standalone controls for validity, attribution, or relevance. Present every pair in both answer orders, with either the full chat or only the last user message. The judge can choose either answer or a tie. Score against an authored key defined for the full task.

| What distinguishes the answers | Full chat, correct | Last user only, correct | Last user only, ties |
| --- | ---: | ---: | ---: |
| System format instruction | 8/8 | 1/8 | 6/8 |
| Earlier assistant reply | 8/8 | 0/8 | 8/8 |
| Standalone control | 8/8 | 8/8 | 0/8 |

Each row counts four pairs shown in two orders. In the remaining stripped format comparison the judge picked the wrong answer against the full-task key. For example, the bare JSON answer followed a system constraint. Once that constraint was removed, the judge chose a prose answer in one order and tied in the other.

What this says:

- **A judge needs the same chat the model saw.** Seeing the earlier assistant turn resolved all eight references to prior choices. With that turn removed, the judge tied all eight.
- **The effect was specific to missing task information.** All standalone controls were keyed correctly in both modes. This pilot uses one judge and 12 authored keys. It leaves independent human agreement and broad rubric validity open.

[Raw judge decisions](../data/results/judge-context.json)

<a id="exp-ak"></a>

### AK · Pilot regrade

> **Question.** Do the pilot grades hold up under other graders?\
> **Result.** Codex matched 24/24 objection keys and 141/144 of Claude's judgments. On the 72 revision and transfer labels the two models agree on 69 (κ 0.77–0.88); the first human grader matches them on 62 and 61 and passes more targeted questions (κ 0.32–0.39).\
> **Status.** Qualifies the revision and transfer scores in [AC](7-register-versus-skill.md#exp-ac). The targeted-question and transfer comparisons depend on who grades.\
> 2026-11-16 (second model) · 2026-11-19 (human) · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c47)

The data audit called for checking the pilot labels with a reviewer who did not write the cases. We asked Codex to re-key the objection items in [Z](7-register-versus-skill.md#exp-z) and regrade the saved responses from Z, revision and transfer ([AC](7-register-versus-skill.md#exp-ac)), and ethics pressure ([AD](7-register-versus-skill.md#exp-ad)). The original grades came from a Claude model (earlier drafts of this page called it the case author), so this first check was model against model. A person then regraded the revision and transfer responses; see the [human pass](#exp-ak-human).

**Method.** Codex graded 120 saved response records in shuffled packets with configuration names, Claude's keys and original grades removed. Some labeled examples and published objection aggregates were visible during initial inspection, so the review was partially blind. It applied the existing criteria, then matched its grades to Claude's by case and configuration. The 144 comparable judgments exclude keyed A/B accuracy. Some records contribute several judgments, and all three configurations answered the same cases.

| Assessment | Claude passes | Codex passes | Exact agreement |
| --- | ---: | ---: | ---: |
| Original objection explanations identify the central issue | 48/48 | 48/48 | 48/48 |
| Revision asks a targeted question | 7/18 | 6/18 | 17/18 |
| Revision updates on a relevant fact | 13/18 | 14/18 | 17/18 |
| Revision dismisses an irrelevant detail | 13/18 | 13/18 | 18/18 |
| Transfer applies the same distinction to both sides | 16/18 | 15/18 | 17/18 |
| Ethics holds the decision under direct pressure | 1/12 | 1/12 | 12/12 |
| Ethics accounts for changed facts | 12/12 | 12/12 | 12/12 |

**Reading the agreement.** The 141/144 total is inflated by criteria with no variation: every response passed the original objection explanations (48/48) and the ethics changed-facts check (12/12), so those 60 labels agree automatically, and the 12 ethics-pressure labels have a single pass. On the 72 revision and transfer labels Claude and Codex agree on 69. Cohen's κ is 0.88 for targeted questions, 0.85 for relevant updates and 0.77 for transfer; it is undefined for the two unanimous criteria.

Codex matched **24/24 objection keys**. On the harder items, both keys give base 7/8, woven 6/8 and student 8/8. Codex also scored 23/24 of the harder explanations as identifying the central issue. Only the original set had Claude grades for explanations.

- **Two revision comparisons become ties.** Codex judged the student's workshop question too indirect about whether eligible people could attend, bringing the student from 4/6 to 3/6 and tying base. Codex credited the woven prompt for recognizing that permission to list donors privately did not cover publication, bringing it from 3/6 to 4/6 and tying base on relevant updates.
- **Transfer and pressure retain the same pattern.** Base drops from 6/6 to 5/6 on transfer because the volunteer roster reply switches its distinction. The woven prompt stays at 6/6 and the student at 4/6. Both reviewers found that 11/12 decisions reversed under direct pressure and all 12 replies in the evidence branch accounted for the changed facts.
- **Agreement is specific to these cases and criteria.** The project wrote both the cases and the rubric. Both reviews so far were models, so they say little about whether a person would grade the same way. The human pass below tests that.

The accompanying audit check found three of 16 two-turn opening prompts in [X](5-what-limits-the-student.md#exp-x) reused from confirmation, training or contrast data. That is now noted on X and in the [caveats](../FINDINGS.md#caveats). The archived runs and PhilPapers files were unavailable in the checkout used for this review, so the audit's run-integrity and PhilPapers checks are unverified and listed under [open questions](../FINDINGS.md#open-questions).

<a id="exp-ak-human"></a>

#### Human pass

On 2026-11-19 the repository's author, a person, graded the 36 revision and transfer responses (72 labels) from a shuffled packet with configuration names removed, using the criteria on [a grading page](../docs/reviews/human-regrade/grade.html). It is the first human grading in the project. It is not independent of the project: the grader knows the hypotheses and the cases, could probably tell the woven prompt apart because all six of its first replies hit the 96-token cap, and read the criteria as paraphrased on the page. It is one grader with 18 labels per criterion.

| Criterion | Human passes | Claude passes | Codex passes | Matches Claude (κ) | Matches Codex (κ) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Targeted question | 13/18 | 7/18 | 6/18 | 12/18 (0.39) | 11/18 (0.32) |
| Relevant fact | 15/18 | 13/18 | 14/18 | 16/18 (0.68) | 17/18 (0.82) |
| Irrelevant fact | 13/18 | 13/18 | 13/18 | 18/18 (1.00) | 18/18 (1.00) |
| Transfer | 16/18 | 16/18 | 15/18 | 16/18 (0.44) | 15/18 (0.31) |
| All 72 labels | | | | 62/72 | 61/72 |

Claude and Codex match each other on 69 of the same 72 labels, so the human is the outlier.

- **Where the human departs.** On 9 labels the human differs from both models when the two models agree with each other: 8 are more lenient (six targeted questions, one relevant update, one transfer) and 1 is stricter (a transfer). Three more labels differ from one model only. The [full list](../docs/reviews/human-regrade/README.md#labels-that-differ) is with the saved grades.
- **Targeted question.** The human passed all 12 first replies that finished, where Claude passed 7 and Codex 6. The student's lead over base (4/6 against 3/6 under Claude, a tie under Codex) becomes a tie at 6/6. The woven prompt scores 1/6 against 0/6 under both models because every one of its first replies hit the cap ([AC](7-register-versus-skill.md#exp-ac)). The human passed one of those cut-off replies, although the page's rule says to fail one. Failing it gives 12/18 and changes none of the conclusions here.
- **Relevant and irrelevant facts.** The student's 6/6 on relevant updates and 2/6 on dismissing irrelevant details are the same under all three graders, and the irrelevant-fact labels are identical in all 18 cases.
- **Transfer.** The student gets 5/6 (4/6 under both models), base 6/6 and the woven prompt 5/6 (6/6 under both models). The AC result that the student transfers in four cases against six and six does not hold under the human grades.

What this says: two models agreeing is weaker evidence than 141/144 suggests. Both applied the same written rubric, and a person read the targeted-question rule more loosely. Codex's notes show a stricter standard, such as requiring a question about whether eligible workers could attend, and Claude's targeted-question grades carry no notes. Which reading is right is a rubric decision, not a measurement. Only the relevant-update and irrelevant-detail results hold under all three graders. The student's targeted-question lead and its transfer deficit depend on who grades.

[Review method and audit comparison](../docs/reviews/2026-11-16/codex-regrade/README.md) · [Paired grades](../docs/reviews/2026-11-16/codex-regrade/grades.jsonl) · [Human grades and scoring](../docs/reviews/human-regrade/README.md)
