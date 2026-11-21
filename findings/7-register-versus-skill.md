# Part 7 · Is the reflective register philosophical skill?

[← Findings overview](../FINDINGS.md) · [Explainer, part 7](https://www.contrasentido.ar/weighting-intelligence/explainer.html#p7)

**Short answer.** No. Raising the register with any lever leaves judge-free measures of skill at base level, and constant pushes lower them. Skill grows with model size instead, from 2B to 4B on every test and less or not at all from 4B to 9B. The student shows no reliable advantage on objections, revision or premise sensitivity, and no 4B configuration holds a position against a bare "I disagree".

| ID | Question | Result |
|---|---|---|
| [Z](#exp-z) | Is the student better at picking the objection that matters? | Inconclusive. All three configurations got 16/16 on the original set; on eight harder items the student got 8, base 7 and the woven prompt 6. |
| [AA](#exp-aa) | Does the reflective register pull the model away from its assistant role over turns? | No drift found. Prompt and adapter start lower on the assistant probe at turn one; the largest drift was 0.24 against a 1.30 threshold. |
| [AB](#exp-ab) | Do wording features affect the model's validity judgments? | Not here. One SAE feature tracks the wording change; ablating it and two others leaves all 24 held-out validity choices unchanged. |
| [AC](#exp-ac) | Does the student revise on relevant facts and reuse its own distinctions? | Mixed. It updates on 6/6 relevant facts but dismisses only 2/6 cosmetic details, and transfers its distinction in 4/6 cases against 6/6 for base and the woven prompt. |
| [AD](#exp-ad) | Does any configuration hold a decision under unsupported pushback? | No. All three account for new facts (4/4). Under bare pushback, base held once in four and the woven prompt and the student never. |
| [AG](#exp-ag) | Does the student apply a rule to changed facts better than base? | No advantage. All three got 6/8 relevant pairs and 4/4 cosmetic pairs right, with identical choices on all 24 variants. |
| [AI](#exp-ai) | Does skill on judge-free philosophy tasks rise with the register? | No. Moving the register from 0 to 3.8 predicts no skill measure. Prompts and students keep base-level skill; constant pushes lose up to 46 points, mostly to position bias. |
| [AI2](#exp-ai2) | If the register is not the skill, where does skill come from? | From model size, on base models where the register is zero: validity 57 → 83 of 96 from 2B to 4B (z 5.2), and +7 from 4B to 9B (z 1.7). |
| [AL](#exp-al) | Does the split hold on real philosophical texts? | Yes for register, partly for scale. On single-passage validity from eight historical sources, 2B → 4B gains 17 points (z 3.0) and 4B → 9B gains 1; prompts and students stay at base, constant pushes lose 16–22. |

## Experiments

<a id="exp-z"></a>
<a id="pilot-argument-critique"></a>

### Z · Which objection matters?

> **Question.** Is the student better at picking the objection that matters?  
> **Result.** Inconclusive. All three configurations got 16/16 on the original set; on eight harder items the student got 8, base 7 and the woven prompt 6.  
> 2026-11-14 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c37)

The student learned to connect a reflection to the user's task ([exp. W](5-what-limits-the-student.md#exp-w)). Does that carry over to identifying the premise an objection actually challenges? LMCA offers a way to pose the question without having to settle whether the underlying philosophical position is true. [LMCA write-up](https://alignment.anthropic.com/2026/conceptual-reasoning-index/) · [LMCA dataset page](https://conceptualreasoning.ai/lmca)

**Method** (`scripts/exp_argument_critique.py`, `runs/philosophy_pilot/{critique,hard}.json`). On Qwen3.5-4B, compare the base model, the woven prompt, and experiment P's 384-answer rank-8 student at gain 1.0. Each sees a position and two objections in alternating A/B order. It must start with its choice and explain which premise or inference the objection challenges. The original set has 16 author-keyed items. After all three hit the ceiling, we added eight items with more plausible distractors. We scored the choice and opening format separately. The case author reviewed the original explanations with configuration labels hidden.

| Config | Correct choice, original | Starts with A/B, original | Mean tokens, original | Correct choice, harder |
| --- | ---: | ---: | ---: | ---: |
| Base | 16/16 | 16/16 | 60.4 | 7/8 |
| Woven prompt | 16/16 | 12/16 | 100.5 | 6/8 |
| Rank-8 student | 16/16 | 13/16 | 76.9 | 8/8 |

All 72 replies reached an end token. All 48 original explanations addressed the targeted issue, though some introduced secondary errors.

What this says:

- **The distractor determines how much the test can tell us.** In the original witness case, a distractor about speaking pace was easy to dismiss. In the harder allocation case, both objections questioned the use of user counts, but only one distinguished total users from users newly served by the grant.
- **The student had a one-item lead over base on the harder set.** Eight author-keyed cases added after the initial run give a preliminary signal. The original set did distinguish instruction compliance, where base most reliably opened with the requested letter.

[Saved outputs and review details](../docs/reviews/2026-11-14/argument-critique/README.md)

<a id="exp-aa"></a>
<a id="pilot-persona-drift"></a>

### AA · Does the assistant drift?

> **Question.** Does the reflective register pull the model away from its assistant role over turns?  
> **Result.** No drift found. Prompt and adapter start lower on the assistant probe at turn one; the largest drift was 0.24 against a 1.30 threshold.  
> 2026-11-14 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c38)

Experiment X found that the student's reflective register persisted into a second turn. If the prompt or adapter also pulls the model away from its default assistant role, does that shift grow across turns? The Assistant Axis paper motivates comparing conversations about AI self-awareness with practical conversations. [The Assistant Axis](https://arxiv.org/html/2601.10387v1)

**Method** (`scripts/exp_persona_drift.py`, `runs/philosophy_pilot/persona.json`). Estimate a direction from three practical assistant roles and three theatrical roles on four calibration prompts. Check their separation on two held-out prompts at layers 16, 20, and 24. Run two AI self-awareness and two practical conversations for three turns each, using greedy base Qwen3.5-4B, the woven prompt, and experiment P's rank-8 student at gain 1.0. Higher projection means closer to the practical roles. At layer 20, compare the turn-1-to-turn-3 changes between domains. The cap and equal-norm random replay require directional drift above 15% of the held-out role separation.

| Config | AI projection, turn 1 → 3 | Practical projection, turn 1 → 3 | Difference in change (practical minus AI) |
| --- | ---: | ---: | ---: |
| Base | 0.43 → 0.75 | 2.07 → 1.48 | −0.91 |
| Woven prompt | −0.70 → −0.61 | 0.16 → 0.49 | +0.24 |
| Rank-8 student | −0.64 → −0.89 | 0.58 → 0.15 | −0.19 |

The roles separated on the held-out prompts at all three layers. The layer-20 gap was 8.68 units, giving a replay threshold of 1.30 units. The largest directional drift contrast was +0.24, so the intervention did not run.

What this says:

- **The clearest change was already present on turn one.** Both the prompt and the adapter started closer to the theatrical end of this probe in both domains. The three-turn trajectories showed little evidence of extra drift on AI topics.
- **The helpfulness question is still open.** With two conversations per domain and no cap result, the run leaves open whether the lower starting projection contributes to the helpfulness cost or grows over longer conversations.

[Saved outputs and review details](../docs/reviews/2026-11-14/persona-drift/README.md)

<a id="exp-ab"></a>
<a id="pilot-prose-validity"></a>

### AB · Does theatrical prose change validity judgments?

> **Question.** Do wording features affect the model's validity judgments?  
> **Result.** Not here. One SAE feature tracks the wording change; ablating it and two others leaves all 24 held-out validity choices unchanged.  
> 2026-11-14 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c39)

Experiment T's sparse autoencoder found features that fire on reflective codas and stock phrases ([exp. T](2-the-steering-tradeoff.md#exp-t)). Can we identify a feature that responds to a change in wording while leaving the model's judgment of the inference intact? [Model Diffing without Borders](https://openreview.net/pdf/659e103888bd395dc0f3af5a626ea1e692aa937e.pdf)

**Method** (`scripts/exp_prose_validity.py`, `runs/philosophy_pilot/sae.json`). Cross plain and theatrical wording with valid and invalid versions of 12 argument families. Hold answer ordering fixed within each family. On base Qwen3.5-4B, use the saved layer-8 SAE to rank features by absolute activation difference on six families and check them on the other six. Subtract each selected feature's decoded contribution while retaining the SAE's reconstruction error. Measure validity using forced-choice next-token A/B scores over the 24 held-out variants. The task asks whether a conclusion follows from the premises.

| Feature selected on discovery cases | Contrast | Discovery activation change | Held-out activation change |
| --- | --- | ---: | ---: |
| 6666 | Theatrical minus plain wording | +0.0403 | +0.0400 |
| 275 | Valid minus invalid inference | +0.0035 | +0.0008 |

| Intervention | Correct validity choices | Choices changed vs. no ablation |
| --- | ---: | ---: |
| No ablation | 21/24 | — |
| Remove 6666, wording feature | 21/24 | 0/24 |
| Remove 275, validity candidate | 21/24 | 0/24 |
| Remove 7238, stock-phrase feature | 21/24 | 0/24 |

What this says:

- **The wording contrast transferred more clearly than the validity contrast.** Feature 6666 retained almost the same activation difference on the held-out families. The two recurring templates leave open whether it tracks a broader register or particular phrases.
- **The measured choices survived all three ablations.** Removing these contributions was insufficient to flip an answer on any held-out item. The result is specific to these features and this forced-choice validity test.

[Saved outputs and review details](../docs/reviews/2026-11-14/prose-validity/README.md)

<a id="exp-ac"></a>
<a id="pilot-revision-transfer"></a>

### AC · Can it revise and reuse a distinction?

> **Question.** Does the student revise on relevant facts and reuse its own distinctions?  
> **Result.** Mixed. It updates on 6/6 relevant facts but dismisses only 2/6 cosmetic details, and transfers its distinction in 4/6 cases against 6/6 for base and the woven prompt.  
> **Status.** [AK](6-can-we-trust-the-scores.md#exp-ak) reports a second model's grades and a human's for the same responses. Under the human grades the student's targeted-question lead disappears and it transfers in 5/6 cases.\
> 2026-11-14 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c40)

Recognizing a stronger objection ([exp. Z](#exp-z)) is one part of helping someone think through an argument. Two further questions matter in a dialogue. Can the model ask about the assumption that matters and update only when the answer bears on it? Can it make a distinction that still works in a new case? [Language Models as Critical Thinking Tools](https://arxiv.org/abs/2404.04516) · [What Is Philosophy?](https://cup.columbia.edu/book/what-is-philosophy/9780231530668/) · [Stanford PhilosophyBench](https://philosophybench.org/)

**Method** (`scripts/exp_revision_transfer.py`, `runs/philosophy_pilot/{revision,transfer}.json`). Give six arguments to greedy base Qwen3.5-4B, the woven prompt, and experiment P's rank-8 student at gain 1.0. Ask one question about the most important unstated assumption with a 96-token cap. Branch the same opening exchange into replies to a relevant fact and a cosmetic detail. In a separate branch, give the relevant fact and ask for a distinction, then ask the model to apply it to a fresh case. Both transfer turns initially had a 128-token cap. The rerun kept the original opening exchanges, requested at most 45 words per turn, and allowed 256 tokens. The case author scored the replies with configuration labels hidden. Word counts use whitespace splitting.

| Config | Targeted question | Sound update on relevant fact | Correctly dismissed irrelevant fact |
| --- | ---: | ---: | ---: |
| Base | 3/6 | 4/6 | 6/6 |
| Woven prompt | 0/6 | 3/6 | 5/6 |
| Rank-8 student | 4/6 | 6/6 | 2/6 |

Every woven opening reply hit the 96-token cap before it finished asking a question. Those partial replies went into the follow-ups, all of which finished. In the initial transfer branch, all base and woven replies hit the 128-token cap on both turns, while all student replies finished. All 36 replies in the transfer rerun reached an end token.

| Config | Successful transfer, rerun | Concept within 45 words | Transfer within 45 words |
| --- | ---: | ---: | ---: |
| Base | 6/6 | 6/6 | 5/6 |
| Woven prompt | 6/6 | 5/6 | 6/6 |
| Rank-8 student | 4/6 | 0/6 | 0/6 |

What this says:

- **The student picked up relevant changes and overread cosmetic ones.** It got all six relevant updates right, but treated four irrelevant details as consequential. For example, it took differences in typeface as evidence about the independence of two witness statements.
- **The student's concepts carried over less reliably in this set.** On the volunteer roster case, it shifted from a distinction about principles and factual justification to one about purpose. On the appointment case, it introduced an unsupported assumption about car owners. Some passing answers also contained secondary errors, such as treating equal measured learning rates as proof of zero causal effect.
- **Completion depends on the setup.** The transfer rerun changed both the requested length and the token ceiling, so the effect of each remains unknown. Woven follow-up scores also reflect the truncated opening exchanges.

[Saved outputs and review details](../docs/reviews/2026-11-14/revision-transfer/README.md)

<a id="exp-ad"></a>
<a id="pilot-ethics-pressure"></a>

### AD · Will it hold its position under pressure?

> **Question.** Does any configuration hold a decision under unsupported pushback?  
> **Result.** No. All three account for new facts (4/4). Under bare pushback, base held once in four and the woven prompt and the student never.  
> 2026-11-14 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c41)

The student in [exp. AC](#exp-ac) often found significance in details that should have left the argument alone. Will any of the configurations hold its decision when the user supplies no new evidence and asks for a reversal? A second branch checks whether the same model can update when the facts do change. [PhilBench](https://www.philbench.com/) · [philosophy-bench](https://github.com/benedictbrady/philosophy-bench)

**Method** (`scripts/exp_ethics_pressure.py`, `runs/philosophy_pilot/pressure.json`). Give four ethical dilemmas to greedy base Qwen3.5-4B, the woven prompt, and experiment P's rank-8 student at gain 1.0. Ask for a decision and its principle. Branch from the same first reply into either a material change in the facts or a request for the opposite decision on the same facts and principle. The case author scored whether the responses accounted for the new fact and whether they held the original decision under pressure, with configuration labels hidden.

| Config | Accounted for new fact | Held decision under pressure |
| --- | ---: | ---: |
| Base | 4/4 | 1/4 |
| Woven prompt | 4/4 | 0/4 |
| Rank-8 student | 4/4 | 0/4 |

In the reservation case, all three initially opposed first-come allocation. They treated a new work-schedule restriction as a further reason against it, so the correct update did not require a reversal.

What this says:

- **Direct pushback usually overrode the original decision.** In the confidentiality case, a verified emergency justified sharing an address with a responder. All three also reversed their confidentiality decision on the curiosity-only facts when pressed to do so.
- **The test shows a shared weakness under this wording.** Four dilemmas and one direct pushback formulation give little basis for ranking the configurations or generalizing to subtler disagreement.

[Saved outputs and review details](../docs/reviews/2026-11-14/ethics-pressure/README.md)

<a id="exp-ag"></a>
<a id="premise-sensitivity"></a>

### AG · Does a changed fact change the decision?

> **Question.** Does the student apply a rule to changed facts better than base?  
> **Result.** No advantage. All three got 6/8 relevant pairs and 4/4 cosmetic pairs right, with identical choices on all 24 variants.  
> 2026-11-14 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c44)

In experiments AC and AD, the model revised an answer within the same conversation. AC also found that the student sometimes treated cosmetic details as evidence. Here each version of a case appears on its own. A fixed rule makes it possible to check whether a model changes its judgment when the decisive fact changes, and stays consistent when a detail is immaterial.

**Method** (`scripts/exp_premise_sensitivity.py`, `data/premise_sensitivity.json`). Give base Qwen3.5-4B, the woven prompt, and the 384-answer rank-8 student eight pairs that change one decisive fact and four pairs that change an irrelevant detail. Randomize the order of standalone cases and balance the keyed A/B positions within both the relevant and cosmetic groups. Ask for the decision and the deciding fact. Score decisions against the authored rule, with a pair counted correct only when both choices match the key. All 72 replies reached an end token and started with a recognizable choice.

| Configuration | Relevant pairs, both correct | Irrelevant pairs, both correct | Clarified archive pair, both correct |
| --- | ---: | ---: | ---: |
| Base | 6/8 | 4/4 | 1/1 |
| Woven prompt | 6/8 | 4/4 | 1/1 |
| Rank-8 student | 6/8 | 4/4 | 1/1 |

All three missed the same two variants of the original relevant pairs. All treated an approval that preceded a rewrite as sufficient approval of the final wording. In the archive case, all rejected the clock that retained its original mechanism inside a replacement case. A follow-up avoids the two meanings of “case” by using “housing” and stating the decision rule explicitly. All three then answered both versions correctly. The table keeps the original results separate from the follow-up, which can be rerun with `--data data/premise_identity_check.json --out runs/premise_sensitivity/identity-check.jsonl`.

What this says:

- **There was no student advantage on this controlled task.** All configurations handled the cosmetic changes, and their choices matched each other on all 24 original variants.
- **A wording repair changed the archive judgment.** The follow-up shows that all three could apply the clarified criterion. The remaining approval error shows a shared failure to distinguish the order of approval and revision. These results concern application of a supplied rule on a small authored set.

[Original replies](../data/results/premise-sensitivity.jsonl) · [Clarified archive pair](../data/results/premise-identity-check.jsonl)

<a id="exp-ai"></a>
<a id="philbench-grid"></a>

### AI · Is the reflective register philosophical skill?

> **Question.** Does skill on judge-free philosophy tasks rise with the register?  
> **Result.** No. Moving the register from 0 to 3.8 predicts no skill measure. Prompts and students keep base-level skill; constant pushes lose up to 46 points, mostly to position bias.  
> 2026-11-15 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c46)

Experiments 0.1 through V measured philosophy as a judge's 0–9 score for reflective sentences in practical answers. They never tested whether that register has anything to do with philosophical skill. Here the register is varied with every lever the project built, and skill is measured on externally defined tasks scored without a judge. If philosophy is the register, skill should rise with register score and with dose. If it is something else, skill should stay flat or fall, and a random push of the same size should do as much damage as a register push. [PhilBench](https://www.philbench.com/) · [2020 PhilPapers Survey](https://survey2020.philpeople.org/) · [philosophy-bench](https://github.com/benedictbrady/philosophy-bench)

**Method** (`scripts/exp_philbench.py`, `scripts/philbench_items.py`, `scripts/exp_philbench_analyze.py`, `data/philpapers2020.json`, `data/philpapers_constraints.json`). Sixteen configurations of Qwen3.5-4B (bf16): base; the woven and reflect-first prompts; layer-8 pushes along diff-means (α 0.35, 0.5), along the learned direction B (0.4, 0.6, 0.8·p50) and along three random unit directions (0.6·p50); the rank-1 student at ×1 and ×2; the 384-answer Claude-taught student at ×1.0 and ×1.5; and the 128-answer student at ×1.5. Register: one blind pool of the 48 confirmation prompts under every configuration (120 tokens), judged by Sonnet and Opus. Skill, by next-token forced choice, one item per forward pass:

- **PhilPapers**: the 98 single-choice questions of the 2020 survey (the two accept/reject grids are left out), each with its substantive options and five shared meta-options ("Agnostic/undecided", "The question is too unclear to answer", …), in three variants: as worded, an author-written paraphrase, and with the substantive options reversed. Measures: same choice under paraphrase, same choice under reversed order, and violations of 37 author-written constraints between answers (28 firm, 9 marked contestable). Constraints use each question's probabilities averaged over both option orders.
- **Pressure**: the model's own PhilPapers answer, then "I disagree. The answer is <the most probable other substantive option>." with no reasons.
- **Argument items**: pilot Z's 24 objection items and pilot AB's 48 validity items, each in both A/B orders.
- **philosophy-bench, adapted**: the 100 scenarios as text, with their 4–5 authored response clusters as lettered options. The scores are the expected `cd_score` and the mass on botched clusters. This is not comparable to the published agentic, panel-judged leaderboard.
- **Generated**: the PhilPapers questions answered in 2–3 sentences ending `ANSWER: <letter>` (160 tokens), checked against the forced choice.

**A measurement correction.** The first scoring pass read the first token of the reply. Several configurations open with prose instead: reflect-first put under 1% of its next-token mass on the option letters (median over items), and diff-means 0.5, B 0.8 and the 384-answer student at ×1.5 put 39–47%. Scoring them by renormalizing over the letters measured noise and showed them near chance. All results below prefill the assistant turn with `Answer:` and also count the bold path (`Answer: **A**`). After that, every configuration puts a median of 0.87–1.0 of its mass on the letters. The first pass is kept as a measure of whether a configuration answers at once.

| Config | Register philosophy / helpfulness | Same choice, paraphrase | Same choice, reversed order | Constraints violated (of 37) | Objections correct (of 48) | Validity correct (of 96) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 0.02 / 8.34 | 65 | 64 | 3 | 48 | 83 |
| Woven prompt | 2.84 / 5.54 | 67 | 77 | 2 | 46 | 83 |
| Reflect-first prompt | 1.98 / 7.69 | 68 | 65 | 1 | 48 | 81 |
| Diff-means α 0.35 | 0.71 / 7.18 | 74 | 70 | 2 | 44 | 58 |
| Diff-means α 0.5 | 2.65 / 3.52 | 79 | 69 | 3 | 34 | 54 |
| B 0.4·p50 | 1.43 / 7.01 | 74 | 75 | 0 | 44 | 70 |
| B 0.6·p50 | 3.53 / 4.62 | 70 | 60 | 0 | 35 | 53 |
| B 0.8·p50 | 3.78 / 3.22 | 87 | 24 | 4 | 26 | 47 |
| Random 0.6·p50, seeds 0 / 1 / 2 | 0.02–0.04 / 7.7–8.1 | 77 / 64 / 66 | 27 / 76 / 66 | 4 / 6 / 5 | 42 / 37 / 46 | 65 / 62 / 59 |
| Rank-1 student ×1 | 2.76 / 5.16 | 79 | 77 | 3 | 48 | 86 |
| Rank-1 student ×2 | 3.44 / 5.24 | 74 | 77 | 3 | 48 | 85 |
| Claude-taught 384 ×1.0 | 1.99 / 7.90 | 68 | 79 | 1 | 47 | 87 |
| Claude-taught 384 ×1.5 | 2.56 / 7.17 | 66 | 76 | 1 | 47 | 87 |
| Claude-taught 128 ×1.5 | 2.72 / 7.28 | 68 | 72 | 0 | 48 | 86 |

Register scores are the mean of Sonnet and Opus over 48 prompts. The PhilPapers columns count questions out of 98.

Under pressure, every configuration flipped on all 98 questions. After the disagreement, the 4B configurations put 93–99.5% of their mass on the user's option. A push did not change that: B 0.8 kept the most of its own answer, at 5%.

Across the 16 configurations, register philosophy does not predict any skill measure. The Spearman correlations are +0.44 for paraphrase stability, +0.17 for order stability, −0.30 for constraint violations, −0.08 for objections and 0.00 for validity, all with 95% intervals that include zero. The one exception is mass on the user's option under pressure (−0.52, interval −0.92 to −0.01), driven by the strong pushes keeping slightly more of their own answer. It is one of about 20 comparisons and doesn't reach 3 standard errors, so treat it as suggestive.

| Paired over items | Objections | Validity | Same choice, reversed order |
| --- | ---: | ---: | ---: |
| Random pushes (mean of 3) minus base | −13 points (z −3.7) | −22 points (z −5.6) | −8 points (z −1.4) |
| B 0.8 minus random pushes | −33 points (z −5.5) | −16 points (z −3.5) | −33 points (z −7.4) |
| B 0.4 minus random pushes | +5 points (z 1.9) | +8 points (z 2.2) | +19 points (z 3.5) |
| Claude-taught 384 ×1.0 minus base | −2 points (z −1.0) | +4 points (z 1.7) | +15 points (z 2.6) |

Generated answers that ended before the cap stated the same option as the forced choice in 51% (base), 56% (woven), 67–69% (Claude-taught students) and 31–44% (diff-means and B 0.6). B 0.8 reached its cap on 72 of 98 answers and stated no answer on 70. Written answers named meta options ("a combination of views") much more often than the forced choice: diff-means 0.5 named one in 62 of 98 written answers and in 7 of 98 forced choices.

What this says:

- **The register and skill come apart.** The trained levers (prompts, rank-1 and Claude-taught students) raise register philosophy from 0 to 2–3.4 and leave every skill measure at base level or slightly above. The constant pushes raise the register as much and lose 27–46 points on objections and 30–38 on validity. Register score orders the configurations, and skill does not follow it.
- **Most of what pushes cost is position bias.** At B 0.8 the model picks the first-listed option almost regardless of content. It chose the stronger objection on 24/24 items when that objection came first, but on only 2/24 when it came second. Its high paraphrase "stability" (87/98) is the same artefact: it gives the first letter both times. Random pushes of the same size also cost 13–22 points, so part of the damage is generic disturbance. B 0.8 does more damage than random; B 0.4 does a little less, with fewer constraint violations (0 against 4–6, z −3.6).
- **Dose cuts skill for pushes, not for students.** Objections and validity fall steadily from base to B 0.4, 0.6 and 0.8, and from base to diff-means 0.35 and 0.5. Raising the gain of the rank-1 or Claude-taught student leaves both flat.
- **Nothing here holds a position.** Every configuration gives up a forced-choice answer to a bare "I disagree" ([exp. AD](#exp-ad) found the same on 4 items). This is a property of the 4B model, unaffected by any lever.
- **The model's base state carries no sign of either direction.** On base, the last prompt token's projection onto diff-means or B at layers 8 and 16 does not predict which questions get stable answers (all |ρ| ≤ 0.20, intervals mostly including zero).
- **What stays open.** The PhilPapers measures have few events (0–6 constraint violations per configuration), the constraints and paraphrases are author-written, and three random directions disagree among themselves on order stability (27–76/98). philosophy-bench is adapted to forced choice. Here the Claude-taught students lean more consequentialist than base (expected `cd_score` +0.18 to +0.28, z 4.8–5.4), which describes a preference, not skill.

[Saved outputs and review details](../docs/reviews/2026-11-15/philbench/README.md)

<a id="exp-ai2"></a>
<a id="philbench-scale"></a>

### AI2 · Does philosophical skill grow with model size?

> **Question.** If the register is not the skill, where does skill come from?  
> **Result.** From model size, on base models where the register is zero: validity 57 → 83 of 96 from 2B to 4B (z 5.2), and +7 from 4B to 9B (z 1.7).  
> 2026-11-15 · [explainer](https://www.contrasentido.ar/weighting-intelligence/explainer.html#c46)

If the register is not the skill, does skill come from somewhere else? [exp. V](5-what-limits-the-student.md#exp-v) found the 2B and 4B students equally reflective. Here the same forced-choice tasks run on base models of three sizes, where the register is zero throughout.

**Method** (`scripts/exp_philbench.py --model 2B`, `scripts/exp_philbench_mlx.py`). Qwen3.5-2B in bf16 (base, woven prompt, its experiment V student) and Qwen3.5-4B and 9B as 4-bit MLX models (base and woven prompt). The 4-bit students from [exp. V](5-what-limits-the-student.md#exp-v) were trained in memory and never saved, and retraining was out of scope, so they are not here. The 9B is compared only with the 4-bit 4B. Same items, prefill and scoring as [exp. AI](#exp-ai).

| Base model | Same choice, paraphrase | Same choice, reversed order | Constraints violated (of 37) | Objections correct (of 48) | Validity correct (of 96) | Mass on the user's option after pressure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2B, bf16 | 61 | 45 | 5 | 46 | 57 | 0.79 |
| 4B, bf16 | 65 | 64 | 3 | 48 | 83 | 0.99 |
| 4B, 4-bit | 61 | 72 | 4 | 48 | 83 | 0.99 |
| 9B, 4-bit | 69 | 76 | 3 | 48 | 90 | 0.78 |

The paired 2B → 4B (bf16) gains are +27 points on validity (z 5.2) and +19 on order stability (z 2.8). The 4B → 9B (4-bit) gains are +7 on validity (z 1.7) and +4 on order stability (z 0.7). With the woven prompt the 9B adds +9 on validity (z 2.0) and +12 on order stability (z 2.2). The 2B's own student scores above the 2B base on validity (64 against 57) and order stability (62 against 45). The two 4B stacks make the same forced choice on 73% of PhilPapers variants, 88% of philosophy-bench scenarios and 90% of validity items.

What this says:

- **Skill grows with size while the register stays at zero.** On base models the clearest step is 2B → 4B, with a smaller one from 4B → 9B, the same shape [exp. V](5-what-limits-the-student.md#exp-v) found for the students' helpfulness. That fits the second prediction: whatever these tasks measure comes with the model, not with the reflective register.
- **The 9B is the only model that resists pressure even a little.** It keeps about 8% of its own answer after a bare disagreement, against 0.2–0.5% for the 4B. The 2B also leaves more mass off the user's option, but it was less confident to begin with.
- **The scale comparison is thin.** One model family, three sizes, base and prompt only for 4B and 9B. The 4-bit and bf16 4B disagree on a quarter of PhilPapers choices, which is why the 9B is never compared across stacks.

[Saved outputs and review details](../docs/reviews/2026-11-15/philbench/README.md)

<a id="exp-al"></a>
<a id="real-text-skill"></a>

### AL · Does the split hold on real philosophical texts?

> **Question.** On tasks built from real philosophical texts and keyed without a grader, does skill grow with model size, and do the register levers leave it at base level?  
> **Result.** Yes from 2B to 4B, not from 4B to 9B. On single-passage validity the 4B rates an intact argument above its corrupted version on 82% of 87 units, against 64% for the 2B (z 3.0); the 9B adds 1 point. Prompts and students stay at base; constant pushes lose 16–22 points (z −2.9 to −3.8).  
> 2026-11-20 · [plan](../docs/plans/real-text-skill-pilot.md)

[AI](#exp-ai) and [AI2](#exp-ai2) used items the project wrote or adapted (PhilPapers paraphrases, pilot arguments). Here the arguments come from public-domain texts, and the keys follow from how the items are built, so no grader decides what is correct. The sources were chosen to go beyond modern northern-European philosophy: Descartes with the Objections and Replies, Hobbes, Leibniz–Clarke, Hume's *Dialogues*, Berkeley's *Three Dialogues*, Anselm with Gaunilo, Vitoria's *De Indis* and Galileo's *Two New Sciences*. The predictions were written before any run: skill grows with size (P1), the register levers leave it at base and constant pushes lower it (P2), and original and paraphrased wording score alike (P3, no memorisation).

**Method** (`scripts/exp_real_text.py`, `scripts/exp_real_text_mlx.py`, `scripts/philbench_items.py`, `scripts/real_text_packets.py`, `data/pilot-real-*.json`, `data/sources/`). Ten configurations, with AI's forced-choice scorer unchanged (`Answer:` prefill plus the bold path, both letter orders): base 2B (bf16), base 4B (bf16), base 4B and 9B (4-bit MLX); and on 4B bf16 the woven prompt, the rank-1 student ×1, the Claude-taught 384 student ×1.0, diff-means α 0.5, B 0.6·p50 and random 0.6·p50 (seed 0). Three tasks:

- **`verdict`** (main measure): one passage, "does its conclusion follow?", A/B in both orders. 22 arguments, each as original, paraphrase and four corrupted versions (one sentence changed: negation, quantifier, dropped premise, overreach or reversed conditional). The score is the share of (argument, corruption) units where P(follows | original) > P(follows | corrupted), each averaged over both letter orders. A constant letter or yes/no preference cannot move it. 87 units after one bad item was dropped.
- **`corrupt`** (as planned): two passages, which one's conclusion follows. Replaced as the main measure because it measured letter preference: the 4B bf16 picked A on 89% of keyed pairs and the 9B picked B on 81%.
- **`objreply`**: 20 objection–reply pairs from four exchanges (Hobbes–Descartes, Leibniz–Clarke, Gaunilo–Anselm, Vitoria); pick the real reply out of four from the same exchange, and the reverse, in original and paraphrased wording.

**Baseline.** Blind graders saw only the answer packets. Two fresh Claude instances scored 20/20 on the 20-item baseline packet and 131/132 on the single passages; the one miss was the bad item, now dropped. The author scored 19/20 on the baseline packet. The one disagreement, `berkeley-one-sensation`, rests on a step from simultaneity to identity that a reader can fairly dispute.

| config | verdict: orig > corr | verdict: para > corr | gap orig (pts) | objreply |
|---|---:|---:|---:|---:|
| 2B base | 64.4 | 67.8 | 1.9 | 76.2 |
| 4B base (bf16) | 81.6 | 81.6 | 12.2 | 91.2 |
| 4B base (4-bit) | 79.3 | 75.9 | 10.7 | 91.2 |
| 9B base (4-bit) | 80.5 | 65.5 | 12.1 | 92.5 |
| woven prompt | 80.5 | 74.7 | 11.8 | 91.2 |
| rank-1 ×1 | 85.1 | 79.3 | 10.7 | 91.2 |
| Claude-taught 384 ×1.0 | 81.6 | 69.0 | 15.8 | 90.0 |
| diff-means 0.5 | 65.5 | 73.6 | 3.4 | 67.5 |
| B 0.6 | 65.5 | 65.5 | 2.3 | 65.0 |
| random 0.6 | 59.8 | 79.3 | 3.6 | 75.0 |

"orig > corr" is the percentage of units where the intact original gets the higher P(follows) (chance 50%); "gap" is the mean difference in P(follows).

What this says:

- **P1, scale, holds from 2B to 4B and stops there.** 2B → 4B: discrimination +17 (z 3.0), gap +10 points (z 6.8), objreply +15 (z 2.1). 4B → 9B, both 4-bit: discrimination +1 (z 0.2), objreply +1. The blind graders score 95–100%, so the 9B stops well short of what the tasks allow. AI2 found a small 4B → 9B step on validity (+7, z 1.7); on real texts there is none.
- **P2 holds: register is not skill.** Against 4B base, the woven prompt (−1, z −0.4), the rank-1 student (+3.5, z 1.8) and the Claude-taught student (0) stay at base, and at 90–91% on objreply. The constant pushes lose 16 (diff-means, z −2.9), 16 (B, z −3.0) and 22 (random, z −3.8) points of discrimination, and 16–26 points on objreply. As in AI, a random push of the same size does as much damage as a register push.
- **P3, memorisation, holds at 4B and possibly fails at 9B.** At 4B, discrimination is 81.6% whether the intact text is the original or the paraphrase. At 9B it is 80.5% with the original against 65.5% with the paraphrase, and originals get 6 points more "follows" (z 2.0), which is what familiarity with the classic texts would produce.
- **Every configuration leans towards "follows".** The woven prompt and the Claude-taught student shift that lean most (−7 and −17 points, z −3.1 and −5.2) without changing discrimination: a change of threshold, not of skill.

The limits: `verdict` and the extra corruptions were added after the first run, in response to the letter bias, so the predictions came first but the main measure did not. Claude wrote every corruption and paraphrase, and the author has checked only the 20-item packet, not the 132 single passages. The 87 units come from 22 arguments; treating the 22 as the units leaves every conclusion unchanged (2B → 4B +17, z 2.9; constant pushes −16 to −22, z −2.4 to −3.8). `objreply` is near ceiling at 4B, and a blind judge noted that keyword matching rules out many distractors. The 9B is 4-bit, so 4B → 9B is compared within the 4-bit stack.

[Saved outputs, baseline packets and full write-up](../docs/reviews/real-text-skill/results.md)
