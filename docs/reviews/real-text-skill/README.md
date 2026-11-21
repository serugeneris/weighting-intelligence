# Pilot AL: real-text skill items

Plan: [docs/plans/real-text-skill-pilot.md](../../plans/real-text-skill-pilot.md). Run on the full scale ladder and register grid on 2026-11-20: results in [results.md](results.md), write-up in [findings part 7, exp. AL](../../../findings/7-register-versus-skill.md#exp-al). The author's baseline (19/20) is in `baseline/answers-MA.json`; the item review below is still open.

## Stage 1: author's item check

[item-review.md](item-review.md) lists every item, with the corrupted sentence in bold. Record verdicts in the `review` field of each item in `data/pilot-real-corruption.json` and `data/pilot-real-objreply.json`, then rerun `python3 scripts/real_text_packets.py review`.

Points that most need a look (also noted per item under *Adaptation*):

- **Reconstructed or joined text:** `galileo-void` (conclusion reworded), `hume-four-hypotheses` (last sentence restored, lost in the OCR), `vitoria-kill-all-guilty` (premise taken from §46), `vit-deuteronomy` and `vit-almain-dilemma` (argument's conclusion made explicit), `lc-3-5-east-west` (last reply sentence from Clarke's §2).
- **Contested validity:** `anselm-proslogium-2`. As a reductio from its stated premises it goes through, but many readers dispute it.
- **Analogical, not deductive:** `hume-design-analogy`.
- **Replies that echo the objection's words** (leaves, painter, lost island, parts) are flagged `echo` and reported separately.
- **Coverage:** 22 arguments rather than the planned ~30. Corruption types: 7 negation, 6 quantifier, 4 drop-premise, 3 overreach, 2 conditional reversal. Task 2 paraphrases run 2–24% shorter than the originals; within an item all options are paraphrased together, so length does not point to the key.

## Stage 2: blind human baseline

`baseline/answer.html` is the 20-item answer sheet (10 per task, fixed seed); graders open it in a browser and download their answers. Keep `baseline/key.json` away from them. Score with:

```bash
python3 scripts/real_text_packets.py score answers-XX.json answers-YY.json
```

Gate: at least 85% agreement with the keys per grader; fix or drop disputed items otherwise.

## Sources

Public-domain OCR texts in `data/sources/`, each with its edition and archive.org URL in a header: Descartes (Haldane & Ross, vols. I–II), Leibniz–Clarke (1717), Hume's *Dialogues* (1779), Berkeley's *Three Dialogues* (1906), Anselm with Gaunilo (Deane, 1903), Vitoria (Bate, 1917), Galileo (Crew & de Salvio, 1914).
