"""Pilot AL items (scripts/philbench_items.corrupt / objreply): keys, orders and rotation, no model download."""
import json
import re
from collections import Counter

from scripts import philbench_items as items_mod

ROOT = items_mod.ROOT
SENTENCE = re.compile(r"(?<=[.?!])\s+(?=[A-Z])")


def corruption_data():
    return json.loads((ROOT / "data/pilot-real-corruption.json").read_text())


def objreply_data():
    return json.loads((ROOT / "data/pilot-real-objreply.json").read_text())


def test_every_corruption_changes_exactly_the_recorded_sentence():
    for it in corruption_data()["items"]:
        orig, corr = SENTENCE.split(it["original"]), SENTENCE.split(it["corrupted"])
        assert len(orig) == len(corr), it["id"]
        changed = [i + 1 for i, (a, b) in enumerate(zip(orig, corr)) if a != b]
        assert changed == [it["corrupted_sentence"]], it["id"]


def test_corruption_items_record_source_type_and_paraphrase_length():
    data = corruption_data()
    sources = {p.stem for p in (ROOT / "data/sources").glob("*.txt")}
    for it in data["items"]:
        assert it["source"] in sources and it["edition"] and it["location"], it["id"]
        assert it["corruption_type"] in data["corruption_types"], it["id"]
        assert 0.85 <= len(it["paraphrase"]) / len(it["original"]) <= 1.15, it["id"]


def test_corrupt_items_cover_three_pairs_in_both_orders():
    items = items_mod.corrupt()
    n = len(corruption_data()["items"])
    assert len(items) == n * 3 * 2
    assert len({(it["item"], it["variant"]) for it in items}) == len(items)
    for it in items:
        assert it["labels"] == ["A", "B"] and it["user"].startswith(items_mod.CORRUPT_ASK)
        if it["pair"] == "orig-para":
            assert it["expected"] is None and it["original_letter"] in "AB"
        else:
            keyed = it["options"][["A", "B"].index(it["expected"])]
            assert keyed in ("original", "paraphrase") and "corrupted" in it["options"]
    keys = Counter(it["expected"] for it in items if it["expected"])
    assert keys["A"] == keys["B"]


def test_corrupt_prompt_shows_the_texts_in_option_order():
    data = {it["id"]: it for it in corruption_data()["items"]}
    for it in items_mod.corrupt():
        src = data[it["item"]]
        a, b = (src[k] for k in it["options"])
        assert f"A. {a}\n\nB. {b}" in it["user"]


def test_objreply_keys_point_at_the_true_partner_and_rotate():
    data = objreply_data()
    pairs = {p["id"]: (ex["id"], p) for ex in data["exchanges"] for p in ex["pairs"]}
    items = items_mod.objreply()
    assert len(items) == len(pairs) * 2 * 2
    assert len({(it["item"], it["variant"]) for it in items}) == len(items)
    for it in items:
        assert len(it["options"]) == len(set(it["options"])) == 4
        assert it["options"][items_mod.LETTERS.index(it["expected"])] == it["item"]
        assert all(pairs[o][0] == it["exchange"] for o in it["options"]), "distractors come from the same exchange"
    by_form = Counter(it["expected"] for it in items if it["form"] == "original")
    assert set(by_form) == set("ABCD") and max(by_form.values()) - min(by_form.values()) <= 1


def test_objreply_paraphrase_form_mirrors_the_original():
    items = {(it["item"], it["variant"]): it for it in items_mod.objreply()}
    for (item, variant), it in items.items():
        direction, form = variant.split("|")
        if form == "original":
            twin = items[(item, f"{direction}|paraphrase")]
            assert twin["options"] == it["options"] and twin["expected"] == it["expected"]
            assert twin["user"] != it["user"]


def test_ai_default_task_set_is_unchanged():
    assert {it["task"] for it in items_mod.all_items(("critique", "validity"))} == {"critique", "validity"}
    assert items_mod.AI_TASKS == ("philpapers", "philbench", "critique", "validity")


def test_extra_corruptions_change_one_sentence_and_dropped_ones_are_skipped():
    extra = json.loads((ROOT / "data/pilot-real-corruption-extra.json").read_text())["items"]
    originals = {it["id"]: it for it in corruption_data()["items"]}
    verdict = items_mod.verdict()
    for item, xs in extra.items():
        sents = SENTENCE.split(originals[item]["original"])
        for k, x in enumerate(xs, 2):
            assert 1 <= x["sentence"] <= len(sents) and x["text"] != sents[x["sentence"] - 1], (item, k)
            assert x["type"] in corruption_data()["corruption_types"], (item, k)
            forms = {it["form"] for it in verdict if it["item"] == item}
            assert (f"corrupted{k}" in forms) == (not x.get("drop")), (item, k)
    for it in verdict:
        assert it["expected"] == (it["follows_letter"] if not it["form"].startswith("corrupted")
                                  else "AB"[1 - "AB".index(it["follows_letter"])])
