import json

import pytest

from scripts import human_regrade as hr


def test_kappa_perfect_none_and_chance():
    assert hr.kappa([True, False, True], [True, False, True]) == 1
    assert hr.kappa([True, True], [True, True]) is None          # no variation: undefined
    assert hr.kappa([True, True, False, False], [True, False, True, False]) == 0


def test_kappa_rejects_unequal_lists():
    with pytest.raises(ValueError):
        hr.kappa([True], [True, False])


def test_packet_is_blind_and_complete():
    packet, key = hr.build_packet()
    assert len(packet["items"]) == 36 and len(key) == 36
    assert "config" not in json.dumps(packet)                      # no configuration name anywhere
    assert {i["id"] for i in packet["items"]} == set(key)
    for set_name in ("revision", "transfer"):
        cells = sorted((k["configuration"], k["case"]) for k in key.values() if k["set"] == set_name)
        assert cells == sorted((c, n) for c in ("base", "student", "woven") for n in range(1, 7))


def test_committed_packet_matches_the_builder():
    packet, key = hr.build_packet()
    assert json.loads((hr.OUT / "packet.json").read_text()) == json.loads(json.dumps(packet))
    assert json.loads((hr.OUT / "key.json").read_text()) == key


def _grades_from(source):
    _, key = hr.build_packet()
    paired = hr.paired_grades()
    return {"packet": hr.PACKET_ID, "grader": "test",
            "grades": {i: dict(paired[(k["set"], k["configuration"], k["case"])][source]) for i, k in key.items()}}


def test_scoring_codex_grades_reproduces_experiment_ak():
    res = hr.score(_grades_from("codex"))
    assert res["missing"] == []
    c = res["criteria"]
    assert [c[k]["agree_author"] for k in ("targeted_question", "relevant_fact", "irrelevant_fact",
                                           "transfers_core_distinction")] == [17, 17, 18, 17]
    assert round(c["targeted_question"]["kappa_author"], 2) == 0.88
    assert round(c["relevant_fact"]["kappa_author"], 2) == 0.85
    assert round(c["transfers_core_distinction"]["kappa_author"], 2) == 0.77
    assert all(v["agree_codex"] == 18 for v in c.values())
    assert c["targeted_question"]["passes"]["student"] == {"human": 3, "author": 4, "codex": 3, "n": 6}


def test_ungraded_labels_are_reported_not_counted():
    grades = _grades_from("author")
    grades["grades"]["R01"]["relevant_fact"] = None
    del grades["grades"]["T01"]
    res = hr.score(grades)
    assert sorted(res["missing"]) == ["R01.relevant_fact", "T01.transfers_core_distinction"]
    assert res["criteria"]["relevant_fact"]["n"] == 17
    assert res["criteria"]["transfers_core_distinction"]["n"] == 17


def test_grades_for_another_packet_are_refused():
    with pytest.raises(ValueError):
        hr.score({"packet": "something-else", "grades": {}})
