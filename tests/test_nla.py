"""Pilot AM helpers in wi/nla.py: marker location, explanation parsing, quote stripping and FVE. No model download."""
import numpy as np
import pytest

from wi import nla


def test_find_marker_uses_template_neighbours():
    ids = [5, 29, 158983, 510, 7, 158983, 9]          # the second marker has the wrong neighbours
    assert nla.find_marker(ids, 158983, 29, 510) == 2
    with pytest.raises(ValueError):
        nla.find_marker([1, 158983, 2], 158983, 29, 510)
    with pytest.raises(ValueError):
        nla.find_marker([29, 158983, 510, 29, 158983, 510, 0], 158983, 29, 510)


def test_parse_explanation_accepts_closed_and_truncated_tags():
    assert nla.parse_explanation("x <explanation>\n a b \n</explanation> y") == "a b"
    assert nla.parse_explanation("<explanation> cut off mid") == "cut off mid"
    assert nla.parse_explanation("no tags here") is None


def test_has_cjk():
    assert nla.has_cjk("概念")
    assert not nla.has_cjk("plain English — with a dash")


def test_strip_quotes_removes_long_copied_runs_only():
    ctx = "Therefore every man is mortal, and Socrates is a man."
    expl = 'Syllogism; ends with "Socrates is a man." Short echo: every man.'
    out = nla.strip_quotes(expl, ctx)
    assert "Socrates is a man" not in out
    assert "every man" in out                          # two words: below the 5-token threshold
    assert "…" in out
    assert nla.strip_quotes("nothing shared at all here", ctx) == "nothing shared at all here"
    assert 0 < nla.quoted_fraction(expl, ctx) < 1


def test_fve_matches_definition():
    rng = np.random.default_rng(0)
    gold = rng.normal(size=(64, 16)) + 3
    f, mse, base = nla.fve(gold * 2.5, gold)            # same directions, other norm: perfect after normalization
    assert f == pytest.approx(1.0) and mse.max() < 1e-12
    mean_pred = np.repeat(nla.normalize(gold).mean(0, keepdims=True), 64, 0)
    f_mean, _, _ = nla.fve(mean_pred, gold)            # the mean direction is close to, not at, the baseline
    assert f_mean < 0.2
    noisy = gold.copy()
    noisy[0] = np.nan
    f_nan, _, _ = nla.fve(noisy, gold)
    assert f_nan == pytest.approx(1.0)
