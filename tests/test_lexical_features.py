"""Invariants of the query-conditioned lexical block.

These are the repo's own documented guarantees, not coverage for its own sake. Each test
below corresponds to a claim `scripts/lexical_features.py` makes in prose about why it is
written the way it is — the point is that the claim stops being only a comment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lexical_features as lf


# --------------------------------------------------------------------------- normalise


def test_normalise_folds_hyphens_and_slashes():
    """The documented reason this exists: briefs write `alkali-activated`, papers write both.

    Folding both sides to spaces is what makes a phrase match on either form. If this
    regresses, every `must`/`nice` term containing a hyphen silently stops matching.
    """
    assert lf.normalise("Alkali-Activated") == "alkali activated"
    assert lf.normalise("CO2/clinker") == "co2 clinker"
    assert lf.normalise("  spaced   out  ") == "spaced out"


@pytest.mark.parametrize("empty", [None, np.nan, float("nan"), ""])
def test_normalise_maps_missing_to_empty_string_not_the_word_nan(empty):
    """NULL is not 0, and it is not the string "nan" either (README.md, conventions).

    `str(np.nan)` is `'nan'`, which would tokenise to a real term and quietly pollute
    every BM25 pool with a token that means "this field was missing".
    """
    assert lf.normalise(empty) == ""


def test_tokenise_drops_punctuation_and_keeps_digits():
    assert lf.tokenise("Reduce CO2 by >50%, vs OPC.") == ["reduce", "co2", "by", "50", "vs", "opc"]


# --------------------------------------------------------------------------- phrase_pattern


def test_phrase_pattern_respects_word_boundaries():
    """The exact false positive named in the docstring: `NER` must not match inside `generic`."""
    pat = lf.phrase_pattern("NER")
    assert pat.search(lf.normalise("transformer based NER models"))
    assert not pat.search(lf.normalise("a generic approach"))


def test_phrase_pattern_matches_across_hyphen_and_whitespace_variants():
    """A hyphenated brief term has to match the spaced form in a paper, and vice versa."""
    pat = lf.phrase_pattern("alkali-activated")
    for variant in ("alkali-activated binders", "alkali activated binders", "alkali  activated"):
        assert pat.search(lf.normalise(variant)), variant


def test_phrase_pattern_on_an_empty_term_never_matches():
    """An empty term must not become a pattern that matches everything.

    A brief with a blank entry in `terms_must_include` would otherwise score every paper
    as a full must-term hit.
    """
    pat = lf.phrase_pattern("   ")
    assert not pat.search("anything at all")
    assert not pat.search("")


# --------------------------------------------------------------------------- brief assembly


def test_build_brief_texts_tolerates_every_shape_a_term_column_arrives_in():
    """Term-list columns arrive as list, ndarray, None or NaN depending on the export."""
    row = pd.Series(
        {
            "objective": "Find alternative binder chemistries.",
            "problem_statement": None,
            "terms_must_include": ["geopolymer", "LC3"],
            "terms_nice_to_have": np.array(["supplementary cementitious"]),
            "terms_exclude": np.nan,
            "domain_industry": "Cement",
            "domain_application": None,
            "domain_technology_focus": ["alternative binders"],
        }
    )
    texts = lf.build_brief_texts(row)

    assert set(texts) == set(lf.BRIEF_KEYS)
    assert "geopolymer" in texts["must"] and "lc3" in texts["must"]
    assert "supplementary cementitious" in texts["nice"]
    assert texts["prob"] == ""                      # None, not "none"
    assert "nan" not in texts["dom"]                # the None field must not leak a token
    assert "cement" in texts["dom"]


def test_collect_briefs_keys_on_use_case_and_carries_term_lists():
    df = pd.DataFrame(
        {
            "use_case_key": ["a", "a", "b"],
            "objective": ["obj A", "obj A", "obj B"],
            "problem_statement": [None, None, None],
            "terms_must_include": [["x"], ["x"], ["y"]],
            "terms_nice_to_have": [[], [], []],
            "terms_exclude": [["z"], ["z"], []],
            "domain_industry": ["i", "i", "i"],
            "domain_application": [None, None, None],
            "domain_technology_focus": [[], [], []],
        }
    )
    briefs = lf.collect_briefs(df)

    assert set(briefs) == {"a", "b"}
    assert briefs["a"]["terms"]["must"] == ["x"]
    assert briefs["a"]["terms"]["exclude"] == ["z"]
    assert briefs["b"]["terms"]["exclude"] == []
    assert briefs["b"]["texts"]["obj"] == "obj b"


# --------------------------------------------------------------------------- the control


def test_derangements_never_leaves_a_use_case_its_own_brief():
    """The falsification control is only valid if no key keeps its own brief.

    README.md's conventions: anything claiming to read the brief must be rebuildable against
    deliberately wrong briefs. A shuffle with one fixed point would leak a correct pairing
    into the control and understate the effect it exists to detect.
    """
    keys = ["carbon_capture", "cement_binders", "ner", "soil_microbiome", "solar_leo",
            "tech_forecasting"]
    for seed in range(25):
        mapping = lf.derangements(keys, seed=seed)
        assert set(mapping) == set(keys)
        assert sorted(mapping.values()) == sorted(keys)      # a permutation, nothing dropped
        assert all(k != v for k, v in mapping.items()), f"fixed point at seed={seed}"


def test_derangements_is_deterministic_for_a_given_seed():
    keys = ["a", "b", "c", "d"]
    assert lf.derangements(keys, seed=7) == lf.derangements(keys, seed=7)
