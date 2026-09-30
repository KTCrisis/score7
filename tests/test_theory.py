"""Tests du modèle d'accord (étiquettes MIREX, réduction, gabarits)."""

import numpy as np

from score7_mcp import theory


def test_parse_and_compact():
    assert theory.parse("A#:maj7") == (10, "maj7")
    assert theory.parse("C") == (0, "maj")
    assert theory.parse("N") is None
    assert theory.compact("F:min7") == "Fm"
    assert theory.compact("A#:maj7") == "A#"
    assert theory.compact("B:hdim7") == "Bm"


def test_pitch_classes_accept_compact_minor():
    assert theory.pitch_classes("Dm") == {2, 5, 9}
    assert theory.pitch_classes("A#:maj7") == {10, 2, 5, 9}


def test_fit_prefers_the_played_chord():
    obs = np.zeros(12)
    obs[[2, 5, 9]] = 1.0                    # ré fa la
    assert theory.fit(obs, "Dm") > theory.fit(obs, "A#")
    assert theory.fit(np.zeros(12), "Dm") == 0.0


def test_mirex_from_compact():
    assert theory.mirex("Am") == "A:min" and theory.mirex("F#") == "F#" and theory.mirex("C#m") == "C#:min"


def test_key_scale_and_in_key():
    em = theory.key_scale("E", "minor")
    assert theory.in_key("Em", em) and theory.in_key("C", em) and theory.in_key("B", em)   # V major via the leading tone
    assert not theory.in_key("F#", em) and not theory.in_key("Dm", em)
