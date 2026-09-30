"""Tests des helpers de la chaîne d'accords deep-learning (sans charger torch/BTC)."""

from score7_mcp import chords_dl


def test_label_to_score7():
    assert chords_dl._label_to_score7("C") == ("C", "C")            # majeur pur
    assert chords_dl._label_to_score7("F:min") == ("Fm", "F:min")
    assert chords_dl._label_to_score7("A#:sus4") == ("A#", "A#:sus4")  # sus → maj compact
    assert chords_dl._label_to_score7("D:min7") == ("Dm", "D:min7")
    assert chords_dl._label_to_score7("G:dim") == ("Gm", "G:dim")   # tierce mineure
    assert chords_dl._label_to_score7("N") == ("N", "N")


def test_segments_to_grid_aligns_and_filters():
    beat_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    segs = [(0.0, 1.0, "Fm", "F:min"), (1.0, 2.5, "C", "C"), (2.5, 3.0, "N", "N")]
    grid = chords_dl._segments_to_grid(segs, beat_times)
    assert [g["chord"] for g in grid] == ["Fm", "C"]   # "N" filtré
    assert grid[0]["start_beat"] == 0 and grid[0]["beats"] >= 1
    assert grid[1]["chord_full"] == "C"


def test_segments_before_first_beat_keep_distinct_times():
    """Plusieurs accords avant le premier beat gardent des timestamps distincts
    (et ne s'écrasent pas tous à bt[0])."""
    beat_times = [1.0, 2.0, 3.0]
    segs = [(0.0, 0.5, "Fm", "F:min"), (0.5, 1.0, "C", "C")]
    grid = chords_dl._segments_to_grid(beat_times=beat_times, segs=segs)
    times = [g["time"] for g in grid]
    assert times == [0.0, 0.5]  # distincts, = vrais débuts de segment


IDX2C = {0: "C", 1: "A:min", 2: "F:maj7", 3: "N"}


def _probs(rows, spf=0.1):
    """Probabilités synthétiques : une ligne par trame, colonnes = IDX2C."""
    import numpy as np
    p = np.asarray(rows, dtype=float)
    return p, spf, p.shape[0] * spf, IDX2C


def test_chain_prefers_btc(monkeypatch):
    monkeypatch.setattr(chords_dl, "btc_probabilities", lambda p: _probs([[0.9, 0.05, 0.05, 0]] * 20))
    grid, src = chords_dl.estimate_chords_chain("x", [0.0, 1.0, 2.0], lambda: [{"chord": "C"}])
    assert src == "btc" and [g["chord"] for g in grid] == ["C"]
    assert grid[0]["time"] == 0.0 and grid[0]["end"] == 2.0 and "start_beat" in grid[0]


def test_chain_falls_back_to_madmom_then_template(monkeypatch):
    monkeypatch.setattr(chords_dl, "btc_probabilities", lambda p: None)
    monkeypatch.setattr(chords_dl, "try_madmom_chords", lambda p, bt: [{"chord": "Am"}])
    grid, src = chords_dl.estimate_chords_chain("x", [0, 1], lambda: [{"chord": "C"}])
    assert src == "madmom"

    monkeypatch.setattr(chords_dl, "try_madmom_chords", lambda p, bt: None)
    grid, src = chords_dl.estimate_chords_chain("x", [0, 1], lambda: [{"chord": "C"}])
    assert src == "template" and grid == [{"chord": "C"}]




def test_grid_publishes_no_fabricated_confidence():
    """madmom ne rend pas de confiance par segment : publier 1.0 se lirait comme une
    certitude mesurée alors que c'est un remplissage. La clé reste absente sur cette
    route ; BTC publie la sienne (`confidence`, tirée de ses probabilités)."""
    from score7_mcp import chords_dl

    segs = [(0.0, 2.0, "C", "C:maj"), (2.0, 4.0, "Am", "A:min")]
    grid = chords_dl._segments_to_grid(segs, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert grid and all("conf" not in s for s in grid)
    assert [s["chord"] for s in grid] == ["C", "Am"]


def test_btc_segments_are_dated_in_seconds_with_confidence_and_candidates():
    rows = [[0.8, 0.15, 0.05, 0]] * 10 + [[0.1, 0.6, 0.3, 0]] * 10
    segs = chords_dl.btc_segments(*_probs(rows))
    assert [s["chord"] for s in segs] == ["C", "Am"]
    assert segs[0]["time"] == 0.0 and abs(segs[1]["end"] - 2.0) < 1e-6
    assert 0.5 < segs[0]["confidence"] <= 1.0
    assert segs[1]["candidates"][0]["chord"] == "Am" and len(segs[1]["candidates"]) >= 2
    assert all(s["source"] == "btc" for s in segs)


def test_btc_segments_absorb_flicker():
    """Une trame isolée d'un autre accord n'est pas un changement d'accord."""
    rows = [[0.9, 0.1, 0, 0]] * 10 + [[0.1, 0.9, 0, 0]] + [[0.9, 0.1, 0, 0]] * 10
    segs = chords_dl.btc_segments(*_probs(rows))
    assert [s["chord"] for s in segs] == ["C"]


def test_rich_label_is_kept_within_the_reduced_class():
    """F:maj7 se réduit à F : l'étiquette riche est gardée pour l'affichage."""
    rows = [[0.05, 0.05, 0.9, 0]] * 10
    segs = chords_dl.btc_segments(*_probs(rows))
    assert segs[0]["chord"] == "F" and segs[0]["chord_full"] == "F:maj7"


def test_tie_break_only_on_uncertain_segments():
    import numpy as np
    chroma = np.zeros((12, 20))
    chroma[[9, 0, 4], :] = 1.0          # la, do, mi : A minor clearly
    ft = np.arange(20) * 0.1
    sure = {"chord": "C", "time": 0.0, "end": 2.0, "confidence": 0.9,
            "candidates": [{"chord": "C", "p": 0.9}, {"chord": "Am", "p": 0.1}], "source": "btc"}
    unsure = {**sure, "confidence": 0.5, "candidates": [{"chord": "C", "p": 0.5}, {"chord": "Am", "p": 0.4}]}
    out = chords_dl.tie_break([dict(sure), dict(unsure)], chroma, ft)
    assert out[0]["chord"] == "C"                                   # BTC sûr : on ne touche pas
    assert out[1]["chord"] == "Am" and out[1]["source"] == "btc+chroma" and out[1]["chord_btc"] == "C"
    assert out[1]["chord_full"] == "A:min"
