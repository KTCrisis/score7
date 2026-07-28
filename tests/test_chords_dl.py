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


def test_chain_prefers_btc(monkeypatch):
    monkeypatch.setattr(chords_dl, "try_btc", lambda p, bt: [{"chord": "Fm"}])
    grid, src = chords_dl.estimate_chords_chain("x", [0, 1], lambda: [{"chord": "C"}])
    assert src == "btc" and grid == [{"chord": "Fm"}]


def test_chain_falls_back_to_madmom_then_template(monkeypatch):
    monkeypatch.setattr(chords_dl, "try_btc", lambda p, bt: None)
    monkeypatch.setattr(chords_dl, "try_madmom_chords", lambda p, bt: [{"chord": "Am"}])
    grid, src = chords_dl.estimate_chords_chain("x", [0, 1], lambda: [{"chord": "C"}])
    assert src == "madmom"

    monkeypatch.setattr(chords_dl, "try_madmom_chords", lambda p, bt: None)
    grid, src = chords_dl.estimate_chords_chain("x", [0, 1], lambda: [{"chord": "C"}])
    assert src == "template" and grid == [{"chord": "C"}]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


def test_grid_publishes_no_fabricated_confidence():
    """BTC et madmom ne rendent pas de confiance par segment. Publier 1.0 se lit
    comme une certitude mesurée sur la route la plus fiable des trois, alors que
    c'est un remplissage : la clé est absente, la route cosinus garde la sienne."""
    from score7_mcp import chords_dl

    segs = [(0.0, 2.0, "C", "C:maj"), (2.0, 4.0, "Am", "A:min")]
    grid = chords_dl._segments_to_grid(segs, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert grid and all("conf" not in s for s in grid)
    assert [s["chord"] for s in grid] == ["C", "Am"]
