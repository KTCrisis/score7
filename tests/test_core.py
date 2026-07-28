"""Tests sur signaux synthétiques — pas de fixtures audio lourds."""

import numpy as np
import pytest

from score7_mcp import core


SR = 22050


def _tone(freqs, dur=3.0, sr=SR):
    """Somme de sinus (un accord) sur `dur` secondes."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    y = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    return (y / np.max(np.abs(y))).astype(np.float32)


def test_spectral_centroid_tracks_pitch():
    """Un sinus grave a un centroïde plus bas qu'un sinus aigu."""
    low = core.spectral_profile(_tone([220.0]), SR)["centroid_hz"]
    high = core.spectral_profile(_tone([2000.0]), SR)["centroid_hz"]
    assert low < high


def test_key_detects_c_major_triad():
    """Un accord Do majeur soutenu (C-E-G) doit donner une tonique Do."""
    y = _tone([261.63, 329.63, 392.0])  # C4 E4 G4
    chroma = np.mean(__import__("librosa").feature.chroma_cqt(y=y, sr=SR), axis=1)
    key = core.estimate_key(chroma)
    assert key["root"] == "C"


def test_chord_vote_corrects_relative_mode():
    """Vote d'accords : une cadence i–iv–V en La mineur doit voter pour A minor,
    pas pour son relatif Do majeur."""
    grid = [
        {"chord": "Am", "beats": 8},   # i
        {"chord": "Dm", "beats": 4},   # iv
        {"chord": "E", "beats": 4},    # V (harmonique)
        {"chord": "Am", "beats": 8},   # i
    ]
    maj, minr = core._chord_vote_scores(grid)
    # A mineur (index 9) doit dominer le vote mineur
    assert int(np.argmax(minr)) == 9
    # et le vote mineur sur A doit battre le vote majeur sur C (relatif)
    assert minr[9] >= maj[0]


def _click_track(times, dur, gain=1.0, freq=1000.0, sr=SR):
    import librosa
    return gain * librosa.clicks(times=times, sr=sr, length=int(sr * dur),
                                 click_freq=freq).astype(np.float32)


def _accented_clicks(bpm, beats_per_bar, dur=40.0, sr=SR):
    """Clics réguliers + clic fort sur le premier temps de chaque mesure."""
    beats = np.arange(0.0, dur, 60.0 / bpm)
    strong = beats[::beats_per_bar]
    return (_click_track(beats, dur, gain=0.35, sr=sr)
            + _click_track(strong, dur, gain=1.0, freq=600.0, sr=sr))


def test_tempo_on_click_track():
    """Un train de clics à 100 BPM doit sortir ~100, et 50/200 en candidats d'octave."""
    bpm = 100.0
    y = _click_track(np.arange(0.0, 30.0, 60.0 / bpm), dur=30.0)
    tempo, beats, beat_times = core.estimate_tempo(y, SR)
    assert abs(tempo["bpm"] - bpm) < 3
    cands = [c["bpm"] for c in tempo["bpm_candidates"]]
    assert any(abs(c - bpm / 2) < 3 for c in cands)
    assert any(abs(c - bpm * 2) < 6 for c in cands)


def test_meter_detects_3_4():
    y = _accented_clicks(120, 3)
    tempo, beats, beat_times = core.estimate_tempo(y, SR)
    meter = core.estimate_meter(y, SR, beats, beat_times)
    assert meter["beats_per_bar"] == 3
    assert meter["meter"] == "3/4"


def test_meter_detects_4_4():
    y = _accented_clicks(120, 4)
    tempo, beats, beat_times = core.estimate_tempo(y, SR)
    meter = core.estimate_meter(y, SR, beats, beat_times)
    assert meter["beats_per_bar"] == 4
    assert meter["meter"] == "4/4"


def test_meter_detects_5_4():
    y = _accented_clicks(120, 5)
    tempo, beats, beat_times = core.estimate_tempo(y, SR)
    meter = core.estimate_meter(y, SR, beats, beat_times)
    assert meter["beats_per_bar"] == 5
    assert meter["meter"] == "5/4"


def test_subdivision_ternary_vs_binary():
    """Sous-clics en triolets → ternaire ; en croches → binaire (base du 6/8)."""
    import librosa
    bpm, dur = 80.0, 40.0
    beats = np.arange(0.0, dur, 60.0 / bpm)
    for divisions, expected in ((3, "ternary"), (2, "binary")):
        subs = np.arange(0.0, dur, 60.0 / bpm / divisions)
        y = _click_track(beats, dur) + _click_track(subs, dur, gain=0.4, freq=1500.0)
        oenv = librosa.onset.onset_strength(y=y, sr=SR)
        assert core._subdivision(oenv, SR, bpm) == expected


def test_stereo_mono_detection():
    y = _tone([440.0])
    assert core.stereo_width(np.stack([y, y]))["mono"] is False  # 2 canaux identiques
    assert core.stereo_width(y[np.newaxis, :])["mono"] is True


def test_rhythm_prefers_beat_this(monkeypatch):
    """Beat This! disponible → il prime sur madmom et sur librosa."""
    fake = {"bpm": 128.0, "beats_per_bar": 4, "source": "beat_this",
            "beat_times": np.arange(0.0, 12.0, 60.0 / 128.0)}
    monkeypatch.setattr(core, "try_beat_this", lambda p: fake)
    monkeypatch.setattr(core, "try_madmom_rhythm",
                        lambda p, **k: {"bpm": 100.0, "beats_per_bar": 3, "source": "madmom",
                                        "beat_times": np.arange(0.0, 12.0, 0.6)})
    y = _click_track(np.arange(0.0, 12.0, 0.6), dur=12.0)
    tempo, meter, beats, bt = core.estimate_rhythm(y, SR, path="x")
    assert tempo["source"] == "beat_this"
    assert tempo["bpm"] == 128.0
    assert meter["beats_per_bar"] == 4


def test_rhythm_falls_back_to_madmom(monkeypatch):
    """Beat This! absent → madmom prend la main avant librosa."""
    monkeypatch.setattr(core, "try_beat_this", lambda p: None)
    monkeypatch.setattr(core, "try_madmom_rhythm",
                        lambda p, **k: {"bpm": 100.0, "beats_per_bar": 4, "source": "madmom",
                                        "beat_times": np.arange(0.0, 12.0, 0.6)})
    y = _click_track(np.arange(0.0, 12.0, 0.6), dur=12.0)
    tempo, meter, beats, bt = core.estimate_rhythm(y, SR, path="x")
    assert tempo["source"] == "madmom"
    assert tempo["bpm"] == 100.0


def test_estimate_key_tags_source():
    """estimate_key marque sa provenance (source unique de vérité pour la tonalité)."""
    import librosa
    y = _tone([261.63, 329.63, 392.0])
    chroma = np.mean(librosa.feature.chroma_cqt(y=y, sr=SR), axis=1)
    assert core.estimate_key(chroma)["source"] == "krumhansl"


def test_reconcile_noop_on_cnn_key():
    """Une clé CNN (source autoritaire) n'est jamais réécrite par la mélodie."""
    key = {"root": "F", "mode": "minor", "source": "madmom_cnn"}
    notes = [{"pitch": 60 + pc} for pc, n in {0: 50, 5: 20, 8: 15}.items() for _ in range(n)]
    assert core.reconcile_key_with_melody(key, notes) is key  # inchangé, pas de réconciliation


def test_norm_key_label_flats_to_sharps():
    """Labels madmom (bémols) normalisés vers NOTE_NAMES (dièses)."""
    assert core._norm_key_label("F minor") == ("F", "minor")
    assert core._norm_key_label("Db major") == ("C#", "major")
    assert core._norm_key_label("Bb minor") == ("A#", "minor")
    assert core._norm_key_label("C") == ("C", "major")  # mode par défaut


def _key(root, mode="major"):
    return {"root": root, "mode": mode, "score": 1.0, "runner_up": "x",
            "krumhansl_only": "x", "corrected": False}


def _notes(counts):
    """counts = {pitch_class: n} → liste de notes (octave 4)."""
    return [{"pitch": 60 + pc} for pc, n in counts.items() for _ in range(n)]


def test_reconcile_keeps_harmonic_tonic_over_fifth():
    """Cas Future Club : harmonie = F, mélodie dominée par le do (quinte de Fm) mais
    le fa est bien présent → la tonique reste F, le mode passe en mineur (tierce G#)."""
    key = _key("F", "major")
    notes = _notes({0: 50, 5: 20, 8: 15})  # C (quinte), F (tonique), G# (tierce min)
    out = core.reconcile_key_with_melody(key, notes)
    assert out["root"] == "F"
    assert out["mode"] == "minor"


def test_reconcile_rewrites_tonic_when_harmony_absent():
    """Si la tonique harmonique est quasi absente de la mélodie, on bascule sur la
    classe de hauteur dominante (garde-fou quand Krumhansl/accords se plantent)."""
    key = _key("D", "major")  # tonique D totalement absente de la mélodie ci-dessous
    notes = _notes({9: 50, 0: 20, 4: 30})  # A, C, E → Am
    out = core.reconcile_key_with_melody(key, notes)
    assert out["root"] == "A"
    assert out["mode"] == "minor"


def test_merge_grid_absorbs_short_segments_without_losing_time():
    """Un segment trop court est rendu au voisin tenu, il n'est plus jeté : la
    grille couvre le morceau de bout en bout, comme celle de `_despike` sur la
    route BTC. Sans cela, une durée d'accord n'est pas comparable d'une route à
    l'autre."""
    seq = [("Am", 0.9)] * 6 + [("F", 0.8)] * 1 + [("Am", 0.9)] * 6
    beat_times = np.arange(13) * 0.5
    grid = core._merge_grid(seq, beat_times, min_beats=2)
    assert all(s["beats"] >= 2 for s in grid)
    assert grid[0]["chord"] == "Am"
    assert sum(s["beats"] for s in grid) == len(seq)


def test_merge_grid_covers_the_track_in_every_shape():
    """Tête courte, suite de courts, bruit intégral : la somme des beats reste
    celle de la séquence. Le cas tout-court garde la grille brute, parce que tout
    absorber en un accord inventerait une tenue que personne n'a jouée."""
    cases = ([("F", 0.8)] + [("Am", 0.9)] * 6 + [("C", 0.9)] * 6,
             [("Am", 0.9)] * 4 + [("F", 0.8), ("G", 0.8)] + [("C", 0.9)] * 4,
             [("Am", 0.9), ("F", 0.8), ("C", 0.9), ("G", 0.9)])
    for seq in cases:
        grid = core._merge_grid(seq, np.arange(len(seq) + 1) * 0.5, min_beats=2)
        assert sum(s["beats"] for s in grid) == len(seq)
        assert grid[0]["start_beat"] == 0
    assert len(core._merge_grid(cases[-1], np.arange(5) * 0.5, min_beats=2)) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_annotate_bass_roots_reads_what_is_played_below(tmp_path):
    """Deux segments, deux fondamentales de basse distinctes : la mesure doit les
    séparer. C'est ce qui distingue un accord de son renversement, et rien ne
    permet de le retrouver depuis la seule grille."""
    import soundfile as sf

    sr = 22050
    def tone(f, d):
        t = np.linspace(0, d, int(sr * d), endpoint=False)
        return (0.4 * np.sin(2 * np.pi * f * t)).astype(np.float32)

    # 2 s de mi grave (E1, 41.2 Hz) puis 2 s de la grave (A1, 55 Hz)
    sf.write(tmp_path / "bass.wav", np.concatenate([tone(41.2, 2.0), tone(55.0, 2.0)]), sr)
    beat_times = np.arange(9) * 0.5          # 8 beats de 0,5 s
    grid = [{"chord": "E", "start_beat": 0, "beats": 4},
            {"chord": "A", "start_beat": 4, "beats": 4}]

    out = core.annotate_bass_roots(grid, str(tmp_path / "bass.wav"), beat_times)
    assert [s["bass_root"] for s in out] == ["E", "A"]


def test_annotate_bass_roots_stays_silent_when_it_cannot_measure(tmp_path):
    """Stem absent : le champ reste absent. Une fondamentale de repli se lirait
    comme une basse observée, ce qu'elle ne serait pas."""
    grid = [{"chord": "E", "start_beat": 0, "beats": 4}]
    out = core.annotate_bass_roots(grid, str(tmp_path / "nope.wav"), np.arange(5) * 0.5)
    assert "bass_root" not in out[0]
    assert core.annotate_bass_roots([], "x", np.arange(3)) == []
