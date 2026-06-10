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


def test_merge_grid_filters_short_segments():
    seq = [("Am", 0.9)] * 6 + [("F", 0.8)] * 1 + [("Am", 0.9)] * 6
    beat_times = np.arange(13) * 0.5
    grid = core._merge_grid(seq, beat_times, min_beats=2)
    assert all(s["beats"] >= 2 for s in grid)
    assert grid[0]["chord"] == "Am"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
