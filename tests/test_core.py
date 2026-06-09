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
