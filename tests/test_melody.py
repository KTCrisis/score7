"""Test de l'extracteur de mélodie PESTO (skip si l'extra [melody] n'est pas installé)."""

import numpy as np
import pytest


def test_melody_pesto_tracks_tone(tmp_path):
    """PESTO sur un timbre harmonique tenu (A3, 220 Hz + harmoniques) doit retrouver La.
    Vérifie au passage le flush de la note finale (note tenue jamais 'changée') et le seuil
    de confiance `>=` (sinon une confiance uniforme à 1.0 ne passe jamais le percentile)."""
    pytest.importorskip("pesto")
    import soundfile as sf

    from score7_mcp import melody as mel

    sr = 22050
    t = np.linspace(0, 2.0, int(sr * 2), endpoint=False)
    y = sum((1.0 / k) * np.sin(2 * np.pi * 220.0 * k * t) for k in (1, 2, 3, 4))  # A3 + harmoniques
    y = (0.3 * y / np.max(np.abs(y))).astype(np.float32)
    f = tmp_path / "a220.wav"
    sf.write(f, y, sr)

    notes = mel._melody_pesto(str(f))
    assert notes, "PESTO devrait segmenter au moins une note sur un timbre stable"
    assert "A" in [n["name"][:-1] for n in notes]  # la classe La domine
    assert all(n["dur"] >= 0.1 for n in notes)  # durée mini respectée


def test_melody_pesto_short_clip_returns_empty(tmp_path):
    """Un clip trop court pour PESTO retombe proprement sur [] au lieu de crasher
    (la STFT interne de PESTO padde sur ~2048 samples)."""
    pytest.importorskip("pesto")
    import soundfile as sf

    from score7_mcp import melody as mel

    sr = 22050
    y = (0.3 * np.sin(2 * np.pi * 220.0 * np.linspace(0, 0.03, int(sr * 0.03),
                                                      endpoint=False))).astype(np.float32)
    f = tmp_path / "tiny.wav"
    sf.write(f, y, sr)
    assert mel._melody_pesto(str(f)) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_split_at_onsets_rearticulates_repeated_notes(tmp_path):
    """Deux attaques de la même hauteur séparées par un silence : la segmentation
    f0 n'y voit qu'une note ; les onsets du signal doivent la couper en deux."""
    import librosa  # noqa: F401 — dépendance cœur, pas d'extra

    from score7_mcp import melody as mel

    sr = 22050
    t1 = np.linspace(0, 0.4, int(sr * 0.4), endpoint=False)
    burst = (0.5 * np.sin(2 * np.pi * 220.0 * t1) * np.hanning(len(t1))).astype(np.float32)
    gap = np.zeros(int(sr * 0.1), dtype=np.float32)
    y = np.concatenate([burst, gap, burst])

    notes = [{"pitch": 57, "start": 0.0, "dur": 0.9, "name": "A3"}]
    out = mel._split_at_onsets(notes, y, sr)
    assert len(out) == 2, f"attendu 2 notes après ré-articulation, obtenu {len(out)}"
    assert out[0]["start"] == 0.0
    assert 0.35 <= out[1]["start"] <= 0.65  # la coupe tombe sur la 2e attaque
    assert all(n["dur"] >= 0.1 for n in out)


def test_velocities_follow_dynamics():
    """Une note dans la partie faible du signal reçoit une vélocité plus basse
    qu'une note dans la partie forte ; bornes [45, 112] respectées."""
    from score7_mcp import melody as mel

    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    soft = 0.05 * np.sin(2 * np.pi * 220.0 * t)
    loud = 0.8 * np.sin(2 * np.pi * 220.0 * t)
    y = np.concatenate([soft, loud]).astype(np.float32)

    notes = [{"pitch": 57, "start": 0.3, "dur": 0.3, "name": "A3"},
             {"pitch": 57, "start": 1.3, "dur": 0.3, "name": "A3"}]
    out = mel._velocities(notes, y, sr)
    assert out[0]["vel"] < out[1]["vel"]
    assert all(45 <= n["vel"] <= 112 for n in out)
