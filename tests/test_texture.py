"""Tests rythme + texture sur signaux synthétiques (+ quelques wav temporaires)."""

import librosa
import numpy as np
import soundfile as sf

from score7_mcp import texture

SR = 22050


def _sine(freq, dur=4.0, sr=SR):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


# --------------------------------------------------------------------------- unités
def test_band_isolates_frequency():
    """La bande kick (20–150 Hz) garde un 60 Hz ; la bande hats (>6 kHz) l'efface."""
    y = _sine(60.0)
    kick = texture._band(y, SR, *texture._BANDS["kick"])
    hats = texture._band(y, SR, *texture._BANDS["hats"])
    assert np.sqrt(np.mean(kick ** 2)) > 0.5 * np.sqrt(np.mean(y ** 2))
    assert np.sqrt(np.mean(hats ** 2)) < 0.05 * np.sqrt(np.mean(y ** 2))


def test_step_times_subdivides_evenly():
    beats = np.array([0.0, 1.0, 2.0])
    steps = texture._step_times(beats, steps_per_beat=4)
    assert len(steps) == 8  # 2 intervalles × 4 pas
    assert np.allclose(np.diff(steps), 0.25)


def test_fold_pattern_recovers_motif():
    """Force concentrée tous les 4 pas → motif replié maximal au pas 0, nul ailleurs."""
    s = np.zeros(64)
    s[::4] = 1.0
    folded = texture._fold_pattern(s, steps_per_bar=16)
    assert folded[0] == 1.0
    assert folded[1] < 0.01 and folded[2] < 0.01


def test_swing_straight_vs_ternary():
    beats = np.arange(0.0, 8.0, 1.0)
    straight = beats[:-1] + 0.5      # croche off pile au milieu
    ternary = beats[:-1] + 2 / 3     # retardée sur le 3e triolet
    assert abs(texture._swing(straight, beats) - 0.5) < 0.05
    assert abs(texture._swing(ternary, beats) - 2 / 3) < 0.05


def test_swing_none_when_sparse():
    beats = np.array([0.0, 1.0, 2.0])
    assert texture._swing(np.array([0.5]), beats) is None


def test_spectral_flux_static_below_moving():
    """Un sinus stationnaire a un flux spectral plus faible qu'un bruit blanc."""
    rng = np.random.default_rng(0)
    tone = texture._spectral_flux(_sine(440.0))
    noise = texture._spectral_flux(rng.standard_normal(SR * 2).astype(np.float32))
    assert tone < noise


# --------------------------------------------------------------------------- intégration (wav)
def _click(times, dur=8.0, freq=1000.0, gain=1.0, sr=SR):
    return (gain * librosa.clicks(times=times, sr=sr, length=int(sr * dur),
                                  click_freq=freq)).astype(np.float32)


def test_rhythm_pattern_kick_on_downbeats(tmp_path):
    """Kick grave sur chaque temps + hats aigus sur les croches : le pattern kick doit
    s'allumer en tête de beat (pas multiples de steps_per_beat), les hats plus densément."""
    bpm, dur = 120.0, 8.0
    beats = np.arange(0.0, dur, 60.0 / bpm)
    eighths = np.arange(0.0, dur, 60.0 / bpm / 2)
    y = _click(beats, dur, freq=80.0, gain=1.0) + _click(eighths, dur, freq=9000.0, gain=0.5)
    drums = tmp_path / "drums.wav"
    sf.write(drums, y, SR)

    rp = texture.rhythm_pattern(str(drums), beats, sr=SR, steps_per_beat=4, beats_per_bar=4)
    assert rp["available"]
    kick = np.array(rp["bands"]["kick"]["pattern"])
    hats = np.array(rp["bands"]["hats"]["pattern"])
    # le kick s'allume en tête de beat (pas 0,4,8,12), pas entre ;
    # les hats marquent aussi les contretemps (croches off : pas 2,6,10,14)
    off = np.delete(np.arange(16), np.arange(0, 16, 4))
    assert kick[::4].mean() > kick[off].mean()
    assert hats[off].mean() > kick[off].mean()


def test_stem_texture_tone_is_sustained(tmp_path):
    """Un sinus soutenu : percussif bas, flux bas, caractère 'soutenu'."""
    f = tmp_path / "other.wav"
    sf.write(f, _sine(330.0, dur=4.0), SR)
    t = texture.stem_texture(str(f), sr=SR)
    assert t["percussive_ratio"] < 0.4
    assert "soutenu" in t["description"]


def test_energy_share_sums_to_one(tmp_path):
    """La part d'énergie des stems somme à 1."""
    sf.write(tmp_path / "drums.wav", _click(np.arange(0, 8, 0.5), freq=80.0), SR)
    sf.write(tmp_path / "bass.wav", 0.5 * _sine(110.0, dur=8.0), SR)
    beats = np.arange(0.0, 8.0, 0.5)
    out = texture.analyze_stems(str(tmp_path), beats, sr=SR)
    shares = [t["energy_share"] for t in out["stem_textures"].values() if "energy_share" in t]
    assert len(shares) == 2
    assert abs(sum(shares) - 1.0) < 1e-6


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


def _drum_grid(offsets_ms, bpm=120.0, dur=8.0, sr=22050):
    """Clics larges bande sur une grille de doubles-croches, décalés à volonté."""
    beat = 60.0 / bpm
    steps = np.arange(0, dur, beat / 4) + np.asarray(offsets_ms) / 1000.0
    rng = np.random.default_rng(0)
    y = np.zeros(int(sr * dur), dtype=np.float32)
    n = int(0.02 * sr)
    env = np.exp(-np.linspace(0, 12, n))
    for t in steps:
        i = int(t * sr)
        y[i:i + n] += (rng.normal(0, 1, n) * env).astype(np.float32) * 0.8
    return y, np.arange(0, dur, beat)


def test_une_grille_droite_ne_produit_pas_de_microtiming(tmp_path):
    """Le défaut d'origine : avec un hop de 512, une grille parfaitement
    quantifiée rapportait 13,3 ms de microtiming. On mesurait la trame
    d'analyse, pas le batteur."""
    y, beats = _drum_grid(np.zeros(64))
    path = tmp_path / "droit.wav"
    sf.write(path, y, 22050)
    r = texture.rhythm_pattern(str(path), beats)
    assert r["microtiming_ms"] < 5.0


def test_un_retard_reel_est_mesure_a_sa_valeur(tmp_path):
    """Et le plancher ne doit pas avoir été gagné en aplatissant la mesure :
    un décalage franc reste lu comme tel."""
    y, beats = _drum_grid(np.full(64, 20.0))
    path = tmp_path / "traine.wav"
    sf.write(path, y, 22050)
    r = texture.rhythm_pattern(str(path), beats)
    assert 15.0 < r["microtiming_ms"] < 25.0
