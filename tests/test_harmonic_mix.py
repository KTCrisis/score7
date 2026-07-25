"""Mix harmonique : lire les accords sans la batterie."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from score7_mcp import harmonic_mix


@pytest.fixture()
def stems(tmp_path):
    """Six stems d'une seconde, chacun à une fréquence reconnaissable."""
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    freqs = {"bass": 110, "piano": 220, "guitar": 330, "other": 440,
             "drums": 55, "vocals": 660}
    for name, f in freqs.items():
        sf.write(str(tmp_path / f"{name}.wav"), 0.2 * np.sin(2 * np.pi * f * t), sr)
    return tmp_path


def test_ne_garde_que_les_stems_harmoniques(stems):
    noms = [p.stem for p in harmonic_mix.available(stems)]
    assert noms == ["bass", "piano", "guitar", "other"]
    assert "drums" not in noms and "vocals" not in noms


def test_construit_un_wav_et_le_renvoie(stems, tmp_path):
    out = tmp_path / "h.wav"
    got = harmonic_mix.build(stems, out)
    assert got == str(out) and out.exists()
    y, sr = sf.read(str(out))
    assert y.size > 0


def test_la_batterie_nentre_pas_dans_le_mix(stems, tmp_path):
    """Le point de tout le module : le 55 Hz de la batterie doit être absent."""
    import librosa
    out = tmp_path / "h.wav"
    harmonic_mix.build(stems, out)
    y, sr = librosa.load(str(out), sr=None, mono=True)
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)

    def energie(f, largeur=6):
        m = (freqs > f - largeur) & (freqs < f + largeur)
        return float(spec[m].max())

    # la basse (110) est là, la batterie (55) et la voix (660) non
    assert energie(110) > 10 * energie(55)
    assert energie(220) > 10 * energie(660)


def test_dossier_sans_stem_harmonique_renvoie_none(tmp_path):
    """Doit dégrader vers le mix d'origine, jamais casser."""
    sr = 22050
    sf.write(str(tmp_path / "drums.wav"), np.zeros(sr), sr)
    assert harmonic_mix.build(tmp_path, tmp_path / "h.wav") is None


def test_dossier_vide_renvoie_none(tmp_path):
    assert harmonic_mix.build(tmp_path, tmp_path / "h.wav") is None


def test_quatre_stems_seulement(tmp_path):
    """htdemucs (4 stems) n'a ni piano ni guitare : on prend ce qu'il y a."""
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    for name, f in (("bass", 110), ("other", 440), ("drums", 55), ("vocals", 660)):
        sf.write(str(tmp_path / f"{name}.wav"), 0.2 * np.sin(2 * np.pi * f * t), sr)
    assert [p.stem for p in harmonic_mix.available(tmp_path)] == ["bass", "other"]
    assert harmonic_mix.build(tmp_path, tmp_path / "h.wav") is not None


def test_la_somme_ne_secrete_pas(tmp_path):
    """Quatre stems forts sommés dépassent 1.0 : on ramène au lieu de clipper."""
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    for name in harmonic_mix.HARMONIC_STEMS:
        sf.write(str(tmp_path / f"{name}.wav"), 0.9 * np.sin(2 * np.pi * 220 * t), sr)
    out = tmp_path / "h.wav"
    harmonic_mix.build(tmp_path, out)
    y, _ = sf.read(str(out))
    assert np.max(np.abs(y)) <= 1.0 + 1e-6
