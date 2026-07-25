"""Trois corrections nées de l'analyse de « A Midsummer Nice Dream »."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from score7_mcp.chords_dl import _despike
from score7_mcp.core import _disputed_octave


# --------------------------------------------------------------------------- #
#  1. le tempo retenu contredit par ses propres candidats
# --------------------------------------------------------------------------- #
def test_le_cas_midsummer_est_signale():
    """230,8 retenu alors que 115,4 est plus fort, confiance 0,375."""
    c = [{"bpm": 115.4, "strength": 0.397},
         {"bpm": 230.8, "strength": 0.375},
         {"bpm": 461.6, "strength": 0.228}]
    d = _disputed_octave(c, 0.375)
    assert d and d["kept"] == 230.8 and d["suggested"] == 115.4


def test_un_tempo_net_nest_pas_signale():
    c = [{"bpm": 60, "strength": 0.1},
         {"bpm": 120, "strength": 0.8},
         {"bpm": 240, "strength": 0.1}]
    assert _disputed_octave(c, 0.8) is None


def test_confiance_basse_mais_retenu_le_plus_fort():
    """Ambigu ne veut pas dire contredit : rien à signaler ici."""
    c = [{"bpm": 60, "strength": 0.30},
         {"bpm": 120, "strength": 0.40},
         {"bpm": 240, "strength": 0.30}]
    assert _disputed_octave(c, 0.40) is None


def test_ecart_negligeable_ignore():
    c = [{"bpm": 60, "strength": 0.334},
         {"bpm": 120, "strength": 0.333},
         {"bpm": 240, "strength": 0.333}]
    assert _disputed_octave(c, 0.333) is None


# --------------------------------------------------------------------------- #
#  2. plancher de niveau : ne pas transcrire une queue de réverbération
# --------------------------------------------------------------------------- #
def test_le_fondu_final_est_repere(tmp_path):
    sr = 22050
    t = np.linspace(0, 6, 6 * sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 220 * t)
    y[4 * sr:] *= 0.005                     # -46 dB sur les 2 dernières secondes
    p = tmp_path / "fade.wav"
    sf.write(str(p), y, sr)

    from score7_mcp.melody import _quiet_spans
    spans = _quiet_spans(str(p))
    assert spans, "le fondu doit être repéré"
    assert any(a >= 3.5 for a, _ in spans)


def test_un_morceau_doux_ne_disparait_pas(tmp_path):
    """Le seuil est relatif : jouer doux du début à la fin n'est pas un fondu."""
    sr = 22050
    t = np.linspace(0, 4, 4 * sr, endpoint=False)
    p = tmp_path / "doux.wav"
    sf.write(str(p), 0.02 * np.sin(2 * np.pi * 220 * t), sr)
    from score7_mcp.melody import _quiet_spans
    assert _quiet_spans(str(p)) == []


def test_signal_vide(tmp_path):
    p = tmp_path / "vide.wav"
    sf.write(str(p), np.zeros(10), 22050)
    from score7_mcp.melody import _quiet_spans
    assert _quiet_spans(str(p)) == []


# --------------------------------------------------------------------------- #
#  3. micro-segments d'accords
# --------------------------------------------------------------------------- #
def _seg(chord, sb, beats):
    return {"chord": chord, "chord_full": chord, "start_beat": sb,
            "beats": beats, "conf": 1.0, "time": float(sb)}


def test_accident_dun_temps_absorbe():
    """Bm … Bm7(1) … Bm : le tenu reprend son bien."""
    grid = [_seg("Bm", 0, 8), _seg("Bm7", 8, 1), _seg("Bm", 9, 8)]
    out = _despike(grid)
    assert [g["chord"] for g in out] == ["Bm"]
    assert out[0]["beats"] == 17


def test_accord_court_entre_deux_differents_conserve():
    """Un vrai passage ne doit pas être effacé au nom de la simplicité."""
    grid = [_seg("Bm", 0, 8), _seg("F#m", 8, 1), _seg("E", 9, 8)]
    assert [g["chord"] for g in out_chords(grid)] == ["Bm", "F#m", "E"]


def out_chords(grid):
    return _despike(grid)


def test_accord_long_entre_deux_pareils_conserve():
    """Quatre temps ne sont pas du jitter : Bm-A-Bm reste une vraie grille."""
    grid = [_seg("Bm", 0, 8), _seg("A", 8, 4), _seg("Bm", 12, 8)]
    assert [g["chord"] for g in _despike(grid)] == ["Bm", "A", "Bm"]


def test_repetitions_consecutives_fusionnees():
    grid = [_seg("Bm", 0, 4), _seg("Bm", 4, 4), _seg("E", 8, 4)]
    out = _despike(grid)
    assert [g["chord"] for g in out] == ["Bm", "E"]
    assert out[0]["beats"] == 8


def test_grille_courte_intacte():
    grid = [_seg("Bm", 0, 4), _seg("E", 4, 4)]
    assert len(_despike(grid)) == 2


def test_le_seuil_epargne_un_creux_modere(tmp_path):
    """Un passage doux à -10 dB fait partie du morceau, pas du fondu."""
    import numpy as np, soundfile as sf
    from score7_mcp.melody import _quiet_spans
    sr = 22050
    t = np.linspace(0, 6, 6 * sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 220 * t)
    y[2 * sr:4 * sr] *= 0.3               # ~-10 dB : un creux, pas une fin
    p = tmp_path / "creux.wav"
    sf.write(str(p), y, sr)
    assert _quiet_spans(str(p)) == []


def test_le_seuil_coupe_un_vrai_fondu(tmp_path):
    """-25 dB : la queue de réverbération, où la transcription invente."""
    import numpy as np, soundfile as sf
    from score7_mcp.melody import _quiet_spans
    sr = 22050
    t = np.linspace(0, 6, 6 * sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 220 * t)
    y[4 * sr:] *= 0.055                   # ~-25 dB
    p = tmp_path / "fondu.wav"
    sf.write(str(p), y, sr)
    assert any(a >= 3.5 for a, _ in _quiet_spans(str(p)))
