"""Couche rythme + texture, par stem (séparation Demucs requise en amont).

Ce que le mix global ne peut pas donner : le pattern de frappe de la batterie
(replié sur la grille de beats) et le caractère sonore de chaque stem pris isolément
(timbre statique vs mouvant, percussif vs soutenu, part dans le mix, largeur stéréo).

Librosa/scipy seulement — aucune dépendance GPU. Ne tourne que sur des stems déjà
posés par `melody.separate` (Demucs). L'étiquetage kick/snare/hats par bande de
fréquence est une heuristique, pas une classification : annoncé comme tel.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

from score7_mcp import core

# Bandes de fréquence ≈ familles de percussions. Grossier mais lisible :
# le grave porte le kick, le médium le corps de la caisse claire, l'aigu les cymbales.
_BANDS = {"kick": (20.0, 150.0), "snare": (150.0, 2000.0), "hats": (6000.0, None)}

# Les 4 stems Demucs (htdemucs), du plus rythmique au plus harmonique.
_STEMS = ("drums", "bass", "other", "vocals")


# --------------------------------------------------------------------------- filtrage
def _band(y: np.ndarray, sr: int, lo: float, hi: float | None) -> np.ndarray:
    """Filtre Butterworth ordre 4, phase nulle (sosfiltfilt) — pas de décalage temporel,
    ce qui est essentiel : on mesure ensuite la position des onsets à la milliseconde."""
    nyq = sr / 2.0
    if hi is None or hi >= nyq:
        sos = butter(4, lo / nyq, btype="high", output="sos")
    elif lo <= 0:
        sos = butter(4, hi / nyq, btype="low", output="sos")
    else:
        sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


# --------------------------------------------------------------------------- pattern batterie
def _step_times(beat_times: np.ndarray, steps_per_beat: int) -> np.ndarray:
    """Subdivise chaque intervalle de beat en `steps_per_beat` pas isochrones.
    La grille n'est pas supposée régulière : on interpole entre beats réels (madmom/librosa),
    donc le pattern reste calé même si le tempo respire."""
    steps = []
    for i in range(len(beat_times) - 1):
        t0, t1 = float(beat_times[i]), float(beat_times[i + 1])
        for j in range(steps_per_beat):
            steps.append(t0 + (t1 - t0) * j / steps_per_beat)
    return np.array(steps)


def _step_strengths(oenv: np.ndarray, sr: int, step_times: np.ndarray, hop: int = 512) -> np.ndarray:
    """Force d'onset à chaque pas : max de l'enveloppe dans une fenêtre ±1 trame autour
    du pas (tolère un placement légèrement en avance/retard sans diluer sur deux pas)."""
    frames = librosa.time_to_frames(step_times, sr=sr, hop_length=hop)
    out = np.zeros(len(frames))
    for k, f in enumerate(frames):
        f0, f1 = max(0, int(f) - 1), min(len(oenv), int(f) + 2)
        if f1 > f0:
            out[k] = float(oenv[f0:f1].max())
    return out


def _fold_pattern(strengths: np.ndarray, steps_per_bar: int) -> np.ndarray:
    """Replie la suite de pas sur une mesure et moyenne — le motif récurrent ressort,
    les frappes accidentelles s'effacent. Normalisé 0..1 par rapport au pas le plus fort."""
    n = (len(strengths) // steps_per_bar) * steps_per_bar
    if n < steps_per_bar:
        return np.zeros(steps_per_bar)
    folded = strengths[:n].reshape(-1, steps_per_bar).mean(axis=0)
    peak = folded.max()
    return folded / peak if peak > 1e-9 else folded


def _grid_str(pattern: np.ndarray, steps_per_beat: int) -> str:
    """Représentation tracker : `X` frappe forte, `x` frappe faible, `.` silence.
    Espace entre chaque beat pour lire la pulsation."""
    cells = ["X" if v > 0.5 else "x" if v > 0.2 else "." for v in pattern]
    return " ".join("".join(cells[i:i + steps_per_beat])
                    for i in range(0, len(cells), steps_per_beat))


def _swing(onset_t: np.ndarray, beat_times: np.ndarray) -> float | None:
    """Position médiane des onsets tombant dans la moitié médiane du beat. 0.5 = binaire
    droit (la croche off est pile au milieu) ; ~0.667 = swing ternaire (croche off retardée
    sur le 3e triolet). None si trop peu de matière pour conclure."""
    rels = []
    for i in range(len(beat_times) - 1):
        t0, t1 = float(beat_times[i]), float(beat_times[i + 1])
        dur = t1 - t0
        if dur <= 0:
            continue
        for ot in onset_t:
            r = (ot - t0) / dur
            if 0.25 < r < 0.75:
                rels.append(r)
    return float(np.median(rels)) if len(rels) >= 4 else None


def rhythm_pattern(drums_path: str, beat_times: np.ndarray, sr: int = 22050,
                   steps_per_beat: int = 4, beats_per_bar: int = 4) -> dict:
    """Pattern de frappe du stem batterie, replié sur une mesure, par bande de fréquence.
    Plus le microtiming (déviation des frappes vs grille isochrone) et le swing."""
    y, _ = librosa.load(drums_path, sr=sr, mono=True)
    dur = len(y) / sr
    if len(beat_times) < 3 or dur < 1.0:
        return {"available": False, "reason": "grille de beats ou stem trop court"}

    step_t = _step_times(beat_times, steps_per_beat)
    steps_per_bar = steps_per_beat * beats_per_bar

    bands = {}
    for name, (lo, hi) in _BANDS.items():
        yb = _band(y, sr, lo, hi)
        oenv = librosa.onset.onset_strength(y=yb, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr)
        pattern = _fold_pattern(_step_strengths(oenv, sr, step_t), steps_per_bar)
        bands[name] = {
            "pattern": [round(float(v), 2) for v in pattern],
            "grid": _grid_str(pattern, steps_per_beat),
            "onsets": int(len(onsets)),
            "density_hz": round(len(onsets) / dur, 2),
        }

    # microtiming & swing : sur l'enveloppe pleine bande du stem batterie
    oenv_full = librosa.onset.onset_strength(y=y, sr=sr)
    onset_t = librosa.frames_to_time(
        librosa.onset.onset_detect(onset_envelope=oenv_full, sr=sr), sr=sr)
    micro_ms = None
    if len(onset_t) and len(step_t):
        # n'évaluer que les onsets couverts par la grille : un onset d'outro (après le
        # dernier beat) snapperait au dernier step à plusieurs secondes et fausserait la moyenne
        in_grid = onset_t[(onset_t >= step_t[0]) & (onset_t <= step_t[-1])]
        if len(in_grid):
            dev = [abs(ot - step_t[int(np.argmin(np.abs(step_t - ot)))]) for ot in in_grid]
            micro_ms = round(float(np.mean(dev)) * 1000, 1)
    swing = _swing(onset_t, beat_times)

    return {
        "available": True,
        "steps_per_beat": steps_per_beat,
        "steps_per_bar": steps_per_bar,
        "bands": bands,
        "onsets_total": int(len(onset_t)),
        "microtiming_ms": micro_ms,
        "swing_ratio": round(swing, 3) if swing is not None else None,
        "swing_pct": (int(np.clip((swing - 0.5) / (2 / 3 - 0.5) * 100, 0, 100))
                      if swing is not None else None),
        "note": "bandes kick/snare/hats = heuristique fréquentielle, pas une classification",
    }


# --------------------------------------------------------------------------- texture par stem
def _spectral_flux(y: np.ndarray, S=None) -> float:
    """Flux spectral demi-rectifié, moyenné : mesure le mouvement du timbre dans le temps.
    Faible = nappe statique ; élevé = son qui évolue (filtre qui bouge, arpège, attaques).
    `S` = magnitude STFT précalculée, réutilisable pour éviter une STFT de plus."""
    if S is None:
        S = np.abs(librosa.stft(y))
    Sn = S / (S.max() + 1e-9)
    diff = np.maximum(np.diff(Sn, axis=1), 0.0)
    return round(float(np.mean(np.sqrt(np.sum(diff ** 2, axis=0)))), 4)


def stem_texture(path: str, sr: int = 22050) -> dict:
    """Caractère sonore d'un stem isolé : profil spectral, mouvement timbral (flux +
    variation du centroïde), équilibre percussif/soutenu (HPSS), niveau et largeur stéréo."""
    y, _, y_stereo, _ = core.load_audio(path, sr=sr)
    rms = float(np.sqrt(np.mean(y ** 2)))
    # une seule magnitude STFT partagée par le profil spectral, le centroïde et le flux
    S = np.abs(librosa.stft(y))
    spec = core.spectral_profile(y, sr, S=S)

    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    centroid_cv = round(float(np.std(cent) / (np.mean(cent) + 1e-9)), 3)

    y_h, y_p = librosa.effects.hpss(y)
    e_p, e_h = float(np.sqrt(np.mean(y_p ** 2))), float(np.sqrt(np.mean(y_h ** 2)))
    perc = round(e_p / (e_p + e_h + 1e-9), 3)

    flux = _spectral_flux(y, S=S)
    movement = ("timbre stable" if flux < 0.02 and centroid_cv < 0.25
                else "timbre mouvant" if flux > 0.05 or centroid_cv > 0.5
                else "modérément animé")
    character = ("percussif / transitoire" if perc > 0.6
                 else "soutenu / tonal" if perc < 0.3 else "mixte")

    return {
        "rms": rms,  # absolu, sert au calcul de la part d'énergie dans analyze_stems
        "rms_dbfs": round(20 * np.log10(rms + 1e-9), 1),
        "spectral": spec,
        "spectral_flux": flux,
        "centroid_cv": centroid_cv,
        "percussive_ratio": perc,
        "stereo": core.stereo_width(y_stereo),
        "description": f"{character} ; {movement}",
    }


# --------------------------------------------------------------------------- orchestration
def analyze_stems(stems_dir: str, beat_times: np.ndarray, sr: int = 22050,
                  beats_per_bar: int = 4) -> dict:
    """Parcourt les stems d'un dossier Demucs : pattern rythmique sur la batterie,
    texture sur chacun, et part d'énergie relative dans le mix (somme des RMS = 100 %)."""
    sd = Path(stems_dir)
    textures = {}
    for name in _STEMS:
        f = sd / f"{name}.wav"
        if f.exists():
            try:
                textures[name] = stem_texture(str(f), sr=sr)
            except Exception as e:
                textures[name] = {"error": str(e)}

    total = sum(t["rms"] for t in textures.values() if "rms" in t) + 1e-9
    for t in textures.values():
        if "rms" in t:
            t["energy_share"] = round(t.pop("rms") / total, 3)

    rhythm = None
    drums = sd / "drums.wav"
    if drums.exists():
        try:
            rhythm = rhythm_pattern(str(drums), beat_times, sr=sr, beats_per_bar=beats_per_bar)
        except Exception as e:
            rhythm = {"available": False, "reason": str(e)}

    return {"rhythm_pattern": rhythm, "stem_textures": textures}
