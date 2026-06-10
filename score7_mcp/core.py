"""Analyses signal (librosa) : tonalité, accords, structure, spectral, stéréo, loudness.

Aucune dépendance lourde ici — librosa/numpy/scipy/soundfile/pyloudnorm.
Les étapes lourdes (séparation, transcription, mélodie) vivent dans melody.py.
"""

from __future__ import annotations

import warnings

import librosa
import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Profils Krumhansl-Kessler (corrélation tonale)
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Qualité des triades diatoniques par degré (en demi-tons depuis la tonique)
# major : I ii iii IV V vi vii°
MAJOR_DEGREES = {0: "maj", 2: "min", 4: "min", 5: "maj", 7: "maj", 9: "min", 11: "dim"}
# minor naturel + V/vii° harmoniques (pour capter la dominante)
MINOR_DEGREES = {0: "min", 2: "dim", 3: "maj", 5: "min", 7: "maj", 8: "maj", 10: "maj",
                 11: "dim"}


# --------------------------------------------------------------------------- chargement
def load_audio(path: str, sr: int = 22050):
    """(y_mono, sr) pour l'analyse + (y_stereo, sr_native) pour stéréo/loudness."""
    y_mono, sr = librosa.load(path, sr=sr, mono=True)
    try:
        y_stereo, sr_native = sf.read(path, always_2d=True)
        y_stereo = y_stereo.T
    except Exception:
        y_stereo, sr_native = librosa.load(path, sr=None, mono=False)
        if y_stereo.ndim == 1:
            y_stereo = np.stack([y_stereo, y_stereo])
    return y_mono, sr, y_stereo, sr_native


# --------------------------------------------------------------------------- tempo & métrique
_METER_NAMES = {2: "2/4", 3: "3/4", 4: "4/4", 5: "5/4", 6: "6/8", 7: "7/8"}


def _tempogram_strengths(oenv, sr, bpms, hop_length=512):
    """Force du tempogramme global (moyenné sur la durée) aux BPM demandés."""
    tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    glob = tg.mean(axis=1)
    freqs = librosa.tempo_frequencies(len(glob), sr=sr, hop_length=hop_length)
    return [float(glob[int(np.argmin(np.abs(freqs - b)))]) for b in bpms]


def _octave_candidates(oenv, sr, bpm):
    """Candidats d'octave (T/2, T, 2T) avec leur part de force tempogramme.
    L'autocorrélation d'onsets ne distingue pas un tempo de son double/moitié —
    on expose l'ambiguïté au lieu de la masquer."""
    cands = [bpm / 2, bpm, bpm * 2]
    strengths = _tempogram_strengths(oenv, sr, cands)
    total = sum(strengths) + 1e-9
    out = [{"bpm": round(b, 1), "strength": round(s / total, 3)}
           for b, s in zip(cands, strengths)]
    return out, out[1]["strength"]


def estimate_tempo(y, sr):
    """BPM : beat tracking librosa pour la grille, mais le BPM retenu vient de la
    médiane des intervalles inter-beats — plus stable que le scalaire de beat_track,
    qui hérite du prior à 120 BPM."""
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    if len(beat_times) > 2:
        bpm = 60.0 / float(np.median(np.diff(beat_times)))
    else:
        bpm = float(np.atleast_1d(tempo)[0])
    candidates, confidence = _octave_candidates(oenv, sr, bpm)
    info = {"bpm": round(bpm, 1), "bpm_candidates": candidates,
            "bpm_confidence": confidence, "source": "librosa"}
    return info, beats, beat_times


def _subdivision(oenv, sr, bpm):
    """Subdivision du beat : binaire (croches) ou ternaire (mesures composées).
    Force tempogramme à 2x vs 3x le tempo ; seuil 1.1 pour biaiser vers binaire,
    et le ternaire n'est retenu que si la subdivision a une énergie réelle
    (sinon, sans subdivision du tout, on comparerait du bruit)."""
    s1, s2, s3 = _tempogram_strengths(oenv, sr, [bpm, 2 * bpm, 3 * bpm])
    return "ternary" if (s3 > s2 * 1.1 and s3 > 0.2 * s1) else "binary"


def _meter_name(beats_per_bar, subdivision):
    """6/8 et 12/8 = pulsation à 2 ou 4 avec subdivision ternaire."""
    if subdivision == "ternary" and beats_per_bar in (2, 4):
        return f"{beats_per_bar * 3}/8"
    return _METER_NAMES.get(beats_per_bar, f"{beats_per_bar}/4")


def estimate_meter(y, sr, beats_frames, beat_times, max_bar=7):
    """Métrique par motif d'accents : l'enveloppe d'onsets échantillonnée sur chaque
    beat est repliée modulo chaque hypothèse de mesure (2..7 beats) ; la bonne
    hypothèse maximise le contraste downbeat vs reste. Une hypothèse bien expliquée
    par son diviseur retombe sur le diviseur (évite 6 quand c'est 3)."""
    null = {"meter": None, "beats_per_bar": None, "subdivision": None,
            "meter_confidence": 0.0, "source": "librosa"}
    if len(beats_frames) < 2 * max_bar:
        return null
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    accents = np.array([float(oenv[min(int(f), len(oenv) - 1)]) for f in beats_frames])
    accents = accents - accents.mean()
    spread = accents.std() + 1e-9

    scores = {}
    for bar in range(2, max_bar + 1):
        n = (len(accents) // bar) * bar
        if n < 2 * bar:
            continue
        profile = accents[:n].reshape(-1, bar).mean(axis=0)
        scores[bar] = float((profile.max() - profile.mean()) / spread)
    if not scores:
        return null
    best = max(scores, key=scores.get)
    for d in sorted(k for k in scores if k < best and best % k == 0):
        if scores[d] >= 0.9 * scores[best]:
            best = d
            break
    others = [v for k, v in scores.items() if k != best]
    conf = 0.0
    if others and scores[best] > 0:
        conf = float(np.clip((scores[best] - max(others)) / scores[best], 0.0, 1.0))

    bpm = 60.0 / float(np.median(np.diff(beat_times)))
    sub = _subdivision(oenv, sr, bpm)
    return {"meter": _meter_name(best, sub), "beats_per_bar": int(best),
            "subdivision": sub, "meter_confidence": round(conf, 3), "source": "librosa"}


def try_madmom_rhythm(path, beats_per_bar=(2, 3, 4, 5, 6, 7)):
    """Beat + downbeat tracking madmom (RNN + DBN), extra [rhythm]. Nettement plus
    fiable que librosa, mais les modèles sont entraînés surtout sur 3/4 et 4/4 —
    les métriques impaires restent indicatives. Renvoie None si madmom est absent
    ou échoue ; l'appelant retombe sur librosa."""
    try:
        from madmom.features.downbeats import (DBNDownBeatTrackingProcessor,
                                               RNNDownBeatProcessor)
    except Exception:
        return None
    try:
        act = RNNDownBeatProcessor()(path)
        db = DBNDownBeatTrackingProcessor(beats_per_bar=list(beats_per_bar), fps=100)(act)
    except Exception:
        return None
    if db is None or len(db) < 8:
        return None
    beat_times = db[:, 0]
    return {"bpm": round(60.0 / float(np.median(np.diff(beat_times))), 1),
            "beats_per_bar": int(round(float(np.max(db[:, 1])))),
            "beat_times": beat_times, "source": "madmom"}


def estimate_rhythm(y, sr, path=None):
    """Tempo + métrique réconciliés : madmom si installé (extra [rhythm]), sinon
    librosa seul. Quand madmom prend la main, sa grille de beats remplace celle de
    librosa (la synchro chroma des accords en profite aussi)."""
    tempo, beats, beat_times = estimate_tempo(y, sr)
    meter = estimate_meter(y, sr, beats, beat_times)
    mm = try_madmom_rhythm(path) if path else None
    if mm:
        oenv = librosa.onset.onset_strength(y=y, sr=sr)
        candidates, confidence = _octave_candidates(oenv, sr, mm["bpm"])
        sub = _subdivision(oenv, sr, mm["bpm"])
        tempo = {"bpm": mm["bpm"], "bpm_candidates": candidates,
                 "bpm_confidence": confidence, "source": "madmom"}
        meter = {"meter": _meter_name(mm["beats_per_bar"], sub),
                 "beats_per_bar": mm["beats_per_bar"], "subdivision": sub,
                 "meter_confidence": None, "source": "madmom"}
        beat_times = mm["beat_times"]
        beats = librosa.time_to_frames(beat_times, sr=sr)
    return tempo, meter, beats, beat_times


# --------------------------------------------------------------------------- tonalité
def _krumhansl_scores(chroma_mean):
    """24 corrélations (12 maj + 12 min) ; renvoie deux vecteurs (12,) normalisés [0,1]."""
    c = chroma_mean - chroma_mean.mean()
    maj = np.array([np.corrcoef(c, np.roll(KK_MAJOR, i) - KK_MAJOR.mean())[0, 1] for i in range(12)])
    minr = np.array([np.corrcoef(c, np.roll(KK_MINOR, i) - KK_MINOR.mean())[0, 1] for i in range(12)])
    return maj, minr


def _chord_vote_scores(chord_grid):
    """Vote par fonction harmonique : chaque accord (root, qualité, durée en beats) vote
    pour toutes les tonalités où il est diatonique. Renvoie (maj_vec, min_vec) (12,).
    Bien plus robuste que la corrélation chroma pour trancher majeur/mineur relatif."""
    maj = np.zeros(12)
    minr = np.zeros(12)
    if not chord_grid:
        return maj, minr
    for seg in chord_grid:
        name = seg["chord"]
        if name == "N":
            continue
        root = NOTE_NAMES.index(name[:-1]) if name.endswith("m") else NOTE_NAMES.index(name)
        qual = "min" if name.endswith("m") else "maj"
        w = seg.get("beats", 1)
        for tonic in range(12):
            deg = (root - tonic) % 12
            if MAJOR_DEGREES.get(deg) == qual:
                maj[tonic] += w
            if MINOR_DEGREES.get(deg) == qual:
                minr[tonic] += w
    return maj, minr


def _norm01(v):
    lo, hi = float(np.min(v)), float(np.max(v))
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def estimate_key(chroma_mean, chord_grid=None, w_chords=1.0):
    """Tonalité réconciliée : Krumhansl (couleur chroma) + vote d'accords (fonction
    harmonique). Le vote d'accords corrige la confusion majeur / mineur relatif que
    fait Krumhansl seul (les deux partagent les mêmes notes)."""
    kmaj, kmin = _krumhansl_scores(chroma_mean)
    cmaj, cmin = _chord_vote_scores(chord_grid)

    # combine : Krumhansl normalisé + vote d'accords normalisé (pondéré)
    smaj = _norm01(kmaj) + w_chords * _norm01(cmaj)
    smin = _norm01(kmin) + w_chords * _norm01(cmin)

    cand = ([(NOTE_NAMES[i], "major", float(smaj[i])) for i in range(12)]
            + [(NOTE_NAMES[i], "minor", float(smin[i])) for i in range(12)])
    cand.sort(key=lambda t: t[2], reverse=True)
    root, mode, score = cand[0]

    # tonalité Krumhansl seule (pour transparence / diagnostic)
    kcand = ([(NOTE_NAMES[i], "major", float(kmaj[i])) for i in range(12)]
             + [(NOTE_NAMES[i], "minor", float(kmin[i])) for i in range(12)])
    kcand.sort(key=lambda t: t[2], reverse=True)

    return {
        "root": root,
        "mode": mode,
        "score": round(score, 3),
        "runner_up": f"{cand[1][0]} {cand[1][1]}",
        "krumhansl_only": f"{kcand[0][0]} {kcand[0][1]}",
        "corrected": kcand[0][:2] != (root, mode),
    }


def reconcile_key_with_melody(key: dict, melody_notes: list) -> dict:
    """Réconcilie la tonalité avec la mélodie — le signal le plus fiable pour le mode.
    Tonique = classe de hauteur la plus fréquente ; mode = tierce mineure (+3) vs
    majeure (+4) au-dessus de la tonique. Corrige les cas où le détecteur d'accords
    étiquette des majeurs parasites (tierce ambiguë sur nappes)."""
    if not melody_notes:
        return key
    pcs = np.bincount([n["pitch"] % 12 for n in melody_notes], minlength=12)
    if pcs.sum() == 0:
        return key
    tonic = int(np.argmax(pcs))
    minor_third = pcs[(tonic + 3) % 12]
    major_third = pcs[(tonic + 4) % 12]
    mode = "minor" if minor_third >= major_third else "major"
    new_root, changed = NOTE_NAMES[tonic], (NOTE_NAMES[tonic], mode) != (key["root"], key["mode"])
    out = dict(key)
    out.update({"root": new_root, "mode": mode, "reconciled_by_melody": True,
                "before_melody": f"{key['root']} {key['mode']}", "changed_by_melody": changed})
    return out


# --------------------------------------------------------------------------- accords
def _chord_templates():
    templates = {}
    for r in range(12):
        maj = np.zeros(12); maj[[r, (r + 4) % 12, (r + 7) % 12]] = 1
        minr = np.zeros(12); minr[[r, (r + 3) % 12, (r + 7) % 12]] = 1
        templates[NOTE_NAMES[r]] = maj / np.linalg.norm(maj)
        templates[NOTE_NAMES[r] + "m"] = minr / np.linalg.norm(minr)
    return templates


def estimate_chords(y, sr, beats_frames, beat_times, min_beats: int = 2):
    """Chroma CENS synchronisé beats → template matching (cosinus) → grille fusionnée."""
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    if len(beats_frames) > 1:
        chroma_sync = librosa.util.sync(chroma, beats_frames, aggregate=np.median)
    else:
        chroma_sync = chroma
    templates = _chord_templates()
    names = list(templates.keys())
    mat = np.array([templates[n] for n in names])

    seq = []
    for t in range(chroma_sync.shape[1]):
        v = chroma_sync[:, t]
        nv = np.linalg.norm(v)
        if nv < 1e-6:
            seq.append(("N", 0.0)); continue
        sims = mat @ (v / nv)
        idx = int(np.argmax(sims))
        seq.append((names[idx], float(sims[idx])))

    return _merge_grid(seq, beat_times, min_beats)


def _merge_grid(seq, beat_times, min_beats):
    if not seq:
        return []
    segs = []
    chord, start, n, confs = seq[0][0], 0, 1, [seq[0][1]]
    for i in range(1, len(seq)):
        if seq[i][0] == chord:
            n += 1; confs.append(seq[i][1])
        else:
            segs.append({"chord": chord, "start_beat": start, "beats": n,
                         "conf": float(np.mean(confs))})
            chord, start, n, confs = seq[i][0], i, 1, [seq[i][1]]
    segs.append({"chord": chord, "start_beat": start, "beats": n, "conf": float(np.mean(confs))})

    filtered = [s for s in segs if s["beats"] >= min_beats] or segs
    for s in filtered:
        bi = s["start_beat"]
        s["time"] = float(beat_times[bi]) if bi < len(beat_times) else None
    return filtered


# --------------------------------------------------------------------------- structure
def dynamic_structure(y, sr, n_sections: int = 8):
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-9, ref=np.max)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    dur = float(times[-1]) if len(times) else 0.0

    edges = np.linspace(0, len(rms), n_sections + 1).astype(int)
    coarse = []
    for k in range(n_sections):
        a, b = edges[k], edges[k + 1]
        if b > a:
            coarse.append({"t_start": round(float(times[a]), 1),
                           "t_end": round(float(times[min(b, len(times) - 1)]), 1),
                           "rms_db": round(float(np.mean(rms_db[a:b])), 1)})

    flux = np.diff(rms_db)
    thr = np.std(flux) * 2.5
    onsets, last = [], -1e9
    for i in range(1, len(flux)):
        t = float(times[i])
        if t < 1.0 or flux[i] <= thr:
            continue
        if t - last >= 3.0:
            onsets.append(round(t, 1)); last = t
    return {"duration_s": round(dur, 1), "coarse": coarse, "layer_onsets": onsets[:12]}


# --------------------------------------------------------------------------- spectral
def spectral_profile(y, sr):
    cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    roll = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)))
    bw = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    desc = ("sombre / froid — peu d'harmoniques hautes" if cent < 1500
            else "équilibré — médiums présents" if cent < 3000
            else "brillant — énergie marquée dans les aigus")
    return {"centroid_hz": round(cent), "rolloff85_hz": round(roll),
            "bandwidth_hz": round(bw), "flatness": round(flat, 4), "description": desc}


# --------------------------------------------------------------------------- stéréo
def stereo_width(y_stereo):
    if y_stereo.shape[0] < 2:
        return {"mono": True, "correlation": 1.0, "side_mid_ratio": 0.0,
                "description": "mono (ou canaux identiques)"}
    L, R = y_stereo[0], y_stereo[1]
    n = min(len(L), len(R)); L, R = L[:n], R[:n]
    corr = float(np.corrcoef(L, R)[0, 1]) if n > 1 else 1.0
    mid, side = (L + R) / 2, (L - R) / 2
    ratio = float(np.sqrt(np.mean(side ** 2))) / (float(np.sqrt(np.mean(mid ** 2))) + 1e-9)
    desc = ("très centré, quasi-mono — espace clos" if ratio < 0.1
            else "stéréo modérée" if ratio < 0.4 else "large, stéréo marquée")
    return {"mono": False, "correlation": round(corr, 3),
            "side_mid_ratio": round(ratio, 3), "description": desc}


# --------------------------------------------------------------------------- loudness
def loudness(y_stereo, sr_native):
    lufs = None
    try:
        import pyloudnorm as pyln
        data = y_stereo.T if y_stereo.ndim == 2 else y_stereo
        lufs = float(pyln.Meter(sr_native).integrated_loudness(data))
    except Exception:
        pass
    flat = y_stereo.flatten()
    peak_db = float(librosa.amplitude_to_db(np.array([np.max(np.abs(flat)) + 1e-9]))[0])
    rms_db = float(librosa.amplitude_to_db(np.array([np.sqrt(np.mean(flat ** 2)) + 1e-9]))[0])
    return {"integrated_lufs": round(lufs, 1) if lufs is not None else None,
            "peak_dbfs": round(peak_db, 1), "rms_dbfs": round(rms_db, 1),
            "crest_factor_db": round(peak_db - rms_db, 1)}
