"""Analyses signal (librosa) : tonalité, accords, structure, spectral, stéréo, loudness.

Aucune dépendance lourde ici — librosa/numpy/scipy/soundfile/pyloudnorm.
Les étapes lourdes (séparation, transcription, mélodie) vivent dans melody.py.
"""

from __future__ import annotations

import functools
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


#: en deçà, l'octave du tempo n'est pas tranchée par le tempogramme
_OCTAVE_CONFIDENCE = 0.5
#: écart de force à partir duquel un autre candidat mérite d'être signalé
_OCTAVE_MARGIN = 0.01


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


def _disputed_octave(candidates, confidence):
    """Le tempo retenu est-il CONTREDIT par ses propres candidats ?

    Le cas ne se voyait pas : sur « A Midsummer Nice Dream » le tempo sortait à
    230,8 alors que 115,4 était le candidat le plus fort (0,397 contre 0,375) et
    la confiance à 0,375, sous le seuil d'ambiguïté. L'information était dans le
    JSON, mais dispersée dans une liste que personne ne relisait.

    On ne CORRIGE pas : diviser le bpm sans décimer `beat_times` casserait la
    cohérence de la grille, et décimer suppose de savoir quel beat sur deux est
    la noire — ce que le tempogramme ne dit pas. Un tracker entraîné qui se
    trompe d'octave reste plus fiable qu'une heuristique qui devine. On rend
    donc le désaccord explicite et exploitable en aval.
    """
    if confidence >= _OCTAVE_CONFIDENCE or len(candidates) < 2:
        return None
    retenu = candidates[1]
    meilleur = max(candidates, key=lambda c: c["strength"])
    if meilleur is retenu or meilleur["strength"] <= retenu["strength"] + _OCTAVE_MARGIN:
        return None
    return {
        "kept": retenu["bpm"], "suggested": meilleur["bpm"],
        "kept_strength": retenu["strength"], "suggested_strength": meilleur["strength"],
        "note": ("le tempogramme préfère un autre multiple ; les durées harmoniques "
                 "sont alors exprimées dans cette subdivision, pas en noires"),
    }


def estimate_tempo(y, sr, oenv=None):
    """BPM : beat tracking librosa pour la grille, mais le BPM retenu vient de la
    médiane des intervalles inter-beats — plus stable que le scalaire de beat_track,
    qui hérite du prior à 120 BPM. `oenv` réutilisable pour éviter de recalculer
    l'enveloppe d'onsets (coûteuse) plusieurs fois dans le même appel."""
    if oenv is None:
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
    disputed = _disputed_octave(candidates, confidence)
    if disputed:
        info["bpm_disputed"] = disputed
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


def estimate_meter(y, sr, beats_frames, beat_times, max_bar=7, oenv=None):
    """Métrique par motif d'accents : l'enveloppe d'onsets échantillonnée sur chaque
    beat est repliée modulo chaque hypothèse de mesure (2..7 beats) ; la bonne
    hypothèse maximise le contraste downbeat vs reste. Une hypothèse bien expliquée
    par son diviseur retombe sur le diviseur (évite 6 quand c'est 3)."""
    null = {"meter": None, "beats_per_bar": None, "subdivision": None,
            "meter_confidence": 0.0, "source": "librosa"}
    if len(beats_frames) < 2 * max_bar:
        return null
    if oenv is None:
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


@functools.lru_cache(maxsize=2)
def _beat_this_model(checkpoint: str, device: str, dbn: bool):
    """Instancie (une fois par process) le tracker Beat This!. Le serveur MCP reste
    vivant entre les appels — on évite de recharger les 77 Mo de poids à chaque analyse."""
    from beat_this.inference import File2Beats
    return File2Beats(checkpoint_path=checkpoint, device=device, dbn=dbn)


def try_beat_this(path, checkpoint="final0"):
    """Beat + downbeat tracking Beat This! (transformer CPJKU, ISMIR 2024 ; extra [rhythm]).
    État de l'art actuel, plus robuste au style et au tempo que madmom RNN+DBN — et il
    fournit les downbeats directement, donc la métrique sans repliement d'accents. Renvoie
    None si le paquet est absent ou échoue ; l'appelant retombe sur madmom puis librosa."""
    try:
        import torch
        from beat_this.inference import File2Beats  # noqa: F401 (présence du paquet)
    except Exception:
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        beats, downbeats = _beat_this_model(checkpoint, device, False)(path)
    except Exception:
        return None
    if beats is None or len(beats) < 8:
        return None
    bpb = None
    if downbeats is not None and len(downbeats) > 1:
        # nombre de beats entre deux downbeats consécutifs = beats par mesure
        idx = np.searchsorted(beats, downbeats)
        spans = np.diff(idx)
        spans = spans[spans > 0]
        if len(spans):
            bpb = int(round(float(np.median(spans))))
    return {"bpm": round(60.0 / float(np.median(np.diff(beats))), 1),
            "beats_per_bar": bpb, "beat_times": np.asarray(beats), "source": "beat_this"}


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


def _apply_external_rhythm(y, sr, ext, meter_fallback, oenv):
    """Construit (tempo, meter, beats, beat_times) à partir d'un tracker externe
    (Beat This! ou madmom) : sa grille de beats remplace celle de librosa — la synchro
    chroma des accords en profite aussi. Candidats d'octave et subdivision restent
    calculés au tempogramme (oenv partagé) pour exposer l'ambiguïté éventuelle. Si le
    tracker n'a pas livré de métrique (beats_per_bar None, downbeats illisibles), on
    conserve l'estimation librosa par repliement d'accents plutôt que de la perdre."""
    candidates, confidence = _octave_candidates(oenv, sr, ext["bpm"])
    sub = _subdivision(oenv, sr, ext["bpm"])
    tempo = {"bpm": ext["bpm"], "bpm_candidates": candidates,
             "bpm_confidence": confidence, "source": ext["source"]}
    disputed = _disputed_octave(candidates, confidence)
    if disputed:
        tempo["bpm_disputed"] = disputed
    bpb = ext.get("beats_per_bar")
    if bpb:
        meter = {"meter": _meter_name(bpb, sub), "beats_per_bar": bpb,
                 "subdivision": sub, "meter_confidence": None, "source": ext["source"]}
    else:
        meter = meter_fallback
    beat_times = np.asarray(ext["beat_times"])
    beats = librosa.time_to_frames(beat_times, sr=sr)
    return tempo, meter, beats, beat_times


def estimate_rhythm(y, sr, path=None):
    """Tempo + métrique. Priorité : Beat This! (transformer, état de l'art) > madmom
    (RNN+DBN) > librosa seul. Le premier tracker disponible prend la main ; sa grille
    de beats remplace celle de librosa. Les deux trackers neuronaux ont besoin du
    chemin fichier (ils relisent l'audio), d'où le fallback librosa quand `path` manque."""
    oenv = librosa.onset.onset_strength(y=y, sr=sr)  # calculé une fois, partagé
    tempo, beats, beat_times = estimate_tempo(y, sr, oenv=oenv)
    meter = estimate_meter(y, sr, beats, beat_times, oenv=oenv)
    ext = (try_beat_this(path) or try_madmom_rhythm(path)) if path else None
    if ext:
        tempo, meter, beats, beat_times = _apply_external_rhythm(y, sr, ext, meter, oenv)
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
        "source": "krumhansl",
    }


def reconcile_key_with_melody(key: dict, melody_notes: list, min_tonic_weight: float = 0.10) -> dict:
    """Réconcilie la tonalité avec la mélodie. La mélodie est le signal le plus fiable
    pour le **mode** (tierce mineure +3 vs majeure +4 au-dessus de la tonique), mais un
    mauvais signal pour la **tonique** : la quinte est souvent aussi fréquente que la
    fondamentale (sur Fm, le do domine). On garde donc la tonique de l'analyse harmonique
    (Krumhansl + vote d'accords) et on ne tranche que le mode au-dessus d'elle. La tonique
    n'est réécrite vers la note dominante que si celle de l'harmonie est quasi absente de
    la mélodie (< min_tonic_weight) — cas où l'analyse harmonique est manifestement à côté."""
    # no-op sur une clé d'un détecteur autoritaire (CNN madmom) : son mode est déjà fiable,
    # la mélodie ne doit pas le réécrire. Le garde-fou vit ici, pas chez l'appelant.
    if not melody_notes or key.get("source") == "madmom_cnn":
        return key
    pcs = np.bincount([n["pitch"] % 12 for n in melody_notes], minlength=12)
    total = pcs.sum()
    if total == 0:
        return key
    key_tonic = NOTE_NAMES.index(key["root"])
    # tonique harmonique conservée si elle est réellement présente dans la mélodie ;
    # sinon repli sur la classe de hauteur dominante
    tonic = key_tonic if pcs[key_tonic] / total >= min_tonic_weight else int(np.argmax(pcs))
    minor_third = pcs[(tonic + 3) % 12]
    major_third = pcs[(tonic + 4) % 12]
    mode = "minor" if minor_third >= major_third else "major"
    new_root = NOTE_NAMES[tonic]
    out = dict(key)
    out.update({"root": new_root, "mode": mode, "reconciled_by_melody": True,
                "before_melody": f"{key['root']} {key['mode']}",
                "changed_by_melody": (new_root, mode) != (key["root"], key["mode"])})
    return out


# madmom écrit les altérations en bémols ; on aligne sur NOTE_NAMES (dièses)
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def _norm_key_label(label: str) -> tuple[str, str]:
    """'F minor' / 'Db major' → ('F', 'minor') / ('C#', 'major')."""
    parts = label.split()
    root = _FLAT_TO_SHARP.get(parts[0], parts[0])
    return root, (parts[1] if len(parts) > 1 else "major")


def try_madmom_key(path):
    """Tonalité par CNN (madmom, Korzeniowski 'genre-agnostic key', 2018 ; extra [rhythm]).
    Tranche maj/min directement sur le signal, sans vote d'accords ni mélodie — plus fiable
    que Krumhansl. Renvoie un dict au format de estimate_key, ou None si madmom est absent
    ou échoue (l'appelant retombe alors sur Krumhansl + vote d'accords)."""
    try:
        from madmom.features.key import CNNKeyRecognitionProcessor, KEY_LABELS
    except Exception:
        return None
    try:
        pred = np.asarray(CNNKeyRecognitionProcessor()(path)).ravel()
    except Exception:
        return None
    order = np.argsort(pred)[::-1]
    root, mode = _norm_key_label(KEY_LABELS[int(order[0])])
    runner_root, runner_mode = _norm_key_label(KEY_LABELS[int(order[1])])
    return {"root": root, "mode": mode, "score": round(float(pred[order[0]]), 3),
            "runner_up": f"{runner_root} {runner_mode}", "source": "madmom_cnn"}


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
def wideband_mono(y_stereo) -> np.ndarray:
    """Mono à la fréquence d'origine, replié depuis le stéréo déjà chargé.

    `load_audio` garde le signal natif pour la stéréo et le loudness : le replier
    ici évite de relire le fichier juste pour retrouver les aigus.
    """
    y = np.asarray(y_stereo)
    return (y.mean(axis=0) if y.ndim == 2 else y).astype(np.float32)


def spectral_profile(y, sr, S=None):
    """Profil spectral — À NOURRIR EN PLEINE BANDE, pas avec le mono d'analyse.

    Les autres couches travaillent à 22 050 Hz, ce qui convient aux notes et aux
    beats mais coupe le spectre à 11 kHz. Le centroïde et le rolloff mesuraient
    donc un signal amputé de ses aigus : sur les deux morceaux du magasin, le
    centroïde monte de 16 à 24 % une fois le signal rendu entier. Le biais était
    systématique et allait toujours vers le sombre.

    Les seuils de description (1500 / 3000 Hz) restent inchangés : ce sont des
    repères de littérature établis sur du signal pleine bande, c'est la mesure
    qui s'en écartait. `sr_hz` est renvoyé pour que deux analyses ne soient
    comparées qu'à largeur de bande égale.

    `S` = magnitude STFT précalculée (np.abs(librosa.stft(y))) ; quand fournie, on
    évite quatre STFT internes (centroïde/rolloff/bande/flatness la recalculent chacune).
    """
    if S is None:
        S = np.abs(librosa.stft(y))
    cent = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    roll = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)))
    bw = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
    flat = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    desc = ("sombre / froid — peu d'harmoniques hautes" if cent < 1500
            else "équilibré — médiums présents" if cent < 3000
            else "brillant — énergie marquée dans les aigus")
    return {"centroid_hz": round(cent), "rolloff85_hz": round(roll),
            "bandwidth_hz": round(bw), "flatness": round(flat, 4),
            "sr_hz": int(sr), "description": desc}


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
