"""Étapes lourdes (GPU) : séparation de stems, transcription, extraction mélodique.

Imports paresseux : ces fonctions ne sont appelées que sur demande, et les dépendances
(demucs, piano_transcription_inference) sont dans l'extra [melody]. Le cœur d'analyse
tourne sans elles.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def separate(path: str, outdir: str, model: str = "htdemucs") -> str:
    """Demucs → stems. `model` choisit l'architecture : htdemucs (4 stems, défaut rapide),
    htdemucs_ft (4 stems fine-tunés, ~2-4× plus lent), htdemucs_6s (6 stems, ajoute
    guitar/piano — peu utile sur du synthé). Renvoie le dossier des stems.

    Demucs écrit dans {outdir}/{model}/{nom}, donc le chemin de retour suit le modèle."""
    print(f"→ Demucs ({model}) sur {Path(path).name}…", file=sys.stderr)
    subprocess.run([sys.executable, "-m", "demucs", "-n", model, "-o", str(outdir), str(path)],
                   check=True)
    return str(Path(outdir) / model / Path(path).stem)


def transcribe(path: str, out_midi: str) -> str:
    """Transcription polyphonique → MIDI (piano_transcription_inference)."""
    from piano_transcription_inference import PianoTranscription, sample_rate
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    PianoTranscription(device="cuda" if _cuda() else "cpu").transcribe(audio, out_midi)
    return out_midi


def _skyline(midi_path: str, lo: int = 60, hi: int = 84, win: float = 0.25):
    """Ligne mélodique depuis un MIDI polyphonique : bande [lo,hi] (C4–C6 par défaut),
    fenêtres de `win` s, note la plus saillante (durée × vélocité) par fenêtre."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    band = [n for inst in pm.instruments for n in inst.notes if lo <= n.pitch <= hi]
    if not band:
        return []
    mel, t, T = [], 0.0, pm.get_end_time()
    while t < T:
        active = [n for n in band if n.start < t + win and n.end > t]
        if active:
            best = max(active, key=lambda n: (n.end - n.start) * n.velocity)
            if not mel or mel[-1]["pitch"] != best.pitch:
                mel.append({"pitch": best.pitch, "start": round(t, 2),
                            "dur": round(best.end - best.start, 2),
                            "name": pretty_midi.note_number_to_name(best.pitch)})
        t += win
    return mel


def _melody_pyin(path: str, sr: int = 22050, fmin="C3", fmax="C7", min_dur=0.1):
    """Fallback monophonique pYIN (peu fiable sur nappes denses)."""
    import pretty_midi
    y, sr = librosa.load(path, sr=sr, mono=True)
    y_h = librosa.effects.harmonic(y, margin=3.0)
    f0, voiced, vprob = librosa.pyin(y_h, fmin=librosa.note_to_hz(fmin),
                                     fmax=librosa.note_to_hz(fmax), sr=sr)
    times = librosa.times_like(f0, sr=sr)
    mf = librosa.hz_to_midi(f0)
    notes, cur, start = [], None, None
    for i in range(len(f0)):
        ok = bool(voiced[i]) and np.isfinite(mf[i]) and vprob[i] > 0.5
        p = int(round(mf[i])) if ok else None
        if p != cur:
            if cur is not None and times[i] - start >= min_dur:
                notes.append({"pitch": cur, "start": float(start), "dur": float(times[i] - start),
                              "name": pretty_midi.note_number_to_name(cur)})
            cur, start = p, (times[i] if p is not None else None)
    return notes


def _highpass(y, sr, fc: float = 180.0):
    """Passe-haut avant suivi de f0. Sur un stem synthé, la pédale grave (bourdon
    de tonique, basse résiduelle) domine la saillance et capture le tracker mono —
    constaté sur du synthwave : 77 % des notes étaient la pédale. Couper sous
    ~180 Hz sort le bourdon du champ, le tracker suit le registre mélodique."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, fc, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def _merge_microgaps(notes: list, max_gap: float = 0.08) -> list:
    """Fusionne les notes de même hauteur séparées par un micro-trou (<80 ms) :
    le vacillement de saillance hachure une tenue en confettis, pas la musique."""
    out = []
    for n in notes:
        prev = out[-1] if out else None
        if (prev and n["pitch"] == prev["pitch"]
                and 0 <= n["start"] - (prev["start"] + prev["dur"]) <= max_gap):
            prev["dur"] = round(n["start"] + n["dur"] - prev["start"], 2)
        else:
            out.append(dict(n))
    return out


def _drop_outliers(notes: list, jump: int = 10, max_dur: float = 0.3) -> list:
    """Éjecte les notes courtes isolées à plus de `jump` demi-tons de leurs DEUX
    voisines : un flip de saillance entre octaves/voix, pas une note jouée."""
    if len(notes) < 3:
        return notes
    keep = [notes[0]]
    for prev, cur, nxt in zip(notes, notes[1:], notes[2:]):
        if (cur["dur"] <= max_dur
                and abs(cur["pitch"] - prev["pitch"]) > jump
                and abs(cur["pitch"] - nxt["pitch"]) > jump):
            continue
        keep.append(cur)
    keep.append(notes[-1])
    return keep


def _melody_pesto(path: str, conf_pct: float = 60.0, min_dur: float = 0.1,
                  hp: float | None = 180.0) -> list:
    """Suivi de hauteur PESTO (transformer self-supervised, ISMIR 2023 ; extra [melody]).
    État de l'art du pitch monophonique : bien plus précis et pur que pYIN (résolution
    ~10 ms, quasi pas de notes hors-gamme). Comme tout pitch tracker mono, il suit la voix
    la plus saillante — sur un stem polyphonique ce n'est pas toujours la mélodie, d'où
    l'intérêt de le lancer sur un stem isolé. Segmente le f0 en notes (quantif. demi-ton,
    seuil de confiance au percentile `conf_pct`, durée mini `min_dur`)."""
    import pesto
    import pretty_midi
    import torch

    y, sr = librosa.load(path, sr=None, mono=True)
    if len(y) < sr * 0.25:  # PESTO padde sa STFT sur ~2048 samples : trop court = crash interne
        return []
    if hp:
        y = _highpass(y, sr, hp)
    ts, pitch, conf = (np.asarray(a) for a in pesto.predict(torch.from_numpy(y).float(), sr)[:3])
    if len(ts) < 2:  # ceinture+bretelles si une version livrait <2 pas
        return []
    midi = librosa.hz_to_midi(pitch)
    # PESTO échantillonne en ms ; on dérive le pas réel (≈10 ms) plutôt que de deviner sur
    # ts.max() (qui casse sur un clip court). Robuste même si une version livrait déjà des s.
    step = float(ts[1] - ts[0])
    dt = step / 1000.0 if step > 1.0 else step
    thr = float(np.percentile(conf, conf_pct))
    notes, cur, start = [], None, 0

    def flush(end):
        if cur is not None and (end - start) * dt >= min_dur:
            notes.append({"pitch": cur, "start": round(start * dt, 2),
                          "dur": round((end - start) * dt, 2),
                          "name": pretty_midi.note_number_to_name(cur)})

    for i in range(len(midi)):
        p = int(round(midi[i])) if (conf[i] >= thr and np.isfinite(midi[i])) else None
        if p != cur:
            flush(i)
            cur, start = p, i
    flush(len(midi))  # ferme la dernière note (sinon une note tenue jusqu'au bout est perdue)
    return notes


def _split_at_onsets(notes: list, y, sr, min_dur: float = 0.1) -> list:
    """Re-articule les notes répétées. La segmentation f0 (PESTO/pYIN) ne coupe
    que sur CHANGEMENT de hauteur : deux croches répétées deviennent une blanche.
    Les onsets du signal (attaques) coupent les tenues au bon endroit ; une coupe
    n'est posée que si les deux segments restent >= min_dur."""
    onset_t = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    out = []
    for n in notes:
        end = n["start"] + n["dur"]
        seg = n["start"]
        for c in onset_t:
            if seg + min_dur <= c <= end - min_dur:
                out.append({**n, "start": round(seg, 2), "dur": round(float(c - seg), 2)})
                seg = float(c)
        out.append({**n, "start": round(seg, 2), "dur": round(end - seg, 2)})
    return out


def _velocities(notes: list, y, sr, lo: int = 45, hi: int = 112) -> list:
    """Vélocité par note depuis l'enveloppe RMS à l'onset, normalisée sur les
    percentiles 10/95 du morceau (robuste au mastering). Sans elle tout sort à
    vélocité fixe — ni le MIDI ni la partition ne portent de nuances."""
    rms = librosa.feature.rms(y=y)[0]
    if not len(rms):
        return notes
    t_rms = librosa.times_like(rms, sr=sr)
    db = librosa.amplitude_to_db(rms + 1e-9)
    p10, p95 = np.percentile(db, 10), np.percentile(db, 95)
    span = max(p95 - p10, 1e-6)
    for n in notes:
        v = float(np.interp(n["start"], t_rms, db))
        x = min(1.0, max(0.0, (v - p10) / span))
        n["vel"] = int(round(lo + x * (hi - lo)))
    return notes


def _melody_basic_pitch(path: str, min_amp: float = 0.15, min_dur: float = 0.1) -> list:
    """Transcription polyphonique basic-pitch (Spotify, ICASSP 2022, backend ONNX).
    Entend TOUTES les notes du stem — c'est la sonde ET la transcription du
    matériau polyphonique. `amp` (0-1) est conservé pour les vélocités."""
    import pretty_midi
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    _, _, events = predict(str(path), ICASSP_2022_MODEL_PATH)
    notes = []
    for s, e, p, a, _bends in events:
        if a < min_amp or (e - s) < min_dur:
            continue
        notes.append({"pitch": int(p), "start": round(float(s), 2),
                      "dur": round(float(e - s), 2), "amp": float(a),
                      "name": pretty_midi.note_number_to_name(int(p))})
    return sorted(notes, key=lambda n: (n["start"], -n["pitch"]))


def _mean_polyphony(notes: list) -> float:
    """Notes simultanées moyennes, échantillonnées juste après chaque onset.
    C'est LE critère d'aiguillage : <=1.2 -> matériau mono (PESTO, plus précis) ;
    au-delà -> polyphonique (skyline sur basic-pitch)."""
    if not notes:
        return 0.0
    total = 0
    for n in notes:
        t = n["start"] + 0.01
        total += sum(1 for m in notes if m["start"] <= t < m["start"] + m["dur"])
    return total / len(notes)


def _split_voices(notes: list) -> tuple[list, list]:
    """Skyline : à chaque instant, la note la plus haute qui sonne = la ligne ;
    tout ce qui sonne SOUS une note plus haute = accompagnement (arpèges, nappes).
    Sur du matériau à voix entrelacées, une voix seule n'est pas le morceau —
    les deux parts gardent le tissage entier."""
    line, acc = [], []
    for n in notes:
        covered = any(m["pitch"] > n["pitch"]
                      and m["start"] < n["start"] + n["dur"] - 0.02
                      and m["start"] + m["dur"] > n["start"] + 0.02
                      for m in notes if m is not n)
        (acc if covered else line).append(n)
    return line, acc


def _amp_velocities(notes: list, lo: int = 45, hi: int = 112) -> list:
    """Vélocités depuis l'amplitude basic-pitch, normalisée percentiles 10/95."""
    if not notes:
        return notes
    amps = sorted(n["amp"] for n in notes)
    p10 = amps[max(0, int(0.10 * len(amps)) - 1)]
    p95 = amps[min(len(amps) - 1, int(0.95 * len(amps)))]
    span = max(p95 - p10, 1e-6)
    for n in notes:
        x = min(1.0, max(0.0, (n["amp"] - p10) / span))
        n["vel"] = int(round(lo + x * (hi - lo)))
    return notes


def extract_melody(path: str, outdir: str, slug: str, band=(60, 84),
                   bpm: float | None = None, poly_threshold: float = 1.2) -> dict:
    """Ligne mélodique → MIDI + séquence. Chaîne : PESTO (pitch mono, état de l'art) >
    skyline (transcription poly) > pYIN. À lancer de préférence sur un stem isolé.
    `bpm` cale le plancher de durée sur le tempo (≈ double-croche) au lieu d'un
    seuil en secondes — 0,1 s à 88 BPM laisse passer des confettis d'un 1/7 de beat."""
    import pretty_midi
    print(f"→ Extraction mélodie sur {Path(path).name}…", file=sys.stderr)
    min_dur = max(0.1, 0.22 * 60.0 / bpm) if bpm else 0.1
    notes, method, voices, poly = None, None, None, None

    # Sonde polyphonique : basic-pitch (~5 s en ONNX) entend TOUTES les notes.
    # Sa mesure arbitre : matériau mono -> PESTO (plus précis sur une voix) ;
    # poly -> skyline sur ses propres notes, en deux voix (ligne / arpèges).
    try:
        bp = _melody_basic_pitch(path, min_dur=min_dur)
        if bp:
            poly = _mean_polyphony(bp)
            print(f"  polyphonie mesurée : {poly:.2f} "
                  f"({'poly' if poly > poly_threshold else 'mono'})", file=sys.stderr)
            if poly > poly_threshold:
                line, acc = _split_voices(bp)
                line = _drop_outliers(_merge_microgaps(line))
                acc = _merge_microgaps(acc)
                _amp_velocities(line)
                _amp_velocities(acc)
                notes = line
                voices = [{"name": "ligne", "notes": line},
                          {"name": "arpèges", "notes": acc}]
                method = f"basic-pitch + skyline (polyphonie {poly:.1f})"
    except Exception as e:
        print(f"  basic-pitch indisponible ({e}) → chaîne mono", file=sys.stderr)

    if notes is None:
        try:
            notes = _melody_pesto(path, min_dur=min_dur)
            if not notes:
                raise ValueError("PESTO n'a renvoyé aucune note")
            method = "PESTO (pitch mono, ISMIR 2023)"
        except Exception as e:
            print(f"  PESTO indisponible ({e}) → skyline", file=sys.stderr)
            try:
                full = str(Path(outdir) / f"{slug}_poly_full.mid")
                transcribe(path, full)
                notes = _skyline(full, lo=band[0], hi=band[1])
                method = "skyline (transcription poly)"
            except Exception as e2:
                print(f"  transcription poly indisponible ({e2}) → fallback pYIN", file=sys.stderr)
                notes = _melody_pyin(path)
                method = "pYIN monophonique (fallback)"

    # post-traitement sur le signal source. Ordre : recoller les confettis
    # (micro-trous de même hauteur), éjecter les flips de saillance isolés,
    # PUIS ré-articuler les vraies répétitions aux onsets (f0 seulement —
    # skyline vient d'une transcription déjà articulée), enfin les nuances RMS.
    # Le chemin basic-pitch porte déjà ses vélocités (amplitude par note).
    try:
        if method and method.startswith(("PESTO", "pYIN")):
            y_src, sr_src = librosa.load(path, sr=22050, mono=True)
            notes = _merge_microgaps(notes)
            notes = _drop_outliers(notes)
            notes = _split_at_onsets(notes, y_src, sr_src, min_dur=min_dur)
            notes = _velocities(notes, y_src, sr_src)
        elif not voices:
            y_src, sr_src = librosa.load(path, sr=22050, mono=True)
            notes = _velocities(notes, y_src, sr_src)
    except Exception as e:
        print(f"  post-traitement mélodie ignoré ({e})", file=sys.stderr)

    pm = pretty_midi.PrettyMIDI()
    for v in (voices or [{"name": "melody", "notes": notes}]):
        inst = pretty_midi.Instrument(program=0, name=v["name"])
        for n in v["notes"]:
            inst.notes.append(pretty_midi.Note(velocity=n.get("vel", 90), pitch=n["pitch"],
                                                start=n["start"], end=n["start"] + max(n["dur"], 0.05)))
        pm.instruments.append(inst)
    midi_path = str(Path(outdir) / f"{slug}_melody.mid")
    pm.write(midi_path)

    # diagnostic : classes de hauteur (sert à vérifier la gamme) — sur la ligne
    from collections import Counter
    pcs = Counter(pretty_midi.note_number_to_name(n["pitch"] % 12 + 60)[:-1] for n in notes)
    out = {"midi": midi_path, "notes": notes, "n_notes": len(notes), "method": method,
           "sequence": " ".join(n["name"] for n in notes),
           "pitch_classes": dict(sorted(pcs.items(), key=lambda kv: -kv[1]))}
    if voices:
        out["voices"] = voices
        out["polyphony"] = round(poly, 2)
    return out
