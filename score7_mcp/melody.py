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


def _melody_pesto(path: str, conf_pct: float = 60.0, min_dur: float = 0.1) -> list:
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


def extract_melody(path: str, outdir: str, slug: str, band=(60, 84)) -> dict:
    """Ligne mélodique → MIDI + séquence. Chaîne : PESTO (pitch mono, état de l'art) >
    skyline (transcription poly) > pYIN. À lancer de préférence sur un stem isolé."""
    import pretty_midi
    print(f"→ Extraction mélodie sur {Path(path).name}…", file=sys.stderr)
    notes, method = None, None
    try:
        notes = _melody_pesto(path)
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

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="melody")
    for n in notes:
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=n["pitch"],
                                            start=n["start"], end=n["start"] + max(n["dur"], 0.05)))
    pm.instruments.append(inst)
    midi_path = str(Path(outdir) / f"{slug}_melody.mid")
    pm.write(midi_path)

    # diagnostic : classes de hauteur (sert à vérifier la gamme)
    from collections import Counter
    pcs = Counter(pretty_midi.note_number_to_name(n["pitch"] % 12 + 60)[:-1] for n in notes)
    return {"midi": midi_path, "notes": notes, "n_notes": len(notes), "method": method,
            "sequence": " ".join(n["name"] for n in notes),
            "pitch_classes": dict(sorted(pcs.items(), key=lambda kv: -kv[1]))}
