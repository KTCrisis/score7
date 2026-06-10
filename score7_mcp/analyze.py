"""Orchestrateur : enchaîne les analyses et renvoie un dict de résultats unique.

C'est le point d'entrée partagé par le CLI et le serveur MCP.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import librosa

from score7_mcp import core


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_")


def analyze_file(path: str, sr: int = 22050, title: str | None = None,
                 separate: bool = False, melody: bool = False,
                 melody_src: str | None = None, outdir: str | None = None) -> dict:
    """Analyse complète. `separate`/`melody` activent les étages lourds (extra [melody])."""
    path = str(Path(path).expanduser().resolve())
    title = title or Path(path).stem
    slug = slugify(title)
    outdir = str(Path(outdir).expanduser()) if outdir else str(Path.home() / "Renoise" / "analyses")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    y, sr, y_stereo, sr_native = core.load_audio(path, sr=sr)
    tempo, meter, beats, beat_times = core.estimate_rhythm(y, sr, path=path)
    chords = core.estimate_chords(y, sr, beats, beat_times)
    chroma_mean = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr), axis=1)
    key = core.estimate_key(chroma_mean, chord_grid=chords)

    results = {
        "title": title, "slug": slug, "file": path, "outdir": outdir,
        "bpm": tempo["bpm"], "tempo": tempo, "meter": meter, "key": key, "chords": chords,
        "structure": core.dynamic_structure(y, sr),
        "spectral": core.spectral_profile(y, sr),
        "stereo": core.stereo_width(y_stereo),
        "loudness": core.loudness(y_stereo, sr_native),
    }

    if separate or melody:
        from score7_mcp import melody as mel_mod
        if separate:
            try:
                results["stems"] = mel_mod.separate(path, outdir)
            except Exception as e:
                results["stems_error"] = str(e)
        if melody:
            src = melody_src
            if src is None and results.get("stems"):
                stems = Path(results["stems"])
                for cand in ("other.wav", "vocals.wav"):
                    if (stems / cand).exists():
                        src = str(stems / cand); break
            src = src or path
            try:
                results["melody"] = mel_mod.extract_melody(src, outdir, slug)
                results["melody"]["source"] = src
                # la mélodie tranche le mode majeur/mineur (plus fiable que les accords)
                results["key"] = core.reconcile_key_with_melody(
                    results["key"], results["melody"]["notes"])
            except Exception as e:
                results["melody_error"] = str(e)

    return results
