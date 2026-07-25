"""Orchestrateur : enchaîne les analyses et renvoie un dict de résultats unique.

C'est le point d'entrée partagé par le CLI et le serveur MCP.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import librosa

from score7_mcp import core


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_")


def _resolve_outdir(outdir: str | None = None) -> str:
    """Dossier de sortie : argument explicite > variable d'env SCORE7_OUT > défaut
    ~/audio_analysis. Le défaut est neutre ; régler SCORE7_OUT pour le rediriger
    (ex. ~/Renoise/analyses pour un workflow Renoise)."""
    chosen = outdir or os.environ.get("SCORE7_OUT") or str(Path.home() / "audio_analysis")
    return str(Path(chosen).expanduser())


def analyze_file(path: str, sr: int = 22050, title: str | None = None,
                 separate: bool = False, melody: bool = False,
                 melody_src: str | None = None, outdir: str | None = None,
                 dl_chords: bool = True, sep_model: str = "htdemucs") -> dict:
    """Analyse complète. `separate`/`melody` activent les étages lourds (extra [melody])."""
    path = str(Path(path).expanduser().resolve())
    title = title or Path(path).stem
    slug = slugify(title)
    outdir = _resolve_outdir(outdir)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    try:
        y, sr, y_stereo, sr_native = core.load_audio(path, sr=sr)
    except Exception as e:
        # fichier corrompu / format non audio : erreur propre plutôt qu'une
        # traceback librosa/audioread que ni le CLI ni le MCP ne savent présenter
        raise ValueError(f"Lecture audio impossible ({path}) : {str(e) or type(e).__name__}") from e
    if y.size == 0:
        raise ValueError(f"Fichier audio vide : {path}")
    tempo, meter, beats, beat_times = core.estimate_rhythm(y, sr, path=path)

    # Séparation d'abord quand elle est demandée : l'harmonie se lit bien mieux
    # sans la batterie, dont l'énergie large bande brouille tout chroma. Le mix
    # harmonique (basse + piano/guitare + nappes) devient la source des accords
    # ET de la tonalité ; le mix d'origine reste le repli.
    results_stems = None
    harm_path = path
    if separate:
        from score7_mcp import melody as mel_mod
        try:
            results_stems = mel_mod.separate(path, outdir, model=sep_model)
        except Exception as e:
            stems_error = str(e)
        else:
            stems_error = None
            try:
                from score7_mcp import harmonic_mix
                built = harmonic_mix.build(results_stems,
                                           Path(outdir) / f"{slug}.harmonic.wav", sr=sr)
                if built:
                    harm_path = built
            except Exception as e:      # un mix absent dégrade, il ne casse pas
                print(f"mix harmonique impossible : {e}", file=sys.stderr)
    else:
        stems_error = None

    # le chroma de repli suit la même source que les détecteurs
    if harm_path != path:
        try:
            y_h, _sr_h, _st, _srn = core.load_audio(harm_path, sr=sr)
        except Exception:
            y_h = y
    else:
        y_h = y

    # Grille cosinus : fallback partagé (tonalité Krumhansl + chaîne d'accords), calculée
    # au plus une fois. Dans le cas nominal (madmom + BTC présents) elle ne sert jamais.
    _cos = []
    def cosine_grid():
        if not _cos:
            _cos.append(core.estimate_chords(y_h, sr, beats, beat_times))
        return _cos[0]

    # Tonalité : CNN madmom (genre-agnostic, tranche maj/min sur le signal) en tête ; sinon
    # Krumhansl + vote d'accords sur la grille cosinus, raffiné par la mélodie plus bas.
    # Chaque détecteur tague key["source"] : une seule source de vérité, dérivée plus bas.
    key = core.try_madmom_key(harm_path)
    if key is None:
        chroma_mean = np.mean(librosa.feature.chroma_cqt(y=y_h, sr=sr), axis=1)
        key = core.estimate_key(chroma_mean, chord_grid=cosine_grid())

    # Accords (affichage) : chaîne BTC > madmom > cosinus, indépendante de la tonalité.
    if dl_chords:
        from score7_mcp import chords_dl
        chords, chords_source = chords_dl.estimate_chords_chain(harm_path, beat_times, cosine_grid)
    else:
        chords, chords_source = cosine_grid(), "template"

    results = {
        "title": title, "slug": slug, "file": path, "outdir": outdir,
        "bpm": tempo["bpm"], "tempo": tempo, "meter": meter, "key": key, "chords": chords,
        "key_source": key.get("source"), "chords_source": chords_source,
        # sur QUOI l'harmonie a été lue : le mix harmonique (sans batterie ni
        # voix) ou le mix d'origine. Deux analyses du même morceau ne sont
        # comparables que si cette source est la même.
        "harmony_source": "harmonic_stems" if harm_path != path else "mix",
        # La grille de beats RÉELLE (beat_this/madmom/librosa) : c'est le référentiel
        # des start_beat d'accords. Exportée pour que les consommateurs (flux7-audio)
        # convertissent les temps en secondes (mélodie) sur la MÊME grille — un
        # simple ×bpm/60 dérive dès que le tempo n'est pas machine ou que le
        # premier beat n'est pas à t=0.
        "rhythm": {"beat_times": [round(float(t), 4) for t in beat_times],
                   "beats_per_bar": (meter or {}).get("beats_per_bar")},
        "structure": core.dynamic_structure(y, sr),
        "spectral": core.spectral_profile(y, sr),
        "stereo": core.stereo_width(y_stereo),
        "loudness": core.loudness(y_stereo, sr_native),
    }

    if separate or melody:
        from score7_mcp import melody as mel_mod
        if separate:
            # déjà séparé plus haut pour l'harmonie : Demucs coûte des minutes,
            # on ne le relance pas
            if results_stems:
                results["stems"] = results_stems
            elif stems_error:
                results["stems_error"] = stems_error
            # couche rythme + texture par stem (librosa seul, pas de GPU)
            if results.get("stems"):
                try:
                    from score7_mcp import texture as tex_mod
                    bpb = (meter or {}).get("beats_per_bar") or 4
                    results.update(tex_mod.analyze_stems(
                        results["stems"], beat_times, sr=sr, beats_per_bar=bpb))
                except Exception as e:
                    results["texture_error"] = str(e)
        if melody:
            src = melody_src
            if src is None and results.get("stems"):
                stems = Path(results["stems"])
                for cand in ("other.wav", "vocals.wav"):
                    if (stems / cand).exists():
                        src = str(stems / cand); break
            src = src or path
            try:
                results["melody"] = mel_mod.extract_melody(src, outdir, slug,
                                                           bpm=tempo["bpm"])
                results["melody"]["source"] = src
                # reconcile_key_with_melody no-op de lui-même sur une clé CNN (mode déjà fiable)
                results["key"] = core.reconcile_key_with_melody(
                    results["key"], results["melody"]["notes"])
            except Exception as e:
                results["melody_error"] = str(e)

    return results
