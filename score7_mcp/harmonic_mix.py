"""Reconstruire un mix HARMONIQUE à partir des stems, pour lire les accords.

Jusqu'ici les accords et la tonalité étaient estimés sur le mix complet, avec la
batterie dedans. C'est le bruit le plus direct qu'on puisse mettre dans une
estimation d'accords : une cymbale étale son énergie sur les douze classes de
hauteur, et un chroma ne sait pas la distinguer d'un cluster.

La séparation étant de toute façon calculée quand `--separate` est demandé, il
suffit de la faire AVANT et de sommer les stems qui portent l'harmonie.

Ce qu'on garde, et pourquoi :

- **bass** : décisif. Deux triades relatives partagent deux notes sur trois ;
  seule la fondamentale les sépare, et c'est la basse qui la donne. Sans elle,
  un détecteur confond un accord avec son renversement et un majeur avec son
  relatif mineur.
- **other** : le lit harmonique dans une séparation à 4 stems (nappes, cordes,
  claviers).
- **guitar**, **piano** (htdemucs_6s) : les instruments qui *énoncent* les
  accords, isolés de ceux qui les brouillent.

Ce qu'on écarte :

- **drums** : énergie large bande, aucune hauteur.
- **vocals** : porte la mélodie, pas l'harmonie, et son vibrato et ses
  portamentos étalent le chroma autour de la note visée.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

#: stems retenus, par ordre de contribution à l'harmonie
HARMONIC_STEMS = ("bass", "piano", "guitar", "other")
#: écartés, quel que soit le modèle de séparation
EXCLUDED_STEMS = ("drums", "vocals")


def available(stems_dir: str | Path) -> list[Path]:
    """Stems harmoniques présents dans le dossier, dans l'ordre canonique."""
    d = Path(stems_dir)
    out = []
    for name in HARMONIC_STEMS:
        p = d / f"{name}.wav"
        if p.exists():
            out.append(p)
    return out


def build(stems_dir: str | Path, out_path: str | Path, *, sr: int | None = None) -> str | None:
    """Somme les stems harmoniques dans un WAV, et renvoie son chemin.

    Renvoie None si aucun stem harmonique n'est disponible : l'appelant retombe
    alors sur le mix d'origine, ce qui est le comportement d'avant. Un mix
    harmonique absent doit dégrader l'analyse, jamais la casser.
    """
    import librosa
    import soundfile as sf

    parts = available(stems_dir)
    if not parts:
        return None

    mix = None
    rate = sr
    for p in parts:
        y, r = librosa.load(str(p), sr=rate, mono=True)
        rate = rate or r
        if mix is None:
            mix = y
        else:
            n = min(len(mix), len(y))     # les stems font la même longueur, mais on ne parie pas
            mix = mix[:n] + y[:n]

    if mix is None or mix.size == 0:
        return None

    peak = float(np.max(np.abs(mix)))
    if peak > 1.0:                        # la somme peut dépasser : on ramène sans écrêter
        mix = mix / peak

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), mix, rate)
    print(f"→ mix harmonique : {' + '.join(p.stem for p in parts)}", file=sys.stderr)
    return str(out)
