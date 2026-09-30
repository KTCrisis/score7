"""Ce qu'est un accord, en un seul endroit.

Les étiquettes MIREX des détecteurs (« A#:maj7 », « F:min », « C ») se lisent ici :
fondamentale, intervalles, réduction majeur/mineur, et chroma attendu. Avant ce
module, la même connaissance vivait en morceaux dans chords_dl (qualités mineures)
et core (gabarits majeur/mineur seulement).
"""

from __future__ import annotations

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

#: intervalles (demi-tons depuis la fondamentale) du vocabulaire BTC large
QUALITIES: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7),
    "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10),
    "minmaj7": (0, 3, 7, 11), "dim7": (0, 3, 6, 9), "hdim7": (0, 3, 6, 10),
    "maj6": (0, 4, 7, 9), "min6": (0, 3, 7, 9),
}

#: qualités à tierce mineure : « m » en notation compacte
MINOR_QUALITIES = {"min", "min6", "min7", "minmaj7", "dim", "dim7", "hdim7"}


def parse(label: str) -> tuple[int, str] | None:
    """« A#:maj7 » → (10, "maj7") ; « C » → (0, "maj") ; « N »/« X » → None.

    Une qualité inconnue du vocabulaire est lue comme majeure plutôt que de
    faire échouer l'analyse ; un renversement (« C/E ») est ignoré ici, la basse
    réellement jouée vient du stem (core.annotate_bass_roots).
    """
    if label in ("N", "X", ""):
        return None
    root, _, qual = label.partition(":")
    qual = qual.split("/")[0] or "maj"
    if root not in NOTE_NAMES:
        return None
    return NOTE_NAMES.index(root), (qual if qual in QUALITIES else "maj")


def compact(label: str) -> str:
    """Notation compacte majeur/mineur (« Fm », « C »), celle du vote de tonalité et des grilles."""
    p = parse(label)
    if p is None:
        return "N"
    root, qual = p
    return NOTE_NAMES[root] + ("m" if qual in MINOR_QUALITIES else "")


def mirex(compact_label: str) -> str:
    """« Am » → « A:min », « F » → « F » : la forme que lisent les consommateurs de chord_full."""
    if compact_label.endswith("m") and compact_label[:-1] in NOTE_NAMES:
        return compact_label[:-1] + ":min"
    return compact_label


def pitch_classes(label: str) -> set[int]:
    """Classes de hauteur de l'accord ; « Fm » (compact) est accepté comme « F:min »."""
    if label.endswith("m") and ":" not in label and label[:-1] in NOTE_NAMES:
        label = label[:-1] + ":min"
    p = parse(label)
    if p is None:
        return set()
    root, qual = p
    return {(root + i) % 12 for i in QUALITIES[qual]}


def template(label: str) -> np.ndarray:
    """Chroma attendu de l'accord : un vecteur unitaire sur ses classes de hauteur."""
    v = np.zeros(12)
    pcs = pitch_classes(label)
    if pcs:
        v[list(pcs)] = 1.0
        v /= np.linalg.norm(v)
    return v


def fit(chroma: np.ndarray, label: str) -> float:
    """Cosinus entre un chroma observé et le gabarit de l'accord (0 si l'un est vide)."""
    t = template(label)
    n = float(np.linalg.norm(chroma))
    return float(chroma @ t / n) if n > 0 and t.any() else 0.0
