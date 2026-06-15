"""Reconnaissance d'accords par deep learning : BTC (transformer) puis madmom (deep
chroma + CRF), avec repli sur le template matching de core.estimate_chords.

BTC (Bi-directional Transformer for Chord recognition, ISMIR 2019, vendorisé sous
score7_mcp/_btc/) sort un vocabulaire riche (maj/min/7/sus/dim, 170 classes) bien plus
fidèle que le cosinus sur chroma. Les poids (~33 Mo) sont téléchargés à la demande depuis
le dépôt d'origine et cachés dans ~/.cache/score7/. Imports lourds (torch) paresseux :
ce module n'est touché que si l'appelant demande la reconnaissance d'accords.
"""

from __future__ import annotations

import functools
import sys
import urllib.request
from pathlib import Path

import numpy as np

_CACHE = Path.home() / ".cache" / "score7"
_BTC_VOCA_URL = ("https://github.com/jayg996/BTC-ISMIR19/raw/master/test/"
                 "btc_model_large_voca.pt")
_BTC_VOCA_NAME = "btc_model_large_voca.pt"

# qualités à tierce mineure → suffixe "m" pour le vote de tonalité et la notation compacte
_MINOR_QUALITIES = {"min", "min6", "min7", "minmaj7", "dim", "dim7", "hdim7"}


# --------------------------------------------------------------------------- labels
def _label_to_score7(label: str) -> tuple[str, str]:
    """Label MIREX ('C', 'F:min', 'A#:sus4'…) → (compact, complet). Le compact réduit à
    maj/min ('Fm', 'C') pour rester compatible avec le vote de tonalité et la grille
    existante ; le complet garde la qualité riche pour l'affichage."""
    if label in ("N", "X"):
        return "N", "N"
    if ":" not in label:
        return label, label  # majeur pur
    root, qual = label.split(":", 1)
    compact = root + "m" if qual in _MINOR_QUALITIES else root
    return compact, label


def _segments_to_grid(segs, beat_times, min_beats: int = 1) -> list:
    """Segments temporels (start_s, end_s, compact, full) → grille alignée sur les beats,
    au format de core.estimate_chords (start_beat / beats / time), + champ chord_full."""
    bt = np.asarray(beat_times, dtype=float)
    out = []
    for s, e, compact, full in segs:
        if compact == "N":
            continue
        sb = max(int(np.searchsorted(bt, s, side="right") - 1), 0)
        eb = max(int(np.searchsorted(bt, e, side="right") - 1), sb)
        beats = max(eb - sb, 1)
        if beats < min_beats:
            continue
        out.append({"chord": compact, "chord_full": full, "start_beat": sb,
                    "beats": beats, "conf": 1.0,
                    "time": float(bt[sb]) if sb < len(bt) else float(s)})
    return out


# --------------------------------------------------------------------------- BTC
def _ensure_btc_weights() -> str:
    """Télécharge (une fois) les poids BTC large-vocabulaire dans ~/.cache/score7/."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    dest = _CACHE / _BTC_VOCA_NAME
    if not dest.exists():
        print(f"→ téléchargement des poids BTC ({_BTC_VOCA_NAME})…", file=sys.stderr)
        urllib.request.urlretrieve(_BTC_VOCA_URL, dest)
    return str(dest)


@functools.lru_cache(maxsize=1)
def _btc_model():
    """Charge (une fois par process) le modèle BTC large-voca + ses stats de normalisation.
    Renvoie (model, mean, std, idx_to_chord, n_timestep, device)."""
    import torch

    from score7_mcp._btc.features import idx2voca_chord
    from score7_mcp._btc.hparams import HParams
    from score7_mcp._btc.model import BTC_model

    cfg_path = Path(__file__).parent / "_btc" / "run_config.yaml"
    cfg = HParams.load(str(cfg_path))
    cfg.feature["large_voca"] = True
    cfg.model["num_chords"] = 170

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BTC_model(cfg.model).to(device)
    # weights_only=False : checkpoint de confiance (vendorisé), il porte mean/std numpy
    ckpt = torch.load(_ensure_btc_weights(), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["mean"], ckpt["std"], idx2voca_chord(), cfg.model["timestep"], device, cfg


def try_btc(path: str, beat_times) -> list | None:
    """Grille d'accords BTC alignée sur les beats, ou None si le paquet/les poids
    manquent ou si l'inférence échoue (l'appelant retombe sur madmom puis le cosinus)."""
    try:
        import torch
        from score7_mcp._btc.features import audio_file_to_features
    except Exception:
        return None
    try:
        model, mean, std, idx2c, nts, device, cfg = _btc_model()
        feat, fps, _ = audio_file_to_features(path, cfg)
        feat = (feat.T - mean) / std
        pad = nts - (feat.shape[0] % nts)
        feat = np.pad(feat, ((0, pad), (0, 0)), "constant")
        ninst = feat.shape[0] // nts

        raw, start, prev = [], 0.0, None
        with torch.no_grad():
            x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
            for t in range(ninst):
                enc, _ = model.self_attn_layers(x[:, nts * t:nts * (t + 1), :])
                pred, _ = model.output_layer(enc)
                pred = pred.squeeze()
                for i in range(nts):
                    idx = int(pred[i].item())
                    if prev is None:
                        prev = idx
                        continue
                    if idx != prev:
                        raw.append((start, fps * (nts * t + i), prev))
                        start = fps * (nts * t + i)
                        prev = idx
            if prev is not None:
                raw.append((start, fps * ninst * nts, prev))
    except Exception:
        return None

    segs = [(s, e, *_label_to_score7(idx2c[idx])) for s, e, idx in raw]
    grid = _segments_to_grid(segs, beat_times)
    return grid or None


# --------------------------------------------------------------------------- madmom
def try_madmom_chords(path: str, beat_times) -> list | None:
    """Grille d'accords madmom (deep chroma + CRF, Korzeniowski/Widmer ; maj/min, extra
    [rhythm]). Fallback si BTC est indisponible. None si madmom absent ou échoue."""
    try:
        from madmom.audio.chroma import DeepChromaProcessor
        from madmom.features.chords import DeepChromaChordRecognitionProcessor
    except Exception:
        return None
    try:
        chords = DeepChromaChordRecognitionProcessor()(DeepChromaProcessor()(path))
    except Exception:
        return None
    segs = [(float(s), float(e), *_label_to_score7(str(lab))) for s, e, lab in chords]
    grid = _segments_to_grid(segs, beat_times)
    return grid or None


# --------------------------------------------------------------------------- chaîne
def estimate_chords_chain(path, beat_times, fallback):
    """BTC > madmom > template matching. `fallback` est un callable sans argument
    (typiquement core.estimate_chords déjà bindé) renvoyant la grille cosinus.
    Renvoie (grille, source)."""
    grid = try_btc(path, beat_times)
    if grid:
        return grid, "btc"
    grid = try_madmom_chords(path, beat_times)
    if grid:
        return grid, "madmom"
    return fallback(), "template"
