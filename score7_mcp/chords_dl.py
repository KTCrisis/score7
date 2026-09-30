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
import os
import shutil
import sys
import tempfile
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
        # time = vrai début du segment (et non bt[sb]) : sinon plusieurs accords avant le
        # premier beat s'écrasent tous à bt[0] avec le même timestamp
        # pas de champ `conf` : BTC et madmom ne renvoient pas de confiance par segment,
        # et un 1.0 constant se lit comme une certitude mesurée alors qu'il n'est qu'un
        # remplissage. Seule la route cosinus en publie une, qui en est vraiment une.
        out.append({"chord": compact, "chord_full": full, "start_beat": sb,
                    "beats": beats, "time": round(float(s), 2)})
    return _despike(out)


def _despike(grid: list, max_beats: int = 1) -> list:
    """Absorbe les micro-segments encadrés par le MÊME accord.

    Un détecteur trame par trame produit des accidents d'un beat au milieu d'un
    accord tenu : sur « A Midsummer Nice Dream », une quinzaine de `Bm7` d'un
    temps ponctuaient des plages de `Bm`. Ce n'est pas une lecture, c'est du
    jitter, et il pollue autant la grille affichée que le calcul de durées.

    On n'absorbe QUE si les deux voisins portent le même accord : un accord
    court entre deux accords différents est peut-être un vrai passage, et
    l'effacer inventerait une harmonie plus simple que la musique.
    """
    if len(grid) < 3:
        return grid
    out = [grid[0]]
    i = 1
    while i < len(grid) - 1:
        cur, prev, nxt = grid[i], out[-1], grid[i + 1]
        if cur["beats"] <= max_beats and prev["chord"] == nxt["chord"] != cur["chord"]:
            prev["beats"] += cur["beats"]          # le tenu absorbe l'accident
            i += 1
            continue
        if cur["chord"] == prev["chord"]:          # même accord de suite : fusionner
            prev["beats"] += cur["beats"]
            i += 1
            continue
        out.append(cur)
        i += 1
    last = grid[-1]
    if out and last["chord"] == out[-1]["chord"]:
        out[-1]["beats"] += last["beats"]
    else:
        out.append(last)
    return out


# --------------------------------------------------------------------------- BTC
def _ensure_btc_weights() -> str:
    """Télécharge (une fois) les poids BTC large-vocabulaire dans ~/.cache/score7/.
    Écriture atomique (fichier temporaire puis rename) + timeout : un download interrompu
    ne laisse jamais un .pt tronqué qui passerait le test d'existence et empoisonnerait
    le cache, forçant torch.load à échouer pour toujours."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    dest = _CACHE / _BTC_VOCA_NAME
    if dest.exists():
        return str(dest)
    print(f"→ téléchargement des poids BTC ({_BTC_VOCA_NAME})…", file=sys.stderr)
    fd, tmp = tempfile.mkstemp(dir=_CACHE, suffix=".part")
    try:
        with urllib.request.urlopen(_BTC_VOCA_URL, timeout=30) as resp, os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(resp, out)
        os.replace(tmp, dest)  # rename atomique : le fichier final n'apparaît que complet
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
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


#: lissage des probabilités BTC avant décision, en secondes : sous ~0,4 s un
#: changement d'accord est du scintillement trame à trame, pas une lecture
_SMOOTH_S = 0.4
#: un segment plus court que ceci est rendu au voisin le plus probable
_MIN_SEG_S = 0.3
#: nombre de candidats publiés par segment
_TOP_K = 3


def btc_probabilities(path: str):
    """Distribution BTC par trame (trames × 170), durée d'une trame (s), durée du
    morceau (s) et table d'étiquettes ; None si BTC est indisponible ou échoue.

    try_btc n'en gardait que l'argmax : la probabilité de la réponse, et les
    réponses suivantes, étaient jetées. Elles disent pourtant à quel point BTC
    est sûr (banc du 30/09 : AUC 0,75-0,78 entre justes et fausses)."""
    try:
        import torch
        import torch.nn.functional as F
        from score7_mcp._btc.features import audio_file_to_features
    except Exception:
        return None
    try:
        model, mean, std, idx2c, nts, device, cfg = _btc_model()
        feat, spf, song_len = audio_file_to_features(path, cfg)
        feat = (feat.T - mean) / std
        pad = nts - (feat.shape[0] % nts)
        feat = np.pad(feat, ((0, pad), (0, 0)), "constant")
        out = []
        with torch.no_grad():
            x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
            for t in range(feat.shape[0] // nts):
                enc, _ = model.self_attn_layers(x[:, nts * t:nts * (t + 1), :])
                logits = model.output_layer.output_projection(enc)
                out.append(F.softmax(logits, -1).squeeze(0).cpu().numpy())
        probs = np.concatenate(out)[: int(np.ceil(song_len / spf))]
        return probs, float(spf), float(song_len), idx2c
    except Exception:
        return None


def _fold_majmin(idx2c) -> tuple[list[str], np.ndarray]:
    """Matrice 170 → étiquettes compactes majeur/mineur (« N » gardé à part)."""
    from score7_mcp import theory
    labels = sorted({theory.compact(idx2c[i]) for i in range(len(idx2c))})
    fold = np.zeros((len(idx2c), len(labels)))
    for i in range(len(idx2c)):
        fold[i, labels.index(theory.compact(idx2c[i]))] = 1.0
    return labels, fold


def btc_segments(probs, spf: float, song_len: float, idx2c) -> list[dict]:
    """Segments datés en SECONDES, chacun avec sa confiance et ses candidats.

    La décision se prend sur les probabilités lissées, pas sur la grille de
    beats : un suivi de beats qui bascule en demi-tempo (L'éveil des Sirènes :
    332 beats trouvés sur 400) ne déplace plus les accords.
    `confidence` = probabilité moyenne de l'étiquette compacte retenue par BTC,
    c'est-à-dire sa certitude sur le segment (un départage par le chroma la laisse
    telle quelle : elle dit que BTC hésitait) ;
    `candidates` = les meilleures étiquettes compactes du segment."""
    labels, fold = _fold_majmin(idx2c)
    n = probs.shape[0]
    w = max(1, int(round(_SMOOTH_S / spf)))
    kernel = np.ones(w) / w
    smooth = np.vstack([np.convolve(probs[:, j], kernel, mode="same") for j in range(probs.shape[1])]).T
    frame_label = (smooth @ fold).argmax(axis=1)

    # segments bruts : changements de l'étiquette compacte
    bounds = [0] + [i for i in range(1, n) if frame_label[i] != frame_label[i - 1]] + [n]
    segs = [[bounds[i], bounds[i + 1]] for i in range(len(bounds) - 1)]

    # un segment trop court rejoint le voisin qui l'explique le mieux
    min_frames = max(1, int(round(_MIN_SEG_S / spf)))
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, (a, b) in enumerate(segs):
            if b - a >= min_frames:
                continue
            mass = (smooth[a:b] @ fold).sum(axis=0)
            left = segs[i - 1] if i > 0 else None
            right = segs[i + 1] if i + 1 < len(segs) else None
            def score(nb):
                return -1.0 if nb is None else float(mass[frame_label[nb[0]]])
            nb = left if score(left) >= score(right) else right
            nb[0], nb[1] = min(nb[0], a), max(nb[1], b)
            segs.pop(i)
            changed = True
            break

    out = []
    for a, b in segs:
        p_full = probs[a:b].mean(axis=0)
        p_mm = p_full @ fold
        order = np.argsort(-p_mm)
        best = labels[int(order[0])]
        if best == "N":
            continue
        # étiquette riche : la classe BTC la plus probable parmi celles qui se réduisent à `best`
        members = [i for i in range(len(idx2c)) if fold[i, labels.index(best)]]
        full = idx2c[max(members, key=lambda i: p_full[i])]
        full = best if full in ("N", "X") else full
        out.append({
            "chord": best, "chord_full": full,
            "time": round(a * spf, 2), "end": round(min(b * spf, song_len), 2),
            "confidence": round(float(p_mm[order[0]]), 3),
            "candidates": [{"chord": labels[int(j)], "p": round(float(p_mm[j]), 3)}
                           for j in order[:_TOP_K] if labels[int(j)] != "N"],
            "source": "btc",
        })
    # deux voisins identiques (séparés par un « N » retiré) : fusion
    merged = []
    for s in out:
        if merged and merged[-1]["chord"] == s["chord"] and abs(merged[-1]["end"] - s["time"]) < 1e-6:
            merged[-1]["end"] = s["end"]
        else:
            merged.append(s)
    return merged


def attach_beats(segs: list[dict], beat_times) -> list[dict]:
    """Ajoute start_beat / beats (affichage, consommateurs existants) à des segments
    datés en secondes ; les secondes restent la référence."""
    bt = np.asarray(beat_times, dtype=float)
    for s in segs:
        if bt.size == 0:
            s["start_beat"], s["beats"] = 0, 1
            continue
        sb = max(int(np.searchsorted(bt, s["time"], side="right") - 1), 0)
        eb = max(int(np.searchsorted(bt, s["end"], side="right") - 1), sb)
        s["start_beat"], s["beats"] = sb, max(eb - sb, 1)
    return segs


def tie_break(segs: list[dict], chroma: np.ndarray, frame_times, threshold: float = 0.8) -> list[dict]:
    """Quand BTC hésite (confiance < threshold), départage ses candidats par le chroma.

    Banc du 30/09 : le chroma seul, en juge, dégrade la grille (le lead et les
    arpèges tirent toujours vers un concurrent) ; en départage parmi les
    candidats de BTC, sur ses seuls segments incertains, il la corrige
    (Split Echo 70,3 → 73,4 %, L'éveil 60,2 → 68,5 %). Tous les candidats sont
    admis, même peu probables : les probabilités basses de BTC sont mal calibrées
    quand il hésite, et un plancher à 0,05 ou 0,10 coûte 1 à 3 points."""
    from score7_mcp import theory
    ft = np.asarray(frame_times, dtype=float)
    for s in segs:
        if s.get("confidence", 1.0) >= threshold or len(s.get("candidates", [])) < 2:
            continue
        sl = chroma[:, (ft >= s["time"]) & (ft < s["end"])]
        if sl.size == 0:
            continue
        obs = sl.mean(axis=1)
        pick = max(s["candidates"], key=lambda c: theory.fit(obs, c["chord"]))["chord"]
        if pick != s["chord"]:
            s["chord_btc"] = s["chord"]
            s["chord"] = pick
            s["chord_full"] = theory.mirex(pick)  # « A:min », pas « Am » : un lecteur de MIREX y verrait une fondamentale « Am »
            s["source"] = "btc+chroma"
    return segs


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
    res = btc_probabilities(path)
    if res is not None:
        segs = btc_segments(*res)
        if segs:
            return attach_beats(segs, beat_times), "btc"
    grid = try_madmom_chords(path, beat_times)
    if grid:
        return grid, "madmom"
    return fallback(), "template"
