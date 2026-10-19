"""Spectre d'un rendu d'instrument : où va l'énergie, et à quoi ça ressemble.

Pensé pour une piste rendue seule (song_render d'un instrument), pas pour un mix :
énergie par bande, centroïde, rolloff, niveaux, largeur stéréo, et un spectrogramme
(+ spectre moyen) en image. Avec plusieurs fichiers, les spectres moyens sont
superposés pour comparer deux réglages d'un même son.

matplotlib n'est requis que pour l'image (extra [plot]) ; les mesures s'en passent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft

# Bandes de mix usuelles : ce qui encombre (sub, grave), ce qui porte (médium),
# ce qui fait passer devant (présence), ce qui fait briller (air).
BANDS = (("sub", 0.0, 60.0), ("grave", 60.0, 250.0), ("medium", 250.0, 2000.0),
         ("presence", 2000.0, 6000.0), ("air", 6000.0, None))

_NPERSEG = 4096


def _load(path: str):
    x, sr = sf.read(str(Path(path).expanduser()), always_2d=True)
    return x, sr


def _power(mono: np.ndarray, sr: int):
    f, t, z = stft(mono, sr, nperseg=_NPERSEG, noverlap=_NPERSEG * 3 // 4)
    return f, t, np.abs(z) ** 2


def measure(path: str) -> dict:
    """Mesures spectrales et de niveau d'un fichier audio."""
    x, sr = _load(path)
    mono = x.mean(axis=1)
    if not np.any(mono):
        raise ValueError(f"fichier silencieux : {path}")
    f, _, p = _power(mono, sr)
    avg = p.mean(axis=1)
    tot = avg.sum()
    bands = {}
    for name, lo, hi in BANDS:
        sel = (f >= lo) & (f < hi) if hi else (f >= lo)
        bands[name] = round(100 * float(avg[sel].sum() / tot), 1)
    cum = np.cumsum(avg) / tot
    corr = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]) if x.shape[1] > 1 else 1.0
    return {
        "file": str(path),
        "duration_s": round(len(mono) / sr, 2),
        "bands_pct": bands,
        "centroid_hz": int(round(float((avg * f).sum() / tot))),
        "rolloff85_hz": int(round(float(f[np.argmax(cum >= 0.85)]))),
        "peak_dbfs": round(20 * np.log10(np.abs(x).max()), 1),
        "rms_dbfs": round(20 * np.log10(np.sqrt(np.mean(mono ** 2))), 1),
        "stereo_corr": round(corr, 2),
    }


def plot(paths: list[str], out_png: str, labels: list[str] | None = None,
         title: str | None = None) -> str:
    """Spectrogramme du premier fichier + spectres moyens superposés de tous."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = labels or [Path(p).stem for p in paths]
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1.6]})
    for i, (path, label) in enumerate(zip(paths, labels)):
        x, sr = _load(path)
        f, t, p = _power(x.mean(axis=1), sr)
        if i == 0:
            db = 10 * np.log10(p + 1e-12)
            ax[0].pcolormesh(t, f, db, shading="auto", cmap="magma",
                             vmin=db.max() - 80, vmax=db.max())
            ax[0].set_yscale("log")
            ax[0].set_ylim(30, sr / 2)
            ax[0].set_ylabel("Hz")
            ax[0].set_xlabel("s")
            ax[0].set_title(title or label)
        avg = p.mean(axis=1)
        ax[1].semilogx(f[1:], 10 * np.log10(avg[1:] + 1e-12), label=label, lw=1)
    for _, lo, _ in BANDS[1:]:
        ax[1].axvline(lo, color="grey", ls=":", lw=0.8)
    ax[1].set_xlim(30, 20000)
    ax[1].set_xlabel("Hz")
    ax[1].set_ylabel("dB moyen")
    ax[1].grid(alpha=0.3)
    if len(paths) > 1:
        ax[1].legend(fontsize=8)
    fig.tight_layout()
    out = Path(out_png).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return str(out)


def main(argv=None):
    import argparse
    import json

    ap = argparse.ArgumentParser(prog="score7-spectrum",
                                 description="Spectre d'un rendu d'instrument (un ou plusieurs wav à comparer).")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--png", default=None, help="image à écrire (spectrogramme + spectres moyens)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--json", action="store_true", help="sortie JSON au lieu du tableau")
    args = ap.parse_args(argv)

    rows = [measure(p) for p in args.files]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        head = f"{'fichier':28s} " + " ".join(f"{n:>8s}" for n, _, _ in BANDS) + \
               "  centroïde  rolloff85  peak  rms  stéréo"
        print(head)
        for r in rows:
            b = " ".join(f"{r['bands_pct'][n]:7.1f}%" for n, _, _ in BANDS)
            print(f"{Path(r['file']).stem[:28]:28s} {b}  {r['centroid_hz']:6d} Hz  {r['rolloff85_hz']:6d} Hz"
                  f"  {r['peak_dbfs']:5.1f} {r['rms_dbfs']:5.1f}  {r['stereo_corr']:5.2f}")
    if args.png:
        print(f"✓ Image : {plot(args.files, args.png, title=args.title)}")


if __name__ == "__main__":
    main()
