"""CLI score7 — analyse un fichier audio et écrit une fiche markdown (+ JSON optionnel)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from score7_mcp.analyze import analyze_file
from score7_mcp.render import render_markdown


def main(argv=None):
    ap = argparse.ArgumentParser(prog="score7",
                                 description="Analyse harmonique et sonore d'un fichier audio.")
    ap.add_argument("file")
    ap.add_argument("--out", default=None, help="dossier de sortie (défaut ~/Renoise/analyses)")
    ap.add_argument("--title", default=None, help="titre + slug des fichiers de sortie")
    ap.add_argument("--json", action="store_true", help="écrit aussi un .json")
    ap.add_argument("--separate", action="store_true", help="séparation stems (Demucs)")
    ap.add_argument("--melody", action="store_true", help="extraction de la ligne mélodique")
    ap.add_argument("--melody-src", default=None, help="stem précis pour la mélodie")
    ap.add_argument("--no-dl-chords", dest="dl_chords", action="store_false",
                    help="accords par template chroma seul (sans BTC/madmom, rapide)")
    ap.add_argument("--sr", type=int, default=22050)
    args = ap.parse_args(argv)

    if not Path(args.file).expanduser().exists():
        sys.exit(f"Fichier introuvable : {args.file}")

    try:
        r = analyze_file(args.file, sr=args.sr, title=args.title, separate=args.separate,
                         melody=args.melody, melody_src=args.melody_src, outdir=args.out,
                         dl_chords=args.dl_chords)
    except ValueError as e:
        sys.exit(f"score7 : {e}")

    outdir = Path(r["outdir"])
    md_path = outdir / f"{r['slug']}.md"
    md_path.write_text(render_markdown(r), encoding="utf-8")
    print(f"\n✓ Fiche : {md_path}")
    if args.json:
        (outdir / f"{r['slug']}.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✓ JSON  : {outdir / (r['slug'] + '.json')}")

    k = r["key"]
    met = (r.get("meter") or {}).get("meter")
    meter_txt = f" · {met}" if met else ""
    print(f"\n{r['title']} — {k['root']} {k['mode']} · {r['bpm']} BPM{meter_txt} · "
          f"accords {r.get('chords_source', 'template')} · "
          f"centroïde {r['spectral']['centroid_hz']} Hz · LUFS {r['loudness']['integrated_lufs']}")
    tp = r.get("tempo") or {}
    if tp.get("bpm_confidence") is not None and tp["bpm_confidence"] < 0.5:
        alts = ", ".join(f"{c['bpm']} ({c['strength']:.0%})" for c in tp["bpm_candidates"])
        print(f"  (tempo ambigu — candidats d'octave : {alts})")
    if k.get("corrected"):
        print(f"  (tonalité corrigée : Krumhansl seul disait {k['krumhansl_only']})")

    # échecs partiels des étages lourds : la fiche existe mais incomplète — le dire
    failed = False
    if r.get("stems_error"):
        print(f"⚠ séparation échouée : {r['stems_error']}", file=sys.stderr)
        failed = True
    if r.get("melody_error"):
        print(f"⚠ mélodie échouée : {r['melody_error']}", file=sys.stderr)
        failed = True
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
