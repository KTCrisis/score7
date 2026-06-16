"""Serveur MCP FastMCP exposant l'analyse audio de score7.

Deux tools :
  - analyze_audio    : analyse harmonique/sonore complète (+ stems/mélodie en option)
  - separate_stems   : séparation Demucs seule

Les options `separate`/`melody` déclenchent des calculs GPU (séparation ~10 s,
transcription plus long) — l'appel bloque le temps du traitement.
"""

from __future__ import annotations

from pathlib import Path

from fastmcp import FastMCP

from score7_mcp.analyze import analyze_file
from score7_mcp.render import render_markdown

mcp = FastMCP(
    "score7",
    instructions=(
        "Analyse harmonique et sonore de fichiers audio (mp3/flac/wav) : tonalité "
        "(Krumhansl + vote d'accords), grille d'accords, structure dynamique, profil "
        "spectral, stéréo, loudness ; séparation de stems (Demucs) et extraction "
        "mélodique (transcription + skyline) en option. Estimation statistique — fiable "
        "sur le spectral/dynamique/loudness/tonalité, approximative sur accords et mélodie."
    ),
)


_CHORDS_RETURN_CAP = 24


def _compact(r: dict) -> dict:
    """Allège le résultat pour un retour MCP. La fiche markdown et le JSON sur disque
    (écrits AVANT cet appel) gardent le détail complet ; le retour, lui, est plafonné pour
    ne pas dépasser la limite de tokens d'un tool result. On retire les tableaux les plus
    verbeux (notes/séquence mélodie, vecteurs de pattern rythmique) et on tronque la grille
    d'accords — le client lit la fiche pour le détail intégral."""
    out = dict(r)

    # mélodie : le MIDI porte le détail ; retirer le tableau de notes ET la séquence verbeuse
    if isinstance(out.get("melody"), dict):
        m = dict(out["melody"])
        m.pop("notes", None)
        m.pop("sequence", None)
        out["melody"] = m

    # grille d'accords : tronquer (BTC en produit des centaines sur un morceau long)
    chords = out.get("chords")
    if isinstance(chords, list) and len(chords) > _CHORDS_RETURN_CAP:
        out["chords"] = chords[:_CHORDS_RETURN_CAP]
        out["chords_total"] = len(chords)  # signale que la fiche en a davantage

    # pattern rythmique : garder les grilles ASCII lisibles, retirer les vecteurs bruts
    rp = out.get("rhythm_pattern")
    if isinstance(rp, dict) and isinstance(rp.get("bands"), dict):
        rp = dict(rp)
        rp["bands"] = {name: {k: v for k, v in band.items() if k != "pattern"}
                       for name, band in rp["bands"].items()}
        out["rhythm_pattern"] = rp

    return out


@mcp.tool
def analyze_audio(
    file_path: str,
    separate: bool = False,
    melody: bool = False,
    melody_src: str | None = None,
    write_fiche: bool = True,
    write_json: bool = False,
    title: str | None = None,
    out_dir: str | None = None,
    dl_chords: bool = True,
    sep_model: str = "htdemucs",
) -> dict:
    """Analyse un fichier audio et renvoie tonalité, accords, structure, spectral, stéréo,
    loudness. `separate`=True ajoute les stems Demucs ; `melody`=True extrait la ligne
    mélodique (utilise un stem isolé si séparé, sinon le mix). `sep_model` choisit le modèle
    de séparation : htdemucs (défaut), htdemucs_ft (plus lent, basse plus propre) ou
    htdemucs_6s (6 stems, +guitar/piano — peu utile sur du synthé). `dl_chords`=True (défaut)
    reconnaît les accords par BTC (transformer) avec repli madmom puis template ; False =
    template chroma seul (rapide, sans GPU). Écrit une fiche markdown dans out_dir (défaut
    ~/audio_analysis, surchargeable via SCORE7_OUT) sauf si write_fiche=False."""
    if not Path(file_path).expanduser().exists():
        return {"error": f"Fichier introuvable : {file_path}"}
    try:
        r = analyze_file(file_path, title=title, separate=separate, melody=melody,
                         melody_src=melody_src, outdir=out_dir, dl_chords=dl_chords,
                         sep_model=sep_model)
    except ValueError as e:
        return {"error": str(e)}
    # échecs partiels (stems/mélodie) : promus en warnings pour qu'un client
    # qui ne lit que le sommet de la réponse ne les prenne pas pour un succès
    warnings = [f"{k.removesuffix('_error')} : {r[k]}"
                for k in ("stems_error", "melody_error") if r.get(k)]
    if warnings:
        r["warnings"] = warnings
    outdir = Path(r["outdir"])
    if write_fiche:
        (outdir / f"{r['slug']}.md").write_text(render_markdown(r), encoding="utf-8")
        r["fiche"] = str(outdir / f"{r['slug']}.md")
    if write_json:
        import json
        (outdir / f"{r['slug']}.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        r["json"] = str(outdir / f"{r['slug']}.json")
    return _compact(r)


@mcp.tool
def separate_stems(file_path: str, out_dir: str | None = None,
                   sep_model: str = "htdemucs") -> dict:
    """Sépare un fichier audio en stems via Demucs. `sep_model` : htdemucs (4 stems
    vocals/other/bass/drums, défaut), htdemucs_ft (4 stems fine-tunés, plus lent) ou
    htdemucs_6s (6 stems, +guitar/piano). Renvoie le dossier des stems. Nécessite l'extra [melody]."""
    if not Path(file_path).expanduser().exists():
        return {"error": f"Fichier introuvable : {file_path}"}
    from score7_mcp import melody as mel_mod
    outdir = str(Path(out_dir).expanduser()) if out_dir else str(Path.home() / "Renoise" / "analyses")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    try:
        return {"stems_dir": mel_mod.separate(str(Path(file_path).expanduser()), outdir,
                                              model=sep_model)}
    except Exception as e:
        return {"error": str(e)}


def main():
    mcp.run()


if __name__ == "__main__":
    main()
