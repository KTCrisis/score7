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


def _compact(r: dict) -> dict:
    """Allège le résultat pour un retour MCP : on retire les tableaux de notes bruts."""
    out = dict(r)
    if "melody" in out and isinstance(out["melody"], dict):
        m = dict(out["melody"])
        m.pop("notes", None)  # le MIDI tient la donnée complète
        out["melody"] = m
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
) -> dict:
    """Analyse un fichier audio et renvoie tonalité, accords, structure, spectral, stéréo,
    loudness. `separate`=True ajoute les stems Demucs ; `melody`=True extrait la ligne
    mélodique (utilise un stem isolé si séparé, sinon le mix). Écrit une fiche markdown
    dans out_dir (défaut ~/Renoise/analyses) sauf si write_fiche=False."""
    if not Path(file_path).expanduser().exists():
        return {"error": f"Fichier introuvable : {file_path}"}
    r = analyze_file(file_path, title=title, separate=separate, melody=melody,
                     melody_src=melody_src, outdir=out_dir)
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
def separate_stems(file_path: str, out_dir: str | None = None) -> dict:
    """Sépare un fichier audio en 4 stems (vocals/other/bass/drums) via Demucs.
    Renvoie le dossier des stems. Nécessite l'extra [melody]."""
    if not Path(file_path).expanduser().exists():
        return {"error": f"Fichier introuvable : {file_path}"}
    from score7_mcp import melody as mel_mod
    outdir = str(Path(out_dir).expanduser()) if out_dir else str(Path.home() / "Renoise" / "analyses")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    try:
        return {"stems_dir": mel_mod.separate(str(Path(file_path).expanduser()), outdir)}
    except Exception as e:
        return {"error": str(e)}


def main():
    mcp.run()


if __name__ == "__main__":
    main()
