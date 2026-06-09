# score7

Analyse harmonique et sonore de fichiers audio (mp3/flac/wav) : tonalité, grille
d'accords, structure dynamique, profil spectral, stéréo, loudness — plus séparation
de stems et extraction mélodique en option. CLI **et** serveur MCP.

C'est de l'**estimation statistique**, pas une transcription exacte. Fiable sur le
spectral, la dynamique, la stéréo, le loudness et la tonalité ; approximatif sur la
grille d'accords (harmonie suspendue) et la mélodie (à nettoyer à l'oreille).

## Installation

```bash
pip install -e .            # cœur (librosa) — analyse harmonique/sonore + MCP
pip install -e ".[melody]"  # + Demucs et transcription (séparation / mélodie, GPU)
pip install -e ".[test]"    # + pytest
```

`basic-pitch` n'est pas utilisé (épingle numpy<1.24, incompatible Python 3.12) :
la transcription passe par `piano_transcription_inference`.

## CLI

```bash
score7 morceau.flac --json
score7 morceau.flac --title mon_titre --separate --melody
score7 morceau.flac --melody --melody-src stems/other.wav
```

Sortie dans `~/Renoise/analyses/` (ou `--out`) : fiche `.md`, `.json` optionnel,
stems, `<slug>_poly_full.mid`, `<slug>_melody.mid`.

## MCP

```bash
python -m score7_mcp        # lance le serveur (stdio)
```

Tools : `analyze_audio` (analyse complète, options `separate`/`melody`) et
`separate_stems`. Branché dans flux7-mesh via :

```yaml
- name: score7
  transport: stdio
  command: /home/fluxart/py_env/bin/python
  args: ["-m", "score7_mcp"]
```

## Méthodes

| Étage | Technique |
|-------|-----------|
| Tonalité | Krumhansl-Schmuckler (chroma CQT) **+ vote par fonction d'accords** (corrige la confusion majeur/mineur relatif) |
| Accords | template matching cosinus sur chroma CENS synchronisé beats, segments fusionnés |
| Structure | RMS par fenêtres + détection de paliers (dérivée d'énergie) |
| Spectral | centroïde, rolloff 85 %, bande passante, flatness |
| Stéréo | corrélation L/R, ratio side/mid |
| Loudness | LUFS intégré (pyloudnorm), crest factor |
| Séparation | Demucs (htdemucs) |
| Mélodie | transcription polyphonique → skyline band-limité (C4–C6, saillance durée×vélocité) |

## Tests

```bash
pytest
```
