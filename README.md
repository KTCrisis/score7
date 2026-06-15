# score7

Analyse harmonique et sonore de fichiers audio (mp3/flac/wav) : tonalité, grille
d'accords, tempo et métrique, structure dynamique, profil spectral, stéréo,
loudness — plus séparation de stems et extraction mélodique en option. CLI **et**
serveur MCP.

C'est de l'**analyse**, pas une transcription fidèle. Chaque dimension difficile
suit une chaîne « modèle entraîné d'abord, méthode signal en repli » : tempo
(Beat This!), accords (BTC), tonalité (CNN), mélodie (PESTO), tout retombant sur
librosa sans les extras. Fiable sur le tempo, la tonalité, l'harmonie, le spectral,
la dynamique, la stéréo et le loudness ; la **mélodie** reste le maillon le plus
fragile (suivi monophonique, à nettoyer à l'oreille) — score7 analyse, il ne
reconstruit pas une partition jouable.

score7 est le maillon **audio** de l'écosystème **keys7** (analyse depuis le son
rendu, là où xrns7 lit le fichier projet et play7 joue le symbolique). Voir
[ECOSYSTEM.md](https://github.com/KTCrisis/keys7/blob/main/ECOSYSTEM.md).

## Installation

```bash
pip install -e .            # cœur (librosa) — analyse harmonique/sonore + MCP
pip install -e ".[melody]"  # + Demucs, PESTO, transcription poly (séparation / mélodie, GPU)
pip install -e ".[rhythm]"  # + Beat This! et madmom (beats/downbeats/tonalité/accords neuronaux)
pip install -e ".[test]"    # + pytest
pip install "madmom @ git+https://github.com/CPJKU/madmom.git"  # voir ci-dessous
```

Les poids des modèles (Beat This! ~77 Mo, BTC ~12 Mo, PESTO, Demucs, transcription)
se téléchargent à la première utilisation. `basic-pitch` n'est pas utilisé (épingle
numpy<1.24, incompatible Python 3.12).

L'extra `[rhythm]` apporte les trackers neuronaux (Beat This! pour les beats,
madmom pour la tonalité CNN et les accords deep-chroma). madmom demande un install
git : le 0.16.1 de PyPI est cassé sur Python 3.12. Sans ces extras, tempo, tonalité
et accords retombent proprement sur les méthodes librosa/Krumhansl/cosinus.

## CLI

```bash
score7 morceau.flac --json
score7 morceau.flac --title mon_titre --separate --melody
score7 morceau.flac --melody --melody-src stems/other.wav
```

Sortie dans `~/Renoise/analyses/` (ou `--out`) : fiche `.md`, `.json` optionnel,
stems, `<slug>_poly_full.mid`, `<slug>_melody.mid`.

Tempo et métrique sont estimés systématiquement (pas de flag) ; quand l'octave
de tempo est ambiguë (confiance < 0.5), la fiche et la console listent les
candidats avec leur force.

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

Les dimensions difficiles suivent une chaîne **modèle entraîné → repli signal** :
le premier maillon disponible gagne, et tout retombe sur librosa sans les extras.
Les auteurs et papiers de chaque modèle sont en [Crédits](#crédits-et-références).

| Étage | Technique | Modèle (référence) |
|-------|-----------|--------------------|
| Tempo / beats | priorité réseau de neurones > librosa ; BPM = médiane des intervalles inter-beats ; candidats d'octave (T/2, T, 2T) exposés avec leur force au tempogramme | **Beat This!** > madmom > librosa |
| Métrique | downbeats du tracker neuronal (temps par mesure) sinon repli sur les accents du beat grid ; subdivision binaire/ternaire (6/8, 12/8 vs 2/4-4/4) | Beat This! / madmom |
| Tonalité | CNN genre-agnostique en tête ; sinon Krumhansl-Schmuckler (chroma CQT) + vote par fonction d'accords + réconciliation par la tonique de la mélodie | **CNN madmom** > Krumhansl |
| Accords | transformer bidirectionnel (vocabulaire large : maj/min/7/sus/dim) ; sinon deep-chroma+CRF ; sinon template matching cosinus sur chroma CENS | **BTC** > madmom > cosinus |
| Mélodie | suivi de hauteur monophonique sur stem isolé (≈10 ms) ; sinon transcription poly → skyline (C4–C6) ; sinon pYIN. Suit la voix dominante, ne sépare pas mélodie/accompagnement | **PESTO** > skyline > pYIN |
| Séparation | séparation de sources hybride spectro-temporelle (4 ou 6 stems, dont piano) | **Demucs** htdemucs |
| Rythme par stem | pattern batterie (bandes kick/snare/hats repliées sur la grille), microtiming, swing | maison (librosa/scipy) |
| Texture par stem | flux spectral, variation du centroïde, ratio percussif (HPSS), part d'énergie, largeur stéréo | maison (librosa) |
| Structure | RMS par fenêtres + détection de paliers (dérivée d'énergie) | maison (librosa) |
| Spectral | centroïde, rolloff 85 %, bande passante, flatness | librosa |
| Stéréo | corrélation L/R, ratio side/mid | maison (numpy) |
| Loudness | LUFS intégré (ITU-R BS.1770), crest factor | pyloudnorm |

score7 n'entraîne aucun modèle : il **assemble** l'état de l'art open-source derrière
une chaîne de repli déterministe. Le crédit revient aux auteurs ci-dessous.

## Crédits et références

| Modèle | Auteurs | Publication |
|--------|---------|-------------|
| **Beat This!** (beats/downbeats) | F. Foscarin, J. Schlüter, G. Widmer (CPJKU) | *Beat This! Accurate Beat Tracking Without DBN Postprocessing*, ISMIR 2024 — [code](https://github.com/CPJKU/beat_this) |
| **BTC** (accords) | J. Park, K. Choi, S. Jeon, D. Kim, J. Park | *A Bi-Directional Transformer for Musical Chord Recognition*, ISMIR 2019 — [code](https://github.com/jayg996/BTC-ISMIR19) (MIT, sous-ensemble inférence vendorisé dans `score7_mcp/_btc/`) |
| **madmom** (beats, downbeats, deep-chroma chords) | S. Böck, F. Korzeniowski, J. Schlüter, F. Krebs, G. Widmer | *madmom: A New Python Audio and Music Signal Processing Library*, ACM MM 2016 |
| **CNN key** (tonalité) | F. Korzeniowski, G. Widmer | *Genre-Agnostic Key Classification with CNN*, ISMIR 2018 |
| **PESTO** (mélodie / pitch) | A. Riou, S. Lattner, G. Hadjeres, G. Peeters | *PESTO: Pitch Estimation with Self-supervised Transposition-equivariant Objective*, ISMIR 2023 |
| **Demucs** (séparation) | S. Rouard, F. Massa, A. Défossez (Meta FAIR) | *Hybrid Transformers for Music Source Separation*, ICASSP 2023 |
| **piano_transcription** (skyline) | Q. Kong, B. Li, X. Song, Y. Wan, Y. Wang (ByteDance) | *High-resolution Piano Transcription with Pedals by Regressing Onset and Offset Times*, IEEE/ACM TASLP 2021 |
| **pYIN** (pitch de repli) | M. Mauch, S. Dixon | *pYIN: A Fundamental Frequency Estimator…*, ICASSP 2014 (via librosa) |
| **Krumhansl-Schmuckler** (profils tonals) | C. L. Krumhansl, E. J. Kessler | *Tracing the Dynamic Changes in Perceived Tonal Organization*, 1982 |
| **librosa** | B. McFee et al. | *librosa: Audio and Music Signal Analysis in Python*, SciPy 2015 |
| **pyloudnorm** | C. J. Steinmetz, J. D. Reiss | implémentation ITU-R BS.1770 |

## Tests

```bash
pytest
```
