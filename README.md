# score7

Harmonic and sonic analysis of audio files (mp3/flac/wav): key, chord grid, tempo
and meter, dynamic structure, spectral profile, stereo, loudness, plus stem
separation and melody extraction as options. CLI **and** MCP server.

This is **analysis**, not faithful transcription. Each hard dimension follows a
"trained model first, signal method as fallback" chain: tempo (Beat This!), chords
(BTC), key (CNN), melody (PESTO), all degrading to librosa without the extras.
Reliable on tempo, key, harmony, spectral, dynamics, stereo and loudness; the
**melody** stays the most fragile link (monophonic tracking, clean up by ear).
score7 analyses, it does not reconstruct a playable score.

score7 is the **audio** link of the **keys7** ecosystem (analysis from rendered
sound, where xrns7 reads the project file and play7 plays the symbolic). See
[ECOSYSTEM.md](https://github.com/KTCrisis/keys7/blob/main/ECOSYSTEM.md).

## Installation

```bash
pip install -e .            # core (librosa): harmonic/sonic analysis + MCP
pip install -e ".[melody]"  # + Demucs, PESTO, poly transcription (separation / melody, GPU)
pip install -e ".[rhythm]"  # + Beat This! and madmom (neural beats/downbeats/key/chords)
pip install -e ".[test]"    # + pytest
pip install "madmom @ git+https://github.com/CPJKU/madmom.git"  # see below
```

Model weights (Beat This! ~77 MB, BTC ~12 MB, PESTO, Demucs, transcription) are
downloaded on first use. `basic-pitch` is not used (it pins numpy<1.24,
incompatible with Python 3.12).

The `[rhythm]` extra brings the neural trackers (Beat This! for beats, madmom for
the CNN key and deep-chroma chords). madmom needs a git install: PyPI 0.16.1 is
broken on Python 3.12. Without these extras, tempo, key and chords fall back
cleanly to the librosa / Krumhansl / cosine methods.

## CLI

```bash
score7 track.flac --json
score7 track.flac --title my_title --separate --melody
score7 track.flac --melody --melody-src stems/other.wav
```

Output goes to `--out` if given, else `$SCORE7_OUT`, else `~/audio_analysis/`: a
`.md` sheet, optional `.json`, stems, `<slug>_poly_full.mid`, `<slug>_melody.mid`.

Tempo and meter are always estimated (no flag); when the tempo octave is ambiguous
(confidence < 0.5), the sheet and console list the candidates with their strength.

## MCP

```bash
python -m score7_mcp        # run the server (stdio)
```

Tools: `analyze_audio` (full analysis, `separate`/`melody` options) and
`separate_stems`. Wired into flux7-mesh via:

```yaml
- name: score7
  transport: stdio
  command: /home/fluxart/py_env/bin/python
  args: ["-m", "score7_mcp"]
```

## Analysis layers

score7 reads a track on four layers, from the most structural to the most granular.

**Harmony.** The key (which scale the piece lives in) and the chord grid (its
harmonic motion). Answers "what is the tonal colour, and how does it move".

**Time.** Tempo and meter (the pulse and its grouping), plus a dynamic structure
that marks where sections rise and fall in energy. Answers "how fast, and how is it
organised over time".

**Sound.** Spectral profile (bright vs dark), stereo image (narrow vs wide), and
loudness with crest factor (level and preserved dynamic range). Answers "what does
it sound like and how is it mixed", independently of the notes.

**Stems** (`--separate`). Demucs splits the mix, then score7 analyses what the full
mix hides: the melodic line on an isolated stem, the drum pattern (kick/snare/hats
folded onto the beat grid, with microtiming and swing), and the timbral texture of
each stem. Answers "what is each instrument actually doing".

The first three layers run on the bare core (CPU, no PyTorch). The stem layer needs
the `[melody]` extra.

## Methods

Hard dimensions follow a **trained-model then signal-fallback** chain: the first
available link wins, and everything falls back to librosa without the extras.
Authors and papers for each model are in [Credits](#credits-and-references).

| Stage | Technique | Model (reference) |
|-------|-----------|-------------------|
| Tempo / beats | neural tracker first, else librosa; BPM = median inter-beat interval; octave candidates (T/2, T, 2T) exposed with their tempogram strength | **Beat This!** > madmom > librosa |
| Meter | downbeats from the neural tracker (beats per bar) else accent folding on the beat grid; binary/ternary subdivision (6/8, 12/8 vs 2/4-4/4) | Beat This! / madmom |
| Key | genre-agnostic CNN first; else Krumhansl-Schmuckler (chroma CQT) + chord-function vote + melody-tonic reconciliation | **madmom CNN** > Krumhansl |
| Chords | bidirectional transformer (large vocabulary: maj/min/7/sus/dim); else deep-chroma+CRF; else cosine template matching on beat-synced CENS chroma | **BTC** > madmom > cosine |
| Melody | monophonic pitch tracking on an isolated stem (~10 ms); else poly transcription to skyline (C4-C6); else pYIN. Follows the dominant voice, does not separate melody from accompaniment | **PESTO** > skyline > pYIN |
| Separation | hybrid spectro-temporal source separation (4 or 6 stems, piano included) | **Demucs** htdemucs |
| Per-stem rhythm | drum pattern (kick/snare/hats bands folded onto the beat grid), microtiming, swing | in-house (librosa/scipy) |
| Per-stem texture | spectral flux, centroid variation, percussive ratio (HPSS), energy share, stereo width | in-house (librosa) |
| Structure | per-window RMS + layer-onset detection (energy derivative) | in-house (librosa) |
| Spectral | centroid, rolloff 85%, bandwidth, flatness | librosa |
| Stereo | L/R correlation, side/mid ratio | in-house (numpy) |
| Loudness | integrated LUFS (ITU-R BS.1770), crest factor | pyloudnorm |

score7 trains no model: it **assembles** open-source state of the art behind a
deterministic fallback chain. Credit belongs to the authors below.

## Credits and references

| Model | Authors | Publication |
|-------|---------|-------------|
| **Beat This!** (beats/downbeats) | F. Foscarin, J. Schlüter, G. Widmer (CPJKU) | *Beat This! Accurate Beat Tracking Without DBN Postprocessing*, ISMIR 2024 - [code](https://github.com/CPJKU/beat_this) |
| **BTC** (chords) | J. Park, K. Choi, S. Jeon, D. Kim, J. Park | *A Bi-Directional Transformer for Musical Chord Recognition*, ISMIR 2019 - [code](https://github.com/jayg996/BTC-ISMIR19) (MIT; inference subset vendored in `score7_mcp/_btc/`) |
| **madmom** (beats, downbeats, deep-chroma chords) | S. Böck, F. Korzeniowski, J. Schlüter, F. Krebs, G. Widmer | *madmom: A New Python Audio and Music Signal Processing Library*, ACM MM 2016 |
| **CNN key** | F. Korzeniowski, G. Widmer | *Genre-Agnostic Key Classification with CNN*, ISMIR 2018 |
| **PESTO** (melody / pitch) | A. Riou, S. Lattner, G. Hadjeres, G. Peeters | *PESTO: Pitch Estimation with Self-supervised Transposition-equivariant Objective*, ISMIR 2023 |
| **Demucs** (separation) | S. Rouard, F. Massa, A. Défossez (Meta FAIR) | *Hybrid Transformers for Music Source Separation*, ICASSP 2023 |
| **piano_transcription** (skyline) | Q. Kong, B. Li, X. Song, Y. Wan, Y. Wang (ByteDance) | *High-resolution Piano Transcription with Pedals by Regressing Onset and Offset Times*, IEEE/ACM TASLP 2021 |
| **pYIN** (fallback pitch) | M. Mauch, S. Dixon | *pYIN: A Fundamental Frequency Estimator…*, ICASSP 2014 (via librosa) |
| **Krumhansl-Schmuckler** (tonal profiles) | C. L. Krumhansl, E. J. Kessler | *Tracing the Dynamic Changes in Perceived Tonal Organization*, 1982 |
| **librosa** | B. McFee et al. | *librosa: Audio and Music Signal Analysis in Python*, SciPy 2015 |
| **pyloudnorm** | C. J. Steinmetz, J. D. Reiss | ITU-R BS.1770 implementation |

## Tests

```bash
pytest
```
