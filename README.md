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
downloaded on first use. `basic-pitch` is *optional* and no extra pulls it in: it
pins numpy<1.24, incompatible with Python 3.12. The melody chain probes for it
and falls through to PESTO when it is missing, which changes what the melody
layer can produce — see [Melody](#melody-what-gets-filtered-and-why).

The `[rhythm]` extra brings the neural trackers (Beat This! for beats, madmom for
the CNN key and deep-chroma chords). madmom needs a git install: PyPI 0.16.1 is
broken on Python 3.12. Without these extras, tempo, key and chords fall back
cleanly to the librosa / Krumhansl / cosine methods.

## CLI

```bash
score7 track.flac --json
score7 track.flac --title my_title --separate --melody
score7 track.flac --separate --sep-model htdemucs_ft     # alternate Demucs model
score7 track.flac --melody --melody-src stems/other.wav
score7 track.flac --no-dl-chords                         # chroma templates only, no GPU
```

`--sep-model` picks the Demucs architecture: `htdemucs` (default, 4 stems, fast),
`htdemucs_ft` (4 stems fine-tuned, cleaner bass/vocals, ~2-4x slower) or
`htdemucs_6s` (6 stems, adds guitar/piano). On dense synth material the default is
enough: `6s` leaves guitar/piano near-empty (synths stay in `other`) and `ft` only
marginally improves `other`. Reserve `ft` for a clean bass, `6s` for acoustic
multi-instrument mixes.

### Harmony is read on the stems, not on the mix

With `--separate`, separation now runs **before** key and chord detection, and
those read a reconstructed *harmonic mix* (`bass + piano + guitar + other`)
instead of the full track.

Drums are the most direct noise one can feed a chord detector: a cymbal spreads
energy across all twelve pitch classes and a chroma cannot tell it from a
cluster. Vocals are dropped too, since they carry the melody rather than the
harmony, and their vibrato smears the chroma around the intended note.

The bass is the decisive one. Two relative triads share two notes out of three;
only the root separates them, and the bass is what states it. Without it a
detector confuses a chord with its inversion and a major with its relative minor.

`harmony_source` in the JSON says which was used, `harmonic_stems` or `mix`: two
analyses of the same track are only comparable when it matches. Separation
failing degrades the analysis back to the mix, it never breaks it, and Demucs
still runs only once.

Output goes to `--out` if given, else `$SCORE7_OUT`, else `~/audio_analysis/`: a
`.md` sheet, optional `.json`, stems, `<slug>.harmonic.wav`,
`<slug>_poly_full.mid`, `<slug>_melody.mid`.

Tempo and meter are always estimated (no flag); when the tempo octave is ambiguous
(confidence < 0.5), the sheet and console list the candidates with their strength.
When one of those candidates is *stronger than the tempo actually kept*, the JSON
carries `tempo.bpm_disputed` with both figures. score7 does not silently correct
it: halving the bpm without decimating `beat_times` would break the grid, and
decimating assumes knowing which beat of two is the quarter note, which the
tempogram does not say. A trained tracker off by an octave still beats a
heuristic guessing. Downstream consumers get the disagreement stated instead of
buried in a list, and a doubled tempo means harmonic durations are expressed in
that subdivision, not in quarter notes.

### Melody: what gets filtered, and why

A pitch tracker returns a curve, not a melody. Turning one into the other takes a
handful of decisions, and every threshold below was set against a real track
rather than picked as a round number. They are stated here because two analyses
are only comparable when the filtering behind them is the same.

**The route is measured when the probe is there.** score7 first tries basic-pitch
(~5 s on the ONNX backend), which transcribes every note it hears; its mean
polyphony then decides what follows. At or below 1.2 the material is monophonic
and PESTO takes over, being more accurate on a single voice; above it, the
skyline of basic-pitch's own notes is used and split into two parts, `ligne` (the
highest note sounding at any instant) and `arpèges` (everything covered by a
higher one). The second part matters on interlocking material, where one voice
alone is not the piece. The split is geometric: score7 follows the dominant
voice, it does not separate melody from accompaniment in the musical sense.

**On a default install that probe never runs.** basic-pitch is in no extra (see
[Installation](#installation)), its import fails, the failure is caught and
reported on stderr, and the chain falls straight through to PESTO. The practical
consequences are worth stating, because they are invisible in the output: the
material is treated as monophonic whatever it actually is, and `voices`,
`polyphony` and the level floor described below never appear. Install basic-pitch
yourself, on a Python where it resolves, to get that route back.

**A high-pass at 180 Hz precedes f0 tracking.** On a synth stem the low pedal (a
tonic drone, residual bass) dominates salience and captures a monophonic tracker:
on synthwave material, 77% of the returned notes were the pedal. Cutting below
~180 Hz takes the drone out of the running.

**The duration floor follows the tempo**, at roughly a sixteenth (`0.22 * 60 /
bpm`), not a fixed number of seconds. A flat 0.1 s means one seventh of a beat at
88 BPM, which lets confetti through where a musical floor would not.

**The level floor sits 18 dB below the body of the track**, on the basic-pitch
route only. Below it nothing is transcribed, because a fade tail is reverberation
rather than played music. The figure is a compromise measured on *A Midsummer
Nice Dream*, whose fade crosses -20 dB around 140 s (where transcription starts
inventing) while its genuinely quiet passages live at -7 dB: low enough to spare
those, high enough to cut the tail. The console reports how many notes were
dropped.

**Cleanup runs in a fixed order, and the order is not interchangeable.** Same-pitch
notes separated by less than 80 ms are merged first, since salience flicker
shreds a held note into confetti rather than into music. Short notes (under
300 ms) more than ten semitones from *both* neighbours are dropped next, being
octave or voice flips rather than played notes. Only then are true repetitions
re-articulated at onsets, and only on the f0 path: skyline notes come from a
transcription that is already articulated. Velocities come last, read from the
RMS envelope on the f0 path and from basic-pitch's per-note amplitude on the
skyline one, normalised on the 10th/95th percentiles into 45-112, so a MIDI
export carries dynamics instead of the flat fallback.

### Key, meter, chords: the arbitrations

Where several signals disagree, score7 does not average them: it decides which
one is competent for which question, and says so in the JSON.

**Key.** The chord vote carries the same weight as the Krumhansl correlation,
because chroma alone cannot separate a major key from its relative minor (same
notes, different function) whereas the chords state the function. When a melody
is available it arbitrates the **mode** only: the third above the tonic is read
directly, so minor and major stop being a matter of correlation. It is not
allowed to move the **tonic**, because a melody is a poor witness there (on Fm
the fifth is often as frequent as the root, so C would win over F). The tonic is
overridden only when the harmonic one is nearly absent from the melody, under
10% of its weight, which means the harmonic analysis is plainly off. And when the
key comes from the madmom CNN, the melody does not touch it at all: an
authoritative detector is not corrected by a weaker signal. The JSON keeps
`krumhansl_only` and `corrected` so the correction can be audited rather than
trusted.

**Meter.** The binary/ternary decision is deliberately biased toward binary:
ternary wins only above 1.1 times the binary strength, and only if the
subdivision carries real energy (without that guard, two flavours of noise get
compared). A wrongly ternary reading changes how the whole piece is written down,
so the bias is toward the more common case.

**Chords.** The published grid is already smoothed, and not the same way on every
route. On the cosine fallback, segments shorter than two beats are dropped,
unless dropping them would empty the grid, in which case the raw segmentation is
kept. On the BTC and madmom routes nothing is dropped for being short; instead a
one-beat segment is absorbed into the surrounding chord only when both of its
neighbours carry the *same* chord — that is jitter from a frame-by-frame
detector, not a reading. A short chord between two *different* ones is left
alone, since it may be a real passage and erasing it would invent a simpler
harmony than the music. `chords_source` says which route produced the grid, and
comparing two analyses means comparing two grids filtered the same way.

**Structure.** `coarse` is eight windows of **equal duration**, not musical
sections: it reports how energy is distributed over time, nothing more. The
musical information sits in `layer_onsets`, where a layer entering is detected as
an RMS derivative above 2.5 standard deviations, ignoring the first second,
keeping onsets at least 3 s apart, capped at twelve.

**Drums.** Bands are fixed at 20-150 Hz (kick), 150-2000 Hz (snare) and above
6000 Hz (hats); the printed grid marks a step `X` above 0.5 and `x` above 0.2 of
the maximum. Swing is reported both raw (0.5 is straight, 0.667 is triplet) and
as a percentage between those two bounds.

Microtiming and swing are read at a finer resolution than the rest (hop 64,
window 1024, plus parabolic interpolation of the onset peak) because they
measure deviations of a few milliseconds. The pattern grid keeps hop 512, which
is plenty for placing a step. Measured against a perfectly quantised grid, whose
true microtiming is zero by construction, the reported figure is about 2.6 ms,
against 13.3 ms with the shared 512 hop: below that resolution one measures the
analysis frame, not the drummer.

The sound layer needs none of this: centroid, 85% rolloff, BS.1770 loudness and
crest factor are definitions, not arbitrations.

## MCP

```bash
python -m score7_mcp        # run the server (stdio)
```

Tools: `analyze_audio` (full analysis, `separate`/`melody`/`sep_model` options) and
`separate_stems` (also takes `sep_model`). Wired into flux7-mesh via:

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
| Melody | an optional basic-pitch probe (absent by default) routes polyphonic material to a two-voice skyline; otherwise monophonic pitch tracking on an isolated stem (~10 ms); else poly transcription to skyline (C4-C6); else pYIN. Follows the dominant voice, does not separate melody from accompaniment | (basic-pitch) > **PESTO** > skyline > pYIN |
| Separation | hybrid spectro-temporal source separation; model selectable via `--sep-model` / `sep_model` (htdemucs 4-stem default, htdemucs_ft fine-tuned, htdemucs_6s 6-stem +guitar/piano) | **Demucs** htdemucs |
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
| **basic-pitch** (optional polyphony probe) | R. M. Bittner, J. J. Bosch, D. Rubinstein, G. Meseguer-Brocal, S. Ewert (Spotify) | *A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation*, ICASSP 2022 |
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
