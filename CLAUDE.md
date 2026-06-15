# CLAUDE.md — score7

## What it is

Harmonic and sonic analysis of audio files (mp3/flac/wav), exposed as both a CLI
and an MCP server. Estimation, not exact transcription: reliable on spectral /
dynamics / stereo / loudness / key, approximate on chord grid (suspended harmony)
and melody (clean up by ear).

Grew out of the local `audio-analyze` Claude skill, which is now a thin front-end
calling this package.

## Quick commands

```bash
pip install -e .                    # core (librosa) — analysis + MCP
pip install -e ".[melody]"          # + Demucs & transcription (GPU, opt-in)
pip install "madmom @ git+https://github.com/CPJKU/madmom.git"  # [rhythm] — PyPI 0.16.1 broken on py3.12
score7 "track.flac" --json          # CLI
python -m score7_mcp                # run MCP server (stdio)
pytest                              # 5 synthetic-signal tests
```

## Structure

```
score7_mcp/
├── core.py        # signal analyses (no heavy deps): key, chords, structure,
│                  #   spectral, stereo, loudness — and key reconciliation
├── melody.py      # heavy/GPU (extra [melody]): Demucs separate, transcribe, skyline
├── texture.py     # per-stem rhythm pattern + texture (librosa only, runs on
│                  #   separated stems): drum step-grid, microtiming/swing, stem timbre
├── chords_dl.py   # deep-learning chord chain: BTC > madmom > template; beat-aligned
├── _btc/          # vendored BTC inference subset (MIT, © 2019 Jonggwon Park)
├── analyze.py     # orchestrator → single results dict (shared by CLI + MCP)
├── render.py      # markdown fiche (Renoise-analyses format)
├── cli.py         # `score7` entry point
├── server.py      # FastMCP server, 2 tools
└── __main__.py    # `python -m score7_mcp` → MCP server (used by flux7-mesh)
```

## MCP Tools

| Tool | Input | Output |
|------|-------|--------|
| `analyze_audio` | file_path (+ separate/melody/write_fiche/write_json flags) | results dict + fiche path |
| `separate_stems` | file_path | stems directory (Demucs) |

## Analysis methods

| Stage | Technique |
|-------|-----------|
| Tempo | priority Beat This! (transformer, ISMIR 2024) > madmom (RNN+DBN) > librosa; BPM = median inter-beat interval; octave candidates (T/2, T, 2T) exposed with tempogram strength |
| Meter | Beat This! / madmom downbeats (beats_per_bar) if installed, else accent-pattern folding on beat grid; binary/ternary subdivision names 6/8 and 12/8 |
| Key | Krumhansl-Schmuckler (chroma CQT) **+ chord-function vote** **+ melody-tonic reconciliation** |
| Chords | priority BTC (Bi-directional Transformer, ISMIR 2019, large-vocab 170 classes: maj/min/7/sus/dim) > madmom deep-chroma+CRF > cosine template matching on beat-synced chroma CENS. Neural grids beat-aligned, with a rich `chord_full` label kept alongside the maj/min compact. Default on (`dl_chords`); weights auto-downloaded to ~/.cache/score7/. **Key vote stays on the cosine grid** (neural detectors label tierce-less power chords as major, which skews the mode vote) — neural grid is display-only. |
| Structure | per-window RMS + layer-onset detection (energy derivative) |
| Spectral | centroid, rolloff 85%, bandwidth, flatness |
| Stereo | L/R correlation, side/mid ratio |
| Loudness | integrated LUFS (pyloudnorm), crest factor |
| Separation | Demucs (htdemucs) |
| Melody | polyphonic transcription → band-limited skyline (C4–C6, salience = duration×velocity) |
| Rhythm pattern | drum stem split into 3 bands (kick/snare/hats), onset envelope folded onto the beat grid (step-sequence), microtiming + swing ratio from onset deviation. Band labels are a frequency heuristic, not classification. Needs `--separate`. |
| Stem texture | per stem: spectral profile + spectral flux + centroid CV (timbre movement), HPSS percussive ratio, RMS energy share, stereo width. Needs `--separate`. |

## Key detection — the important subtlety

Krumhansl alone confuses a key with its relative (same notes). The chord-function
vote fixes most cases but inherits the chord detector's major/minor third errors on
pads. The melody is the **reliable discriminator for the mode**, not the tonic: when
`--melody` runs, `core.reconcile_key_with_melody` keeps the harmonic tonic (Krumhansl
+ chord vote) and only decides minor-third vs major-third above it. It rewrites the
tonic toward the dominant melody pitch class **only** if the harmonic tonic is nearly
absent from the melody (< 10 %), i.e. the harmony is clearly off.

→ Why not "tonic = most frequent melody note": the fifth is often as frequent as the
root (on Fm the C dominates), so that rule mislabels the tonic. Validated on Perturbator
"Future Club": chords said F minor (correct per Tunebat/SongBPM), the old rule wrongly
flipped it to C minor; the current rule keeps F minor.

→ For trustworthy mode on pad-heavy material, run with `--melody`.

## flux7-mesh

Registered in `~/flux7-mesh/my-flow.local.yaml` as a stdio MCP server
(`python -m score7_mcp`). Reloading the mesh is required to pick it up
(briefly drops all MCP servers).

## Don't

- Don't add heavy deps (torch/demucs/transcription) to core — keep them in `[melody]`.
- Don't use `basic-pitch` — it pins numpy<1.24, incompatible with Python 3.12.
  Use `piano_transcription_inference` instead.
- Don't trust the auto chord major/minor labels on sustained pads — cross-check
  against the melody pitch-class histogram in the JSON.
- Don't run melody extraction on the full mix when stems exist — use the `other` stem.
