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
| Tempo | madmom RNN+DBN beats if installed, else librosa; BPM = median inter-beat interval; octave candidates (T/2, T, 2T) exposed with tempogram strength |
| Meter | madmom DBN downbeats (beats_per_bar 2–7) if installed, else accent-pattern folding on beat grid; binary/ternary subdivision names 6/8 and 12/8 |
| Key | Krumhansl-Schmuckler (chroma CQT) **+ chord-function vote** **+ melody-tonic reconciliation** |
| Chords | cosine template matching on beat-synced chroma CENS, merged segments |
| Structure | per-window RMS + layer-onset detection (energy derivative) |
| Spectral | centroid, rolloff 85%, bandwidth, flatness |
| Stereo | L/R correlation, side/mid ratio |
| Loudness | integrated LUFS (pyloudnorm), crest factor |
| Separation | Demucs (htdemucs) |
| Melody | polyphonic transcription → band-limited skyline (C4–C6, salience = duration×velocity) |

## Key detection — the important subtlety

Krumhansl alone confuses a key with its relative (same notes). The chord-function
vote fixes most cases but inherits the chord detector's major/minor third errors on
pads. The **reliable** discriminator is the melody tonic: when `--melody` runs,
`core.reconcile_key_with_melody` sets the tonic to the most frequent melody pitch
class and the mode from minor-third vs major-third presence above it.

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
