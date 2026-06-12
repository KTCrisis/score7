"""Tests de l'orchestrateur : erreurs propres, échecs partiels visibles."""

import sys
import types

import numpy as np
import pytest
import soundfile as sf

from score7_mcp.analyze import analyze_file


def _write_tone(path, seconds=2.0, sr=22050):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, y, sr)


def test_corrupted_file_raises_clean_valueerror(tmp_path):
    bad = tmp_path / "not_audio.wav"
    bad.write_bytes(b"definitely not a RIFF header")
    with pytest.raises(ValueError, match="Lecture audio impossible"):
        analyze_file(str(bad), outdir=str(tmp_path))


def test_separation_failure_surfaces_as_stems_error(tmp_path, monkeypatch):
    wav = tmp_path / "tone.wav"
    _write_tone(wav)

    # stub du module melody : l'import réel tire torch/demucs (extra [melody])
    stub = types.ModuleType("score7_mcp.melody")
    stub.separate = lambda path, outdir: (_ for _ in ()).throw(RuntimeError("demucs absent"))
    monkeypatch.setitem(sys.modules, "score7_mcp.melody", stub)

    r = analyze_file(str(wav), separate=True, outdir=str(tmp_path))
    assert r["stems_error"] == "demucs absent"
    assert "stems" not in r
    assert r["key"]["root"]  # l'analyse cœur a quand même abouti
