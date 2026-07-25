"""Tests de l'orchestrateur : erreurs propres, échecs partiels visibles."""

import sys
import types

import numpy as np
import pytest
import soundfile as sf

from score7_mcp.analyze import _resolve_outdir, analyze_file


def _write_tone(path, seconds=2.0, sr=22050):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, y, sr)


def test_resolve_outdir_precedence(monkeypatch):
    """Priorité : argument explicite > $SCORE7_OUT > défaut ~/audio_analysis."""
    from pathlib import Path

    monkeypatch.setenv("SCORE7_OUT", "/tmp/score7_env")
    assert _resolve_outdir("/explicit/dir") == "/explicit/dir"   # explicite gagne
    assert _resolve_outdir(None) == "/tmp/score7_env"            # sinon env
    monkeypatch.delenv("SCORE7_OUT", raising=False)
    assert _resolve_outdir(None) == str(Path.home() / "audio_analysis")  # sinon défaut neutre


def test_corrupted_file_raises_clean_valueerror(tmp_path):
    bad = tmp_path / "not_audio.wav"
    bad.write_bytes(b"definitely not a RIFF header")
    with pytest.raises(ValueError, match="Lecture audio impossible"):
        analyze_file(str(bad), outdir=str(tmp_path))


def test_separation_failure_surfaces_as_stems_error(tmp_path, monkeypatch):
    wav = tmp_path / "tone.wav"
    _write_tone(wav)

    # stub du module melody : l'import réel tire torch/demucs (extra [melody]).
    # On patche sys.modules ET l'attribut du package : `from score7_mcp import
    # melody` lit l'attribut dès qu'il existe, donc sys.modules seul ne suffit
    # plus une fois qu'un autre test a importé le vrai module.
    import score7_mcp
    stub = types.ModuleType("score7_mcp.melody")
    stub.separate = lambda path, outdir, model="htdemucs": (_ for _ in ()).throw(RuntimeError("demucs absent"))
    monkeypatch.setitem(sys.modules, "score7_mcp.melody", stub)
    monkeypatch.setattr(score7_mcp, "melody", stub, raising=False)

    r = analyze_file(str(wav), separate=True, outdir=str(tmp_path))
    assert r["stems_error"] == "demucs absent"
    assert "stems" not in r
    assert r["key"]["root"]  # l'analyse cœur a quand même abouti


# --------------------------------------------------------------------------- #
#  L'harmonie se lit sur les stems, pas sur le mix
# --------------------------------------------------------------------------- #
def _fake_stems(dirpath, names, sr=22050, seconds=2.0):
    """Écrit des stems jouables dans `dirpath`."""
    from pathlib import Path
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    for i, n in enumerate(names):
        sf.write(str(d / f"{n}.wav"), 0.2 * np.sin(2 * np.pi * (110 * (i + 1)) * t), sr)
    return str(d)


def test_chords_are_read_from_the_harmonic_mix(tmp_path, monkeypatch):
    """Le changement de fond : ce n'est plus le mix d'origine qui part au
    détecteur d'accords, mais la somme des stems harmoniques."""
    src = tmp_path / "song.wav"
    _write_tone(src)

    stems = _fake_stems(tmp_path / "stems", ["bass", "other", "drums", "vocals"])
    monkeypatch.setattr("score7_mcp.melody.separate", lambda p, o, model="htdemucs": stems)

    vu = {}
    def fake_chain(path, beat_times, fallback):
        vu["path"] = path
        return [], "btc"
    monkeypatch.setattr("score7_mcp.chords_dl.estimate_chords_chain", fake_chain)
    monkeypatch.setattr("score7_mcp.core.try_madmom_key", lambda p: None)

    res = analyze_file(str(src), outdir=str(tmp_path / "out"), separate=True, melody=False)

    assert vu["path"] != str(src)              # pas le mix d'origine
    assert vu["path"].endswith(".harmonic.wav")
    assert res["harmony_source"] == "harmonic_stems"


def test_falls_back_to_the_mix_when_separation_fails(tmp_path, monkeypatch):
    """Un mix harmonique absent doit dégrader l'analyse, jamais la casser."""
    src = tmp_path / "song.wav"
    _write_tone(src)

    def boom(p, o, model="htdemucs"):
        raise RuntimeError("demucs absent")
    monkeypatch.setattr("score7_mcp.melody.separate", boom)

    vu = {}
    def fake_chain(path, beat_times, fallback):
        vu["path"] = path
        return [], "btc"
    monkeypatch.setattr("score7_mcp.chords_dl.estimate_chords_chain", fake_chain)
    monkeypatch.setattr("score7_mcp.core.try_madmom_key", lambda p: None)

    res = analyze_file(str(src), outdir=str(tmp_path / "out"), separate=True, melody=False)

    assert vu["path"] == str(src)              # repli sur le mix
    assert res["harmony_source"] == "mix"
    assert "demucs absent" in res.get("stems_error", "")


def test_separation_runs_once(tmp_path, monkeypatch):
    """Demucs coûte des minutes : le remonter avant les accords ne doit pas
    conduire à le lancer deux fois."""
    src = tmp_path / "song.wav"
    _write_tone(src)
    stems = _fake_stems(tmp_path / "stems", ["bass", "other"])

    appels = []
    def once(p, o, model="htdemucs"):
        appels.append(model)
        return stems
    monkeypatch.setattr("score7_mcp.melody.separate", once)
    monkeypatch.setattr("score7_mcp.chords_dl.estimate_chords_chain",
                        lambda p, b, f: ([], "btc"))
    monkeypatch.setattr("score7_mcp.core.try_madmom_key", lambda p: None)

    res = analyze_file(str(src), outdir=str(tmp_path / "out"), separate=True, melody=False)
    assert len(appels) == 1
    assert res["stems"] == stems


def test_no_separation_keeps_the_mix(tmp_path, monkeypatch):
    src = tmp_path / "song.wav"
    _write_tone(src)
    vu = {}
    monkeypatch.setattr("score7_mcp.chords_dl.estimate_chords_chain",
                        lambda p, b, f: (vu.setdefault("path", p), [])[1] or ([], "btc"))
    monkeypatch.setattr("score7_mcp.core.try_madmom_key", lambda p: None)

    res = analyze_file(str(src), outdir=str(tmp_path / "out"), separate=False, melody=False)
    assert res["harmony_source"] == "mix"
