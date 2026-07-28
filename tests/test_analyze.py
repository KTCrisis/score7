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


def _write_wideband(path, sr=48000, seconds=2.0):
    """Un grave dominant, plus une composante à 15 kHz — au-dessus du Nyquist
    de 22 050 Hz, donc absente dès que le signal est ramené à la bande
    d'analyse. C'est ce qui rend le test capable de distinguer les deux."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = np.sin(2 * np.pi * 200 * t) + 0.9 * np.sin(2 * np.pi * 15000 * t)
    sf.write(path, (y / np.max(np.abs(y))).astype(np.float32), sr)


def test_spectral_profile_reads_the_whole_band(tmp_path):
    """Le profil spectral doit voir le signal natif, pas le mono d'analyse.

    Nourri à 22 050 Hz, Nyquist coupait à 11 kHz : la composante à 15 kHz
    disparaissait et le centroïde retombait vers le fondamental, un biais
    systématique vers le sombre. Le test échoue sur l'ancien code deux fois —
    `sr_hz` n'existait pas, et le centroïde restait proche de 200 Hz.
    """
    wav = tmp_path / "wideband.wav"
    _write_wideband(wav)
    res = analyze_file(str(wav), outdir=str(tmp_path / "out"),
                       separate=False, melody=False)
    spectral = res["spectral"]

    # La mesure d'abord : c'est elle que le correctif répare. Si `sr_hz` était
    # vérifié en premier, un ajout du champ sans réparation de la bande ferait
    # passer le test pour la mauvaise raison.
    # Le seuil est loin des deux valeurs en jeu (~200 Hz amputé, plusieurs
    # kHz entier) : il tranche sans dépendre de la fenêtre d'analyse.
    assert spectral["centroid_hz"] > 2000
    # Le taux natif est rendu : deux analyses ne se comparent qu'à bande égale.
    assert spectral["sr_hz"] == 48000


def test_melody_src_accepts_a_stem_name(tmp_path):
    """`--melody-src vocals` désigne un stem, pas un fichier du dossier courant :
    le chemin du dossier de stems n'est connu qu'après la séparation, donc
    personne ne peut le donner au moment de régler l'analyse."""
    from score7_mcp.analyze import _resolve_melody_src

    stems = tmp_path / "htdemucs" / "track"
    stems.mkdir(parents=True)
    for name in ("vocals", "other", "bass", "drums"):
        (stems / f"{name}.wav").write_bytes(b"")

    assert _resolve_melody_src("vocals", str(stems), "mix.wav") == str(stems / "vocals.wav")
    assert _resolve_melody_src("vocals.wav", str(stems), "mix.wav") == str(stems / "vocals.wav")
    # sans consigne : other d'abord, la voix seulement s'il manque
    assert _resolve_melody_src(None, str(stems), "mix.wav") == str(stems / "other.wav")
    (stems / "other.wav").unlink()
    assert _resolve_melody_src(None, str(stems), "mix.wav") == str(stems / "vocals.wav")


def test_melody_src_names_what_exists_instead_of_failing_late(tmp_path):
    """Un stem inexistant échoue tout de suite, en disant lesquels existent —
    plutôt qu'une erreur de lecture audio à l'autre bout de la chaîne."""
    import pytest

    from score7_mcp.analyze import _resolve_melody_src

    stems = tmp_path / "stems"
    stems.mkdir()
    (stems / "other.wav").write_bytes(b"")

    with pytest.raises(ValueError) as e:
        _resolve_melody_src("piano", str(stems), "mix.wav")
    assert "piano" in str(e.value) and "other" in str(e.value)

    # un vrai fichier garde la priorité sur l'interprétation « nom de stem »
    real = tmp_path / "piano.wav"
    real.write_bytes(b"")
    assert _resolve_melody_src(str(real), str(stems), "mix.wav") == str(real)

    # pas de séparation : sans stems, le mix reste la source
    assert _resolve_melody_src(None, None, "mix.wav") == "mix.wav"
