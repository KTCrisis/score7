"""Tests du spectre d'instrument sur signaux synthétiques."""

import numpy as np
import pytest
import soundfile as sf

from score7_mcp import spectrum

SR = 48000


def _wav(tmp_path, name, left, right=None):
    data = left if right is None else np.stack([left, right], axis=1)
    path = tmp_path / name
    sf.write(path, data, SR)
    return str(path)


def _sine(freq, dur=2.0, amp=0.5):
    t = np.arange(int(SR * dur)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def test_sine_lands_in_its_band(tmp_path):
    """Un 1 kHz pur met presque toute l'énergie dans le médium, centroïde vers 1 kHz."""
    m = spectrum.measure(_wav(tmp_path, "s.wav", _sine(1000.0)))
    assert m["bands_pct"]["medium"] > 99
    assert abs(m["centroid_hz"] - 1000) < 30
    assert sum(m["bands_pct"].values()) == pytest.approx(100, abs=0.5)


def test_two_tones_split_energy(tmp_path):
    """50 Hz et 4 kHz à amplitude égale : moitié sub, moitié présence."""
    m = spectrum.measure(_wav(tmp_path, "t.wav", _sine(50.0) + _sine(4000.0)))
    assert m["bands_pct"]["sub"] == pytest.approx(50, abs=3)
    assert m["bands_pct"]["presence"] == pytest.approx(50, abs=3)


def test_levels_and_stereo(tmp_path):
    """Sinus 0.5 : crête -6 dBFS, RMS -9 dBFS ; canaux opposés = corrélation -1."""
    s = _sine(440.0)
    m = spectrum.measure(_wav(tmp_path, "st.wav", s, -s))
    assert m["peak_dbfs"] == pytest.approx(-6.0, abs=0.1)
    assert m["stereo_corr"] == pytest.approx(-1.0, abs=0.01)
    mono = spectrum.measure(_wav(tmp_path, "m.wav", s))
    assert mono["rms_dbfs"] == pytest.approx(-9.0, abs=0.1)
    assert mono["stereo_corr"] == 1.0


def test_silence_is_refused(tmp_path):
    with pytest.raises(ValueError):
        spectrum.measure(_wav(tmp_path, "z.wav", np.zeros(SR)))


def test_plot_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    a = _wav(tmp_path, "a.wav", _sine(300.0))
    b = _wav(tmp_path, "b.wav", _sine(3000.0))
    out = spectrum.plot([a, b], str(tmp_path / "cmp.png"))
    assert (tmp_path / "cmp.png").stat().st_size > 10_000
    assert out.endswith("cmp.png")
