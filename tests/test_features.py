import numpy as np

from biosignal import band_power, feature_vector, wavelet_energy
from biosignal.features import feature_names


def _sine(freq, fs=128.0, duration=4.0):
    t = np.arange(int(fs * duration)) / fs
    return np.sin(2 * np.pi * freq * t)


def test_band_power_finds_the_right_band():
    bp = band_power(_sine(10.0), fs=128.0)      # 10 Hz -> alpha (8-13)
    assert max(bp, key=bp.get) == "alpha"
    assert abs(sum(bp.values()) - 1.0) < 0.05   # relative powers sum to ~1


def test_band_power_shifts_with_frequency():
    assert max(band_power(_sine(6.0), fs=128.0), key=lambda k: band_power(_sine(6.0), 128.0)[k]) == "theta"
    assert max(band_power(_sine(20.0), fs=128.0), key=lambda k: band_power(_sine(20.0), 128.0)[k]) == "beta"


def test_wavelet_energy_normalizes():
    we = wavelet_energy(_sine(10.0))
    assert abs(sum(we.values()) - 1.0) < 1e-6


def test_feature_vector_matches_declared_names():
    feats = feature_vector(_sine(10.0), fs=128.0, use_wavelet=True)
    assert set(feats) == set(feature_names(use_wavelet=True))
