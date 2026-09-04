"""Synthetic 1-D signals for demos and tests.

Three classes, each dominated by oscillations in a different frequency band on
top of broadband noise. No recorded or third-party data is used; the point is to
exercise the feature extraction and show that band/wavelet features separate
signals by their spectral content.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# class label -> dominant frequency (Hz)
CLASS_FREQS = {0: 6.0, 1: 10.0, 2: 20.0}  # theta-ish, alpha-ish, beta-ish


def make_signal(freq: float, fs: float, duration: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(fs * duration)) / fs
    amp = rng.uniform(0.8, 1.2)
    phase = rng.uniform(0, 2 * np.pi)
    signal = amp * np.sin(2 * np.pi * freq * t + phase)
    signal += 0.3 * np.sin(2 * np.pi * (2 * freq) * t)  # a harmonic
    signal += rng.normal(0, 0.5, t.size)                # broadband noise
    return signal


def make_dataset(
    n_per_class: int = 100, fs: float = 128.0, duration: float = 4.0, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    signals, labels = [], []
    for label, freq in CLASS_FREQS.items():
        for _ in range(n_per_class):
            signals.append(make_signal(freq, fs, duration, rng))
            labels.append(label)
    return np.array(signals), np.array(labels)
