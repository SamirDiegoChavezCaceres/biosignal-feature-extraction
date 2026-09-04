"""Feature extraction from 1-D signals.

Two classic families, useful for any biosignal or sensor stream:

* **Spectral band power** (FFT): how much of the signal's energy sits in each
  frequency band. Pure numpy, always available.
* **Wavelet energy** (discrete wavelet transform): energy per decomposition
  level, which captures transient, non-stationary structure the FFT smears out.
  Needs PyWavelets (the ``wavelet`` extra).

The default frequency bands below are the usual EEG bands, but nothing here is
tied to a particular signal; pass your own ``bands`` for ECG, vibration, audio,
etc.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

# name -> (low_hz, high_hz)
DEFAULT_BANDS: Dict[str, tuple] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}


def band_power(signal: np.ndarray, fs: float, bands: Dict[str, tuple] = None) -> Dict[str, float]:
    """Relative power of ``signal`` in each frequency band (sums to ~1)."""
    bands = bands or DEFAULT_BANDS
    signal = np.asarray(signal, dtype=float)
    signal = signal - signal.mean()
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(signal)) ** 2
    total = psd.sum() or 1.0
    out = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        out[name] = float(psd[mask].sum() / total)
    return out


def wavelet_energy(signal: np.ndarray, wavelet: str = "db4", level: int = 4) -> Dict[str, float]:
    """Relative energy of ``signal`` at each wavelet decomposition level."""
    import pywt  # lazy: only needed for this feature family

    signal = np.asarray(signal, dtype=float)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    energies = [float(np.sum(np.square(c))) for c in coeffs]
    total = sum(energies) or 1.0
    # coeffs[0] = approximation, then detail levels from coarse to fine.
    names = ["approx"] + [f"detail_{i}" for i in range(len(coeffs) - 1, 0, -1)]
    return {name: e / total for name, e in zip(names, energies)}


def feature_vector(signal: np.ndarray, fs: float, use_wavelet: bool = True) -> Dict[str, float]:
    feats = {f"bp_{k}": v for k, v in band_power(signal, fs).items()}
    if use_wavelet:
        feats.update({f"we_{k}": v for k, v in wavelet_energy(signal).items()})
    return feats


def feature_names(use_wavelet: bool = True) -> List[str]:
    names = [f"bp_{b}" for b in DEFAULT_BANDS]
    if use_wavelet:
        names += ["we_approx"] + [f"we_detail_{i}" for i in range(4, 0, -1)]
    return names
