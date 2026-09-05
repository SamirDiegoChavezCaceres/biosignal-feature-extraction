"""Walkthrough: band power per signal class, then classify.

    python scripts/demo.py
"""

from __future__ import annotations

import numpy as np

from biosignal import band_power, make_dataset, make_signal, train
from biosignal.signals import CLASS_FREQS

FS = 128.0


def rule(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    rule("1. Band power separates signals by their dominant frequency")
    rng = np.random.default_rng(0)
    header = "  class (freq)   " + "".join(f"{b:>8}" for b in ("delta", "theta", "alpha", "beta", "gamma"))
    print(header)
    for label, freq in CLASS_FREQS.items():
        sig = make_signal(freq, FS, 4.0, rng)
        bp = band_power(sig, FS)
        row = "".join(f"{bp[b]:>8.2f}" for b in ("delta", "theta", "alpha", "beta", "gamma"))
        dominant = max(bp, key=bp.get)
        print(f"  {label} ({freq:>4.0f} Hz)  {row}   <- {dominant}")

    rule("2. Classify the three spectral classes")
    signals, labels = make_dataset(n_per_class=100)
    _, metrics = train(signals, labels)
    print(f"  features per signal: {metrics['n_features']} (5 FFT bands + 5 wavelet levels)")
    print(f"  accuracy: {metrics['accuracy']}")


if __name__ == "__main__":
    main()
