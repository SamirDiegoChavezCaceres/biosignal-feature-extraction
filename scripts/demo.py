"""Extract features from synthetic signals and classify them.

    python scripts/demo.py
"""

from __future__ import annotations

from biosignal import band_power, make_dataset, make_signal, train
import numpy as np


def main() -> None:
    rng = np.random.default_rng(0)
    sig = make_signal(freq=10.0, fs=128.0, duration=4.0, rng=rng)
    bp = band_power(sig, fs=128.0)
    print("band power of a 10 Hz signal:")
    for name, value in bp.items():
        print(f"  {name:<6} {value:.3f}")
    print("  -> dominant band:", max(bp, key=bp.get))

    signals, labels = make_dataset(n_per_class=100)
    _, metrics = train(signals, labels)
    print("\nclassification of three spectral classes:", metrics)


if __name__ == "__main__":
    main()
