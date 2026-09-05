# biosignal-feature-extraction

Turn a raw 1-D signal into features a model can use: **spectral band power**
(FFT) and **wavelet energy** (DWT). A compact, well-tested take on the classic
first step of any biosignal or sensor pipeline.

Everything runs on synthetic signals, so there is nothing to download.

## What it shows

- **Spectral band power** - the share of a signal's energy in each frequency
  band, via the FFT. Pure numpy. The default bands are the usual EEG bands, but
  you pass your own for ECG, vibration, audio, etc.
- **Wavelet energy** - energy per DWT level (PyWavelets), which captures
  transient, non-stationary structure that a plain FFT smears across time.
- **End to end** - signals become a feature matrix and a classifier that tells
  three spectral classes apart with >90% accuracy, so you can see the features
  actually carry signal.

## Run it

```bash
pip install -e .
python scripts/demo.py
# band power of a 10 Hz signal: ... -> dominant band: alpha
# classification of three spectral classes: {'accuracy': 0.9x, 'n_features': 10}
```

```python
from biosignal import band_power, wavelet_energy
band_power(signal, fs=128)      # {'delta':.., 'theta':.., 'alpha':.., 'beta':.., 'gamma':..}
wavelet_energy(signal, "db4")   # energy per decomposition level
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Covers that band power lands in the correct band for known frequencies, that
wavelet energies normalize, and that the extracted features separate the classes
(with and without the wavelet family).

## License

MIT.
