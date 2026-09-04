"""Spectral and wavelet feature extraction from 1-D signals, with a classifier."""

from .classify import build_matrix, train
from .features import DEFAULT_BANDS, band_power, feature_names, feature_vector, wavelet_energy
from .signals import make_dataset, make_signal

__all__ = [
    "band_power",
    "wavelet_energy",
    "feature_vector",
    "feature_names",
    "DEFAULT_BANDS",
    "make_signal",
    "make_dataset",
    "build_matrix",
    "train",
]
