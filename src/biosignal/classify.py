"""Turn signals into feature vectors and train a classifier on them."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .features import feature_names, feature_vector


def build_matrix(signals: np.ndarray, fs: float, use_wavelet: bool = True) -> np.ndarray:
    names = feature_names(use_wavelet)
    rows = []
    for sig in signals:
        feats = feature_vector(sig, fs, use_wavelet=use_wavelet)
        rows.append([feats[n] for n in names])
    return np.array(rows)


def train(
    signals: np.ndarray, labels: np.ndarray, fs: float = 128.0, use_wavelet: bool = True
) -> Tuple[RandomForestClassifier, dict]:
    X = build_matrix(signals, fs, use_wavelet=use_wavelet)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, labels, test_size=0.3, random_state=0, stratify=labels
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    return clf, {"accuracy": round(float(acc), 4), "n_features": X.shape[1]}
