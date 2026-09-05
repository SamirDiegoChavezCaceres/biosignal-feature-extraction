from biosignal import make_dataset, train


def test_spectral_classes_are_separable():
    signals, labels = make_dataset(n_per_class=80, seed=1)
    _, metrics = train(signals, labels)
    assert metrics["accuracy"] > 0.9


def test_works_without_wavelet_features():
    signals, labels = make_dataset(n_per_class=80, seed=2)
    _, metrics = train(signals, labels, use_wavelet=False)
    assert metrics["n_features"] == 5   # five FFT bands only
    assert metrics["accuracy"] > 0.9
