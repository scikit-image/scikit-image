import numpy as np

from _wavelet import wavelet_filter


def test_null_sigma():
    a = np.zeros((4, 4))
    assert np.all(wavelet_filter(a, 1) == a)


def test_energy_decrease():
    a = np.random.randn(10,10)
    wavelet_a = wavelet_filter(a, 100)
    assert wavelet_a.std() < a.std()


if __name__ == "__main__":
    from numpy import testing
    testing.run_module_suite()
