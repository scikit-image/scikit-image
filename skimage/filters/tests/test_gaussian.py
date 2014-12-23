import numpy as np
from skimage.filters._gaussian import gaussian_filter
from skimage._shared._warnings import expected_warnings


def test_null_sigma():
    a = np.zeros((3, 3))
    a[1, 1] = 1.
    assert np.all(gaussian_filter(a, 0) == a)


def test_energy_decrease():
    a = np.zeros((3, 3))
    a[1, 1] = 1.
    gaussian_a = gaussian_filter(a, sigma=1, mode='reflect')
    assert gaussian_a.std() < a.std()


def test_multichannel():
    a = np.zeros((5, 5, 3))
    a[1, 1] = np.arange(1, 4)
    gaussian_rgb_a = gaussian_filter(a, sigma=1, mode='reflect',
                                     multichannel=True)
    # Check that the mean value is conserved in each channel
    # (color channels are not mixed together)
    assert np.allclose([a[..., i].mean() for i in range(3)],
                        [gaussian_rgb_a[..., i].mean() for i in range(3)])
    # Test multichannel = None
    with expected_warnings(['multichannel']):
        gaussian_rgb_a = gaussian_filter(a, sigma=1, mode='reflect')
    # Check that the mean value is conserved in each channel
    # (color channels are not mixed together)
    assert np.allclose([a[..., i].mean() for i in range(3)],
                        [gaussian_rgb_a[..., i].mean() for i in range(3)])
    # Iterable sigma
    gaussian_rgb_a = gaussian_filter(a, sigma=[1, 2], mode='reflect',
                                     multichannel=True)
    assert np.allclose([a[..., i].mean() for i in range(3)],
                        [gaussian_rgb_a[..., i].mean() for i in range(3)])


if __name__ == "__main__":
    from numpy import testing
    testing.run_module_suite()
