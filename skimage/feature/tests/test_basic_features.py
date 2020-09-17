import pytest
import numpy as np
from skimage.feature import multiscale_basic_features


@pytest.mark.parametrize('edges', (False, True))
@pytest.mark.parametrize('texture', (False, True))
def test_multiscale_basic_features(edges, texture):
    img = np.zeros((20, 20, 3))
    img[:10] = 1
    img += 0.05 * np.random.randn(*img.shape)
    features = multiscale_basic_features(img, edges=edges, texture=texture, multichannel=True)
    n_sigmas = 6
    intensity = True
    assert features.shape[-1] == 3 * n_sigmas * (int(intensity) + int(edges) + 2 * int(texture))
    assert features.shape[:-1] == img.shape[:-1]


def test_multiscale_basic_features_channel():
    img = np.zeros((10, 10, 5))
    img[:10] = 1
    img += 0.05 * np.random.randn(*img.shape)
    n_sigmas = 2
    features = multiscale_basic_features(img, sigma_min=1, sigma_max=2, multichannel=True)
    assert features.shape[-1] == 5 * n_sigmas * 4
    assert features.shape[:-1] == img.shape[:-1]
    # Consider last axis as spatial dimension
    features = multiscale_basic_features(img, sigma_min=1, sigma_max=2)
    assert features.shape[-1] == n_sigmas * 5
    assert features.shape[:-1] == img.shape
