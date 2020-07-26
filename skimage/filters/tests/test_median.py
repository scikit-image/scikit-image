import pytest

import numpy as np
from scipy import ndimage

from skimage.filters import median
from skimage.filters import rank
from skimage._shared.testing import assert_allclose


@pytest.fixture
def image():
    return np.array([[1, 2, 3, 2, 1],
                     [1, 1, 2, 2, 3],
                     [3, 2, 1, 2, 1],
                     [3, 2, 1, 1, 1],
                     [1, 2, 1, 2, 3]],
                    dtype=np.uint8)


@pytest.mark.parametrize(
    "mode, cval, behavior, n_warning, warning_type",
    [('nearest', 0.0, 'ndimage', 0, []),
     ('constant', 0.0, 'rank', 1, (UserWarning,)),
     ('nearest', 0.0, 'rank', 0, []),
     ('nearest', 0.0, 'ndimage', 0, [])]
)
def test_median_warning(image, mode, cval, behavior,
                        n_warning, warning_type):

    with pytest.warns(None) as records:
        median(image, mode=mode, behavior=behavior)

    assert len(records) == n_warning
    for rec in records:
        assert isinstance(rec.message, warning_type)


@pytest.mark.parametrize(
    "behavior, func, params",
    [('ndimage', ndimage.median_filter, {'size': (3, 3)}),
     ('rank', rank.median, {'selem': np.ones((3, 3), dtype=np.uint8)})]
)
def test_median_behavior(image, behavior, func, params):
    assert_allclose(median(image, behavior=behavior), func(image, **params))


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint16, np.float32, np.float64]
)
def test_median_preserve_dtype(image, dtype):
    median_image = median(image.astype(dtype), behavior='ndimage')
    assert median_image.dtype == dtype


def test_median_error_ndim():
    img = np.random.randint(0, 10, size=(5, 5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        median(img, behavior='rank')


@pytest.mark.parametrize(
    "img, behavior",
    [(np.random.randint(0, 10, size=(3, 3), dtype=np.uint8), 'rank'),
     (np.random.randint(0, 10, size=(3, 3), dtype=np.uint8), 'ndimage'),
     (np.random.randint(0, 10, size=(3, 3, 3), dtype=np.uint8), 'ndimage')]
)
def test_median(img, behavior):
    median(img, behavior=behavior)
