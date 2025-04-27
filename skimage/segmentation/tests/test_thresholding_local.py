import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose

import skimage as ski
from skimage.segmentation import (
    threshold_local,
    threshold_local_niblack,
    threshold_local_sauvola,
    threshold_labels_hysteresis,
)
from skimage._shared.utils import _supported_float_type


class Test_threshold_local:
    def setup_method(self):
        self.image = np.array(
            [
                [0, 0, 1, 3, 5],
                [0, 1, 4, 3, 4],
                [1, 2, 5, 4, 1],
                [2, 4, 5, 2, 1],
                [4, 5, 1, 0, 0],
            ],
            dtype=int,
        )

    @pytest.mark.parametrize('ndim', [2, 3])
    def test_gaussian(self, ndim):
        ref = np.array(
            [
                [False, False, False, False, True],
                [False, False, True, False, True],
                [False, False, True, True, False],
                [False, True, True, False, False],
                [True, True, False, False, False],
            ]
        )
        if ndim == 2:
            image = self.image
            block_sizes = [3, (3,) * image.ndim]
        else:
            image = np.stack((self.image,) * 5, axis=-1)
            ref = np.stack((ref,) * 5, axis=-1)
            block_sizes = [3, (3,) * image.ndim, (3,) * (image.ndim - 1) + (1,)]

        for block_size in block_sizes:
            out = threshold_local(
                image, block_size=block_size, method='gaussian', mode='reflect'
            )
            assert_equal(ref, image > out)

        out = threshold_local(
            image, block_size=3, method='gaussian', mode='reflect', param=1 / 3
        )
        assert_equal(ref, image > out)

    @pytest.mark.parametrize('ndim', [2, 3])
    def test_mean(self, ndim):
        ref = np.array(
            [
                [False, False, False, False, True],
                [False, False, True, False, True],
                [False, False, True, True, False],
                [False, True, True, False, False],
                [True, True, False, False, False],
            ]
        )
        if ndim == 2:
            image = self.image
            block_sizes = [3, (3,) * image.ndim]
        else:
            image = np.stack((self.image,) * 5, axis=-1)
            ref = np.stack((ref,) * 5, axis=-1)
            # Given the same data at each z location, the following block sizes
            # will all give an equivalent result.
            block_sizes = [3, (3,) * image.ndim, (3,) * (image.ndim - 1) + (1,)]
        for block_size in block_sizes:
            out = threshold_local(
                image, block_size=block_size, method='mean', mode='reflect'
            )
            assert_equal(ref, image > out)

    @pytest.mark.parametrize('block_size', [(3,), (3, 3, 3)])
    def test_invalid_block_size(self, block_size):
        # len(block_size) != image.ndim
        with pytest.raises(ValueError):
            threshold_local(self.image, block_size=block_size, method='mean')

    @pytest.mark.parametrize('ndim', [2, 3])
    def test_median(self, ndim):
        ref = np.array(
            [
                [False, False, False, False, True],
                [False, False, True, False, False],
                [False, False, True, False, False],
                [False, False, True, True, False],
                [False, True, False, False, False],
            ]
        )
        if ndim == 2:
            image = self.image
        else:
            image = np.stack((self.image,) * 5, axis=-1)
            ref = np.stack((ref,) * 5, axis=-1)
        out = threshold_local(image, block_size=3, method='median', mode='reflect')
        assert_equal(ref, image > out)

    def test_median_constant_mode(self):
        out = threshold_local(
            self.image, block_size=3, method='median', mode='constant', cval=20
        )
        expected = np.array(
            [
                [20.0, 1.0, 3.0, 4.0, 20.0],
                [1.0, 1.0, 3.0, 4.0, 4.0],
                [2.0, 2.0, 4.0, 4.0, 4.0],
                [4.0, 4.0, 4.0, 1.0, 2.0],
                [20.0, 5.0, 5.0, 2.0, 20.0],
            ]
        )
        assert_equal(expected, out)

    def test_even_block_size_error(self):
        img = ski.data.camera()
        with pytest.raises(ValueError):
            threshold_local(img, block_size=4)

    @pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float16, np.float32])
    def test_dtypes(self, dtype):
        r = 255 * np.random.rand(32, 16)
        r = r.astype(dtype, copy=False)

        # use double precision result as a reference
        expected = threshold_local(r.astype(float), block_size=9)

        out = threshold_local(r, block_size=9)
        assert out.dtype == _supported_float_type(dtype)
        assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
