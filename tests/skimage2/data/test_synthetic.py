import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from skimage2.data import binary_blobs
from skimage._shared.testing import assert_stacklevel


class Test_binary_blobs:
    def test_volume_fraction(self):
        blobs = binary_blobs(shape=(128, 128))
        assert_almost_equal(blobs.mean(), 0.5, decimal=1)

        blobs = binary_blobs(shape=(128, 128), volume_fraction=0.25)
        assert_almost_equal(blobs.mean(), 0.25, decimal=1)

        blobs = binary_blobs(shape=(32, 32, 32), volume_fraction=0.25)
        assert_almost_equal(blobs.mean(), 0.25, decimal=1)

        other_realization = binary_blobs(shape=(32, 32, 32), volume_fraction=0.25)
        assert not np.all(blobs == other_realization)

    def test_boundary(self):
        # Assert that `boundary_mode="wrap"` decreases the pixel difference on
        # opposing borders compared to `boundary_mode="nearest"`
        blobs_near = binary_blobs(shape=(300, 300), boundary_mode="nearest", rng=3)
        blobs_wrap = binary_blobs(shape=(300, 300), boundary_mode="wrap", rng=3)

        diff_near_ax0 = blobs_near[0, :] ^ blobs_near[-1, :]
        diff_wrap_ax0 = blobs_wrap[0, :] ^ blobs_wrap[-1, :]
        assert diff_wrap_ax0.sum() < diff_near_ax0.sum()

        diff_near_ax1 = blobs_near[:, 0] ^ blobs_near[:, -1]
        diff_wrap_ax1 = blobs_wrap[:, 0] ^ blobs_wrap[:, -1]
        assert diff_wrap_ax1.sum() < diff_near_ax1.sum()

    def test_small_blob_size(self):
        # A very small `blob_size_fraction` in relation to `length` will allocate
        # excessive memory and likely leads to unexpected results. Check that this
        # is gracefully handled
        regex = ".* Clamping to .* blob size of 0.1 pixels"
        with pytest.warns(RuntimeWarning, match=regex) as record:
            result = binary_blobs(shape=(100, 100), rng=3, blob_size_fraction=0.0009)
        assert_stacklevel(record)
        np.testing.assert_equal(result, 1)
