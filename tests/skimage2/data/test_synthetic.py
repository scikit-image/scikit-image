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

    @pytest.mark.parametrize("blob_size", [0.9, 0.4, 0.3, 0.1])
    @pytest.mark.parametrize("volume_fraction", [0.9, 0.1])
    def test_small_blob_size(self, blob_size, volume_fraction):
        regex = r"Requested `blob_size` .* is smaller than 1"
        with pytest.warns(RuntimeWarning, match=regex) as record:
            result = binary_blobs(
                shape=(100, 100),
                rng=3,
                blob_size=blob_size,
                volume_fraction=volume_fraction,
            )
        assert_stacklevel(record, offset=-6)
        if blob_size >= 0.4:
            assert not np.all(result == 1)  # Features still exist
        else:
            np.testing.assert_equal(result, 1)

    @pytest.mark.filterwarnings("ignore:Requested `blob_size` .* is smaller than 1")
    def test_blob_size_clamping(self):
        # A very small `blob_size` will allocate excessive memory
        # Check that this is gracefully handled
        regex = r"Clamping to `blob_size=0.1`"
        with pytest.warns(RuntimeWarning, match=regex) as record:
            result = binary_blobs(shape=(100, 100), rng=3, blob_size=0.09)
        assert_stacklevel(record)
        np.testing.assert_equal(result, 1)
