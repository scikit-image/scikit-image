import numpy as np
from numpy.testing import assert_array_almost_equal as assert_close

from skimage.feature import match_template, peak_local_max


def test_template():
    size = 100
    # Type conversion of image and target not required but prevents warnings.
    image = np.zeros((400, 400), dtype=np.float32)
    target = np.tri(size) + np.tri(size)[::-1]
    target = target.astype(np.float32)
    target_positions = [(50, 50), (200, 200)]
    for x, y in target_positions:
        image[x:x + size, y:y + size] = target
    np.random.seed(1)
    image += np.random.randn(400, 400) * 2

    result = match_template(image, target)
    delta = 5

    positions = peak_local_max(result, min_distance=delta)

    if len(positions) > 2:
        # Keep the two maximum peaks.
        intensities = result[tuple(positions.T)]
        i_maxsort = np.argsort(intensities)[::-1]
        positions = positions[i_maxsort][:2]

    # Sort so that order matches `target_positions`.
    positions = positions[np.argsort(positions[:, 0])]

    for xy_target, xy in zip(target_positions, positions):
        yield assert_close, xy, xy_target


def test_normalization():
    """Test that `match_template` gives the correct normalization.

    Normalization gives 1 for a perfect match and -1 for an inverted-match.
    This test adds positive and negative squares to a zero-array and matches
    the array with a positive template.
    """
    n = 5
    N = 20
    ipos, jpos = (2, 3)
    ineg, jneg = (12, 11)
    image = np.zeros((N, N))
    image[ipos:ipos + n, jpos:jpos + n] = 10
    image[ineg:ineg + n, jneg:jneg + n] = -10

    # white square with a black border
    template = np.zeros((n+2, n+2))
    template[1:1+n, 1:1+n] = 1

    result = match_template(image, template)

    # get the max and min results.
    sorted_result = np.argsort(result.flat)
    iflat_min = sorted_result[0]
    iflat_max = sorted_result[-1]
    min_result = np.unravel_index(iflat_min, result.shape)
    max_result = np.unravel_index(iflat_max, result.shape)

    # shift result by 1 because of template border
    assert np.all((np.array(min_result) + 1) == (ineg, jneg))
    assert np.all((np.array(max_result) + 1) == (ipos, jpos))

    assert np.allclose(result.flat[iflat_min], -1)
    assert np.allclose(result.flat[iflat_max], 1)


def test_no_nans():
    """Test that `match_template` doesn't return NaN values.

    When image values are only slightly different, floating-point errors can
    cause a subtraction inside of a square root to go negative (without an
    explicit check that was added to `match_template`).
    """
    np.random.seed(1)
    image = 10000 + np.random.normal(size=(20, 20))
    template = np.ones((6, 6))
    template[:3, :] = 0
    result = match_template(image, template)
    assert not np.any(np.isnan(result))


def test_switched_arguments():
    image = np.ones((5, 5))
    template = np.ones((3, 3))
    np.testing.assert_raises(ValueError, match_template, template, image)


if __name__ == "__main__":
    from numpy import testing
    testing.run_module_suite()

