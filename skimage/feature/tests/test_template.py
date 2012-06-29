import numpy as np
from numpy.testing import assert_array_almost_equal as assert_close

from skimage.morphology import diamond
from skimage.feature import match_template, peak_local_max


def test_template():
    size = 100
    # Float prefactors ensure that image range is between 0 and 1
    image = 0.5 * np.ones((400, 400))
    target = 0.1 * (np.tri(size) + np.tri(size)[::-1])
    target_positions = [(50, 50), (200, 200)]
    for x, y in target_positions:
        image[x:x + size, y:y + size] = target
    np.random.seed(1)
    image += 0.1 * np.random.uniform(size=(400, 400))

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
    image = 0.5 * np.ones((N, N))
    image[ipos:ipos + n, jpos:jpos + n] = 1
    image[ineg:ineg + n, jneg:jneg + n] = 0

    # white square with a black border
    template = np.zeros((n + 2, n + 2))
    template[1:1 + n, 1:1 + n] = 1

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
    image = 0.5 + 1e-9 * np.random.normal(size=(20, 20))
    template = np.ones((6, 6))
    template[:3, :] = 0
    result = match_template(image, template)
    assert not np.any(np.isnan(result))


def test_switched_arguments():
    image = np.ones((5, 5))
    template = np.ones((3, 3))
    np.testing.assert_raises(ValueError, match_template, template, image)


def test_pad_input():
    """Test `match_template` when `pad_input=True`.

    This test places two full templates (one with values lower than the image
    mean, the other higher) and two half templates, which are on the edges of
    the image. The two full templates should score the top (positive and
    negative) matches and the centers of the half templates should score 2nd.
    """
    # Float prefactors ensure that image range is between 0 and 1
    template = 0.5 * diamond(2)
    image = 0.5 * np.ones((9, 19))
    mid = slice(2, 7)
    image[mid, :3] -= template[:, -3:]  # half min template centered at 0
    image[mid, 4:9] += template         # full max template centered at 6
    image[mid, -9:-4] -= template       # full min template centered at 12
    image[mid, -3:] += template[:, :3]  # half max template centered at 18

    result = match_template(image, template, pad_input=True)

    # get the max and min results.
    sorted_result = np.argsort(result.flat)
    i, j = np.unravel_index(sorted_result[:2], result.shape)
    assert_close(j, (12, 0))
    i, j = np.unravel_index(sorted_result[-2:], result.shape)
    assert_close(j, (18, 6))


if __name__ == "__main__":
    from numpy import testing
    testing.run_module_suite()
