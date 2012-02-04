import numpy as np
from numpy.random import randn
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
    image += randn(400, 400) * 2

    for method in ["norm-corr", "norm-coeff"]:
        result = match_template(image, target, method=method)
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


if __name__ == "__main__":
    from numpy import testing
    testing.run_module_suite()

