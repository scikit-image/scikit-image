import numpy as np
from numpy.testing import assert_array_almost_equal

from skimage import transform as tf, data, img_as_float


def test_roundtrip():
    image = img_as_float(data.checkerboard())

    swirl_params = {'radius': 80, 'rotation': 0, 'order': 2, 'mode': 'reflect'}
    swirled = tf.swirl(image, strength=10, **swirl_params)
    unswirled = tf.swirl(swirled, strength=-10, **swirl_params)

    assert np.mean(np.abs(image - unswirled)) < 0.01

if __name__ == "__main__":
    np.testing.run_module_suite()
