"""Testing code for converting between different types of data.

Mark Harfouche
Feb 13, 2018
"""

import numpy as np

from skimage.color.data_convert import *

from skimage._shared import testing
from skimage._shared.testing import (assert_array_almost_equal,
                                     assert_array_equal, assert_warns)


def test_bogus_dtype():
    """Ensures that image conversion will raise errors on bad types.

    im2integer should fail if passed a float
    im2float should fail if passed an int.
    """
    image = np.ones((3, 3))

    with testing.raises(TypeError):
        im2integer(image, dtype=np.float)

    with testing.raises(TypeError):
        im2float(image, dtype=np.uint8)


def test_im2type():
    image = np.array([[1, 1, 1, 2],
                      [2, 2, 3, 3],
                      [4, 4, 6, 6]], dtype=np.uint8)
    image_max = 255

    types = [np.float64, np.int16, np.float32,
             np.uint16, np.uint8]

    functions = [im2double, im2int16, im2single,
                 im2uint16, im2uint8]

    for input_type in types:
        for output_type, func in zip(types, functions):
            input_image = image.astype(input_type)

            output_image_estimate = image.astype(output_type)
            if issubclass(output_type, np.floating):
                output_image_estimate = output_image_estimate / image_max

            output_image = im2type(input_image, output_type)

            if issubclass(output_type, np.floating):
                assert_array_almost_equal(output_image, output_image_estimate)
            else:
                assert_array_equal(output_image, output_image_estimate)

            output_image = func(input_image)

            if issubclass(output_type, np.floating):
                assert_array_almost_equal(output_image, output_image_estimate)
            else:
                assert_array_equal(output_image, output_image_estimate)
