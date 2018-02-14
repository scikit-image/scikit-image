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
    image = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [10, 12, 14, 16]]) / 16

    types = [np.float64, np.int16, np.float32,
             np.uint16, np.uint8]

    functions = [im2double, im2int16, im2single,
                 im2uint16, im2uint8]

    for input_type in types:
        for output_type, func in zip(types, functions):
            err_message = "input_type: " + str(input_type) + ", " + \
                          "output_type: " + str(output_type)
            if issubclass(input_type, np.integer):
                input_image = \
                    (image * np.iinfo(input_type).max).astype(input_type)
                bits_input = np.iinfo(input_type).bits
                if issubclass(input_type, np.signedinteger):
                    bits_input = bits_input - 1
            else:
                input_image = image.astype(input_type)

            if issubclass(output_type, np.integer):
                output_image_estimate = \
                    (image * np.iinfo(output_type).max).astype(output_type)
                bits_output = np.iinfo(output_type).bits
                if issubclass(output_type, np.signedinteger):
                    bits_output = bits_output - 1
            else:
                output_image_estimate = image.astype(output_type)


            output_image = im2type(input_image, output_type)

            float_decimals = 4
            if issubclass(input_type, np.uint8):
                float_decimals = 2
            if issubclass(output_type, np.floating):
                assert_array_almost_equal(output_image, output_image_estimate,
                                          decimal=float_decimals, err_msg=err_message)
            elif issubclass(input_type, np.integer):
                assert_array_almost_equal(output_image, output_image_estimate,
                                          decimal=-np.abs(bits_input - bits_output),
                                          err_msg=err_message)
            else:
                assert_array_equal(output_image, output_image_estimate)

            output_image = func(input_image)

            if issubclass(output_type, np.floating):
                assert_array_almost_equal(output_image, output_image_estimate,
                                          decimal=float_decimals, err_msg=err_message)
            elif issubclass(input_type, np.integer):
                assert_array_almost_equal(output_image, output_image_estimate,
                                          decimal=-np.abs(bits_input - bits_output),
                                          err_msg=err_message)
            else:
                assert_array_equal(output_image, output_image_estimate)

