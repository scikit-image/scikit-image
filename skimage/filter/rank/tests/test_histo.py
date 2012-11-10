import numpy as np
from numpy.testing import run_module_suite, assert_array_equal

from skimage.filter.rank import _crank8, _crank16


def test_trivial_selem8():
    # check that min, max and mean returns identity if structuring element
    # contains only central pixel

    image = np.zeros((5, 5), dtype=np.uint8)
    out = np.zeros_like(image)
    mask = np.ones_like(image, dtype=np.uint8)
    image[2,2] = 255
    image[2,3] = 128
    image[1,2] = 16

    elem = np.array([[0, 0, 0], [0, 1, 0],[0, 0, 0]], dtype=np.uint8)
    _crank8.mean(image=image, selem=elem, out=out, mask=mask,
                 shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    _crank8.minimum(image=image, selem=elem, out=out, mask=mask,
                    shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    _crank8.maximum(image=image, selem=elem, out=out, mask=mask,
                    shift_x=0, shift_y=0)
    assert_array_equal(image, out)


def test_trivial_selem16():
    # check that min, max and mean returns identity if structuring element
    # contains only central pixel

    image = np.zeros((5, 5), dtype=np.uint16)
    out = np.zeros_like(image)
    mask = np.ones_like(image, dtype=np.uint8)
    image[2,2] = 255
    image[2,3] = 128
    image[1,2] = 16

    elem = np.array([[0, 0, 0], [0, 1, 0],[0, 0, 0]], dtype=np.uint8)
    _crank16.mean(image=image, selem=elem, out=out, mask=mask,
                  shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    _crank16.minimum(image=image, selem=elem, out=out, mask=mask,
                     shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    _crank16.maximum(image=image, selem=elem, out=out, mask=mask,
                     shift_x=0, shift_y=0)
    assert_array_equal(image, out)


def test_smallest_selem8():
    # check that min, max and mean returns identity if structuring element
    # contains only central pixel

    image = np.zeros((5, 5), dtype=np.uint8)
    out = np.zeros_like(image)
    mask = np.ones_like(image, dtype=np.uint8)
    image[2,2] = 255
    image[2,3] = 128
    image[1,2] = 16

    elem = np.array([[1]], dtype=np.uint8)
    _crank8.mean(image=image, selem=elem, out=out, mask=mask,
                 shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    _crank8.minimum(image=image, selem=elem, out=out, mask=mask,
                    shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    _crank8.maximum(image=image, selem=elem, out=out, mask=mask,
                    shift_x=0, shift_y=0)
    assert_array_equal(image, out)


def test_smallest_selem16():
    # check that min, max and mean returns identity if structuring element
    # contains only central pixel

    image = np.zeros((5, 5), dtype=np.uint16)
    out = np.zeros_like(image)
    mask = np.ones_like(image, dtype=np.uint8)
    image[2,2] = 255
    image[2,3] = 128
    image[1,2] = 16

    elem = np.array([[1]], dtype=np.uint8)
    _crank16.mean(image=image, selem=elem, out=out, mask=mask,
                  shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    _crank16.minimum(image=image, selem=elem, out=out, mask=mask,
                     shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    _crank16.maximum(image=image, selem=elem, out=out, mask=mask,
                     shift_x=0, shift_y=0)
    assert_array_equal(image, out)


if __name__ == "__main__":
    run_module_suite()
