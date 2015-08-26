from __future__ import print_function, division

import numpy as np
from numpy.testing import (run_module_suite, assert_allclose,
                           assert_raises)
from skimage.restoration import inpaint


def test_inpaint_biharmonic():
    row = np.square(np.linspace(0, 1, 5))
    img = np.repeat(np.reshape(row, (1, 5)), 5, axis=0)
    mask = np.zeros_like(img)
    mask[2, 2:] = 1
    mask[1, 3:] = 1
    mask[0, 4:] = 1
    out = inpaint.inpaint_biharmonic(img, mask)
    ref = [[0., 0.0625, 0.25, 0.5625, 0.671875],
           [0., 0.0625, 0.25, 0.5390625, 0.78125],
           [0., 0.0625, 0.2578125, 0.5625, 0.890625],
           [0., 0.0625, 0.25, 0.5625, 1.],
           [0., 0.0625, 0.25, 0.5625, 1.]]
    assert_allclose(ref, out)


def test_invalid_input():
    assert_raises(TypeError, inpaint.inpaint_biharmonic, np.zeros([]))

    img, mask = np.zeros([]), np.zeros([])
    assert_raises(ValueError, inpaint.inpaint_biharmonic, [img, mask])

    img, mask = np.zeros((2, 2)), np.zeros((4, 1))
    assert_raises(ValueError, inpaint.inpaint_biharmonic, [img, mask])

    img = np.ma.array(np.zeros(2, 2), mask=[[0, 0], [0, 0]])
    mask = np.zeros((2, 2))
    assert_raises(ValueError, inpaint.inpaint_biharmonic, [img, mask])


if __name__ == '__main__':
    run_module_suite()
