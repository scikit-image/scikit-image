from __future__ import print_function, division

import numpy as np
from numpy.testing import (run_module_suite, assert_allclose,
                           assert_raises)
from skimage.restoration import inpaint


def test_inpaint_biharmonic():
    img = np.tile(np.square(np.linspace(0, 1, 5)), (5, 1))
    mask = np.zeros_like(img)
    mask[2, 2:] = 1
    mask[1, 3:] = 1
    mask[0, 4:] = 1
    out = inpaint.inpaint_biharmonic(img, mask)
    ref = np.array(
        [[0., 0.0625, 0.25, 0.5625, 0.56947314],
         [0., 0.0625, 0.25, 0.47029959, 0.57644628],
         [0., 0.0625, 0.24664256, 0.49225207, 0.68956612],
         [0., 0.0625, 0.25, 0.5625, 1.],
         [0., 0.0625, 0.25, 0.5625, 1.]]
    )
    assert_allclose(ref, out)


def test_inpaint_edges():
    img = np.tile(np.square(np.linspace(0, 1, 5)), (5, 1))
    mask = np.zeros_like(img)
    mask[[0, -1], :] = 1
    mask[:, [0, -1]] = 1
    out = inpaint.inpaint_biharmonic(img, mask)
    ref = np.array(
        [[0.12199519, 0.15599245, 0.28348214, 0.44445398, 0.48737981],
         [0.08799794, 0.0625, 0.25, 0.5625, 0.53030563],
         [0.07949863, 0.0625, 0.25, 0.5625, 0.54103709],
         [0.08799794, 0.0625, 0.25, 0.5625, 0.53030563],
         [0.12199519, 0.15599245, 0.28348214, 0.44445398, 0.48737981]]
    )
    assert_allclose(ref, out)


def test_invalid_input():
    img, mask = np.zeros([]), np.zeros([])
    assert_raises(ValueError, inpaint.inpaint_biharmonic, img, mask)

    img, mask = np.zeros((2, 2)), np.zeros((4, 1))
    assert_raises(ValueError, inpaint.inpaint_biharmonic, img, mask)

    img = np.ma.array(np.zeros((2, 2)), mask=[[0, 0], [0, 0]])
    mask = np.zeros((2, 2))
    assert_raises(TypeError, inpaint.inpaint_biharmonic, img, mask)


if __name__ == '__main__':
    run_module_suite()
