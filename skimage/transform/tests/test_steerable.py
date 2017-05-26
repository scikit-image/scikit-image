from numpy.testing import assert_equal, assert_array_equal,\
    assert_array_almost_equal, run_module_suite
from skimage.transform import steerable
import numpy as np
from skimage import img_as_float


def test_steerable_shape():
    im = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    coeff = steerable.build_steerable(im)

    assert_array_equal(coeff[0].shape, [128, 128])
    assert_array_equal(coeff[-1].shape, [16, 16])

    for i in range(1, 4):
        assert_equal(len(coeff[i]), 4)
        for j in range(4):
            m = 2**(8 - i)
            assert_array_equal(coeff[i][j].shape, np.array([m, m]))


def test_different_orientation_height():
    im = np.random.randint(0, 255, (113, 29), dtype=np.uint8)
    coeff = steerable.build_steerable(im, height=3, nbands=6)
    out = steerable.recon_steerable(coeff)

    assert_array_almost_equal(img_as_float(im), out, decimal=1)


def test_steerable_reconstruction_power_of_two():
    im = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    coeff = steerable.build_steerable(im)
    out = steerable.recon_steerable(coeff)

    assert_array_almost_equal(img_as_float(im), out, decimal=2)


def test_steerable_reconstruction_symmetric():
    im = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    coeff = steerable.build_steerable(im)
    out = steerable.recon_steerable(coeff)

    assert_array_almost_equal(img_as_float(im), out, decimal=2)


def test_steerable_reconstruction_asymmetric():
    im = np.random.randint(0, 255, (113, 29), dtype=np.uint8)
    coeff = steerable.build_steerable(im)
    out = steerable.recon_steerable(coeff)

    assert_array_almost_equal(img_as_float(im), out, decimal=1)


def test_steerable_reconstruction_power_of_two_float():
    im = np.random.uniform(0, 1, size=(128, 128))
    im = im.astype(np.float)
    coeff = steerable.build_steerable(im)
    out = steerable.recon_steerable(coeff)

    assert_array_almost_equal(img_as_float(im), out, decimal=2)


def test_steerable_reconstruction_symmetric_float():
    im = np.random.uniform(0, 1, size=(128, 128))
    im = im.astype(np.float)

    coeff = steerable.build_steerable(im)
    out = steerable.recon_steerable(coeff)

    assert_array_almost_equal(img_as_float(im), out, decimal=2)


def test_steerable_reconstruction_asymmetric_float():
    im = np.random.uniform(0, 1, size=(128, 128))
    im = im.astype(np.float)

    coeff = steerable.build_steerable(im)
    out = steerable.recon_steerable(coeff)

    assert_array_almost_equal(img_as_float(im), out, decimal=1)


def test_steerable_reconstruction_asymmetric_binary():
    im = np.random.uniform(0, 1, size=(128, 128))
    im = im > 0.5

    coeff = steerable.build_steerable(im)
    out = steerable.recon_steerable(coeff)

    assert_array_almost_equal(img_as_float(im), out, decimal=1)


if __name__ == "__main__":
    run_module_suite()
