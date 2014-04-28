import numpy as np
from numpy.testing import assert_allclose, assert_raises

from skimage.feature.register_translation import register_translation,\
    fourier_shift
from skimage.data import camera


def test_correlation():
    image = camera()
    shift = (-7, 12)
    shifted_image = fourier_shift(image, shift)

    # pixel precision
    result, error, diffphase = register_translation(image, shifted_image)

    assert_allclose(result[:2], np.array(shift))


def test_subpixel_precision():
    reference_image = camera()
    subpixel_shift = (-2.4, 1.32)
    shifted_image = fourier_shift(reference_image, subpixel_shift)

    # subpixel precision
    result, error, diffphase = register_translation(reference_image,
                                                    shifted_image, 100)

    assert_allclose(result[:2], np.array(subpixel_shift), atol=0.05)


def test_3d_input():
    # TODO: this test case is waiting on a Phantom data set to be added to the
    #    data module.
    # pixel precision
    # result, error, diffphase = register_translation(ref_image, shifted_image)

    # assert_allclose(np.array(result[:2]), np.array(shift))
    pass


def test_wrong_input():
    # Dimensionality mismatch
    image = np.ones((5, 5, 1))
    template = np.ones((5, 5))
    assert_raises(ValueError, register_translation, template, image)

    # Greater than 2 dimensions does not support subpixel precision
    #   (TODO: should support 3D at some point.)
    image = np.ones((5, 5, 5))
    template = np.ones((5, 5, 5))
    assert_raises(NotImplementedError, register_translation,
                  template, image, 2)

    # Size mismatch
    image = np.ones((5, 5))
    template = np.ones((4, 4))
    assert_raises(ValueError, register_translation, template, image)


if __name__ == "__main__":
    from numpy import testing
    testing.run_module_suite()
