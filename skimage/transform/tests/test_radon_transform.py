from __future__ import print_function

import numpy as np
from numpy.testing import *
from skimage.transform import *


def rescale(x):
    x = x.astype(float)
    x -= x.min()
    x /= x.max()
    return x


def test_radon_iradon():
    size = 100
    debug = False
    image = np.tri(size) + np.tri(size)[::-1]
    for filter_type in ["ramp", "shepp-logan", "cosine", "hamming", "hann"]:
        reconstructed = iradon(radon(image), filter=filter_type)

        image = rescale(image)
        reconstructed = rescale(reconstructed)
        delta = np.mean(np.abs(image - reconstructed))

        if debug:
            print(delta)
            import matplotlib.pyplot as plt
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(image, cmap=plt.cm.gray)
            ax2.imshow(reconstructed, cmap=plt.cm.gray)
            plt.show()

        assert delta < 0.05

    reconstructed = iradon(radon(image), filter="ramp", interpolation="nearest")
    delta = np.mean(abs(image - reconstructed))
    assert delta < 0.05
    size = 20
    image = np.tri(size) + np.tri(size)[::-1]
    reconstructed = iradon(radon(image), filter="ramp", interpolation="nearest")


def test_iradon_angles():
    """
    Test with different number of projections
    """
    size = 100
    # Synthetic data
    image = np.tri(size) + np.tri(size)[::-1]
    # Large number of projections: a good quality is expected
    nb_angles = 200
    radon_image_200 = radon(image, theta=np.linspace(0, 180, nb_angles,
                    endpoint=False))
    reconstructed = iradon(radon_image_200)
    delta_200 = np.mean(abs(rescale(image) - rescale(reconstructed)))
    assert delta_200 < 0.03
    # Lower number of projections
    nb_angles = 80
    radon_image_80 = radon(image, theta=np.linspace(0, 180, nb_angles,
                    endpoint=False))
    # Test whether the sum of all projections is approximately the same
    s = radon_image_80.sum(axis=0)
    assert np.allclose(s, s[0], rtol=0.01)
    reconstructed = iradon(radon_image_80)
    delta_80 = np.mean(abs(image / np.max(image) -
                           reconstructed / np.max(reconstructed)))
    # Loss of quality when the number of projections is reduced
    assert delta_80 > delta_200


def test_radon_minimal():
    """
    Test for small images for various angles
    """
    thetas = [np.arange(180)]
    for theta in thetas:
        a = np.zeros((3, 3))
        a[1, 1] = 1
        p = radon(a, theta)
        reconstructed = iradon(p, theta)
        reconstructed /= np.max(reconstructed)
        assert np.all(abs(a - reconstructed) < 0.4)

        b = np.zeros((4, 4))
        b[1:3, 1:3] = 1
        p = radon(b, theta)
        reconstructed = iradon(p, theta)
        reconstructed /= np.max(reconstructed)
        assert np.all(abs(b - reconstructed) < 0.4)

        c = np.zeros((5, 5))
        c[1:3, 1:3] = 1
        p = radon(c, theta)
        reconstructed = iradon(p, theta)
        reconstructed /= np.max(reconstructed)
        assert np.all(abs(c - reconstructed) < 0.4)


def test_reconstruct_with_wrong_angles():
    a = np.zeros((3, 3))
    p = radon(a, theta=[0, 1, 2])
    iradon(p, theta=[0, 1, 2])
    assert_raises(ValueError, iradon, p, theta=[0, 1, 2, 3])


if __name__ == "__main__":
    run_module_suite()
