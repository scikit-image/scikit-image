from __future__ import print_function
from __future__ import division

import numpy as np
from numpy.testing import *
from numpy.fft import ifftshift, ifftn
import itertools
from skimage.transform import *


def rescale(x):
    x = x.astype(float)
    x -= x.min()
    x /= x.max()
    return x


def check_radon_center(shape, circle):
    # Determine the center of an array as defined by the fft
    ft_image = np.abs(ifftshift(ifftn(np.ones(shape))))**2
    fft_center = np.unravel_index(np.argmax(ft_image), shape)
    print('fft_center =', fft_center)
    # Create a test image with only a single non-zero pixel at the origin
    image = np.zeros(shape, dtype=np.float)
    image[fft_center] = 1.
    # Calculate the sinogram
    theta = np.linspace(0., 180., max(shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=circle)
    # The sinogram should be a straight, horizontal line
    sinogram_max = np.argmax(sinogram, axis=0)
    print(sinogram_max)
    assert np.std(sinogram_max) < 1e-6


def test_radon_center():
    shapes = [(16, 16), (17, 17)]
    circles = [False, True]
    for shape, circle in itertools.product(shapes, circles):
        yield check_radon_center, shape, circle
    rectangular_shapes = [(32, 16), (33, 17)]
    for shape in rectangular_shapes:
        yield check_radon_center, shape, False


def check_iradon_center(size, theta, circle):
    debug = False
    # Create a test sinogram corresponding to a single projection
    # with a single non-zero pixel at the rotation center
    if circle:
        sinogram = np.zeros((size, 1), dtype=np.float)
        sinogram[size // 2, 0] = 1.
    else:
        diagonal = int(np.ceil(np.sqrt(2) * size))
        sinogram = np.zeros((diagonal, 1), dtype=np.float)
        sinogram[sinogram.shape[0] // 2, 0] = 1.
    maxpoint = np.unravel_index(np.argmax(sinogram), sinogram.shape)
    print('shape of generated sinogram', sinogram.shape)
    print('maximum in generated sinogram', maxpoint)
    # Compare reconstructions for theta=angle and theta=angle + 180;
    # these should be exactly equal
    reconstruction = iradon(sinogram, theta=[theta], circle=circle)
    reconstruction_opposite = iradon(sinogram, theta=[theta + 180],
                                     circle=circle)
    print('rms deviance:',
          np.sqrt(np.mean((reconstruction_opposite - reconstruction)**2)))
    if debug:
        import matplotlib.pyplot as plt
        imkwargs = dict(cmap='gray', interpolation='nearest')
        plt.figure()
        plt.subplot(221)
        plt.imshow(sinogram, **imkwargs)
        plt.subplot(222)
        plt.imshow(reconstruction_opposite - reconstruction, **imkwargs)
        plt.subplot(223)
        plt.imshow(reconstruction, **imkwargs)
        plt.subplot(224)
        plt.imshow(reconstruction_opposite, **imkwargs)
        plt.show()

    assert np.allclose(reconstruction, reconstruction_opposite)


def test_iradon_center():
    sizes = [16, 17]
    thetas = [0, 90]
    circles = [False, True]
    for size, theta, circle in itertools.product(sizes, thetas, circles):
        yield check_iradon_center, size, theta, circle


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

    reconstructed = iradon(radon(image), filter="ramp",
                           interpolation="nearest")
    delta = np.mean(abs(image - reconstructed))
    assert delta < 0.05
    size = 20
    image = np.tri(size) + np.tri(size)[::-1]
    reconstructed = iradon(radon(image), filter="ramp",
                           interpolation="nearest")


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


def _random_circle(shape):
    # Synthetic random data, zero outside reconstruction circle
    np.random.seed(98312871)
    image = np.random.rand(*shape)
    c0, c1 = np.ogrid[0:shape[0], 0:shape[1]]
    r = np.sqrt((c0 - shape[0] // 2)**2 + (c1 - shape[1] // 2)**2)
    radius = min(shape) // 2
    image[r >= radius] = 0.
    return image


def test_radon_circle():
    a = np.ones((10, 10))
    assert_raises(ValueError, radon, a, circle=True)

    # Synthetic data, circular symmetry
    shape = (61, 79)
    c0, c1 = np.ogrid[0:shape[0], 0:shape[1]]
    r = np.sqrt((c0 - shape[0] // 2)**2 + (c1 - shape[1] // 2)**2)
    radius = min(shape) // 2
    image = np.clip(radius - r, 0, np.inf)
    image = rescale(image)
    angles = np.linspace(0, 180, min(shape), endpoint=False)
    sinogram = radon(image, theta=angles, circle=True)
    assert np.all(sinogram.std(axis=1) < 1e-2)

    # Synthetic data, random
    image = _random_circle(shape)
    sinogram = radon(image, theta=angles, circle=True)
    mass = sinogram.sum(axis=0)
    average_mass = mass.mean()
    relative_error = np.abs(mass - average_mass) / average_mass
    print(relative_error.max(), relative_error.mean())
    assert np.all(relative_error < 3e-3)


def test_radon_iradon_circle():
    shape = (61, 79)
    radius = min(shape) // 2
    image = _random_circle(shape)
    interpolations = ('nearest', 'linear')
    output_sizes = (None, min(shape), max(shape), 97)

    for interpolation, output_size in itertools.product(interpolations,
                                                        output_sizes):
        print('interpolation =', interpolation)
        print('output_size =', output_size)
        # Forward and inverse radon on synthetic data
        sinogram_rectangle = radon(image, circle=False)
        reconstruction_rectangle = iradon(sinogram_rectangle,
                                          output_size=output_size,
                                          interpolation=interpolation,
                                          circle=False)
        sinogram_circle = radon(image, circle=True)
        reconstruction_circle = iradon(sinogram_circle,
                                       output_size=output_size,
                                       interpolation=interpolation,
                                       circle=True)
        # Crop rectangular reconstruction to match circle=True reconstruction
        width = reconstruction_circle.shape[0]
        excess = int(np.ceil((reconstruction_rectangle.shape[0] - width) / 2))
        s = np.s_[excess:width + excess, excess:width + excess]
        reconstruction_rectangle = reconstruction_rectangle[s]
        # Find the reconstruction circle, set reconstruction to zero outside
        c0, c1 = np.ogrid[0:width, 0:width]
        r = np.sqrt((c0 - width // 2)**2 + (c1 - width // 2)**2)
        reconstruction_rectangle[r >= radius] = 0.
        print(reconstruction_circle.shape)
        print(reconstruction_rectangle.shape)
        np.allclose(reconstruction_rectangle, reconstruction_circle)


if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
