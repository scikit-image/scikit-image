"""
radon.py - Radon and inverse radon transforms

Based on code of Justin K. Romberg
(http://www.clear.rice.edu/elec431/projects96/DSP/bpanalysis.html)
J. Gillam and Chris Griffin.

References:
    -B.R. Ramesh, N. Srinivasa, K. Rajgopal, "An Algorithm for Computing
    the Discrete Radon Transform With Some Applications", Proceedings of
    the Fourth IEEE Region 10 International Conference, TENCON '89, 1989.
    -A. C. Kak, Malcolm Slaney, "Principles of Computerized Tomographic
    Imaging", IEEE Press 1988.
"""
from __future__ import division
import numpy as np
from scipy.fftpack import fftshift, fft, ifft
from ._warps_cy import _warp_fast

__all__ = ["radon", "iradon"]


def radon(image, theta=None, circle=False):
    """
    Calculates the radon transform of an image given specified
    projection angles.

    Parameters
    ----------
    image : array_like, dtype=float
        Input image.
    theta : array_like, dtype=float, optional (default np.arange(180))
        Projection angles (in degrees).
    circle : boolean, optional
        Assume image is zero outside the inscribed circle, making the
        width of each projection (the first dimension of the sinogram)
        equal to ``min(image.shape)``.

    Returns
    -------
    output : ndarray
        Radon transform (sinogram).

    Raises
    ------
    ValueError
        If called with ``circle=True`` and ``image != 0`` outside the inscribed
        circle
    """
    if image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    if theta is None:
        theta = np.arange(180)

    if circle:
        radius = min(image.shape) // 2
        c0, c1 = np.ogrid[0:image.shape[0], 0:image.shape[1]]
        reconstruction_circle = ((c0 - image.shape[0] // 2)**2
                                 + (c1 - image.shape[1] // 2)**2) < radius**2
        if not np.all(reconstruction_circle | (image == 0)):
            raise ValueError('Image must be zero outside the reconstruction'
                             ' circle')
        slices = []
        for d in (0, 1):
            if image.shape[d] > min(image.shape):
                excess = image.shape[d] - min(image.shape)
                slices.append(slice(int(np.ceil(excess / 2)),
                                    int(np.ceil(excess / 2)
                                        + min(image.shape))))
            else:
                slices.append(slice(None))
        slices = tuple(slices)
        padded_image = image[slices]
        out = np.zeros((min(padded_image.shape), len(theta)))
        dh = padded_image.shape[0] // 2
        dw = padded_image.shape[1] // 2
    else:
        height, width = image.shape
        diagonal = np.sqrt(2) * max(image.shape)
        heightpad = int(np.ceil(diagonal - height))
        widthpad = int(np.ceil(diagonal - width))
        padded_image = np.zeros((int(height + heightpad),
                                int(width + widthpad)))
        y0 = heightpad // 2
        y1 = y0 + height
        x0 = widthpad // 2
        x1 = x0 + width
        padded_image[y0:y1, x0:x1] = image
        out = np.zeros((max(padded_image.shape), len(theta)))
        dh = y0 + height // 2
        dw = x0 + width // 2

    shift0 = np.array([[1, 0, -dw],
                       [0, 1, -dh],
                       [0, 0, 1]])
    shift1 = np.array([[1, 0, dw],
                       [0, 1, dh],
                       [0, 0, 1]])

    def build_rotation(theta):
        T = np.deg2rad(theta)
        R = np.array([[np.cos(T), np.sin(T), 0],
                      [-np.sin(T), np.cos(T), 0],
                      [0, 0, 1]])
        return shift1.dot(R).dot(shift0)

    for i in range(len(theta)):
        rotated = _warp_fast(padded_image, build_rotation(theta[i]))
        out[:, i] = rotated.sum(0)
    return out


def _sinogram_circle_to_square(sinogram):
    size = int(np.ceil(np.sqrt(2) * sinogram.shape[0]))
    sinogram_padded = np.zeros((size, sinogram.shape[1]))
    pad = (size - sinogram.shape[0]) // 2
    sinogram_padded[pad:pad + sinogram.shape[0], :] = sinogram
    return sinogram_padded


def iradon(radon_image, theta=None, output_size=None,
           filter="ramp", interpolation="linear", circle=False):
    """
    Inverse radon transform.

    Reconstruct an image from the radon transform, using the filtered
    back projection algorithm.

    Parameters
    ----------
    radon_image : array_like, dtype=float
        Image containing radon transform (sinogram). Each column of
        the image corresponds to a projection along a different angle.
    theta : array_like, dtype=float, optional
        Reconstruction angles (in degrees). Default: m angles evenly spaced
        between 0 and 180 (if the shape of `radon_image` is (N, M)).
    output_size : int
        Number of rows and columns in the reconstruction.
    filter : str, optional (default ramp)
        Filter used in frequency domain filtering. Ramp filter used by default.
        Filters available: ramp, shepp-logan, cosine, hamming, hann
        Assign None to use no filter.
    interpolation : str, optional (default linear)
        Interpolation method used in reconstruction.
        Methods available: nearest, linear.
    circle : boolean, optional
        Assume the reconstructed image is zero outside the inscribed circle.
        Also changes the default output_size to match the behaviour of
        ``radon`` called with ``circle=True``.

    Returns
    -------
    output : ndarray
      Reconstructed image.

    Notes
    -----
    It applies the Fourier slice theorem to reconstruct an image by
    multiplying the frequency domain of the filter with the FFT of the
    projection data. This algorithm is called filtered back projection.

    """
    if radon_image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    if theta is None:
        m, n = radon_image.shape
        theta = np.linspace(0, 180, n, endpoint=False)
    else:
        theta = np.asarray(theta)
    if len(theta) != radon_image.shape[1]:
        raise ValueError("The given ``theta`` does not match the number of "
                         "projections in ``radon_image``.")
    if not output_size:
        # If output size not specified, estimate from input radon image
        if circle:
            output_size = radon_image.shape[0]
        else:
            output_size = int(np.floor(np.sqrt((radon_image.shape[0])**2
                                               / 2.0)))
    if circle:
        radon_image = _sinogram_circle_to_square(radon_image)

    th = (np.pi / 180.0) * theta
    n = radon_image.shape[0]
    img = radon_image.copy()
    # resize image to next power of two for fourier analysis
    # speeds up fourier and lessens artifacts
    order = max(64., 2**np.ceil(np.log(2 * n) / np.log(2)))
    # zero pad input image
    img.resize((order, img.shape[1]))

    # Construct the Fourier filter
    f = fftshift(abs(np.mgrid[-1:1:2 / order])).reshape(-1, 1)
    w = 2 * np.pi * f
    # Start from first element to avoid divide by zero
    if filter == "ramp":
        pass
    elif filter == "shepp-logan":
        f[1:] = f[1:] * np.sin(w[1:] / 2) / (w[1:] / 2)
    elif filter == "cosine":
        f[1:] = f[1:] * np.cos(w[1:] / 2)
    elif filter == "hamming":
        f[1:] = f[1:] * (0.54 + 0.46 * np.cos(w[1:]))
    elif filter == "hann":
        f[1:] = f[1:] * (1 + np.cos(w[1:])) / 2
    elif filter is None:
        f[1:] = 1
    else:
        raise ValueError("Unknown filter: %s" % filter)
    filter_ft = np.tile(f, (1, len(theta)))
    # Apply filter in Fourier domain
    projection = fft(img, axis=0) * filter_ft
    radon_filtered = np.real(ifft(projection, axis=0))

    # Resize filtered image back to original size
    radon_filtered = radon_filtered[:radon_image.shape[0], :]
    reconstructed = np.zeros((output_size, output_size))
    # Determine the center of the projections (= center of sinogram)
    circle_size = int(np.floor(radon_image.shape[0] / np.sqrt(2)))
    square_size = radon_image.shape[0]
    mid_index = (square_size - circle_size) // 2 + circle_size // 2 + 1

    x = output_size
    y = output_size
    [X, Y] = np.mgrid[0.0:x, 0.0:y]
    xpr = X - int(output_size) // 2
    ypr = Y - int(output_size) // 2
    if circle:
        radius = (output_size - 1) // 2
        reconstruction_circle = (xpr**2 + ypr**2) < radius**2

    # Reconstruct image by interpolation
    if interpolation == "nearest":
        for i in range(len(theta)):
            k = np.round(mid_index + ypr * np.cos(th[i]) - xpr * np.sin(th[i]))
            backprojected = radon_filtered[
                ((((k > 0) & (k < n)) * k) - 1).astype(np.int), i]
            if circle:
                backprojected[~reconstruction_circle] = 0.
            reconstructed += backprojected
    elif interpolation == "linear":
        for i in range(len(theta)):
            t = ypr * np.cos(th[i]) - xpr * np.sin(th[i])
            a = np.floor(t)
            b = mid_index + a
            b0 = ((((b + 1 > 0) & (b + 1 < n)) * (b + 1)) - 1).astype(np.int)
            b1 = ((((b > 0) & (b < n)) * b) - 1).astype(np.int)
            backprojected = (t - a) * radon_filtered[b0, i] + \
                            (a - t + 1) * radon_filtered[b1, i]
            if circle:
                backprojected[~reconstruction_circle] = 0.
            reconstructed += backprojected
    else:
        raise ValueError("Unknown interpolation: %s" % interpolation)

    return reconstructed * np.pi / (2 * len(th))
