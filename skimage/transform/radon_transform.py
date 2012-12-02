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


def radon(image, theta=None):
    """
    Calculates the radon transform of an image given specified
    projection angles.

    Parameters
    ----------
    image : array_like, dtype=float
        Input image.
    theta : array_like, dtype=float, optional (default np.arange(180))
        Projection angles (in degrees).

    Returns
    -------
    output : ndarray
        Radon transform (sinogram).

    """
    if image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    if theta is None:
        theta = np.arange(180)

    height, width = image.shape
    diagonal = np.sqrt(height**2 + width**2)
    heightpad = np.ceil(diagonal - height)
    widthpad = np.ceil(diagonal - width)
    padded_image = np.zeros((int(height + heightpad),
                             int(width + widthpad)))
    y0, y1 = int(np.ceil(heightpad / 2)), \
             int((np.ceil(heightpad / 2) + height))
    x0, x1 = int((np.ceil(widthpad / 2))), \
             int((np.ceil(widthpad / 2) + width))

    padded_image[y0:y1, x0:x1] = image
    out = np.zeros((max(padded_image.shape), len(theta)))

    h, w = padded_image.shape
    dh, dw = h // 2, w // 2
    shift0 = np.array([[1, 0, -dw],
                       [0, 1, -dh],
                       [0, 0, 1]])

    shift1 = np.array([[1, 0, dw],
                       [0, 1, dh],
                       [0, 0, 1]])

    def build_rotation(theta):
        T = -np.deg2rad(theta)

        R = np.array([[np.cos(T), -np.sin(T), 0],
                      [np.sin(T), np.cos(T), 0],
                      [0, 0, 1]])

        return shift1.dot(R).dot(shift0)

    for i in range(len(theta)):
        rotated = _warp_fast(padded_image,
                             np.linalg.inv(build_rotation(-theta[i])))

        out[:, i] = rotated.sum(0)[::-1]

    return out


def iradon(radon_image, theta=None, output_size=None,
           filter="ramp", interpolation="linear"):
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

    Returns
    -------
    output : ndarray
      Reconstructed image.

    Notes
    -----
    It applies the fourier slice theorem to reconstruct an image by
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

    th = (np.pi / 180.0) * theta
    # if output size not specified, estimate from input radon image
    if not output_size:
        output_size = int(np.floor(np.sqrt((radon_image.shape[0])**2 / 2.0)))
    n = radon_image.shape[0]

    img = radon_image.copy()
    # resize image to next power of two for fourier analysis
    # speeds up fourier and lessens artifacts
    order = max(64., 2**np.ceil(np.log(2 * n) / np.log(2)))
    # zero pad input image
    img.resize((order, img.shape[1]))
    # construct the fourier filter

    f = fftshift(abs(np.mgrid[-1:1:2 / order])).reshape(-1, 1)
    w = 2 * np.pi * f
    # start from first element to avoid divide by zero
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
    elif filter == None:
        f[1:] = 1
    else:
        raise ValueError("Unknown filter: %s" % filter)

    filter_ft = np.tile(f, (1, len(theta)))

    # apply filter in fourier domain
    projection = fft(img, axis=0) * filter_ft
    radon_filtered = np.real(ifft(projection, axis=0))

    # resize filtered image back to original size
    radon_filtered = radon_filtered[:radon_image.shape[0], :]
    reconstructed = np.zeros((output_size, output_size))
    mid_index = np.ceil(n / 2.0)

    x = output_size
    y = output_size
    [X, Y] = np.mgrid[0.0:x, 0.0:y]
    xpr = X - int(output_size) // 2
    ypr = Y - int(output_size) // 2

    # reconstruct image by interpolation
    if interpolation == "nearest":
        for i in range(len(theta)):
            k = np.round(mid_index + xpr * np.sin(th[i]) - ypr * np.cos(th[i]))
            reconstructed += radon_filtered[
                ((((k > 0) & (k < n)) * k) - 1).astype(np.int), i]

    elif interpolation == "linear":
        for i in range(len(theta)):
            t = xpr * np.sin(th[i]) - ypr * np.cos(th[i])
            a = np.floor(t)
            b = mid_index + a
            b0 = ((((b + 1 > 0) & (b + 1 < n)) * (b + 1)) - 1).astype(np.int)
            b1 = ((((b > 0) & (b < n)) * b) - 1).astype(np.int)
            reconstructed += (t - a) * radon_filtered[b0, i] + \
                             (a - t + 1) * radon_filtered[b1, i]

    else:
        raise ValueError("Unknown interpolation: %s" % interpolation)

    return reconstructed * np.pi / (2 * len(th))
