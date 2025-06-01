from numpy import divide, absolute, zeros_like, ndarray
from scipy.fftpack import fft2, ifft2
from skimage.exposure.exposure import rescale_intensity


def alpha_rooting(image: ndarray,
                  alpha: float = 0.9) -> ndarray:
    """Contrast enhancement using 2D Fourier transform, introduced in [1]_.

    Parameters
    ----------
    image : ndarray
        Input image.
        RGB or grayscale input image.
    alpha : float
        The exponent for computing the root of modulus.
        If the value is below 1, the image is sharpened,
        otherwise, the image is blurred.

    Returns
    -------
    out : float
        Inverse Fourier transformed image with new absolute value.

    References
    ----------
    .. [1] A. K. Jain, Fundamentals of Digital Image Processing.
           Upper SaddleRiver, NJ: Prentice-Hall, 1989.

    """
    fft_transformed = fft2(image)
    magnitude = absolute(fft_transformed) ** alpha
    out = ifft2(magnitude * divide(fft_transformed,
                                   absolute(fft_transformed),
                                   out=zeros_like(fft_transformed),
                                   where=absolute(fft_transformed) != 0)
                )
    out = rescale_intensity(out.real, out_range=(0., 1.))
    return out
