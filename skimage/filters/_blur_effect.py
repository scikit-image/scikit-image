import numpy as np
import scipy.ndimage as ndi

from ..color import rgb2gray
from . import sobel
from ..util import img_as_float


__all__ = ['blur_effect']


def blur_effect(image, h_size=11, channel_axis=None):
    """
    Compute a metric that indicates the strength of blur in an image (0 for no
    blur, 1 for maximal blur).

    Parameters
    ----------
    image : ndarray
        RGB or grayscale 2D image. The input image is converted to grayscale
        before computing the blur metric.
    h_size : int, optional
        Size of the re-blurring filter. Default is 11.
    channel_axis : int or None, optional
        If None, the image is assumed to be grayscale (single-channel).
        Otherwise, this parameter indicates which axis of the array
        corresponds to color channels.

    Returns
    -------

    blur : float (0 to 1)
        Blur metric: the maximum of horizontal (Bx) and vertical (By) metrics.
    blur_table : ndarray [Bx, By]
        Bx (resp. By) is the blur metric in the horizontal (resp. vertical)
        direction.

    Notes
    -----

    `h_size` must keep the same value in order to compare results between
    images. Most of the time, the default size (11) is enough. This means that
    the metric can clearly discriminate blur up to an average 11x11 filter; if
    blur is higher, the metric still gives good results but its values tend
    towards an asymptote.

    References
    ----------
    .. [1] Frederique Crete, Thierry Dolmiere, Patricia Ladret, and Marina
       Nicolas "The blur effect: perception and estimation with a new
       no-reference perceptual blur metric" Proc. SPIE 6492, Human Vision and
       Electronic Imaging XII, 64920I (2007)
       https://hal.archives-ouvertes.fr/hal-00232709
       :DOI:`10.1117/12.702790`
    """
    if image.ndim not in (2, 3):
        raise ValueError('image must be 2-dimensional')

    if channel_axis is not None:
        if not isinstance(channel_axis, int) or channel_axis >= image.ndim:
            raise ValueError('channel_axis value is invalid')
        else:
            # ensure color channels are at the final dimension to use rgb2gray
            image = np.moveaxis(image, channel_axis, -1)
            image = rgb2gray(image)
    image = img_as_float(image)
    shape = image.shape

    # vertical blur
    ver = ndi.uniform_filter1d(image, h_size, axis=0)
    im_sharp = np.abs(sobel(image, axis=0))
    im_blur = np.abs(sobel(ver, axis=0))

    T = np.fmax(np.zeros_like(image), im_sharp - im_blur)
    M1 = np.sum(im_sharp[2:shape[0] - 1, 2:shape[1] - 1])
    M2 = np.sum(T[2:shape[0] - 1, 2:shape[1] - 1])

    By = np.abs((M1 - M2)) / M1

    # horizontal blur
    hor = ndi.uniform_filter1d(image, h_size, axis=1)
    im_sharp = np.abs(sobel(image, axis=1))
    im_blur = np.abs(sobel(hor, axis=1))

    T = np.fmax(np.zeros_like(image), im_sharp - im_blur)
    M1 = np.sum(im_sharp[2:shape[0] - 1, 2:shape[1] - 1])
    M2 = np.sum(T[2:shape[0] - 1, 2:shape[1] - 1])

    Bx = np.abs((M1 - M2)) / M1

    B = np.array([Bx, By])
    return B.max(), B
