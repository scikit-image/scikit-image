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
        RGB or grayscale nD image. The input image is converted to grayscale
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
        Blur metric: the maximum of blur metrics along all axes.
    blur_table : list of floats
        The i-th element is the blur metric along the i-th axis.

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

    if channel_axis is not None:
        try:
            # ensure color channels are in the final dimension
            image = np.moveaxis(image, channel_axis, -1)
        except np.AxisError:
            print('channel_axis must be one of the image array dimensions')
            raise
        except TypeError:
            print('channel_axis must be an integer')
            raise
        image = rgb2gray(image)
    n_axes = image.ndim
    image = img_as_float(image)
    shape = image.shape
    B = []

    for a in range(n_axes):
        filt_im = ndi.uniform_filter1d(image, h_size, axis=a)
        im_sharp = np.abs(sobel(image, axis=a))
        im_blur = np.abs(sobel(filt_im, axis=a))
        T = np.maximum(0, im_sharp - im_blur)
        slices = [slice(2, shape[ax] - 1) for ax in range(n_axes)]
        M1 = np.sum(im_sharp[tuple(slices)])
        M2 = np.sum(T[tuple(slices)])
        B.append(np.abs((M1 - M2)) / M1)

    return np.max(B), B
