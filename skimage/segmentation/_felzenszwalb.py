import numpy as np

from .._shared.utils import warn
from ._felzenszwalb_cy import _felzenszwalb_grey, _felzenszwalb_rgb


def felzenszwalb(image, scale=1, sigma=0.8, min_size=20):
    """Computes Felsenszwalb's efficient graph based image segmentation.

    Produces an oversegmentation of a multichannel (i.e. RGB) image
    using a fast, minimum spanning tree based clustering on the image grid.
    The parameter ``scale`` sets an observation level. Higher scale means
    less and larger segments. ``sigma`` is the diameter of a Gaussian kernel,
    used for smoothing the image prior to segmentation.

    The number of produced segments as well as their size can only be
    controlled indirectly through ``scale``. Segment size within an image can
    vary greatly depending on local contrast.

    For RGB images, the algorithm computes a separate segmentation for each
    channel and then combines these. The combined segmentation is the
    intersection of the separate segmentations on the color channels.

    Parameters
    ----------
    image : (width, height, 3) or (width, height) ndarray
        Input image.
    scale : float
        Free parameter. Higher means larger clusters.
    sigma : float
        Width of Gaussian kernel used in preprocessing.
    min_size : int
        Minimum component size. Enforced using postprocessing.

    Returns
    -------
    segment_mask : (width, height) ndarray
        Integer mask indicating segment labels.

    References
    ----------
    .. [1] Efficient graph-based image segmentation, Felzenszwalb, P.F. and
           Huttenlocher, D.P.  International Journal of Computer Vision, 2004
    """

    if image.ndim == 2:
        # assume single channel image
        return _felzenszwalb_grey(image, scale=scale, sigma=sigma,
                                  min_size=min_size)

    elif image.ndim != 3:
        raise ValueError("Felzenswalb segmentation can only operate on RGB and"
                         " grey images, but input array of ndim %d given."
                         % image.ndim)
    return _felzenszwalb_rgb(image, scale=scale, sigma=sigma,
                             min_size=min_size)
