import warnings
import numpy as np

from ._felzenszwalb_cy import _felzenszwalb_grey


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

    # assume we got 2d image with multiple channels
    n_channels = image.shape[2]
    if n_channels != 3:
        warnings.warn("Got image with %d channels. Is that really what you"
                      " wanted?" % image.shape[2])
    segmentations = []
    # compute quickshift for each channel
    for c in range(n_channels):
        channel = np.ascontiguousarray(image[:, :, c])
        s = _felzenszwalb_grey(channel, scale=scale, sigma=sigma,
                               min_size=min_size)
        segmentations.append(s)

    # put pixels in same segment only if in the same segment in all images
    # we do this by combining the channels to one number
    n0 = segmentations[0].max() + 1
    n1 = segmentations[1].max() + 1
    segmentation = (segmentations[0] + segmentations[1] * n0
                    + segmentations[2] * n0 * n1)
    # make segment labels consecutive numbers starting at 0
    labels = np.unique(segmentation, return_inverse=True)[1]
    return labels.reshape(image.shape[:2])
