import warnings
import numpy as np

from ._felzenszwalb import felzenszwalb_segmentation_grey

from IPython.core.debugger import Tracer
tracer = Tracer()


def felzenszwalb_segmentation(image, scale=200, sigma=0.8):
    """Computes Felsenszwalb's segmentation for multi channel images.

    Calls the algorithm on each channel separately, then combines
    using "and", i.e. two pixels are in the same segment if they are
    in the same segment for each channel.

    Parameters
    ----------
    image: ndarray, [width, height]
        Input image

    scale: float
        Free parameter. Higher means larger clusters.
        For 0-255 data, hundereds are good.

    sigma: float
        Width of Gaussian kernel used in preprocessing.

    Returns
    -------
    segment_mask: ndarray, [width, height]
        Integer mask indicating segment labels.
    """

    #image = img_as_float(image)
    if image.ndim == 2:
        # assume single channel image
        return felzenszwalb_segmentation_grey(image, scale=scale, sigma=sigma)

    elif image.ndim != 3:
        raise ValueError("Got image with ndim=%d, don't know"
                " what to do." % image.ndim)

    # assume we got 2d image with multiple channels
    n_channels = image.shape[2]
    if n_channels != 3:
        warnings.warn("Got image with %d channels. Is that really what you"
                " wanted?" % image.shape[2])
    segmentations = []
    # compute quickshift for each channel
    for c in xrange(n_channels):
        channel = np.ascontiguousarray(image[:, :, c])
        seg = felzenszwalb_segmentation_grey(channel, scale=scale, sigma=sigma)
        segmentations.append(seg)

    # put pixels in same segment only if in the same segment in all images
    # we do this by combining the channels to one number
    segmentations = [np.unique(s, return_inverse=True)[1] for s in
            segmentations]
    n0 = max(segmentations[0])
    n1 = max(segmentations[1])
    hasher = np.array([n1 * n0, n0, 1])
    segmentations = np.dstack(segmentations).reshape(-1, n_channels)
    segmentation = np.dot(segmentations, hasher)
    # make segment labels consecutive numbers starting at 0
    labels = np.unique(segmentation, return_inverse=True)[1]
    return labels.reshape(image.shape[:2])
