# coding=utf-8

import collections as coll
import numpy as np
from scipy import ndimage
import warnings

from skimage.util import img_as_float, regular_grid
from skimage.segmentation._slic import _slic_cython
from skimage.color import rgb2lab


def slic(image, n_segments=100, compactness=10., max_iter=20, sigma=1,
         multichannel=True, convert2lab=True, ratio=None):
    """Segments image using k-means clustering in Color-(x,y,z) space.

    Parameters
    ----------
    image : 2D, 3D or 4D ndarray
        Input image, which can be 2D or 3D, and grayscale or multichannel
        (see `multichannel` parameter).
    n_segments : int, optional
        The (approximate) number of labels in the segmented output image.
    compactness : float, optional
        Balances color-space proximity and image-space proximity. Higher
        values give more weight to image-space. As `compactness` tends to
        infinity, superpixel shapes become square/cubic.
    max_iter : int, optional
        Maximum number of iterations of k-means.
    sigma : float or (3,) array of floats, optional
        Width of Gaussian smoothing kernel for pre-processing for each
        dimension of the image. The same sigma is applied to each dimension in
        case of a scalar value. Zero means no smoothing.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension.
    convert2lab : bool, optional
        Whether the input should be converted to Lab colorspace prior to
        segmentation. For this purpose, the input is assumed to be RGB. Highly
        recommended.
    ratio : float, optional
        Synonym for `compactness`. This keyword is deprecated.

    Returns
    -------
    labels : 2D or 3D array
        Integer mask indicating segment labels.

    Raises
    ------
    ValueError
        If:
            - the image dimension is not 2 or 3 and `multichannel == False`, OR
            - the image dimension is not 3 or 4 and `multichannel == True`, OR
            - `multichannel == True` and the length of the last dimension of
            the image is not 3, OR

    Notes
    -----
    If `sigma > 0` as is default, the image is smoothed using a Gaussian kernel
    prior to segmentation.

    The image is rescaled to be in [0, 1] prior to processing.

    Images of shape (M, N, 3) are interpreted as 2D RGB images by default. To
    interpret them as 3D with the last dimension having length 3, use
    `multichannel=False`.

    References
    ----------
    .. [1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi,
        Pascal Fua, and Sabine Süsstrunk, SLIC Superpixels Compared to
        State-of-the-art Superpixel Methods, TPAMI, May 2012.

    Examples
    --------
    >>> from skimage.segmentation import slic
    >>> from skimage.data import lena
    >>> img = lena()
    >>> segments = slic(img, n_segments=100, ratio=10)
    >>> # Increasing the ratio parameter yields more square regions
    >>> segments = slic(img, n_segments=100, ratio=20)
    """

    if ratio is not None:
        msg = 'Keyword `ratio` is deprecated. Use `compactness` instead.'
        warnings.warn(msg)
        compactness = ratio

    image = img_as_float(image)
    image = np.atleast_3d(image)

    if image.ndim == 3:
        if multichannel:
            # Make 2D image 3D with depth = 1
            image = image[np.newaxis, ...]
        else:
            # Add channel as single last dimension
            image = image[..., np.newaxis]

    if not isinstance(sigma, coll.Iterable):
        sigma = np.array([sigma, sigma, sigma])
    if (sigma > 0).any():
        sigma = list(sigma) + [0]
        image = ndimage.gaussian_filter(image, sigma)

    if convert2lab:

        if not multichannel or image.shape[3] != 3:
            raise ValueError("Lab colorspace conversion requires a RGB image.")
        image = rgb2lab(image)

    depth, height, width = image.shape[:3]

    # initialize cluster centroids for desired number of segments
    grid_z, grid_y, grid_x = np.mgrid[:depth, :height, :width]
    slices = regular_grid(image.shape[:3], n_segments)
    step_z, step_y, step_x = [int(s.step) for s in slices]
    segments_z = grid_z[slices]
    segments_y = grid_y[slices]
    segments_x = grid_x[slices]

    segments_color = np.zeros(segments_z.shape + (image.shape[3],))
    segments = np.concatenate([segments_z[..., np.newaxis],
                               segments_y[..., np.newaxis],
                               segments_x[..., np.newaxis],
                               segments_color
                              ], axis=-1).reshape(-1, 3 + image.shape[3])
    segments = np.ascontiguousarray(segments)

    # we do the scaling of ratio in the same way as in the SLIC paper
    # so the values have the same meaning
    ratio = float(max((step_z, step_y, step_x))) / compactness
    image = np.ascontiguousarray(image * ratio)

    labels = _slic_cython(image, segments, max_iter)

    if labels.shape[0] == 1:
        labels = labels[0]

    return labels
