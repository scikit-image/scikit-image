# coding=utf-8

import collections as coll
import numpy as np
from scipy import ndimage
import warnings

from ..util import img_as_float, regular_grid
from ..color import rgb2lab, gray2rgb, guess_spatial_dimensions
from ._slic import _slic_cython


def slic(image, n_segments=100, compactness=10., max_iter=10, sigma=1,
         multichannel=None, convert2lab=True, ratio=None):
    """Segments image using k-means clustering in Color-(x,y) space.

    Parameters
    ----------
    image : (width, height [, depth] [, 3]) ndarray
        Input image, which can be 2D or 3D, and grayscale or multi-channel
        (see `multichannel` parameter).
    n_segments : int, optional (default: 100)
        The (approximate) number of labels in the segmented output image.
    compactness : float, optional (default: 10)
        Balances color-space proximity and image-space proximity. Higher
        values give more weight to image-space. As `compactness` tends to
        infinity, superpixel shapes become square/cubic.
    max_iter : int, optional (default: 10)
        Maximum number of iterations of k-means.
    sigma : float or array of floats, optional (default: 1)
        Width of Gaussian smoothing kernel for pre-processing for each
        dimension of the image. The same sigma is applied to each dimension in
        case of a scalar value. Zero means no smoothing.
    multichannel : bool, optional (default: None)
        Whether the last axis of the image is to be interpreted as multiple
        channels. Only 3 channels are supported. If `None`, the function will
        attempt to guess this, and raise a warning if ambiguous, when the
        array has shape (M, N, 3).
    convert2lab : bool, optional (default: True)
        Whether the input should be converted to Lab colorspace prior to
        segmentation. For this purpose, the input is assumed to be RGB. Highly
        recommended.
    ratio : float, optional
        Synonym for `compactness`. This keyword is deprecated.

    Returns
    -------
    labels : (width, height, depth) array
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
    spatial_dims = guess_spatial_dimensions(image)
    if spatial_dims is None and multichannel is None:
        msg = ("Images with dimensions (M, N, 3) are interpreted as 2D+RGB" +
                   " by default. Use `multichannel=False` to interpret as " +
                   " 3D image with last dimension of length 3.")
        warnings.warn(RuntimeWarning(msg))
        multichannel = True
    elif multichannel is None:
        multichannel = (spatial_dims + 1 == image.ndim)
    if ((not multichannel and image.ndim not in [2, 3]) or
            (multichannel and image.ndim not in [3, 4]) or
            (multichannel and image.shape[-1] != 3)):
        ValueError("Only 1- or 3-channel 2- or 3-D images are supported.")

    image = img_as_float(image)
    image = np.atleast_3d(image)

    if image.ndim == 3:
        # See 2D RGB image as 3D RGB image with Z = 1
        image = image[np.newaxis, ...]

    if not isinstance(sigma, coll.Iterable):
        sigma = np.array([sigma, sigma, sigma, 0])
    if (sigma > 0).any():
        image = ndimage.gaussian_filter(image, sigma)

    if image.shape[3] == 3 and convert2lab:
        image = rgb2lab(image)

    depth, height, width = image.shape[:3]

    # initialize cluster centroids for desired number of segments
    grid_z, grid_y, grid_x = np.mgrid[:depth, :height, :width]
    slices = regular_grid(image.shape[:3], n_segments)
    step_z, step_y, step_x = [int(s.step) for s in slices]
    clusters_z = grid_z[slices]
    clusters_y = grid_y[slices]
    clusters_x = grid_x[slices]

    clusters_color = np.zeros(clusters_z.shape + (image.shape[3],))
    clusters = np.concatenate([clusters_z[..., np.newaxis],
                               clusters_y[..., np.newaxis],
                               clusters_x[..., np.newaxis],
                               clusters_color
                              ], axis=-1).reshape(-1, 3 + image.shape[3])
    clusters = np.ascontiguousarray(clusters)

    # we do the scaling of ratio in the same way as in the SLIC paper
    # so the values have the same meaning
    ratio = float(max((step_z, step_y, step_x))) / compactness
    image = image * ratio

    labels = _slic_cython(image, clusters, max_iter)

    if labels.shape[0] == 1:
        labels = labels[0]

    return labels
