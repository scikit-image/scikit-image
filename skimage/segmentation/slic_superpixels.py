import warnings
import functools
import collections as coll
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist, squareform

from ._slic import (_slic_cython, _enforce_label_connectivity_cython)
from ..util import img_as_float, regular_grid
from ..color import rgb2lab


def _get_mask_centroids(mask, n_centroids, spacing=None):
    """Find regularly spaced centroids on a mask.

    Parameters
    ----------
    mask : 3D ndarray
        The mask within which the centroids must be positioned.
    n_centroids : int
        The number of centroids to be returned.
    spacing : sequence of same, optional,
        Spacing of elements along each dimension.

    Returns
    -------
    centroids : 2D ndarray
        The coordinates of the centroids with shape (n_centroids, 3).
    steps : 1D ndarray
        The approximate distance between two seeds in all dimensions.

    """

    # Get tight ROI around the mask to optimize
    coord = np.asarray(np.nonzero(mask))
    bbox = coord.min(-1), coord.max(-1) + 1
    roi = mask[bbox[0][0]: bbox[1][0],
               bbox[0][1]: bbox[1][1],
               bbox[0][2]: bbox[1][2]].copy()

    if spacing is None:
        # Chamfer distance transform (faster then euclidean distance).
        dist_map = np.empty_like(roi, dtype=np.int32)
        update_dist_map = functools.partial(ndi.distance_transform_cdt,
                                            distances=dist_map)
    else:
        # Exact euclidean distance transform.
        dist_map = np.empty_like(roi, dtype=np.float64)
        spacing = np.ascontiguousarray(spacing, dtype=np.float64)
        update_dist_map = functools.partial(ndi.distance_transform_edt,
                                            distances=dist_map,
                                            sampling=spacing)

    centroids = np.repeat([bbox[0]], n_centroids, axis=0)

    # Iteratively place centroids in the mask
    for idx in range(n_centroids):
        # Compute distance map
        update_dist_map(roi)

        # Set the point of furthest distance to be a centroid
        coord = dist_map.argmax()
        centroids[idx, :] += np.unravel_index(coord, roi.shape)

        # Remove the detected centroid from the mask
        roi.ravel()[coord] = 0

    # Compute the minimum distance of each centroid to the others
    dist = squareform(pdist(centroids))
    np.fill_diagonal(dist, np.inf)
    closest_pts = dist.argmin(-1)
    steps = abs(centroids - centroids[closest_pts, :]).mean(0)

    return centroids, steps


def _get_grid_centroids(image, n_centroids):
    """Find regularly spaced centroids on the image.

    Parameters
    ----------
    image : 2D, 3D or 4D ndarray
        Input image, which can be 2D or 3D, and grayscale or
        multichannel.
    n_centroids : int
        The (approximate) number of centroids to be returned.

    Returns
    -------
    centroids : 2D ndarray
        The coordinates of the centroids with shape (~n_centroids, 3).
    steps : 1D ndarray
        The approximate distance between two seeds in all dimensions.

    """
    d, h, w = image.shape[:3]

    grid_z, grid_y, grid_x = np.mgrid[:d, :h, :w]
    slices = regular_grid(image.shape[:3], n_centroids)

    centroids_z = grid_z[slices].ravel()[..., np.newaxis]
    centroids_y = grid_y[slices].ravel()[..., np.newaxis]
    centroids_x = grid_x[slices].ravel()[..., np.newaxis]

    centroids = np.concatenate([centroids_z, centroids_y, centroids_x],
                               axis=-1)

    steps = np.asarray([float(s.step) if s.step is not None else 1.0
                        for s in slices])
    return centroids, steps


def slic(image, n_segments=100, compactness=10., max_iter=10, sigma=0,
         spacing=None, multichannel=True, convert2lab=None,
         enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3,
         slic_zero=False, start_label=None, mask=None):
    """Segments image using k-means clustering in Color-(x,y,z) space.

    Parameters
    ----------
    image : 2D, 3D or 4D ndarray
        Input image, which can be 2D or 3D, and grayscale or multichannel
        (see `multichannel` parameter).
    n_segments : int, optional
        The (approximate) number of labels in the segmented output image.
    compactness : float, optional
        Balances color proximity and space proximity. Higher values give
        more weight to space proximity, making superpixel shapes more
        square/cubic. In SLICO mode, this is the initial compactness.
        This parameter depends strongly on image contrast and on the
        shapes of objects in the image. We recommend exploring possible
        values on a log scale, e.g., 0.01, 0.1, 1, 10, 100, before
        refining around a chosen value.
    max_iter : int, optional
        Maximum number of iterations of k-means.
    sigma : float or (3,) array-like of floats, optional
        Width of Gaussian smoothing kernel for pre-processing for each
        dimension of the image. The same sigma is applied to each dimension in
        case of a scalar value. Zero means no smoothing.
        Note, that `sigma` is automatically scaled if it is scalar and a
        manual voxel spacing is provided (see Notes section).
    spacing : (3,) array-like of floats, optional
        The voxel spacing along each image dimension. By default, `slic`
        assumes uniform spacing (same voxel resolution along z, y and x).
        This parameter controls the weights of the distances along z, y,
        and x during k-means clustering.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension.
    convert2lab : bool, optional
        Whether the input should be converted to Lab colorspace prior to
        segmentation. The input image *must* be RGB. Highly recommended.
        This option defaults to ``True`` when ``multichannel=True`` *and*
        ``image.shape[-1] == 3``.
    enforce_connectivity : bool, optional
        Whether the generated segments are connected or not
    min_size_factor : float, optional
        Proportion of the minimum segment size to be removed with respect
        to the supposed segment size ```depth*width*height/n_segments```
    max_size_factor : float, optional
        Proportion of the maximum connected segment size. A value of 3 works
        in most of the cases.
    slic_zero : bool, optional
        Run SLIC-zero, the zero-parameter mode of SLIC. [2]_
    start_label: int, optional
        The labels' index start. Should be 0 or 1.
    mask : 2D ndarray, optional
        if provided, superpixels are computed only where mask is True,
        and seed points are iteratively placed over the masked area to
        ensure their homogeneous spacial distribution over the mask
        (see [3]_ for more details).

    Returns
    -------
    labels : 2D or 3D array
        Integer mask indicating segment labels.

    Raises
    ------
    ValueError
        If ``convert2lab`` is set to ``True`` but the last array
        dimension is not of length 3.
    ValueError
        If ``start_label`` is not 0 or 1.

    Notes
    -----
    * If `sigma > 0`, the image is smoothed using a Gaussian kernel prior to
      segmentation.

    * If `sigma` is scalar and `spacing` is provided, the kernel width is
      divided along each dimension by the spacing. For example, if ``sigma=1``
      and ``spacing=[5, 1, 1]``, the effective `sigma` is ``[0.2, 1, 1]``. This
      ensures sensible smoothing for anisotropic images.

    * The image is rescaled to be in [0, 1] prior to processing.

    * Images of shape (M, N, 3) are interpreted as 2D RGB images by default. To
      interpret them as 3D with the last dimension having length 3, use
      `multichannel=False`.

    * If spacing is None and mask is provided, chamfer distance
      transform is used to homogeneously place seed points over the
      mask, otherwise, the exact euclidian distance transform is used.

    * `start_label` is introduced to handle the issue [4]_. The labels
      indexing starting at 0 will be deprecated in future versions. If
      `mask` is not `None` labels indexing starts at 1 and masked area
      is set to 0.

    References
    ----------
    .. [1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi,
        Pascal Fua, and Sabine Süsstrunk, SLIC Superpixels Compared to
        State-of-the-art Superpixel Methods, TPAMI, May 2012.
    .. [2] http://ivrg.epfl.ch/research/superpixels#SLICO
    .. [3] https://arxiv.org/pdf/1606.09518.pdf
    .. [4] https://github.com/scikit-image/scikit-image/issues/3722

    Examples
    --------
    >>> from skimage.segmentation import slic
    >>> from skimage.data import astronaut
    >>> img = astronaut()
    >>> segments = slic(img, n_segments=100, compactness=10)

    Increasing the compactness parameter yields more square regions:

    >>> segments = slic(img, n_segments=100, compactness=20)

    """

    image = img_as_float(image)
    use_mask = mask is not None
    dtype = image.dtype

    is_2d = False

    if image.ndim == 2:
        # 2D grayscale image
        image = image[np.newaxis, ..., np.newaxis]
        is_2d = True
    elif image.ndim == 3 and multichannel:
        # Make 2D multichannel image 3D with depth = 1
        image = image[np.newaxis, ...]
        is_2d = True
    elif image.ndim == 3 and not multichannel:
        # Add channel as single last dimension
        image = image[..., np.newaxis]

    if multichannel and (convert2lab or convert2lab is None):
        if image.shape[-1] != 3 and convert2lab:
            raise ValueError("Lab colorspace conversion requires a RGB image.")
        elif image.shape[-1] == 3:
            image = rgb2lab(image)

    if start_label is None:
        if use_mask:
            start_label = 1
        else:
            warnings.warn("Labels' indexing start from 0. " +
                          "In future version it will start from 1. " +
                          "To disable this warning, explicitely " +
                          "specify the start_label parameter.",
                          FutureWarning, stacklevel=2)
            start_label = 0

    if start_label not in [0, 1]:
        raise ValueError("start_label should be 0 or 1.")

    # initialize cluster centroids for desired number of segments
    update_centroids = False
    if use_mask:
        mask = np.ascontiguousarray(mask, dtype=np.bool).view('uint8')
        if mask.ndim == 2:
            mask = np.ascontiguousarray(mask[np.newaxis, ...])
        if mask.shape != image.shape[:3]:
            raise ValueError("image and mask should have the same shape.")
        # Step 1 of the algorithm [3]_
        centroids, steps = _get_mask_centroids(mask, n_segments, spacing)
        update_centroids = True
    else:
        centroids, steps = _get_grid_centroids(image, n_segments)

    if spacing is None:
        spacing = np.ones(3, dtype=dtype)
    elif isinstance(spacing, (list, tuple)):
        spacing = np.ascontiguousarray(spacing, dtype=dtype)

    if not isinstance(sigma, coll.Iterable):
        sigma = np.array([sigma, sigma, sigma], dtype=dtype)
        sigma /= spacing.astype(dtype)
    elif isinstance(sigma, (list, tuple)):
        sigma = np.array(sigma, dtype=dtype)
    if (sigma > 0).any():
        # add zero smoothing for multichannel dimension
        sigma = list(sigma) + [0]
        image = ndi.gaussian_filter(image, sigma)

    n_centroids = centroids.shape[0]
    segments = np.ascontiguousarray(np.concatenate(
        [centroids, np.zeros((n_centroids, image.shape[3]))],
        axis=-1))

    # Scaling of ratio in the same way as in the SLIC paper so the
    # values have the same meaning
    step = max(steps)
    ratio = 1.0 / compactness

    image = np.ascontiguousarray(image * ratio, dtype=np.double)

    if update_centroids:
        # Step 2 of the algorithm [3]_
        _slic_cython(image, mask, segments, step, max_iter, spacing,
                     slic_zero, ignore_color=True,
                     start_label=start_label)

    labels = _slic_cython(image, mask, segments, step, max_iter,
                          spacing, slic_zero, ignore_color=False,
                          start_label=start_label)

    if enforce_connectivity:
        if use_mask:
            segment_size = mask.sum() / n_centroids
        else:
            segment_size = np.prod(image.shape[:3]) / n_centroids
        min_size = int(min_size_factor * segment_size)
        max_size = int(max_size_factor * segment_size)
        labels = _enforce_label_connectivity_cython(
            labels, min_size, max_size, start_label=start_label)

    if is_2d:
        labels = labels[0]

    return labels
