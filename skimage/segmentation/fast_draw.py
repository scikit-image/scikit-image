import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import cg as cg_solver

from ..transform import resize
from ..measure import label as find_label
from ..morphology import binary_dilation, disk
from ..segmentation import random_walker
from .. import img_as_float
from ..segmentation import random_walker_segmentation

from .._shared.utils import warn

def _check_parameters(beta, image, labels, target_label):
    """Assert validity of `beta`, `labels` and `target_label` inputs"""

    if (image.shape != labels.shape):
        raise ValueError('image and labels must be the same shape')

    if downsampled_size > min(image.shape):
        warn('The size of the downsampled image if larger than the '
             'original image. The computation time could be affected.')

    assert (beta > 0), 'beta should be positive.'

    if target_label not in np.unique(labels):
        warn('The target label '+str(target_label)+ \
             ' does not match any label')
        return 1
    if (labels != 0).all():
        warn('The segmentation is computed on the unlabeled area '
             '(labels == 0). No zero valued areas in labels were '
             'found. Returning provided labels.')
        return -1
    return 1

def _build_graph(image, beta, spacing, multichannel):
    # This algorithm expects 4-D arrays of floats, where the first three
    # dimensions are spatial and the final denotes channels. 2-D images have
    # a singleton placeholder dimension added for the third spatial dimension,
    # and single channel images likewise have a singleton added for channels.
    # The following block ensures valid input and coerces it to the correct
    # form.
    if not multichannel:
        if image.ndim < 2 or image.ndim > 3:
            raise ValueError('For non-multichannel input, image must be of '
                             'dimension 2 or 3.')
        dims = image.shape  # To reshape final labeled result
        image = np.atleast_3d(img_as_float(image))[..., np.newaxis]
    else:
        if image.ndim < 3:
            raise ValueError('For multichannel input, image must have 3 or 4 '
                             'dimensions.')
        dims = image[..., 0].shape  # To reshape final labeled result
        image = img_as_float(image)
        if image.ndim == 3:  # 2D multispectral, needs singleton in 3rd axis
            image = image[:, :, np.newaxis, :]

    # Spacing kwarg checks
    if spacing is None:
        spacing = np.asarray((1.,) * 3)
    elif len(spacing) == len(dims):
        if len(spacing) == 2:  # Need a dummy spacing for singleton 3rd dim
            spacing = np.r_[spacing, 1.]
        else:                  # Convert to array
            spacing = np.asarray(spacing)
    else:
        raise ValueError('Input argument `spacing` incorrect, should be an '
                         'iterable with one number per spatial dimension.')

    ## Compute the graph's Laplacian for both the full resolution `L`
    ## and the down-sampled `ds_L` images
    L = random_walker_segmentation._build_laplacian(image, spacing,
                                        beta=beta, multichannel=multichannel)

    return L



def _compute_relevance_map(labels, ds_labels, full_to_ds_ratio):
    """Computes the relevance map from labels and initialize
    down-sampled label image `ds_labels`.

    The relevance map assumes that the object boundary is more likely to
    be located somwhere between different label categories, i.e. two labels
    of the same category generate a low energy, where two labels of
    different categories generate high energy, therefore precluding
    redundent label information. The relevance map is computed using the
    sum of the distance transforms for each label category."""

    ds_relevance_map = 0
    for i in np.unique(labels):
        if i != 0:
            # 2.1- Compute the coarse label image
            y, x = np.where(labels == i)
            ds_labels[np.int32(y*full_to_ds_ratio[0]),
                      np.int32(x*full_to_ds_ratio[1])] = i
            # 2.2- Compute the energy map
            M = np.ones_like(ds_labels)
            M[ds_labels == i] = 0
            distance_map = distance_transform_edt(M)
            ds_relevance_map +=  distance_map

    # 2.3- Normalize the energy map and compute the ROI
    ds_relevance_map = ds_relevance_map / ds_relevance_map.max()
    return ds_relevance_map, ds_labels

def _coarse_random_walker(target_label, ds_L, ds_maskROI, ds_labels):
    """Performs a coarse random walker segmentation on the down-sampled
    image.

    Parameters
    ----------
    target_label : int
        The label category to comput the segmentation for. `labels` should
        contain at least one pixel with value `target_label`

    Returns
    -------
    ds_probability : ndarray of the same size as `ds_image`
        Array of the probability between [0.0, 1.0], of each pixel
        to belong to `target_label`
    """
    unlabeled = np.ravel_multi_index(np.where((ds_labels == 0) & \
                    (ds_maskROI)), ds_labels.shape)
    labeled = np.ravel_multi_index(np.where((ds_labels > 0) & \
                    (ds_maskROI)), ds_labels.shape)
    # 3.1- Preparing the right handside of the equation BT xs
    B = ds_L[unlabeled][:, labeled]
    mask = ds_labels.flatten()[labeled] == target_label
    fs = csr_matrix(mask)
    fs = fs.transpose()
    rhs = B * fs
    # 3.2- Preparing the left handside of the equation Lu
    Lu = ds_L[unlabeled][:, unlabeled]
    # 3.3- Solve the linear equation Lu xu = -BT xs
    xu = cg_solver(Lu, -rhs.todense(), tol=1e-3, maxiter=120)[0]

    ds_probability = np.zeros_like(ds_labels, dtype=np.float32)
    ds_probability[(ds_labels == 0) & (ds_maskROI)] = xu
    ds_probability[(ds_labels == target_label) & (ds_maskROI)] = 1

    return ds_probability

def _refinement_random_walker(target_label, L, labels, maskROI, ds_labels):
    """Performs a random walker segmentation over a small region
    `self.maskROI` around the coarse contour on the full resolution image.

    Requires `target_label` and `labels`

    Returns
    -------
    probability : ndarray of the same size as `image`
        Array of the probability between [0.0, 1.0], of each pixel
        to belong to `target_label`
    """
    labeledImage = find_label(maskROI, background=True)
    ds_added_labels = ds_labels
    # for pixels outside the refinement region (ring), if their connected
    # region contains `target_label` pixels, than all pixels of the region
    # should be labeled as `target_label`.
    # TODO : this code could be optimized
    for area in np.unique(labeledImage):
        if area != -1:
            if target_label in ds_labels[labeledImage == area]:
                ds_added_labels[labeledImage == area] = target_label

    added_labels = resize(ds_added_labels, labels.shape, order=0,
                          preserve_range=True)

    maskROI = resize(maskROI, labels.shape, order=0,
                          preserve_range=True)
    maskROI = maskROI.astype(np.bool)

    # Extract labelled and unlabelled vertices
    m_unlabeled = (added_labels == 0) & (maskROI)
    m_foreground = (added_labels == target_label)

    unlabeled = np.ravel_multi_index(np.where(m_unlabeled), labels.shape)
    labeled = np.ravel_multi_index(np.where((m_foreground) | \
                             (added_labels > 0)), labels.shape)

    # Preparing the right handside of the equation BT xs
    B = L[unlabeled][:, labeled]
    mask = (added_labels[added_labels > 0]).flatten() == target_label
    fs = csr_matrix(mask).transpose()
    rhs = B * fs

    # Preparing the left handside of the equation Lu
    Lu = L[unlabeled][:, unlabeled]

    # Solve the linear equation Lu xu = -BT xs
    xu = cg_solver(Lu, -rhs.todense(), tol=1e-3, maxiter=120)[0]

    probability = np.zeros_like(labels, dtype=np.float32)
    probability[m_unlabeled] = xu
    probability[m_foreground] = 1

    return probability

def fast_draw(image, labels, target_label=1, beta=300, downsampled_size=100,
              k=1.0, spacing=None, multichannel=False):
    """A fast segmentation approach based on the random walker algorithm.

    FastDRaW implemented for 2D images as described in [1].
    Like the random walker algorithm, FastDRaW uses `labels` that can be drawn
    by the user on the image (e.g., few pixels inside the object labeled as
    foreground and few pixels outside the object labeled as background).
    The algorithm performs in a two-step coarse-to-fine segmetnation. In the
    first step, a random walker segmentation is performed on a small
    (down-sampled) version of the image to obtain a coarse segmentation
    contour. In the second step, the result is refined by applying a second
    random walker segmentation over a narrow strip around the coarse contour.

    Parameters
    ----------
    image : array_like
        Image to be segmented. If `image` is multi-channel `multichannel` must
        be set to True.
    labels : array of ints, of same shape as `image`
        Array of seed markers labeled with different positive integers
        (each label category is represented with an integer value).
        Zero-labeled pixels represent unlabeled pixels.
    target_label : int, default value 1
        The label category to comput the segmentation for. `labels` should
        contain at least one pixel with value `target_label`.
    beta : float, default value 300
        Penalization coefficient for the random walker motion
        (the greater `beta`, the more difficult the diffusion).
    downsampled_size : int, default value 100
        The size of the down-sampled image. Should be smaller than the
        size of the original image. Recommended values between 100 and 200.
    k : float, default value 1.0
        Control the size of the region of interest (ROI). Large positive
        value of `k` allows a larger ROI.
    spacing : iterable of floats, default None
        Spacing between voxels in each spatial dimension. If `None`, then
        the spacing between pixels/voxels in each dimension is assumed 1.
    multichannel : bool, default False
        If True, input image is parsed as multichannel image (see 'image' above
        for proper input format in this case)

    Returns
    ----------
    output : ndarray of bools of same shape as `image`
        in which pixels have been labeled True if they belong to
        `target_label`, and False otherwise.

    See also
    --------
    skimage.segmentation.random_walker: random walker segmentation
        The original random walker algorithm.

    References
    ----------
    [1] H.-E. Gueziri, L. Lakhdar, M. J. McGuffin and C. Laporte,
    "FastDRaW - Fast Delineation by Random Walker: application to large
    images", MICCAI Workshop on Interactive Medical Image Computing (IMIC),
    Athens, Greece, (2016).

    Examples
    --------
    >>> from skimage.data import coins
    >>> import matplotlib.pyplot as plt
    >>> image = coins()
    >>> labels = np.zeros_like(image)
    >>> labels[[129, 199], [155, 155]] = 1 # label some pixels as foreground
    >>> labels[[162, 224], [131, 184]] = 2 # label some pixels as background
    >>> segm = fast_draw(image, labels, beta=100, downsampled_size=100)
    >>> plt.imshow(image,'gray')
    >>> plt.imshow(segm, alpha=0.7)
    """

    ## 1- Checking if inputs are valide
    if _check_parameters(beta, image, labels, target_label) == -1:
        return labels == target_label

    ## 2- Prepare graph's Laplacian matrices and down-sample the image
    ## this pre-processing can be done offline for more efficiency

    ## Build the graph's Laplacian for the original image
    L = _build_graph(image, beta, spacing, multichannel)

    ## Down-sample `image`
    ratio = float(image.shape[0]) / image.shape[1]
    ds_dim = [int(downsampled_size * ratio), ds_dim.append(downsampled_size)]
    if multichannel:
        ds_dim.append(image.shape[2])
    ds_image = resize(image, ds_dim, order=0, preserve_range=True)

    ## Build the graph's Laplacian for the down-sampled image
    ds_L = _build_graph(ds_image, beta, spacing, multichannel)

    ## Initialize the ROI to zero
    ds_maskROI = np.zeros((ds_dim[0], ds_dim[1]), dtype=np.bool)
    ## `full_to_ds_ratio` is used to convert labels from full resolution
    ## image to down-sampled image
    full_to_ds_ratio = (float(ds_dim[0])/image.shape[0],
                        float(ds_dim[1])/image.shape[1])

    ## 3- Compute the relevance map
    ds_labels = np.zeros((ds_dim[0], ds_dim[1]))
    ds_relevance_map, ds_labels = _compute_relevance_map(labels, ds_labels,
                                                         full_to_ds_ratio)

    # Threshold the energy map and append new region to the existing ROI
    threshold = ds_relevance_map.mean() + k*ds_relevance_map.std()
    ds_maskROI = ds_maskROI | (ds_relevance_map <= threshold)

    ## 4- Performe a corse RW segmentation on the down-sampled image
    ds_probability = _coarse_random_walker(target_label, ds_L,
                                           ds_maskROI, ds_labels)

    # Compute the corse segmentation result
    mask = ds_probability >= 0.5
    mask = (binary_dilation(mask, disk(1)) - mask).astype(np.bool)
    # Compute the refinement region around the corse result
    maskROI = binary_dilation(mask, disk(3))

    ## 5- Performe a fine RW segmentation on the full resolution image
    ##    only on the refinement region
    probability = _refinement_random_walker(target_label, L, labels,
                                            maskROI, ds_labels)

    # 6- threshold the probability map above 0.5
    segm = (probability >= 0.5)
    return segm


