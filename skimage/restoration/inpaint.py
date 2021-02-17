import functools
import operator

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.ndimage as ndi
from scipy.ndimage.filters import laplace

import skimage
from ..measure import label, regionprops_table
from ._inpaint import _build_matrix_inner


def _get_neighborhood(nd_idx, radius, nd_shape):
    bounds_lo = tuple(max(p - radius, 0) for p in nd_idx)
    bounds_hi = tuple(min(p + radius + 1, s) for p, s in zip(nd_idx, nd_shape))
    return bounds_lo, bounds_hi


def _get_neigh_coef(shape, center, coef_cache={}):
    key = (shape, center)
    if key in coef_cache:
        neigh_coef = coef_cache[key]
    else:
        # Create biharmonic coefficients ndarray
        neigh_coef = np.zeros(shape)
        neigh_coef[center] = 1
        neigh_coef = laplace(laplace(neigh_coef))
        coef_cache[key] = neigh_coef
    return neigh_coef


def _inpaint_biharmonic_single_region(image, mask, out, neigh_coef_full,
                                      coef_vals, raveled_offsets, limits):
    # Initialize sparse matrices

    nchannels = out.shape[-1]
    radius = neigh_coef_full.shape[0] // 2

    edge_mask = np.ones(mask.shape, dtype=bool)
    edge_mask[(slice(2, -2),) * mask.ndim] = 0
    boundary_mask = edge_mask * mask
    center_mask = ~edge_mask * mask

    boundary_pts = np.where(boundary_mask)
    boundary_i = np.where(boundary_mask.ravel())[0]
    center_i = np.where(center_mask.ravel())[0]
    mask_i = np.concatenate((boundary_i, center_i))

    center_pts = np.where(center_mask)
    mask_pts = tuple(
        [np.concatenate((b, c)) for b, c in zip(boundary_pts, center_pts)]
    )

    # Trick: Use convolution predetermine the number of non-zero entries in the
    #        sparse system matrix.
    structure = neigh_coef_full != 0
    mask_uint8 = mask.astype(np.uint8)
    tmp = mask * ndi.convolve(mask_uint8, structure, mode='constant')
    nnz_matrix = tmp.sum()

    # Need to estimate the number of zeros for the right hand side vector.
    # The computation below will slightly overestimate the true number of zeros
    # due to edge effects (the kernel itself gets shrunk in size near the
    # edges, but that isn't accounted for here). We can trim any trailing zeros
    # later.
    nnz_rhs_vector_max = mask.sum() - (tmp==structure.sum()).sum()

    # pre-allocate arrays storing sparse matrix indices and values
    row_idx_known = np.empty(nnz_rhs_vector_max, dtype=np.intp)
    data_known = np.empty((nnz_rhs_vector_max, nchannels), dtype=out.dtype)
    row_idx_unknown = np.empty(nnz_matrix, dtype=np.intp)
    col_idx_unknown = np.empty(nnz_matrix, dtype=np.intp)
    data_unknown = np.empty(nnz_matrix, dtype=out.dtype)

    # cache the various small, non-square Laplacians used near the boundary
    coef_cache = {}

    # Iterate over masked points near the boundary
    mask_flat = mask.ravel()
    out_flat = np.ascontiguousarray(out.reshape((-1, nchannels)))
    idx_known = 0
    idx_unknown = 0
    mask_pt_n = -1
    for mask_pt_n, mask_pt_idx in enumerate(zip(*boundary_pts)):
        # Get bounded neighborhood of selected radius
        b_lo, b_hi = _get_neighborhood(mask_pt_idx, radius, out.shape)

        coef_shape = tuple(hi - lo for lo, hi in zip(b_lo, b_hi))
        coef_center = tuple(p - lo for p, lo in zip(mask_pt_idx, b_lo))
        neigh_coef = _get_neigh_coef(coef_shape, coef_center, coef_cache)

        # Iterate over masked point's neighborhood
        it_inner = np.nditer(neigh_coef, flags=['multi_index'])
        vals_known = [0] * nchannels
        for coef in it_inner:
            if coef == 0:
                continue
            tmp_pt_idx = np.add(b_lo, it_inner.multi_index)
            tmp_pt_i = np.ravel_multi_index(tmp_pt_idx, mask.shape)

            if mask_flat[tmp_pt_i]:
                row_idx_unknown[idx_unknown] = mask_pt_n
                col_idx_unknown[idx_unknown] = tmp_pt_i
                data_unknown[idx_unknown] = coef
                idx_unknown += 1
            else:
                for ch in range(nchannels):
                    vals_known[ch] -= coef * out_flat[tmp_pt_i, ch]

        if any(val_known != 0 for val_known in vals_known):
            row_idx_known[idx_known] = mask_pt_n
            for ch in range(nchannels):
                data_known[idx_known, ch] = vals_known[ch]
            idx_known += 1

    # Now use an efficient Cython-based implementation for all interior points
    row_start = mask_pt_n + 1
    known_start_idx = idx_known
    unknown_start_idx = idx_unknown

    nnz_rhs = _build_matrix_inner(
        # starting indices
        row_start, known_start_idx, unknown_start_idx,
        # input arrays
        center_i, raveled_offsets, coef_vals, mask_flat.astype(np.int16),
        out_flat,
        # output arrays
        row_idx_known, data_known, row_idx_unknown, col_idx_unknown,
        data_unknown
    )

    # trim RHS vector values and indices to the exact length
    row_idx_known = row_idx_known[:nnz_rhs]
    data_known = data_known[:nnz_rhs, :]

    # form sparse matrix of unknown values
    sp_shape = (np.sum(mask), out.size)
    matrix_unknown = sparse.coo_matrix(
        (data_unknown, (row_idx_unknown, col_idx_unknown)), shape=sp_shape
    ).tocsr()

    # Solve linear system for masked points
    matrix_unknown = matrix_unknown[:, mask_i]

    col_idx_known = np.zeros_like(row_idx_known)
    for ch in range(nchannels):
        # form sparse vector representing the right hand side
        rhs = sparse.coo_matrix(
            (data_known[:, ch], (row_idx_known, col_idx_known)),
            shape=(np.sum(mask), 1)
        ).tocsr()

        result = spsolve(matrix_unknown, rhs)

        # Handle enormous values on a per-channel basis
        result = np.clip(result, *limits[ch])

        out[..., ch][mask_pts] = result.ravel()

    return out


def inpaint_biharmonic(image, mask, multichannel=False, *,
                       split_into_regions=False):
    """Inpaint masked points in image with biharmonic equations.

    Parameters
    ----------
    image : (M[, N[, ..., P]][, C]) ndarray
        Input image.
    mask : (M[, N[, ..., P]]) ndarray
        Array of pixels to be inpainted. Have to be the same shape as one
        of the 'image' channels. Unknown pixels have to be represented with 1,
        known pixels - with 0.
    multichannel : boolean, optional
        If True, the last `image` dimension is considered as a color channel,
        otherwise as spatial.

    Returns
    -------
    out : (M[, N[, ..., P]][, C]) ndarray
        Input image with masked pixels inpainted.

    References
    ----------
    .. [1]  N.S.Hoang, S.B.Damelin, "On surface completion and image inpainting
            by biharmonic functions: numerical aspects",
            :arXiv:`1707.06567`
    .. [2]  C. K. Chui and H. N. Mhaskar, MRA Contextual-Recovery Extension of
            Smooth Functions on Manifolds, Appl. and Comp. Harmonic Anal.,
            28 (2010), 104-113,
            :DOI:`10.1016/j.acha.2009.04.004`

    Examples
    --------
    >>> img = np.tile(np.square(np.linspace(0, 1, 5)), (5, 1))
    >>> mask = np.zeros_like(img)
    >>> mask[2, 2:] = 1
    >>> mask[1, 3:] = 1
    >>> mask[0, 4:] = 1
    >>> out = inpaint_biharmonic(img, mask)
    """

    if image.ndim < 1:
        raise ValueError('Input array has to be at least 1D')

    img_baseshape = image.shape[:-1] if multichannel else image.shape
    if img_baseshape != mask.shape:
        raise ValueError('Input arrays have to be the same shape')

    if np.ma.isMaskedArray(image):
        raise TypeError('Masked arrays are not supported')

    image = skimage.img_as_float(image)
    mask = mask.astype(bool)

    if split_into_regions:
        # Split inpainting mask into independent regions
        mask = mask.astype(bool, copy=False)
        kernel = ndi.morphology.generate_binary_structure(mask.ndim, 1)
        mask_dilated = ndi.morphology.binary_dilation(mask, structure=kernel)
        mask_labeled, num_labels = label(mask_dilated, return_num=True)
        mask_labeled *= mask

        # create slices for cropping to each labeled region
        props = regionprops_table(mask_labeled)
        ndim = len(img_baseshape)
        def _get_ax_slices(axis, ndim, size, radius=2):
            return [
                slice(max(ax_start - radius, 0), min(ax_end + radius, size))
                for ax_start, ax_end
                in zip(props[f'bbox-{axis}'], props[f'bbox-{axis + ndim}'])
            ]
        bbox_slices = list(
            zip(*[_get_ax_slices(ax, ndim, img_baseshape[ax])
                  for ax in range(ndim)])
        )

    if not multichannel:
        image = image[..., np.newaxis]
    out = np.copy(image)

    # Create biharmonic coefficients ndarray
    radius = 2
    coef_shape = (2*radius + 1,) * mask.ndim
    coef_center = (radius,) * mask.ndim
    neigh_coef_full = _get_neigh_coef(coef_shape, coef_center)

    # stride for the last spatial dimension
    channel_stride_bytes = out.strides[-2]

    # offsets to all neighboring non-zero elements in the footprint
    idx_coef = np.where(neigh_coef_full)
    coef_vals = neigh_coef_full[idx_coef]
    offsets = tuple(c - radius for c in idx_coef)

    # determine per-channel intensity limits
    limits = []
    for ch in range(out.shape[-1]):
        known_points= image[..., ch][~mask]
        limits.append((np.min(known_points), np.max(known_points)))

    if split_into_regions:
        for idx_region in range(1, num_labels + 1):

            # extract only the region surrounding the label of interest
            roi_sl = bbox_slices[idx_region - 1]
            mask_region = mask_labeled[roi_sl] == idx_region
            # add slice for axes
            roi_sl =  tuple(list(roi_sl) + [slice(None,)])
            otmp = out[roi_sl].copy()

            # compute raveled offsets for the ROI
            ostrides = tuple(s // channel_stride_bytes
                             for s in otmp[..., 0].strides)
            raveled_offsets = tuple(
                ax_off * ax_stride
                for ax_off, ax_stride in zip(offsets, ostrides)
            )
            raveled_offsets = functools.reduce(operator.add, raveled_offsets)

            _inpaint_biharmonic_single_region(
                image[roi_sl], mask_region, otmp,
                neigh_coef_full, coef_vals, raveled_offsets, limits
            )
            # assign output to the
            out[roi_sl] = otmp
    else:
        ostrides = tuple(s // channel_stride_bytes
                         for s in out[..., 0].strides)

        # compute raveled offsets for output image
        raveled_offsets = tuple(
            ax_off * ax_stride
            for ax_off, ax_stride in zip(offsets, ostrides)
        )
        raveled_offsets = functools.reduce(operator.add, raveled_offsets)

        _inpaint_biharmonic_single_region(
            image, mask, out, neigh_coef_full, coef_vals, raveled_offsets,
            limits
        )

    if not multichannel:
        out = out[..., 0]

    return out
