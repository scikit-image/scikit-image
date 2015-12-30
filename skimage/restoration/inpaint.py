from __future__ import print_function, division

import numpy as np
import skimage
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage.filters import laplace


def inpaint_biharmonic(img, mask, multichannel=False):
    """Inpaint masked points in image with biharmonic equations.

    Parameters
    ----------
    img : nD{+color channel} np.ndarray
        Input image.
    mask : nD np.ndarray
        Array of pixels to be inpainted. Have to be the same size as one
        of the 'img' channels. Unknown pixels has to be represented with 1, 
        known pixels - with 0.
    multichannel : boolean, optional
        If True, the last `img` dimension is considered as a color channel.

    Returns
    -------
    out : nD{+color channel} np.array
        Input image with masked pixels inpainted.

    Example
    -------
    >>> img = np.tile(np.square(np.linspace(0, 1, 5)), (5, 1))
    >>> mask = np.zeros_like(img)
    >>> mask[2, 2:] = 1
    >>> mask[1, 3:] = 1
    >>> mask[0, 4:] = 1
    >>> out = inpaint_biharmonic(img, mask)

    References
    ----------
    Algorithm is based on:
    .. [1]  N.S.Hoang, S.B.Damelin, "On surface completion and image inpainting
            by biharmonic functions: numerical aspects",
            http://www.ima.umn.edu/~damelin/biharmonic
    """
    
    def _inpaint(img, mask):
        out = np.copy(img)

        # Initialize sparse matrices
        matrix_unknown = sparse.lil_matrix((np.sum(mask), out.size))
        matrix_known = sparse.lil_matrix((np.sum(mask), out.size))

        def _get_neighborhood(idx, radii):
            bounds_lo = (idx - radii).clip(min=0)
            bounds_hi = (idx + np.add(radii, 1)).clip(max=out.shape)
            return bounds_lo, bounds_hi

        # Find indexes of masked points in flatten array
        mask_i = np.ravel_multi_index(np.where(mask), mask.shape)

        # Find masked points and prepare them to be easily enumerate over
        mask_pts = np.array(np.where(mask)).T

        # Iterate over masked points
        for mask_pt_n, mask_pt_idx in enumerate(mask_pts):
            # Get bounded neighborhood of selected radii
            b_lo, b_hi = _get_neighborhood(mask_pt_idx, radii=np.array([2]))

            # Create biharmonic coefficients ndarray
            neigh_coef = np.zeros(b_hi - b_lo)
            neigh_coef[tuple(mask_pt_idx - b_lo)] = 1
            neigh_coef = laplace(laplace(neigh_coef))

            # Iterate over masked point's neighborhood
            it_inner = np.nditer(neigh_coef, flags=['multi_index'])
            for coef in it_inner:
                if coef == 0:
                    continue
                tmp_pt_idx = np.add(b_lo, it_inner.multi_index)
                tmp_pt_i = np.ravel_multi_index(tmp_pt_idx, mask.shape)

                if mask[tuple(tmp_pt_idx)]:
                    matrix_unknown[mask_pt_n, tmp_pt_i] = coef
                else:
                    matrix_known[mask_pt_n, tmp_pt_i] = coef

        # Prepare diagonal matrix
        flat_diag_image = sparse.dia_matrix((out.flatten(), np.array([0])),
                                            shape=(out.size, out.size))

        # Calculate right hand side as a sum of known matrix's columns
        matrix_known = matrix_known.tocsr()
        rhs = -(matrix_known * flat_diag_image).sum(axis=1)

        # Solve linear system for masked points
        matrix_unknown = matrix_unknown[:, mask_i]
        matrix_unknown = sparse.csr_matrix(matrix_unknown)
        result = spsolve(matrix_unknown, rhs)

        # Handle enormous values
        # TODO: consider images in [-1:1] scale
        result[np.where(result < 0)] = 0
        result[np.where(result > 1)] = 1

        result = result.ravel()

        # Substitute masked points with inpainted versions
        for mask_pt_n, mask_pt_idx in enumerate(mask_pts):
            out[tuple(mask_pt_idx)] = result[mask_pt_n]

        return out

    img_baseshape = img.shape[:-1] if multichannel else img.shape
    if img_baseshape != mask.shape:
        raise ValueError('Input arrays have to be the same shape')
    
    if np.ma.isMaskedArray(img):
        raise TypeError('Masked arrays are not supported')

    img = skimage.img_as_float(img)
    mask = mask.astype(np.bool)
    
    if not multichannel:
        img = img.reshape(img.shape + (1,))

    out = np.zeros_like(img)
    
    for i in range(img.shape[-1]):
        out[..., i] = _inpaint(img[..., i], mask)

    if not multichannel:
        out = out.reshape(out.shape[:-1])

    return out
