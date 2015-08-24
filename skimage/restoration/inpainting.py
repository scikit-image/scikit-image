# -*- coding: utf-8 -*-

import numpy as np
import skimage
from scipy import sparse
from scipy.sparse.linalg import spsolve


def biharmonic_inpaint(img, mask):
    # XXX: check and/or rework examples
    """Inpaint masked points in image, using system of biharmonic
    equations.

    Parameters
    ----------
    img: 2-D ndarray
        Input image.
    mask: 2-D ndarray
        Array of pixels to be inpainted. Has to have the same size as 'img'.
        Unknown pixels has to be represented with 1, known - with 0.

    Returns
    -------
    out: 2-D ndarray
        Input image with masked pixels inpainted.

    Example
    -------
    # >>> import numpy as np
    # >>> from skimage.restoration.inpainting import biharmonic_inpaint
    # >>> image_in = np.ones((5, 5))
    # >>> image_in[:, :2] = 1
    # >>> image_in[:, 2]  = 2
    # >>> image_in[:, 3:] = 3
    # >>> image_in
    # array([[ 1.,  1.,  2.,  3.,  3.],
    #        [ 1.,  1.,  2.,  3.,  3.],
    #        [ 1.,  1.,  2.,  3.,  3.],
    #        [ 1.,  1.,  2.,  3.,  3.],
    #        [ 1.,  1.,  2.,  3.,  3.]])
    # >>> mask = np.zeros_like(image_in)
    # >>> mask[1:3, 2:] = 1
    # >>> mask
    # array([[ 0.,  0.,  0.,  0.,  0.],
    #        [ 0.,  0.,  1.,  1.,  1.],
    #        [ 0.,  0.,  1.,  1.,  1.],
    #        [ 0.,  0.,  0.,  0.,  0.],
    #        [ 0.,  0.,  0.,  0.,  0.]])
    # >>> image_in = image_in + mask * 100
    # >>> image_in
    # array([[ 1.,  1.,    2.,    3.,    3.],
    #        [ 1.,  1.,  102.,  103.,  103.],
    #        [ 1.,  1.,  102.,  103.,  103.],
    #        [ 1.,  1.,    2.,    3.,    3.],
    #        [ 1.,  1.,    2.,    3.,    3.]])
    # >>> image_out = biharmonic_inpaint(image_in, mask)
    # >>> image_out
    # array([[ 1.,  1.,  2.,  3.,  3.],
    #        [ 1.,  1.,  2.,  3.,  3.],
    #        [ 1.,  1.,  2.,  3.,  3.],
    #        [ 1.,  1.,  2.,  3.,  3.],

    References
    ----------
    Algorithm is based on:
    .. [1]  N.S.Hoang, S.B.Damelin, "On surface completion and image inpainting
            by biharmonic functions: numerical aspects",
            http://www.ima.umn.edu/~damelin/biharmonic

    Realization is based on:
    .. [2]  John D'Errico,
        http://www.mathworks.com/matlabcentral/fileexchange/4551-inpaint-nans,
            method 3
    """

    # TODO: add sufficient conditions (e.g. unknown area has to be <= 1/16)

    img = skimage.img_as_float(img)
    mask = skimage.img_as_bool(mask)

    out = np.copy(img)
    out_h, out_w = out.shape

    # Find indexes of masked points in flatten array
    mask_mn = np.array(np.where(mask)).T
    mask_i = np.ravel_multi_index(np.where(mask), mask.shape)

    # Initialize sparse matrix
    # TODO: Only points required for computation could be considered
    matrix_unknown = sparse.lil_matrix((np.sum(mask), out.size), dtype=np.int32)
    matrix_known = sparse.lil_matrix((np.sum(mask), out.size), dtype=np.int32)

    # INFO: kernels can be reworked using scipy.signal.convolve2d
    #       and np.array([0, 1, 0], [1, -4, 1], [0, 1, 0])

    # 1 stage. Find points 2 or more pixels far from bounds
    # kernel = [        1
    #               2  -8   2
    #           1  -8  20  -8   1
    #               2  -8   2
    #                   1       ]
    #
    kernel = [1, 2, -8, 2, 1, -8, 20, -8, 1, 2, -8, 2, 1]
    offset = [-2 * out_w, -out_w - 1, -out_w, -out_w + 1,
              -2, -1, 0, 1, 2, out_w - 1, out_w, out_w + 1, 2 * out_w]

    for idx, (i, (m, n)) in enumerate(zip(mask_i, mask_mn)):
        if 2 <= m <= out_h - 3 and 2 <= n <= out_w - 3:
            for k, o in zip(kernel, offset):
                if i + o in mask_i:
                    matrix_unknown[idx, i + o] = k
                else:
                    matrix_known[idx, i + o] = k

    # 2 stage. Find points 1 pixel far from bounds
    # kernel = [     1
    #            1  -4  1
    #                1     ]
    #
    kernel = [1, 1, -4, 1, 1]
    offset = [-out_w, -1, 0, 1, out_w]

    for idx, (i, (m, n)) in enumerate(zip(mask_i, mask_mn)):
        if (m in [1, out_h - 2] and 1 <= n <= out_h - 2) or \
           (n in [1, out_w - 2] and 1 <= m <= out_w - 2):
            for k, o in zip(kernel, offset):
                if i + o in mask_i:
                    matrix_unknown[idx, i + o] = k
                else:
                    matrix_known[idx, i + o] = k

    # 3 stage. Find points on the horizontal bounds
    # kernel = [ 1, -2, 1 ]
    #
    kernel = [1, -2, 1]
    offset = [-1, 0, 1]

    for idx, (i, (m, n)) in enumerate(zip(mask_i, mask_mn)):
        if m in [0, out_h - 1] and 1 <= n <= out_w - 1:
            for k, o in zip(kernel, offset):
                if i + o in mask_i:
                    matrix_unknown[idx, i + o] = k
                else:
                    matrix_known[idx, i + o] = k

    # 4 stage. Find points on the vertical bounds
    # kernel = [  1,
    #            -2,
    #             1  ]
    #
    kernel = [1, -2, 1]
    offset = [-out_w, 0, out_w]

    for idx, (i, (m, n)) in enumerate(zip(mask_i, mask_mn)):
        if n in [0, out_w - 1] and 1 <= m <= out_h - 1:
            for k, o in zip(kernel, offset):
                if i + o in mask_i:
                    matrix_unknown[idx, i + o] = k
                else:
                    matrix_known[idx, i + o] = k

    # Prepare diagonal matrix
    flat_diag_image = sparse.dia_matrix((out.flatten(), np.array([0])),
                                        shape=(out.size, out.size))

    # Calculate right hand side as a sum of known matrix columns
    matrix_known = matrix_known.tocsr()
    rhs = -(matrix_known * flat_diag_image).sum(axis=1)

    # Solve linear system over defect points
    matrix_unknown = matrix_unknown[:, mask_i]
    matrix_unknown = sparse.csr_matrix(matrix_unknown)
    result = spsolve(matrix_unknown, rhs)

    # Handle enormous values
    out[np.where(out < -1)] = -1
    out[np.where(out > 1)] = 1

    # Put calculated points into the image
    for idx, (m, n) in enumerate(mask_mn):
        out[m, n] = result[idx]

    return skimage.img_as_uint(out)
