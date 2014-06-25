# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

__author__ = "Egor Panfilov"


def biharmonic_inpaint(img, mask):
    """Inpaint masked points in image, using system of biharmonic
    equations.

    Parameters
    ----------
    img: 2-D ndarray
        Input image.
    mask: 2-D ndarray
        Array of pixels to inpaint. Should have the same size as 'img'.
        Unknown pixels should be represented with 1, known - with 0.

    Returns
    -------
    out: 2-D ndarray
        Image with unknown regions inpainted.

    Example
    -------
    >>> import numpy as np
    >>> from inpainting import biharmonic_inpaint
    >>> image_in = np.ones((5, 5))
    >>> image_in[:, :2] = 1
    >>> image_in[:, 2]  = 2
    >>> image_in[:, 3:] = 3
    >>> image_in
    array([[ 1.,  1.,  2.,  3.,  3.],
           [ 1.,  1.,  2.,  3.,  3.],
           [ 1.,  1.,  2.,  3.,  3.],
           [ 1.,  1.,  2.,  3.,  3.],
           [ 1.,  1.,  2.,  3.,  3.]])
    >>> mask = np.zeros_like(image_in)
    >>> mask[1:3, 2:] = 1
    >>> mask
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.],
           [ 0.,  0.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> image_in = image_in + mask * 100
    >>> image_in
    array([[ 1.,  1.,    2.,    3.,    3.],
           [ 1.,  1.,  102.,  103.,  103.],
           [ 1.,  1.,  102.,  103.,  103.],
           [ 1.,  1.,    2.,    3.,    3.],
           [ 1.,  1.,    2.,    3.,    3.]])
    >>> image_out = biharmonic_inpaint(image_in, mask)
    >>> image_out
    array([[ 1.,  1.,  2.,  3.,  3.],
           [ 1.,  1.,  2.,  3.,  3.],
           [ 1.,  1.,  2.,  3.,  3.],
           [ 1.,  1.,  2.,  3.,  3.],

    References
    ----------
    Algorithm is based on:
    .. [1] N.S.Hoang, S.B.Damelin, "On surface completion and image inpainting
           by biharmonic functions: numerical aspects",
           http://www.ima.umn.edu/~damelin/biharmonic

    Realization is based on:
    .. [2] John D'Errico,
           http://www.mathworks.com/matlabcentral/fileexchange/4551-inpaint-nans,
           method 3
    """

    out = np.copy(img)
    img_height, img_width = out.shape
    img_pixnum = img_height * img_width

    i, j = np.where(mask)
    defect_points_num = i.size

    # Defect points position in flatten array
    n = [i[n] * img_width + j[n] for n in range(defect_points_num)]
    defect_list = {e[0]: e[1:] for e in zip(n, i, j)}

    # Create list of neighbor points to be used
    # Possible relative indexes
    eps = [            [-2, 0],
             [-1, -1], [-1, 0], [-1, 1],
    [0, -2], [ 0, -1],          [ 0, 1], [0, 2],
             [ 1, -1], [ 1, 0], [ 1, 1],
                       [ 2, 0]
    ]

    neighbor_list = {}
    for defect_pnt in defect_list:
        for eps_pnt in eps:
            # Check if point coordinates are inside the image image
            # TODO: Shouldn't add [0,-2], [2,0], etc points for edge defects
            #       and so on. It'll increase perfomance a little.
            p_i = eps_pnt[0] + defect_list[defect_pnt][0]
            p_j = eps_pnt[1] + defect_list[defect_pnt][1]
            if 0 <= p_i <= (img_height - 1) and 0 <= p_j <= (img_width - 1):
                p_n = p_i * img_width + p_j

                # Add point. No duplicates will be added
                neighbor_list.update({p_n: (p_i, p_j)})

    # Initialize sparse matrix
    coef_matrix = sparse.dok_matrix((defect_points_num, img_pixnum))

    # 1 stage. Find points 2 or more pixels far from bounds
    # kernel = [        1
    #               2  -8   2
    #           1  -8  20  -8   1
    #               2  -8   2
    #                   1       ]
    #
    kernel = [1, 2, -8, 2, 1, -8, 20, -8, 1, 2, -8, 2, 1]
    offset = [-2 * img_width, -img_width - 1, -img_width, -img_width + 1,
              -2, -1, 0, 1, 2, img_width - 1, img_width, img_width + 1, 2 * img_width]

    for idx, (seq_num, coord) in enumerate(zip(defect_list.keys(), defect_list.values())):
        if 2 <= coord[0] <= img_height - 3 and 2 <= coord[1] <= img_width - 3:
            for k in range(len(kernel)):
                coef_matrix[idx, seq_num + offset[k]] = kernel[k]

    # 2 stage. Find points 1 pixel far from bounds
    # kernel = [     1
    #            1  -4  1
    #                1     ]
    #
    kernel = [1, 1, -4, 1, 1]
    offset = [-img_width, -1, 0, 1, img_width]

    for idx, (seq_num, coord) in enumerate(zip(defect_list.keys(), defect_list.values())):
        if ((coord[0] == 1 or coord[0] == img_height - 2) and 1 <= coord[1] <= img_height - 2) or\
           ((coord[1] == 1 or coord[1] == img_width - 2) and 1 <= coord[0] <= img_width - 2):
            for k in range(len(kernel)):
                coef_matrix[idx, seq_num + offset[k]] = kernel[k]

    # 3 stage. Find points on the horizontal bounds
    # kernel = [ 1, -2, 1 ]
    #
    kernel = [1, -2, 1]
    offset = [-1, 0, 1]

    for idx, (seq_num, coord) in enumerate(zip(defect_list.keys(), defect_list.values())):
        if (coord[0] == 0 or coord[0] == img_height - 1) and 1 <= coord[1] <= img_width - 1:
            for k in range(len(kernel)):
                coef_matrix[idx, seq_num + offset[k]] = kernel[k]

    # 4 stage. Find points on the vertical bounds
    # kernel = [  1,
    #            -2,
    #             1  ]
    #
    kernel = [1, -2, 1]
    offset = [-img_width, 0, img_width]

    for idx, (seq_num, coord) in enumerate(zip(defect_list.keys(), defect_list.values())):
        if (coord[1] == 0 or coord[1] == img_width - 1) and 1 <= coord[0] <= img_height - 1:
            for k in range(len(kernel)):
                coef_matrix[idx, seq_num + offset[k]] = kernel[k]

    # Separate known and unknown elements
    knowns_matrix = coef_matrix.copy()
    knowns_matrix[:, defect_list.keys()] = 0

    unknowns_matrix = coef_matrix.copy()
    unknowns_matrix = unknowns_matrix[:, defect_list.keys()]

    # Put known image values into the matrix (multiply matrix by known values of image)
    flat_diag_image = sparse.dia_matrix((out.flatten(), np.array([0])), shape=(img_pixnum, img_pixnum))

    # Get right hand side by sum knowns matrix columns
    rhs = -(knowns_matrix * flat_diag_image).sum(axis=1)

    # Solve linear system over defect points
    unknowns_matrix = sparse.csr_matrix(unknowns_matrix)
    result = spsolve(unknowns_matrix, rhs)

    # Image can contain only integers. Rounding so
    result = result.round()

    # Put calculated points into the image
    for idx, defect_coords in enumerate(defect_list.values()):
        out[defect_coords] = result[idx]

    return out

