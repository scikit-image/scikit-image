# -*- coding: utf-8 -*-

import scipy
from scipy import sparse
from scipy.sparse.linalg import spsolve

__author__ = "Egor Panfilov"


def biharmonic_inpaint(img_in, mask_in):
    """ Inpaint points in image, specified with mask, using biharmonic inpainting.

    Parameters
    ----------
    img_in: 2-D ndarray
        Input image;
    mask_in: 2-D ndarray
        Array of pixels to inpaint. Should have the same size as 'img_in'.
        Unknown pixels should be represented with 1, known - with 0.

    Returns
    -------
    img_out: 2-D ndarray
        Image with unknown regions inpainted.

    Example
    -------
    image_in = scipy.ones((10, 10))
    image_in[:, 6:] = 2
    mask = scipy.zeros_like(image_in)
    mask[4, 5:] = 1
    image_out = biharmonic_inpaint(image_in, mask)

    image_in =
       [[ 1.  1.  1.  1.  1.  1.  2.  2.  2.  2.]
        [ 1.  1.  1.  1.  1.  1.  2.  2.  2.  2.]
        [ 1.  1.  1.  1.  1.  1.  2.  2.  2.  2.]
        [ 1.  1.  1.  1.  1.  1.  2.  2.  2.  2.]
        [ 1.  1.  1.  1.  1.  1.  2.  2.  2.  2.]
        [ 1.  1.  1.  1.  1.  1.  2.  2.  2.  2.]
        [ 1.  1.  1.  1.  1.  1.  2.  2.  2.  2.]
        [ 1.  1.  1.  1.  1.  1.  2.  2.  2.  2.]
        [ 1.  1.  1.  1.  1.  1.  2.  2.  2.  2.]
        [ 1.  1.  1.  1.  1.  1.  2.  2.  2.  2.]]

    mask =
        [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  1.  1.  1.  1.  1.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]

    img_out (__before rounding__) =
        [[ 1.  1.  1.  1.  1.  1.      2.      2.      2.      2.]
         [ 1.  1.  1.  1.  1.  1.      2.      2.      2.      2.]
         [ 1.  1.  1.  1.  1.  1.      2.      2.      2.      2.]
         [ 1.  1.  1.  1.  1.  1.      2.      2.      2.      2.]
         [ 1.  1.  1.  1.  1.  1.10... 1.89... 2.00... 2.00... 2.]
         [ 1.  1.  1.  1.  1.  1.      2.      2.      2.      2.]
         [ 1.  1.  1.  1.  1.  1.      2.      2.      2.      2.]
         [ 1.  1.  1.  1.  1.  1.      2.      2.      2.      2.]
         [ 1.  1.  1.  1.  1.  1.      2.      2.      2.      2.]
         [ 1.  1.  1.  1.  1.  1.      2.      2.      2.      2.]]

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

    img = scipy.copy(img_in)
    mask = scipy.copy(mask_in)
    img_height, img_width = img.shape
    img_pixnum = img_height * img_width

    i, j = scipy.where(mask)
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
            # TODO: drop some elements in eps for near-the-edge defect points
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

    # =============================== Time to do math =========================
    # Separate known and unknown elements
    knowns_matrix = coef_matrix.copy()
    knowns_matrix[:, defect_list.keys()] = 0

    unknowns_matrix = coef_matrix.copy()
    unknowns_matrix = unknowns_matrix[:, defect_list.keys()]

    # Put known image values into the matrix (multiply matrix by known values of image)
    flat_diag_image = sparse.dia_matrix((img.flatten(), scipy.array([0])), shape=(img_pixnum, img_pixnum))

    # Get right hand side by sum knowns matrix columns
    rhs = -(knowns_matrix * flat_diag_image).sum(axis=1)

    # Solve linear system over defect points
    unknowns_matrix = sparse.csr_matrix(unknowns_matrix)
    result = spsolve(unknowns_matrix, rhs)

    # Image can contain only integers. Rounding so
    result = result.round()

    # Put calculated points into the image
    for idx, defect_coords in enumerate(defect_list.values()):
        img[defect_coords] = result[idx]

    img_out = img
    return img_out


if __name__ == "__main__":
    image_in = scipy.ones((10, 10))
    image_in[:, 6:] = 2
    mask = scipy.zeros_like(image_in)
    mask[4, 5:] = 1
    print(image_in)
    print("\n")
    print(mask)
    print("\n")

    image_out = biharmonic_inpaint(image_in, mask)
    print(image_out)
    print("\n")
