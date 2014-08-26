# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import convolve2d


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
    >>> from skimage.restoration.inpainting import biharmonic_inpaint
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
    .. [1]  N.S.Hoang, S.B.Damelin, "On surface completion and image inpainting
            by biharmonic functions: numerical aspects",
            http://www.ima.umn.edu/~damelin/biharmonic

    Realization is based on:
    .. [2]  John D'Errico,
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
    # eps = [            [-2, 0],
    #          [-1, -1], [-1, 0], [-1, 1],
    # [0, -2], [ 0, -1],          [ 0, 1], [0, 2],
    #          [ 1, -1], [ 1, 0], [ 1, 1],
    #                    [ 2, 0]                ]
    #
    eps = [[-2, 0], [-1, -1], [-1, 0], [-1, 1], [0, -2], [0, -1],
           [0, 1], [0, 2], [1, -1], [1, 0], [1, 1], [2, 0]]

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
              -2, -1, 0, 1, 2, img_width - 1, img_width, img_width + 1,
              2 * img_width]

    for idx, (seq_num, coord) in enumerate(zip(defect_list.keys(),
                                               defect_list.values())):
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

    for idx, (seq_num, coord) in enumerate(zip(defect_list.keys(),
                                               defect_list.values())):
        if ((coord[0] == 1 or coord[0] == img_height - 2) and
           1 <= coord[1] <= img_height - 2) or \
           ((coord[1] == 1 or coord[1] == img_width - 2) and
           1 <= coord[0] <= img_width - 2):
            for k in range(len(kernel)):
                coef_matrix[idx, seq_num + offset[k]] = kernel[k]

    # 3 stage. Find points on the horizontal bounds
    # kernel = [ 1, -2, 1 ]
    #
    kernel = [1, -2, 1]
    offset = [-1, 0, 1]

    for idx, (seq_num, coord) in enumerate(zip(defect_list.keys(),
                                               defect_list.values())):
        if (coord[0] == 0 or coord[0] == img_height - 1) and \
           1 <= coord[1] <= img_width - 1:
            for k in range(len(kernel)):
                coef_matrix[idx, seq_num + offset[k]] = kernel[k]

    # 4 stage. Find points on the vertical bounds
    # kernel = [  1,
    #            -2,
    #             1  ]
    #
    kernel = [1, -2, 1]
    offset = [-img_width, 0, img_width]

    for idx, (seq_num, coord) in enumerate(zip(defect_list.keys(),
                                               defect_list.values())):
        if (coord[1] == 0 or coord[1] == img_width - 1) and \
           1 <= coord[0] <= img_height - 1:
            for k in range(len(kernel)):
                coef_matrix[idx, seq_num + offset[k]] = kernel[k]

    # Separate known and unknown elements
    knowns_matrix = coef_matrix.copy()
    knowns_matrix[:, defect_list.keys()] = 0

    unknowns_matrix = coef_matrix.copy()
    unknowns_matrix = unknowns_matrix[:, defect_list.keys()]

    # Put known image values into the matrix (multiply matrix by known values)
    flat_diag_image = sparse.dia_matrix((out.flatten(), np.array([0])),
                                        shape=(img_pixnum, img_pixnum))

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


def exemplar_inpaint(img, mask, patch_size=(9, 9), search_region=(0, 0)):
    """Inpaint masked points in image, using exemplar-based method.

    Parameters
    ----------
    img: 2-D ndarray
        Input image.
    mask: 2-D ndarray
        Array of pixels to inpaint. Should have the same size as 'img'.
        Unknown pixels should be represented with 1, known - with 0.
    patch_size: (odd uint, odd uint)
        '(Height, width)' of patch in pixels.
    search_region: (odd uint, odd uint)
        '(Height, width)' of search region in pixels.
        Should be more, than patch dimensions. Use (0, 0) for global search.

    Returns
    -------
    out: 2-D ndarray
        Image with unknown regions inpainted.

    Example
    -------
    >>> import numpy as np
    >>> from skimage.restoration.inpainting import exemplar_inpaint
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
    >>> image_out = exemplar_inpaint(image_in, mask)

    References
    ----------
    Algorithm is based on:
    .. [1]  A. Criminisi, P. Perez, K. Toyama
        "Region Filling and Object Removal by Exemplar-Based Image Inpainting
        http://research.microsoft.com/apps/pubs/default.aspx?id=67276

    Realization is based on:
    .. [2]  Sooraj Bhat,
           "Object Removal by Exemplar-based Inpainting -
            A CS7495 Final Project by Sooraj Bhat"
        http://white.stanford.edu/teach/images/5/55/ExemplarBasedInpainting.zip
    """
    # TODO: example

    def getpatch(img_height, img_width, point, patch_radii):
        # 'Point' is a middle of patch
        r = patch_radii
        x = point[0]
        y = point[1]

        # Generate indexes grid
        (patch_i, patch_j) = \
            np.mgrid[max(x - r[0], 0):min(x + r[1], img_height),
                     max(y - r[1], 0):min(y + r[1], img_width)]
        return patch_i, patch_j

    out = np.copy(img)
    img_height, img_width = out.shape

    unknowns_mask = mask.copy()
    knowns_mask = np.ones_like(unknowns_mask) - unknowns_mask

    patch_size = list(patch_size)
    patch_radii = (np.array(patch_size) - 1) / 2

    # TODO: check fucking logic
    search_radii = ()
    for idx in range(len(search_region)):
        if search_region[idx] == 0:
            search_radii += (img_height, ) if idx == 0 else (img_width, )
        elif search_region[idx] < patch_size[idx]:
            raise ValueError("Search region is less than patch size")
        else:
            search_radii += ((search_region[idx] - 1) / 2, )

    (grad_x, grad_y) = np.gradient(out)

    while unknowns_mask.any():
        print(np.sum(unknowns_mask))  # FIXME
        edge_kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        defect_edge = convolve2d(unknowns_mask, edge_kernel, mode='same')
        p_i, p_j = np.nonzero(defect_edge)
        p_n = np.ravel_multi_index((p_i, p_j), (img_height, img_width))

        # Create list of defect region edge points
        defect_edge_list = {}
        for k in range(len(p_n)):
            defect_edge_list.update({p_n[k]: (p_i[k], p_j[k])})

        (norm_x, norm_y) = np.gradient(knowns_mask)

        fill_rate = np.array([])
        data_rate = np.array([])
        prior_keys = np.array([])
        # Among edge points find one to start with
        for seq_num, coord in zip(defect_edge_list.keys(),
                                  defect_edge_list.values()):
            (tmp_i, tmp_j) = getpatch(img_height, img_width, coord, patch_radii)
            fill_rate = np.append(fill_rate, np.sum(knowns_mask[tmp_i, tmp_j]) /
                                  tmp_i.size)
            data_rate = np.append(data_rate, abs(grad_y[coord] * norm_x[coord] +
                                  grad_x[coord] * norm_y[coord]) + 0.001)
            prior_keys = np.append(prior_keys, seq_num)

        priorities = np.multiply(fill_rate, data_rate)

        # Proceed with region, centered at point with maximum weight
        prior_pnt_i, prior_pnt_j = \
            defect_edge_list[prior_keys[np.argmax(priorities)]]
        # Get prior region coordinates
        (prior_i, prior_j) = getpatch(img_height, img_width,
                                      (prior_pnt_i, prior_pnt_j), patch_radii)

        best_error = np.iinfo(np.uint64).max
        best_i, best_j = np.array([]), np.array([])

        # Search for max correlated patch in known pixels region
        # TODO: port this part to cython. It's the first bottleneck
        for dx in range(
                max(patch_radii[0], prior_pnt_i - search_radii[0]),
                min(img_height - patch_radii[0],
                    prior_pnt_i + search_radii[0])):
            for dy in range(
                    max(patch_radii[1], prior_pnt_j - search_radii[1]),
                    min(img_width - patch_radii[1],
                        prior_pnt_j + search_radii[1])):

                tmp_i, tmp_j = \
                    getpatch(img_height, img_width, (dx, dy), patch_radii)

                # If moving window consists unknown points - skipping it
                if knowns_mask[tmp_i, tmp_j].all():
                    # Calculate diff between moving and prior windows image
                    # over known pixels only
                    tmp_mask = knowns_mask[prior_i, prior_j]
                    diff = np.multiply(out[tmp_i, tmp_j], tmp_mask) - \
                           np.multiply(out[prior_i, prior_j], tmp_mask)
                    # Preventing overflow in np.sum
                    diff = diff.astype('float64')
                    error = np.sum((diff ** 2)) / np.sum(tmp_mask)
                else:
                    error = None

                if error is not None and error < best_error:
                    best_error = error
                    best_i, best_j = tmp_i, tmp_j

        # Inpaint patch from best-fitting region
        # TODO: leave the best case
        # copy_logic = 'wise'
        copy_logic = 'dumb'

        if copy_logic == 'wise':
            # Copy _only_ unknown values to prior edge point patch
            tmp_mask = np.nonzero(unknowns_mask[prior_i, prior_j])
            out[prior_i[tmp_mask], prior_j[tmp_mask]] = \
                out[best_i[tmp_mask], best_j[tmp_mask]]

            # Update masks
            knowns_mask[prior_i, prior_j] = 1
            unknowns_mask[prior_i, prior_j] = 0
            # Update gradient field
            # TODO: For graient we need to dilate masked region 1 point
            #       towards every direction. Honest solution is to recalculate
            #       gradient points in a patch and 1 pix around it. For
            #       simplier solution using gradient from "best" region
            grad_x[prior_i, prior_j] = grad_x[best_i, best_j]
            grad_y[prior_i, prior_j] = grad_y[best_i, best_j]
            # grad_x[prior_i[unk], prior_j[unk]] = \
            #   grad_x[best_i[unk], best_j[unk]]
            # grad_y[prior_i[unk], prior_j[unk]] = \
            #   grad_y[best_i[unk], best_j[unk]]
        else:
            # FIXME: Due to previous comment: Remove it.
            # Copying whole "best" patch to "prior" location
            out[prior_i, prior_j] = out[best_i, best_j]
            # Update masks
            knowns_mask[prior_i, prior_j] = 1
            unknowns_mask[prior_i, prior_j] = 0
            # Update gradient field
            grad_x[prior_i, prior_j] = grad_x[best_i, best_j]
            grad_y[prior_i, prior_j] = grad_y[best_i, best_j]

    return out
