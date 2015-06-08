import numpy as np
cimport numpy as cnp


cdef cnp.double_t DBL_MAX = np.finfo(np.double).max


cdef _find_seam_v(cnp.double_t[:, ::1] energy_img, cnp.int8_t[:, ::1] track_img,
                 cnp.double_t[::1] current_cost, cnp.double_t[::1] prev_cost,
                 Py_ssize_t cols):
    """Find a single vertical seam in an image that will be removed.

    Parameters
    ----------
    energy_img : (M, N) ndarray
        The energy image where a higher value signifies a pixel of more
        importance. Pixels with a lower value will be cropped first.
    track_img : (M, N) ndarray
        The image used to store the optimal decision made at each point while
        finding a minimum cost path. For each pixel it stores the offset that
        produced that least cost.
    current_cost : (N,) ndarray
        An array to store the current cost of the optimal path for each column
        in row currently being processed.
    prev_cost : (N,) ndarray
        An array to store the current cost of the optimal path for each column
        in row prior to the one being processed.
    cols : int
        The number of cols to process for seam carving. Columns with indices
        more than `cols` are ignored.


    Returns
    -------
    seam : (M, ) ndarray of int
        An array containing the index of the row of the pixel to be removed
        for each column in the image.

    Notes
    -----
    `track_img`, `current_cost` and `prev_cost` are passed as arguments to
    avoid memory allocation at each iteration of `_seam_carve_v`.
    """

    cdef Py_ssize_t rows, row, col
    rows = energy_img.shape[0]
    cdef cnp.double_t tmp, min_cost
    cdef Py_ssize_t offset, idx, offset_clip

    cdef Py_ssize_t[::1] seam = np.zeros(rows, dtype=np.int)

    for idx in range(cols):
        prev_cost[idx] = energy_img[0, idx]

    for row in range(1, rows):
        for col in range(0, cols):

            min_cost = DBL_MAX
            for offset in range(-1, 2):
                idx = col + offset

                if idx > cols - 1 or idx < 0:
                    continue

                if prev_cost[idx] < min_cost:
                    min_cost = prev_cost[idx]
                    track_img[row, col] = offset

            current_cost[col] = min_cost + energy_img[row, col]

        prev_cost[:] = current_cost

    seam[rows-1] = np.argmin(current_cost)

    for row in range(rows-2, -1, -1):
        col = seam[row + 1]
        offset = track_img[row, col]
        seam[row] = seam[row + 1] + offset

    return seam


cdef remove_seam_v(cnp.double_t[:, :, ::1] img, Py_ssize_t[::1] seam,
                      Py_ssize_t cols):
    """ Removes one horizontal seam from the image.

    The method modifies `img` so that all pixels to the right of the vertical
    seam are pushed one place left.

    image : (M, N, 3) ndarray
        Input image whose vertical seam is to be removed.
    seam : (M, ) ndarray
        An array use to store the index of the column in the seam for each row.
    cols : int
        Number of columns in the input image to process. Column indices more
        than `cols` are ingored.

    Notes
    -----
    `seam` is passed as an argument so that we don't have to reallocate it for
    each iteration in `_seam_carve_v`.
    """
    cdef Py_ssize_t rows, row, col, idx
    rows = img.shape[0]

    for row in range(rows):
        for idx in range(seam[row], cols - 1):
            img[row, idx, :] = img[row, idx + 1, :]


cdef _preprocess_image(cnp.double_t[:, ::1] energy_img,
                       cnp.double_t[:, ::1] cumulative_img,
                       cnp.int8_t[:, ::1] track_img,
                       Py_ssize_t cols):

    cdef Py_ssize_t r, c, offset, c_idx
    cdef Py_ssize_t rows = energy_img.shape[0]
    cdef cnp.double_t min_cost = DBL_MAX

    for c in range(cols):
        cumulative_img[0, c] = energy_img[0, c]


    for r in range(1, rows):
        for c in range(cols):
            min_cost = DBL_MAX
            for offset in range(-1, 2):

                c_idx = c + offset
                if (c_idx > cols - 1) or (c_idx < 0) :
                    continue

                if cumulative_img[r-1, c_idx] < min_cost:
                    min_cost = cumulative_img[r-1, c_idx]
                    track_img[r, c] = offset

            #print "min_cost = ", min_cost
            cumulative_img[r,c] = min_cost + energy_img[r, c]

    #print "-------Cumulative Image --------"
    #print np.array(cumulative_img)
    #print "-------Energy Image --------"
    #print np.array(energy_img)

cdef cnp.uint8_t mark_seam(cnp.int8_t[:, ::1] track_img, Py_ssize_t start_index,
                          cnp.uint8_t[:, ::1] seam_map):

    cdef Py_ssize_t rows = track_img.shape[0]
    cdef Py_ssize_t[::1] current_seam_indices = np.zeros(rows, dtype=np.int)
    cdef Py_ssize_t row, col
    cdef cnp.int8_t offset
    cdef Py_ssize_t seams

    current_seam_indices[rows - 1] = start_index
    for row in range(rows - 2, -1, -1):
        col = current_seam_indices[row+1]
        offset = track_img[row, col]
        col = col + offset
        current_seam_indices[row] = col

        if seam_map[row, col]:
            return 0


    for row in range(rows):
        col = current_seam_indices[row]
        seam_map[row, col] = 1

    return 1
def _seam_carve_v(img, iters, energy_func, extra_args , extra_kwargs, border):
    """ Carve vertical seams off an image.

    Carves out vertical seams off an image while using the given energy
    function to decide the importance of each pixel.[1]

    Parameters
    ----------
    img : (M, N) or (M, N, 3) ndarray
        Input image whose vertical seams are to be removed.
    iters : int
        Number of vertical seams are to be removed.
    energy_func : callable
        The function used to decide the importance of each pixel. The higher
        the value corresponding to a pixel, the more the algorithm will try
        to keep it in the image. For every iteration `energy_func` is called
        as `energy_func(image, *extra_args, **extra_kwargs)`, where `image`
        is the cropped image during each iteration and is expected to return a
        (M, N) ndarray depicting each pixel's importance.
    extra_args : iterable
        The extra arguments supplied to `energy_func`.
    extra_kwargs : dict
        The extra keyword arguments supplied to `energy_func`.
    border : int
        The number of pixels in the right and left end of the image to be
        excluded from being considered for a seam. This is important as certain
        filters just ignore image boundaries and set them to `0`.

    Returns
    -------
    image : (M, N - iters) or (M, N - iters, 3) ndarray
        The cropped image with the vertical seams removed.

    References
    ----------
    .. [1] Shai Avidan and Ariel Shamir
           "Seam Carving for Content-Aware Image Resizing"
           http://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Avidan07.pdf
    """
    last_row_obj = np.zeros(img.shape[1], dtype=np.float)
    seam_map_obj = np.zeros(img.shape[0:2], dtype=np.uint8)

    cdef cnp.double_t[::1] last_row = last_row_obj
    cdef Py_ssize_t[::1] sorted_indices
    cdef cnp.uint8_t[:, ::1] seam_map = seam_map_obj
    cdef Py_ssize_t cols = img.shape[1]

    cdef cnp.double_t[:, :, ::1] image = img
    cdef cnp.int8_t[:, ::1] track_img = np.zeros(img.shape[0:2], dtype=np.int8)
    cdef cnp.double_t[:, ::1] cumulative_img = np.zeros(img.shape[0:2], dtype=np.float)
    cdef cnp.double_t[:, ::1] energy_img

    energy_img_obj = energy_func(np.squeeze(img))
    energy_img = energy_img_obj

    energy_img_obj[:, 0:border] = DBL_MAX
    energy_img_obj[:, cols-border:cols] = DBL_MAX

    _preprocess_image(energy_img, cumulative_img, track_img, cols)
    last_row[...] = cumulative_img[-1, :]
    sorted_indices = np.argsort(last_row_obj)
    #print "Sorted Indices = ", np.array(sorted_indices)
    #print "First sorted Index = ", sorted_indices[0]
    #print "Last Row = ", np.array(energy_img[-1, :])
    #print np.array()


    from skimage import io
    io.imshow(seam_map_obj*255)
    io.show()
    return img[:, 0:cols]
