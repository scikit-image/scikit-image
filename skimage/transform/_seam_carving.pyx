# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as cnp


cdef cnp.double_t ABSOLUTE_MAX = np.finfo(np.double).max


cdef find_seam_v(cnp.double_t[:, ::1] energy_img, cnp.int8_t[:, ::1] track_img,
                 cnp.double_t[::1] current_cost, cnp.double_t[::1] prev_cost,
                 Py_ssize_t cols):
    """Find a single vertical seam in an image that will be removed.
    
    Parameters
    ----------
    energy_img : (M, N) ndarray
        The energy image where a higher value signifies a pixel of more
        importance.
    track_img : (M, N) ndarray
        The image used to store the optimal decision made at each point while
        finding a minimum cost path.
    current_cost : (N, ) ndarray
        An array to store the current cost of the optimal path for each column
        in row currently being processed.
    prev_cost : (N, ) ndarray
        An array to store the current cost of the optimal path for each column
        in row prior to the one being processed.
    cols : int
        The number of cols to process for seam carving. Columns with indices
        more than `cols` are ignored.

        
    Returns
    -------
    seam : (M, ) ndarray
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
            
            min_cost = ABSOLUTE_MAX
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
        #print offset
        seam[row] = seam[row + 1] + offset

    return seam


cdef remove_seam_v_2d(cnp.double_t[:, ::1] img, Py_ssize_t[::1] seam,
                      Py_ssize_t cols):
    cdef Py_ssize_t rows, row, col, idx
    rows = img.shape[0]
    """ Removes one horizontal seam from the image.

    The method modifies `img` so that all pixels to the right of the vertical
    seam are pushed one place left.

    image : (M, N) ndarray
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
    
    for row in range(rows):
        for idx in range(seam[row], cols - 1):
            img[row, idx] = img[row, idx + 1]


cdef remove_seam_v_3d(cnp.double_t[:, :, ::1] img, Py_ssize_t[::1] seam,
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


def _seam_carve_v(img, iters, energy_func, extra_args , extra_kwargs, border):
    """ Carve vertical seams off an image.

    Carves out vertical seams off an image while using the given energy
    function to decide the importance of each pixel.[1]

    Parameters
    ----------
    image : (M, N) or (M, N, 3) ndarray
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
    cdef Py_ssize_t[::1] seam
    cdef Py_ssize_t ndim = img.ndim
    cdef Py_ssize_t cols = img.shape[1]

    track_img = np.zeros(img.shape[0:2], dtype=np.int8)

    current_cost = np.zeros_like(track_img[0], dtype = img.dtype)
    prev_cost = np.zeros_like(track_img[0], dtype = img.dtype)

    for i in range(iters):

        sliced_img = img[:, 0:cols]
        energy_img = energy_func(sliced_img, *extra_args, **extra_kwargs)
        
        # So that borders are ignored.
        energy_img[:, 0:border] = ABSOLUTE_MAX
        energy_img[:, cols-border:cols] = ABSOLUTE_MAX
        
        seam = find_seam_v(energy_img, track_img, current_cost, prev_cost,
                           cols)

        if ndim == 2:
            remove_seam_v_2d(img, seam, cols)
        elif ndim == 3:
            remove_seam_v_3d(img, seam, cols)

        cols -= 1

    return img[:, 0:cols]
