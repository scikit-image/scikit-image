#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp
from libc.float cimport DBL_MAX

from skimage.color import rgb2grey
from skimage.util import img_as_float


def moravec(image, int window_size=1):
    """Compute Moravec response image.

    This interest operator is comparatively fast but not rotation invariant.

    Parameters
    ----------
    image : ndarray
        Input image.
    window_size : int, optional
        Window size.

    Returns
    -------
    response : ndarray
        Moravec response image.

    References
    ----------
    ..[1] http://kiwi.cs.dal.ca/~dparks/CornerDetection/moravec.htm
    ..[2] http://en.wikipedia.org/wiki/Corner_detection

    Examples
    -------
    >>> from skimage.feature import moravec, peak_local_max
    >>> square = np.zeros([7, 7])
    >>> square[3, 3] = 1
    >>> square
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    >>> moravec(square)
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  2.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    """

    cdef int rows = image.shape[0]
    cdef int cols = image.shape[1]

    cdef cnp.ndarray[dtype=cnp.double_t, ndim=2, mode='c'] cimage, out

    if image.ndim == 3:
        cimage = rgb2grey(image)
    cimage = np.ascontiguousarray(img_as_float(image))

    out = np.zeros(image.shape, dtype=np.double)

    cdef double* image_data = <double*>cimage.data
    cdef double* out_data = <double*>out.data

    cdef double msum, min_msum
    cdef int r, c, br, bc, mr, mc, a, b
    for r in range(2 * window_size, rows - 2 * window_size):
        for c in range(2 * window_size, cols - 2 * window_size):
            min_msum = DBL_MAX
            for br in range(r - window_size, r + window_size + 1):
                for bc in range(c - window_size, c + window_size + 1):
                    if br != r and bc != c:
                        msum = 0
                        for mr in range(- window_size, window_size + 1):
                            for mc in range(- window_size, window_size + 1):
                                a = (r + mr) * cols + c + mc
                                b = (br + mr) * cols + bc + mc
                                msum += (image_data[a] - image_data[b]) ** 2
                        min_msum = min(msum, min_msum)

            out_data[r * cols + c] = min_msum

    return out
