import numpy as np


def unique_rows(ar):
    """Remove repeated rows from a 2D array.

    Parameters
    ----------
    ar : 2D np.ndarray
        The input array.

    Returns
    -------
    ar_out : 2D np.ndarray
        A copy of the input array with repeated rows removed.

    Raises
    ------
    ValueError : if `ar` is not two-dimensional.

    Examples
    --------
    >>> ar = np.array([[1, 0, 1],
                       [0, 1, 0],
                       [1, 0, 1]], np.uint8)
    >>> aru = unique_rows(ar)
    array([[0, 1, 0],
           [1, 0, 1]], dtype=uint8)
    """
    if ar.ndim != 2:
        raise ValueError("unique_rows() only makes sense for 2D arrays, "
                         "got %dd" % ar.ndim)
    ar_row_view = ar.view('|S%d' % (ar.itemsize * ar.shape[1]))
    _, unique_row_indices = np.unique(ar_row_view, return_index=True)
    ar_out = ar[unique_row_indices]
    return ar_out
