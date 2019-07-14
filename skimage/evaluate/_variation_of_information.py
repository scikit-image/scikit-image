import numpy as np
import scipy.sparse as sparse
from utils import _contigency_table

__all__ = ['variation_of_information']

def compare_variation_of_information(im_true, im_test):
    """Return the symmetric conditional entropies associated with the VI.
    The variation of information is defined as VI(X,Y) = H(X|Y) + H(Y|X).
    If Y is the ground-truth segmentation, then H(Y|X) can be interpreted
    as the amount of under-segmentation of Y and H(X|Y) as the amount
    of over-segmentation. In other words, a perfect over-segmentation
    will have H(Y|X)=0 and a perfect under-segmentation will have H(X|Y)=0.
    Parameters
    ----------
    im_true, im_test : ndarray of int
        Label images / segmentations.
    Returns
    -------
    vi : ndarray of float, shape (2,)
        The conditional entropies of im_test|im_true and im_true|im_test.
    """
    hxgy, hygx = _vi_tables(im_true, im_test)
    # false splits, false merges
    return np.array([hygx.sum(), hxgy.sum()])

def _xlogx(x):
    """Compute x * log_2(x).
    We define 0 * log_2(0) = 0
    Parameters
    ----------
    x : ndarray or scipy.sparse.csc_matrix or csr_matrix
        The input array.
    Returns
    -------
    y : same type as x
        Result of x * log_2(x).
    """
    y = x.copy()
    if isinstance(y, sparse.csc_matrix) or isinstance(y, sparse.csr_matrix):
        z = y.data
    else:
        z = np.asarray(y)  # ensure np.matrix converted to np.array
    nz = z.nonzero()
    z[nz] *= np.log2(z[nz])
    return y


def _divide_rows(matrix, column):
    """Divide each row of `matrix` by the corresponding element in `column`.
    The result is as follows: out[i, j] = matrix[i, j] / column[i]
    Parameters
    ----------
    matrix : ndarray, scipy.sparse.csc_matrix or csr_matrix, shape (M, N)
        The input matrix.
    column : a 1D ndarray, shape (M,)
        The column dividing `matrix`.
    Returns
    -------
    out : same type as `matrix`
        The result of the row-wise division.
    """
    out = matrix.copy()
    if type(out) in [sparse.csc_matrix, sparse.csr_matrix]:
        if type(out) == sparse.csr_matrix:
            convert_to_csr = True
            out = out.tocsc()
        else:
            convert_to_csr = False
        column_repeated = np.take(column, out.indices)
        nz = out.data.nonzero()
        out.data[nz] /= column_repeated[nz]
        if convert_to_csr:
            out = out.tocsr()
    else:
        out /= column[:, np.newaxis]
    return out


def _divide_columns(matrix, row):
    """Divide each column of `matrix` by the corresponding element in `row`.
    The result is as follows: out[i, j] = matrix[i, j] / row[j]
    Parameters
    ----------
    matrix : ndarray, scipy.sparse.csc_matrix or csr_matrix, shape (M, N)
        The input matrix.
    column : a 1D ndarray, shape (N,)
        The row dividing `matrix`.
    Returns
    -------
    out : same type as `matrix`
        The result of the row-wise division.
    """
    out = matrix.copy()
    if type(out) in [sparse.csc_matrix, sparse.csr_matrix]:
        if type(out) == sparse.csc_matrix:
            convert_to_csc = True
            out = out.tocsr()
        else:
            convert_to_csc = False
        row_repeated = np.take(row, out.indices)
        nz = out.data.nonzero()
        out.data[nz] /= row_repeated[nz]
        if convert_to_csc:
            out = out.tocsc()
    else:
        out /= row[np.newaxis, :]
    return out


def _vi_tables(im_true, im_test):
    """Compute probability tables used for calculating VI.
    Parameters
    ----------
    im_true, im_test : ndarray of int
        Input label images, any dimensionality.
    Returns
    -------
    hxgy, hygx : ndarray of float
        Per-segment conditional entropies of ``im_true`` given ``im_test`` and
        vice-versa.
    """
    # normalize, since it is an identity op if already done
    pxy = sparse.coo_matrix((np.full(im_true.size, 1/im_true.size), 
                            (im_true.ravel(), im_test.ravel())),
                            dtype=float).tocsr()

    # compute marginal probabilities, converting to 1D array
    px = np.ravel(pxy.sum(axis=1))
    py = np.ravel(pxy.sum(axis=0))

    # use sparse matrix linear algebra to compute VI
    # first, compute the inverse diagonal matrices
    px_inv = sparse.diags(_invert_nonzero(px))
    py_inv = sparse.diags(_invert_nonzero(py))

    # then, compute the entropies
    hygx = -px @ _xlogx(px_inv @ pxy).sum(axis=1)
    hxgy = -_xlogx(pxy @ py_inv).sum(axis=0) @ py

    return list(map(np.asarray, [hxgy, hygx]))


def _invert_nonzero(arr):
    """Compute the inverse of the non-zero elements of arr, not changing 0.
    Parameters
    ----------
    arr : ndarray
    Returns
    -------
    arr_inv : ndarray
        Array containing the inverse of the non-zero elements of arr, and
        zero elsewhere.
    """
    arr_inv = arr.copy()
    nz = np.nonzero(arr)
    arr_inv[nz] = 1 / arr[nz]
    return arr_inv
