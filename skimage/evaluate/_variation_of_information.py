import numpy as np
import scipy.sparse as sparse
from ._contingency_table import contingency_table

__all__ = ['variation_of_information']


def variation_of_information(im_true=None, im_test=None, *, table=None,
                             ignore_labels=[], normalize=False):
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
    hxgy, hygx = _vi_tables(im_true, im_test, table,
                            ignore_labels, normalize=normalize)
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


def _vi_tables(im_true, im_test, table=None, ignore_labels=[],
               normalize=False):
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
    if table is None:
        # normalize, since it is an identity op if already done
        pxy = contingency_table(
            im_true, im_test, ignore_labels, normalize=normalize)

    else:
        pxy = table

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
