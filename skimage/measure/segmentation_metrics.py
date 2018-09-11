import numpy as np
import multiprocessing
from .simple_metrics import _assert_compatible
import scipy.sparse as sparse
from scipy.ndimage.measurements import label

__all__ = [ 'compare_adapted_rand_error',
            'compare_split_variation_of_information',
            'compare_variation_of_information',
          ]

def compare_adapted_rand_error(im_true, im_test):
    """Compute Adapted Rand error as defined by the SNEMI3D contest [1]

    Formula is given as 1 - the maximal F-score of the Rand index
    (excluding the zero component of the original labels). Adapted
    from the SNEMI3D MATLAB script, hence the strange style.

    Parameters
    ----------
    im_true : ndarray of int
        Ground-truth label image.
    im_test : ndarray of int
        Test image.

    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float
        The adapted Rand precision.
    rec : float
        The adapted Rand recall.

    References
    ----------
    [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
    """
    _assert_compatible(im_true, im_test)

    # segA is query, segB is truth
    segA = im_test
    segB = im_true

    n = segA.size

    # This is the contingency table obtained from segA and segB, we obtain
    # the marginal probabilities from the table.
    p_ij = _contingency_table(segB, segA)

    # Sum of the joint distribution squared
    sum_p_ij = p_ij.data @ p_ij.data

    # These are the axix-wise sums (np.sumaxis)
    a_i = p_ij.sum(axis=0).A.ravel()
    b_i = p_ij.sum(axis=1).A.ravel()

    # Sum of the segment labeled 'A'
    sum_a = a_i @ a_i
    # Sum of the segment labeled 'B'
    sum_b = b_i @ b_i

    precision = (sum_p_ij - n)/ (sum_a - n)
    recall = (sum_p_ij - n)/ (sum_b - n)

    fscore = 2. * precision * recall / (precision + recall)
    are = 1. - fscore

    return (are, precision, recall)

def compare_variation_of_information(im_true, im_test, *, weights=np.ones(2)):
    """Return the variation of information metric. [1]

    VI(X, Y) = H(X | Y) + H(Y | X), where H(.|.) denotes the conditional
    entropy.

    Parameters
    ----------
    im_true, im_test : ndarray of int
        Image.  Any dimensionality.
    weights : ndarray of float, shape (2,), optional
        The weights of the conditional entropies of `im_true` and `im_test`. Equal weights
        are the default.

    Returns
    -------
    v : float
        The variation of information between `im_true` and `im_test`.

    References
    ----------
    [1] Meila, M. (2007). Comparing clusterings - an information based
    distance. Journal of Multivariate Analysis 98, 873-895.
    """
    return weights @ compare_split_variation_of_information(im_true, im_test)

def compare_split_variation_of_information(im_true, im_test):
    """Return the symmetric conditional entropies associated with the VI.

    The variation of information is defined as VI(X,Y) = H(X|Y) + H(Y|X).
    If Y is the ground-truth segmentation, then H(Y|X) can be interpreted
    as the amount of under-segmentation of Y and H(X|Y) is then the amount
    of over-segmentation.  In other words, a perfect over-segmentation
    will have H(Y|X)=0 and a perfect under-segmentation will have H(X|Y)=0.

    Parameters
    ----------
    im_true, im_test : ndarray of int
        Image.  Any dimensionality.

    Returns
    -------
    sv : ndarray of float, shape (2,)
        The conditional entropies of im_test|im_true and im_true|im_test.

    See Also
    --------
    compare_variation_of_information
    """
    hxgy, hygx = _vi_tables(im_true, im_test)
    # false splits, false merges
    return np.array([hygx.sum(), hxgy.sum()])

def _contingency_table(im_true, im_test):
    """Return the contingency table for all regions in matched segmentations.

    Parameters
    ----------
    im_true : ndarray of int
        Ground-truth label image.
    im_test : ndarray of int
        Test image.

    Returns
    -------
    cont : scipy.sparse.csr_matrix
        A contingency table. `cont[i, j]` will equal the number of voxels
        labeled `i` in `im_test` and `j` in `im_true`.
    """
    im_test_r = im_test.ravel()
    im_true_r = im_true.ravel()
    ignored = np.zeros(im_test_r.shape, np.bool)
    data = np.ones(im_true_r.shape)
    ignored[im_test_r == 0] = True
    ignored[im_true_r == 0] = True
    data[ignored] = 0
    cont = sparse.coo_matrix((data, (im_test_r, im_true_r))).tocsr()
    return cont

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
    """Return probability tables used for calculating VI.

    Parameters
    ----------
    im_true, im_test : ndarray of int
        Image.  Any dimensionality.

    Returns
    -------
    pxy : sparse.csc_matrix of float
        The normalized contingency table.
    hxgy, hygx : ndarray of float
        the per-segment conditional entropies of `im_true`
         given `im_test` and vice-versa
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
    """Returns the inverse of the non-zero elements of arr

    Parameters
    ----------
    arr : ndarray

    Returns
    -------
    arr_inv : ndarray
         the inverse of the non-zero elements of arr
    """
    arr_inv = arr.copy()
    nz = np.nonzero(arr)
    arr_inv[nz] = 1 / arr[nz]
    return arr_inv
