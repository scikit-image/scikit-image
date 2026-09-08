import scipy.sparse as sparse

import _skimage2 as ski2

from .._migration import ski2_migration_decorator

__all__ = ['contingency_table']


@ski2_migration_decorator(
    """\
    ``%(qname_old)s`` is deprecated in favor of
    ``%(qname_new)s`` with new behavior:

    * Parameter `sparse_type` was removed
    * Returned `cont` is always a `scipy.sparse.csr_array`

    SciPy 2.0 deprecates the sparse matrix classes and will remove them no
    earlier than SciPy 2.2. The two types define ``*`` differently: matrix
    multiplication for `scipy.sparse.csr_matrix`, elementwise multiplication
    for `scipy.sparse.csr_array`. ``@`` is matrix multiplication for both.

    To keep the old (``skimage``, v1.x) behavior, convert the result::

        cont = skimage2.metrics.contingency_table(im_true, im_test, ...)
        cont = scipy.sparse.csr_matrix(cont)
    """,
    qname_old="skimage.metrics.contingency_table",
)
def contingency_table(
    im_true, im_test, *, ignore_labels=None, normalize=False, sparse_type="matrix"
):
    """
    Return the contingency table for all regions in matched segmentations.

    Parameters
    ----------
    im_true : ndarray of int
        Ground-truth label image, same shape as im_test.
    im_test : ndarray of int
        Test image.
    ignore_labels : sequence of int, optional
        Labels to ignore. Any part of the true image labeled with any of these
        values will not be counted in the score.
    normalize : bool
        Determines if the contingency table is normalized by pixel count.
    sparse_type : {"matrix", "array"}, optional
        The return type of `cont`, either `scipy.sparse.csr_array` or
        `scipy.sparse.csr_matrix` (default).

    Returns
    -------
    cont : scipy.sparse.csr_matrix or scipy.sparse.csr_array
        A contingency table. `cont[i, j]` will equal the number of voxels
        labeled `i` in `im_true` and `j` in `im_test`. Depending on `sparse_type`,
        this can be returned as a `scipy.sparse.csr_array`.
    """
    if sparse_type not in ("array", "matrix"):
        msg = f"`sparse_type` must be 'array' or 'matrix', got {sparse_type}"
        raise ValueError(msg)

    cont = ski2.metrics.contingency_table(
        im_true,
        im_test,
        ignore_labels=ignore_labels,
        normalize=normalize,
    )
    if sparse_type == "matrix":
        cont = sparse.csr_matrix(cont)

    return cont


from skimage._doctest_adapters import adapt_doctests  # noqa: E402

adapt_doctests(globals())
