"""
Random walker segmentation algorithm

from *Random walks for image segmentation*, Leo Grady, IEEE Trans
Pattern Anal Mach Intell. 2006 Nov;28(11):1768-83.

Installing pyamg and using the 'cg_mg' mode of random_walker improves
significantly the performance.
"""

import warnings

import numpy as np
from scipy import sparse, ndimage
try:
    from scipy.sparse.linalg.dsolve import umfpack
    u = umfpack.UmfpackContext()
except:
    warnings.warn("""Scipy was built without UMFPACK. Consider rebuilding
    Scipy with UMFPACK, this will greatly speed up the random walker
    functions. You may also install pyamg and run the random walker function
    in cg_mg mode (see the docstrings)
    """)
try:
    from pyamg import ruge_stuben_solver
    amg_loaded = True
except ImportError:
    amg_loaded = False
from scipy.sparse.linalg import cg
from ..util import img_as_float
from ..filter import rank_order

#-----------Laplacian--------------------


def _make_graph_edges_3d(n_x, n_y, n_z):
    """
    Returns a list of edges for a 3D image.

    Parameters
    ----------
    n_x: integer
        The size of the grid in the x direction.
    n_y: integer
        The size of the grid in the y direction
    n_z: integer
        The size of the grid in the z direction

    Returns
    -------
    edges : (2, N) ndarray
        with the total number of edges N = n_x * n_y * (nz - 1) +
                                           n_x * (n_y - 1) * nz +
                                           (n_x - 1) * n_y * nz
        Graph edges with each column describing a node-id pair.
    """
    vertices = np.arange(n_x * n_y * n_z).reshape((n_x, n_y, n_z))
    edges_deep = np.vstack((vertices[:, :, :-1].ravel(),
                            vertices[:, :, 1:].ravel()))
    edges_right = np.vstack((vertices[:, :-1].ravel(),
                             vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
    edges = np.hstack((edges_deep, edges_right, edges_down))
    return edges


def _compute_weights_3d(data, beta=130, eps=1.e-6, depth=1.,
                        multichannel=False):
    # Weight calculation is main difference in multispectral version
    # Original gradient**2 replaced with sum of gradients ** 2
    gradients = 0
    for channel in range(0, data.shape[-1]):
        gradients += _compute_gradients_3d(data[..., channel],
                                               depth=depth) ** 2
    # All channels considered together in this standard deviation
    beta /= 10 * data.std()
    if multichannel:
        # New final term in beta to give == results in trivial case where
        # multiple identical spectra are passed.
        beta /= np.sqrt(data.shape[-1])
    gradients *= beta
    weights = np.exp(- gradients)
    weights += eps
    return weights


def _compute_gradients_3d(data, depth=1.):
    gr_deep = np.abs(data[:, :, :-1] - data[:, :, 1:]).ravel() / depth
    gr_right = np.abs(data[:, :-1] - data[:, 1:]).ravel()
    gr_down = np.abs(data[:-1] - data[1:]).ravel()
    return np.r_[gr_deep, gr_right, gr_down]


def _make_laplacian_sparse(edges, weights):
    """
    Sparse implementation
    """
    pixel_nb = edges.max() + 1
    diag = np.arange(pixel_nb)
    i_indices = np.hstack((edges[0], edges[1]))
    j_indices = np.hstack((edges[1], edges[0]))
    data = np.hstack((-weights, -weights))
    lap = sparse.coo_matrix((data, (i_indices, j_indices)),
                            shape=(pixel_nb, pixel_nb))
    connect = - np.ravel(lap.sum(axis=1))
    lap = sparse.coo_matrix((np.hstack((data, connect)),
                (np.hstack((i_indices, diag)), np.hstack((j_indices, diag)))),
                shape=(pixel_nb, pixel_nb))
    return lap.tocsr()


def _clean_labels_ar(X, labels, copy=False):
    X = X.astype(labels.dtype)
    if copy:
        labels = np.copy(labels)
    labels = np.ravel(labels)
    labels[labels == 0] = X
    return labels


def _buildAB(lap_sparse, labels):
    """
    Build the matrix A and rhs B of the linear system to solve.
    A and B are two block of the laplacian of the image graph.
    """
    labels = labels[labels >= 0]
    indices = np.arange(labels.size)
    unlabeled_indices = indices[labels == 0]
    seeds_indices = indices[labels > 0]
    # The following two lines take most of the time in this function
    B = lap_sparse[unlabeled_indices][:, seeds_indices]
    lap_sparse = lap_sparse[unlabeled_indices][:, unlabeled_indices]
    nlabels = labels.max()
    rhs = []
    for lab in range(1, nlabels + 1):
        mask = (labels[seeds_indices] == lab)
        fs = sparse.csr_matrix(mask)
        fs = fs.transpose()
        rhs.append(B * fs)
    return lap_sparse, rhs


def _mask_edges_weights(edges, weights, mask):
    """
    Remove edges of the graph connected to masked nodes, as well as
    corresponding weights of the edges.
    """
    mask0 = np.hstack((mask[:, :, :-1].ravel(), mask[:, :-1].ravel(),
                       mask[:-1].ravel()))
    mask1 = np.hstack((mask[:, :, 1:].ravel(), mask[:, 1:].ravel(),
                       mask[1:].ravel()))
    ind_mask = np.logical_and(mask0, mask1)
    edges, weights = edges[:, ind_mask], weights[ind_mask]
    max_node_index = edges.max()
    # Reassign edges labels to 0, 1, ... edges_number - 1
    order = np.searchsorted(np.unique(edges.ravel()),
                            np.arange(max_node_index + 1))
    edges = order[edges]
    return edges, weights


def _build_laplacian(data, mask=None, beta=50, depth=1., multichannel=False):
    l_x, l_y, l_z = data.shape[:3]
    edges = _make_graph_edges_3d(l_x, l_y, l_z)
    weights = _compute_weights_3d(data, beta=beta, eps=1.e-10, depth=depth,
                                  multichannel=multichannel)
    if mask is not None:
        edges, weights = _mask_edges_weights(edges, weights, mask)
    lap = _make_laplacian_sparse(edges, weights)
    del edges, weights
    return lap


#----------- Random walker algorithm --------------------------------


def random_walker(data, labels, beta=130, mode='bf', tol=1.e-3, copy=True,
                  multichannel=False, return_full_prob=False, depth=1.):
    """
    Random walker algorithm for segmentation from markers, for gray-level or
    multichannel images.

    Parameters
    ----------

    data : array_like
        Image to be segmented in phases. Gray-level `data` can be two- or
        three-dimensional; multichannel data can be three- or four-
        dimensional (multichannel=True) with the highest dimension denoting
        channels. Data spacing is assumed isotropic unless depth keyword
        argument is used.

    labels : array of ints, of same shape as `data` without channels dimension
        Array of seed markers labeled with different positive integers
        for different phases. Zero-labeled pixels are unlabeled pixels.
        Negative labels correspond to inactive pixels that are not taken
        into account (they are removed from the graph). If labels are not
        consecutive integers, the labels array will be transformed so that
        labels are consecutive. In the multichannel case, `labels` should have
        the same shape as a single channel of `data`, i.e. without the final
        dimension denoting channels.

    beta : float
        Penalization coefficient for the random walker motion
        (the greater `beta`, the more difficult the diffusion).

    mode : {'bf', 'cg_mg', 'cg'} (default: 'bf')
        Mode for solving the linear system in the random walker
        algorithm.

        - 'bf' (brute force, default): an LU factorization of the Laplacian is
          computed. This is fast for small images (<1024x1024), but very slow
          (due to the memory cost) and memory-consuming for big images (in 3-D
          for example).

        - 'cg' (conjugate gradient): the linear system is solved iteratively
          using the Conjugate Gradient method from scipy.sparse.linalg. This is
          less memory-consuming than the brute force method for large images,
          but it is quite slow.

        - 'cg_mg' (conjugate gradient with multigrid preconditioner): a
          preconditioner is computed using a multigrid solver, then the
          solution is computed with the Conjugate Gradient method.  This mode
          requires that the pyamg module (http://code.google.com/p/pyamg/) is
          installed. For images of size > 512x512, this is the recommended
          (fastest) mode.

    tol : float
        tolerance to achieve when solving the linear system, in
        cg' and 'cg_mg' modes.

    copy : bool
        If copy is False, the `labels` array will be overwritten with
        the result of the segmentation. Use copy=False if you want to
        save on memory.

    multichannel : bool, default False
        If True, input data is parsed as multichannel data (see 'data' above
        for proper input format in this case)

    return_full_prob : bool, default False
        If True, the probability that a pixel belongs to each of the labels
        will be returned, instead of only the most likely label.

    depth : float, default 1.
        Correction for non-isotropic voxel depths in 3D volumes.
        Default (1.) implies isotropy.  This factor is derived as follows:
        depth = (out-of-plane voxel spacing) / (in-plane voxel spacing), where
        in-plane voxel spacing represents the first two spatial dimensions and
        out-of-plane voxel spacing represents the third spatial dimension.

    Returns
    -------

    output : ndarray
        If `return_full_prob` is False, array of ints of same shape as `data`,
        in which each pixel has been labeled according to the marker that
        reached the pixel first by anisotropic diffusion.
        If `return_full_prob` is True, array of floats of shape
        `(nlabels, data.shape)`. `output[label_nb, i, j]` is the probability
        that label `label_nb` reaches the pixel `(i, j)` first.

    See also
    --------

    skimage.morphology.watershed: watershed segmentation
        A segmentation algorithm based on mathematical morphology
        and "flooding" of regions from markers.

    Notes
    -----

    Multichannel inputs are scaled with all channel data combined. Ensure all
    channels are separately normalized prior to running this algorithm.

    The `depth` argument is specifically for certain types of 3-dimensional
    volumes which, due to how they were acquired, have different spacing
    along in-plane and out-of-plane dimensions. This is commonly encountered
    in medical imaging. The `depth` argument corrects gradients calculated
    along the third spatial dimension for the otherwise inherent assumption
    that all points are equally spaced.

    The algorithm was first proposed in *Random walks for image
    segmentation*, Leo Grady, IEEE Trans Pattern Anal Mach Intell.
    2006 Nov;28(11):1768-83.

    The algorithm solves the diffusion equation at infinite times for
    sources placed on markers of each phase in turn. A pixel is labeled with
    the phase that has the greatest probability to diffuse first to the pixel.

    The diffusion equation is solved by minimizing x.T L x for each phase,
    where L is the Laplacian of the weighted graph of the image, and x is
    the probability that a marker of the given phase arrives first at a pixel
    by diffusion (x=1 on markers of the phase, x=0 on the other markers, and
    the other coefficients are looked for). Each pixel is attributed the label
    for which it has a maximal value of x. The Laplacian L of the image
    is defined as:

       - L_ii = d_i, the number of neighbors of pixel i (the degree of i)
       - L_ij = -w_ij if i and j are adjacent pixels

    The weight w_ij is a decreasing function of the norm of the local gradient.
    This ensures that diffusion is easier between pixels of similar values.

    When the Laplacian is decomposed into blocks of marked and unmarked
    pixels::

        L = M B.T
            B A

    with first indices corresponding to marked pixels, and then to unmarked
    pixels, minimizing x.T L x for one phase amount to solving::

        A x = - B x_m

    where x_m = 1 on markers of the given phase, and 0 on other markers.
    This linear system is solved in the algorithm using a direct method for
    small images, and an iterative method for larger images.

    Examples
    --------

    >>> a = np.zeros((10, 10)) + 0.2*np.random.random((10, 10))
    >>> a[5:8, 5:8] += 1
    >>> b = np.zeros_like(a)
    >>> b[3,3] = 1 #Marker for first phase
    >>> b[6,6] = 2 #Marker for second phase
    >>> random_walker(a, b)
    array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],
           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],
           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=int32)

    """
    # Parse input data
    if not multichannel:
        # We work with 4-D arrays of floats
        dims = data.shape
        data = np.atleast_3d(img_as_float(data))
        data.shape += (1,)
    else:
        dims = data[..., 0].shape
        assert multichannel and data.ndim > 2, 'For multichannel input, data \
                                                must have >= 3 dimensions.'
        data = img_as_float(data)
        if data.ndim == 3:
            data.shape += (1,)
            data = data.transpose((0, 1, 3, 2))

    if copy:
        labels = np.copy(labels)
    label_values = np.unique(labels)
    # Reorder label values to have consecutive integers (no gaps)
    if np.any(np.diff(label_values) != 1):
        mask = labels >= 0
        labels[mask] = rank_order(labels[mask])[0].astype(labels.dtype)
    labels = labels.astype(np.int32)
    # If the array has pruned zones, be sure that no isolated pixels
    # exist between pruned zones (they could not be determined)
    if np.any(labels < 0):
        filled = ndimage.binary_propagation(labels > 0, mask=labels >= 0)
        labels[np.logical_and(np.logical_not(filled), labels == 0)] = -1
        del filled
    labels = np.atleast_3d(labels)
    if np.any(labels < 0):
        lap_sparse = _build_laplacian(data, mask=labels >= 0, beta=beta,
                                      depth=depth, multichannel=multichannel)
    else:
        lap_sparse = _build_laplacian(data, beta=beta, depth=depth,
                                      multichannel=multichannel)
    lap_sparse, B = _buildAB(lap_sparse, labels)
    # We solve the linear system
    # lap_sparse X = B
    # where X[i, j] is the probability that a marker of label i arrives
    # first at pixel j by anisotropic diffusion.
    if mode == 'cg':
        X = _solve_cg(lap_sparse, B, tol=tol,
                      return_full_prob=return_full_prob)
    if mode == 'cg_mg':
        if not amg_loaded:
            warnings.warn(
            """pyamg (http://code.google.com/p/pyamg/)) is needed to use
            this mode, but is not installed. The 'cg' mode will be used
            instead.""")
            X = _solve_cg(lap_sparse, B, tol=tol,
                          return_full_prob=return_full_prob)
        else:
            X = _solve_cg_mg(lap_sparse, B, tol=tol,
                             return_full_prob=return_full_prob)
    if mode == 'bf':
        X = _solve_bf(lap_sparse, B,
                      return_full_prob=return_full_prob)
    # Clean up results
    if return_full_prob:
        labels = labels.astype(np.float)
        X = np.array([_clean_labels_ar(Xline, labels,
                     copy=True).reshape(dims) for Xline in X])
        for i in range(1, int(labels.max()) + 1):
            mask_i = np.squeeze(labels == i)
            X[:, mask_i] = 0
            X[i - 1, mask_i] = 1
    else:
        X = _clean_labels_ar(X + 1, labels).reshape(dims)
    return X


def _solve_bf(lap_sparse, B, return_full_prob=False):
    """
    solves lap_sparse X_i = B_i for each phase i. An LU decomposition
    of lap_sparse is computed first. For each pixel, the label i
    corresponding to the maximal X_i is returned.
    """
    lap_sparse = lap_sparse.tocsc()
    solver = sparse.linalg.factorized(lap_sparse.astype(np.double))
    X = np.array([solver(np.array((-B[i]).todense()).ravel())\
                  for i in range(len(B))])
    if not return_full_prob:
        X = np.argmax(X, axis=0)
    return X


def _solve_cg(lap_sparse, B, tol, return_full_prob=False):
    """
    solves lap_sparse X_i = B_i for each phase i, using the conjugate
    gradient method. For each pixel, the label i corresponding to the
    maximal X_i is returned.
    """
    lap_sparse = lap_sparse.tocsc()
    X = []
    for i in range(len(B)):
        x0 = cg(lap_sparse, -B[i].todense(), tol=tol)[0]
        X.append(x0)
    if not return_full_prob:
        X = np.array(X)
        X = np.argmax(X, axis=0)
    return X


def _solve_cg_mg(lap_sparse, B, tol, return_full_prob=False):
    """
    solves lap_sparse X_i = B_i for each phase i, using the conjugate
    gradient method with a multigrid preconditioner (ruge-stuben from
    pyamg). For each pixel, the label i corresponding to the maximal
    X_i is returned.
    """
    X = []
    ml = ruge_stuben_solver(lap_sparse)
    M = ml.aspreconditioner(cycle='V')
    for i in range(len(B)):
        x0 = cg(lap_sparse, -B[i].todense(), tol=tol, M=M, maxiter=30)[0]
        X.append(x0)
    if not return_full_prob:
        X = np.array(X)
        X = np.argmax(X, axis=0)
    return X
