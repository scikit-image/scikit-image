"""
Random walker segmentation algorithm

from *Random walks for image segmentation*, Leo Grady, IEEE Trans
Pattern Anal Mach Intell. 2006 Nov;28(11):1768-83.

Installing pyamg and using the 'cg_mg' mode of random_walker improves
significantly the performance.
"""

import warnings
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import lobpcg
from scipy.sparse.linalg.eigen.lobpcg.lobpcg import symeig
from scipy.sparse.linalg import eigsh
from scipy import linalg

from random_walker_segmentation import _make_graph_edges_3d, _mask_edges_weights, _compute_gradients_3d
from ..util import img_as_float

def norm(v):
    v = np.asarray(v)
    __nrm2, = linalg.get_blas_funcs(['nrm2'], [v])
    return __nrm2(v)


def _graph_laplacian_sparse(graph, normed=False, return_diag=False):
    n_nodes = graph.shape[0]
    if not graph.format == 'coo':
        lap = (-graph).tocoo()
    else:
        lap = -graph.copy()
    diag_mask = (lap.row == lap.col)
    if not diag_mask.sum() == n_nodes:
        # The sparsity pattern of the matrix has holes on the diagonal,
        # we need to fix that
        diag_idx = lap.row[diag_mask]

        lap = lap.tolil()

        diagonal_holes = list(set(range(n_nodes)).difference(
                                diag_idx))
        lap[diagonal_holes, diagonal_holes] = 1
        lap = lap.tocoo()
        diag_mask = (lap.row == lap.col)
    lap.data[diag_mask] = 0
    w = -np.asarray(lap.sum(axis=1)).squeeze()
    if normed:
        w = np.sqrt(w)
        w_zeros = w == 0
        w[w_zeros] = 1
        lap.data /= w[lap.row]
        lap.data /= w[lap.col]
        lap.data[diag_mask] = (1 - w_zeros).astype(lap.data.dtype)
    else:
        lap.data[diag_mask] = w[lap.row[diag_mask]]
    if return_diag:
        return lap, w
    return lap


def graph_laplacian(graph, normed=False, return_diag=False):
    """ Return the Laplacian of the given graph.
    """
    if normed and (np.issubdtype(graph.dtype, np.int)
                    or np.issubdtype(graph.dtype, np.uint)):
        graph = graph.astype(np.float)
    if sparse.isspmatrix(graph):
        return _graph_laplacian_sparse(graph, normed=normed,
                                       return_diag=return_diag)
    else:
        # We have a numpy array
        return _graph_laplacian_dense(graph, normed=normed,
                                       return_diag=return_diag)


def img_to_graph(img, mask=None, return_as=sparse.coo_matrix, dtype=None):
    """Graph of the pixel-to-pixel gradient connections

    Edges are weighted with the gradient values.

    Parameters
    ===========
    img: ndarray, 2D or 3D
        2D or 3D image
    mask : ndarray of booleans, optional
        An optional mask of the image, to consider only part of the
        pixels.
    return_as: np.ndarray or a sparse matrix class, optional
        The class to use to build the returned adjacency matrix.
    dtype: None or dtype, optional
        The data of the returned sparse matrix. By default it is the
        dtype of img
    """
    img = np.atleast_3d(img)
    n_x, n_y, n_z = img.shape
    return _to_graph(n_x, n_y, n_z, mask, img, return_as, dtype)


def _compute_gradient_3d(edges, img):
    n_x, n_y, n_z = img.shape
    gradient = np.abs(img[edges[0] // (n_y * n_z),
                                (edges[0] % (n_y * n_z)) // n_z,
                                (edges[0] % (n_y * n_z)) % n_z] -
                                img[edges[1] // (n_y * n_z),
                                (edges[1] % (n_y * n_z)) // n_z,
                                (edges[1] % (n_y * n_z)) % n_z])
    return gradient


def _to_graph(n_x, n_y, n_z, mask=None, img=None,
              return_as=sparse.coo_matrix, dtype=None):
    """Auxiliary function for img_to_graph and grid_to_graph
    """
    edges = _make_graph_edges_3d(n_x, n_y, n_z)

    if dtype is None:
        if img is None:
            dtype = np.int
        else:
            dtype = img.dtype

    if img is not None:
        img = np.atleast_3d(img)
        weights = _compute_gradient_3d(edges, img)
        if mask is not None:
            edges, weights = _mask_edges_weights(mask, edges, weights)
            diag = img.squeeze()[mask]
        else:
            diag = img.ravel()
        n_voxels = diag.size
    else:
        if mask is not None:
            mask = mask.astype(np.bool)
            edges = _mask_edges_weights(mask, edges)
            n_voxels = np.sum(mask)
        else:
            n_voxels = n_x * n_y * n_z
        weights = np.ones(edges.shape[1], dtype=dtype)
        diag = np.ones(n_voxels, dtype=dtype)

    diag_idx = np.arange(n_voxels)
    i_idx = np.hstack((edges[0], edges[1]))
    j_idx = np.hstack((edges[1], edges[0]))
    graph = sparse.coo_matrix((np.hstack((weights, weights, diag)),
                              (np.hstack((i_idx, diag_idx)),
                               np.hstack((j_idx, diag_idx)))),
                              (n_voxels, n_voxels),
                              dtype=dtype)
    if return_as is np.ndarray:
        return graph.todense()
    return return_as(graph)


def discretize(vectors, copy=True, max_svd_restarts=30, n_iter_max=20,
               random_state=None):
    """Search for a partition matrix (clustering) which is closest to the
    eigenvector embedding.

    Parameters
    ----------
    vectors : array-like, shape: (n_samples, n_clusters)
        The embedding space of the samples.

    copy : boolean, optional, default: True
        Whether to copy vectors, or perform in-place normalization.

    max_svd_restarts : int, optional, default: 30
        Maximum number of attempts to restart SVD if convergence fails

    n_iter_max : int, optional, default: 30
        Maximum number of iterations to attempt in rotation and partition
        matrix search if machine precision convergence is not reached

    random_state: int seed, RandomState instance, or None (default)
        A pseudo random number generator used for the initialization of the
        of the rotation matrix

    Returns
    -------
    labels : array of integers, shape: n_samples
        The labels of the clusters.

    References
    ----------

    - Multiclass spectral clustering, 2003
      Stella X. Yu, Jianbo Shi
      http://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf

    Notes
    -----

    The eigenvector embedding is used to iteratively search for the
    closest discrete partition.  First, the eigenvector embedding is
    normalized to the space of partition matrices. An optimal discrete
    partition matrix closest to this normalized embedding multiplied by
    an initial rotation is calculated.  Fixing this discrete partition
    matrix, an optimal rotation matrix is calculated.  These two
    calculations are performed until convergence.  The discrete partition
    matrix is returned as the clustering solution.  Used in spectral
    clustering, this method tends to be faster and more robust to random
    initialization than k-means.

    """

    from scipy.sparse import csc_matrix
    from scipy.linalg import LinAlgError

    random_state = np.random.RandomState(random_state)

    #    vectors = as_float_array(vectors, copy=copy)

    eps = np.finfo(float).eps
    n_samples, n_components = vectors.shape

    # Normalize the eigenvectors to an equal length of a vector of ones.
    # Reorient the eigenvectors to point in the negative direction with respect
    # to the first element.  This may have to do with constraining the
    # eigenvectors to lie in a specific quadrant to make the discretization
    # search easier.
    norm_ones = np.sqrt(n_samples)
    for i in range(vectors.shape[1]):
        vectors[:, i] = (vectors[:, i] / norm(vectors[:, i])) \
            * norm_ones
        if vectors[0, i] != 0:
            vectors[:, i] = -1 * vectors[:, i] * np.sign(vectors[0, i])

    # Normalize the rows of the eigenvectors.  Samples should lie on the unit
    # hypersphere centered at the origin.  This transforms the samples in the
    # embedding space to the space of partition matrices.
    vectors = vectors / np.sqrt((vectors ** 2).sum(axis=1))[:, np.newaxis]

    svd_restarts = 0
    has_converged = False

    # If there is an exception we try to randomize and rerun SVD again
    # do this max_svd_restarts times.
    while (svd_restarts < max_svd_restarts) and not has_converged:

        # Initialize first column of rotation matrix with a row of the
        # eigenvectors
        rotation = np.zeros((n_components, n_components))
        rotation[:, 0] = vectors[random_state.randint(n_samples), :].T

        # To initialize the rest of the rotation matrix, find the rows
        # of the eigenvectors that are as orthogonal to each other as
        # possible
        c = np.zeros(n_samples)
        for j in range(1, n_components):
            # Accumulate c to ensure row is as orthogonal as possible to
            # previous picks as well as current one
            c += np.abs(np.dot(vectors, rotation[:, j - 1]))
            rotation[:, j] = vectors[c.argmin(), :].T

        last_objective_value = 0.0
        n_iter = 0

        while not has_converged:
            n_iter += 1

            t_discrete = np.dot(vectors, rotation)

            labels = t_discrete.argmax(axis=1)
            vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_components))

            t_svd = vectors_discrete.T * vectors

            try:
                U, S, Vh = np.linalg.svd(t_svd)
                svd_restarts += 1
            except LinAlgError:
                print "SVD did not converge, randomizing and trying again"
                break

            ncut_value = 2.0 * (n_samples - S.sum())
            if ((abs(ncut_value - last_objective_value) < eps) or
               (n_iter > n_iter_max)):
                has_converged = True
            else:
                # otherwise calculate rotation and continue
                last_objective_value = ncut_value
                rotation = np.dot(Vh.T, U.T)

    if not has_converged:
        raise LinAlgError('SVD did not converge')
    return labels

def _set_diag(laplacian, value):
    """Set the diagonal of the laplacian matrix and convert it to a
    sparse format well suited for eigenvalue decomposition

    Parameters
    ----------
    laplacian : array or sparse matrix
        The graph laplacian
    value : float
        The value of the diagonal

    Returns
    -------
    laplacian : array or sparse matrix
        An array of matrix in a form that is well suited to fast
        eigenvalue decomposition, depending on the band width of the
        matrix.
    """
    n_nodes = laplacian.shape[0]
    # We need all entries in the diagonal to values
    if not sparse.isspmatrix(laplacian):
        laplacian.flat[::n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        diag_idx = (laplacian.row == laplacian.col)
        laplacian.data[diag_idx] = value
        # If the matrix has a small number of diagonals (as in the
        # case of structured matrices comming from images), the
        # dia format might be best suited for matvec products:
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            # 3 or less outer diagonals on each side
            laplacian = laplacian.todia()
        else:
            # csr has the fastest matvec and is thus best suited to
            # arpack
            laplacian = laplacian.tocsr()
    return laplacian


def _spectral_embedding(adjacency, n_components=8, eigen_solver=None,
                       random_state=None, eigen_tol=0.0,
                       norm_laplacian=True, drop_first=True,
                       mode=None):
    """Project the sample on the first eigen vectors of the graph Laplacian

    The adjacency matrix is used to compute a normalized graph Laplacian
    whose spectrum (especially the eigen vectors associated to the
    smallest eigen values) has an interpretation in terms of minimal
    number of cuts necessary to split the graph into comparably sized
    components.

    This embedding can also 'work' even if the ``adjacency`` variable is
    not strictly the adjacency matrix of a graph but more generally
    an affinity or similarity matrix between samples (for instance the
    heat kernel of a euclidean distance matrix or a k-NN matrix).

    However care must taken to always make the affinity matrix symmetric
    so that the eigen vector decomposition works as expected.

    Parameters
    ----------
    adjacency : array-like or sparse matrix, shape: (n_samples, n_samples)
        The adjacency matrix of the graph to embed.

    n_components : integer, optional
        The dimension of the projection subspace.

    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities

    random_state : int seed, RandomState instance, or None (default)
        A pseudo random number generator used for the initialization of the
        lobpcg eigen vectors decomposition when eigen_solver == 'amg'.
        By default, arpack is used.

    eigen_tol : float, optional, default: 0.0
        Stopping criterion for eigendecomposition of the Laplacian matrix
        when using arpack eigen_solver.

    drop_first : bool, optional, default: True
        Whether to drop the first eigenvector. For spectral embedding, this
        should be True as the first eigenvector should be constant vector for
        connected graph, but for spectral clustering, this should be kept as
        False to retain the first eigenvector.

    Returns
    -------
    embedding : array, shape: (n_samples, n_components)
        The reduced samples

    Notes
    -----
    Spectral embedding is most useful when the graph has one connected
    component. If there graph has many components, the first few
    eigenvectors will simply uncover the connected components of the graph.

    References
    ----------
    [1] http://en.wikipedia.org/wiki/LOBPCG
    [2] Toward the Optimal Preconditioned Eigensolver: Locally Optimal
        Block Preconditioned Conjugate Gradient Method
        Andrew V. Knyazev
        http://dx.doi.org/10.1137%2FS1064827500366124
    """

    try:
        from pyamg import smoothed_aggregation_solver
    except ImportError:
        if eigen_solver == "amg":
            raise ValueError("The eigen_solver was set to 'amg', but pyamg is "
                             "not available.")

    if not mode is None:
        warnings.warn("'mode' was renamed to eigen_solver "
                      "and will be removed in 0.15.",
                      DeprecationWarning)
        eigen_solver = mode

    random_state = np.random.RandomState(random_state)


    n_nodes = adjacency.shape[0]
    # Whether to drop the first eigenvector
    if drop_first:
        n_components = n_components + 1
    # Check that the matrices given is symmetric
    if ((not sparse.isspmatrix(adjacency) and
         not np.all((adjacency - adjacency.T) < 1e-10)) or
        (sparse.isspmatrix(adjacency) and
         (adjacency - adjacency.T).nnz > 0)):
        warnings.warn("Graph adjacency matrix should be symmetric. "
                      "Converted to be symmetric by average with its "
                      "transpose.")
    adjacency = .5 * (adjacency + adjacency.T)


    if eigen_solver is None:
        eigen_solver = 'arpack'
    elif not eigen_solver in ('arpack', 'lobpcg', 'amg'):
        raise ValueError("Unknown value for eigen_solver: '%s'."
                         "Should be 'amg', 'arpack', or 'lobpcg'"
                         % eigen_solver)
    laplacian, dd = graph_laplacian(adjacency,
                                    normed=norm_laplacian, return_diag=True)
    if (eigen_solver == 'arpack'
        or eigen_solver != 'lobpcg' and
            (not sparse.isspmatrix(laplacian)
             or n_nodes < 5 * n_components)):
        # lobpcg used with eigen_solver='amg' has bugs for low number of nodes
        # for details see the source code in scipy:
        # https://github.com/scipy/scipy/blob/v0.11.0/scipy/sparse/linalg/eigen
        # /lobpcg/lobpcg.py#L237
        # or matlab:
        # http://www.mathworks.com/matlabcentral/fileexchange/48-lobpcg-m
        laplacian = _set_diag(laplacian, 1)

        # Here we'll use shift-invert mode for fast eigenvalues
        # (see http://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
        #  for a short explanation of what this means)
        # Because the normalized Laplacian has eigenvalues between 0 and 2,
        # I - L has eigenvalues between -1 and 1.  ARPACK is most efficient
        # when finding eigenvalues of largest magnitude (keyword which='LM')
        # and when these eigenvalues are very large compared to the rest.
        # For very large, very sparse graphs, I - L can have many, many
        # eigenvalues very near 1.0.  This leads to slow convergence.  So
        # instead, we'll use ARPACK's shift-invert mode, asking for the
        # eigenvalues near 1.0.  This effectively spreads-out the spectrum
        # near 1.0 and leads to much faster convergence: potentially an
        # orders-of-magnitude speedup over simply using keyword which='LA'
        # in standard mode.
        try:
            lambdas, diffusion_map = eigsh(-laplacian, k=n_components,
                                           sigma=1.0, which='LM',
                                           tol=eigen_tol)
            embedding = diffusion_map.T[n_components::-1] * dd
        except RuntimeError:
            # When submatrices are exactly singular, an LU decomposition
            # in arpack fails. We fallback to lobpcg
            eigen_solver = "lobpcg"

    if eigen_solver == 'amg':
        # Use AMG to get a preconditioner and speed up the eigenvalue
        # problem.
        if not sparse.issparse(laplacian):
            warnings.warn("AMG works better for sparse matrices")
        laplacian = laplacian.astype(np.float)  # lobpcg needs native floats
        laplacian = _set_diag(laplacian, 1)
        ml = smoothed_aggregation_solver(atleast2d_or_csr(laplacian))
        M = ml.aspreconditioner()
        X = random_state.rand(laplacian.shape[0], n_components + 1)
        X[:, 0] = dd.ravel()
        lambdas, diffusion_map = lobpcg(laplacian, X, M=M, tol=1.e-12,
                                        largest=False)
        embedding = diffusion_map.T * dd
        if embedding.shape[0] == 1:
            raise ValueError

    elif eigen_solver == "lobpcg":
        laplacian = laplacian.astype(np.float)  # lobpcg needs native floats
        if n_nodes < 5 * n_components + 1:
            # see note above under arpack why lobpcg has problems with small
            # number of nodes
            # lobpcg will fallback to symeig, so we short circuit it
            if sparse.isspmatrix(laplacian):
                laplacian = laplacian.todense()
            lambdas, diffusion_map = symeig(laplacian)
            embedding = diffusion_map.T[:n_components] * dd
        else:
            # lobpcg needs native floats
            laplacian = laplacian.astype(np.float)
            laplacian = _set_diag(laplacian, 1)
            # We increase the number of eigenvectors requested, as lobpcg
            # doesn't behave well in low dimension
            X = random_state.rand(laplacian.shape[0], n_components + 1)
            X[:, 0] = dd.ravel()
            lambdas, diffusion_map = lobpcg(laplacian, X, tol=1e-15,
                                            largest=False, maxiter=2000)
            embedding = diffusion_map.T[:n_components] * dd
            if embedding.shape[0] == 1:
                raise ValueError
    if drop_first:
        return embedding[1:n_components].T
    else:
        return embedding[:n_components].T

    
def _normalized_cut(affinity, n_cluster = 8):
    embedding = _spectral_embedding(affinity, n_components = n_cluster, drop_first = False)
    return discretize(embedding)


def _normalized_cut_segmentation(image, n_cluster = 8):
    graph = img_to_graph(image)
    beta = 8
    eps = 1e-6
    graph.data = np.exp(-beta * graph.data / image.std()) + eps
    label = _normalized_cut(graph, n_cluster = n_cluster)
    return label.reshape(image.shape)

def _with_slic_init(n_init_cluster = 100):
    
