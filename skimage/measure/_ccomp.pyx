#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
import warnings

cimport numpy as cnp

"""
See also:

  Christophe Fiorio and Jens Gustedt,
  "Two linear time Union-Find strategies for image processing",
  Theoretical Computer Science 154 (1996), pp. 165-181.

  Kensheng Wu, Ekow Otoo and Arie Shoshani,
  "Optimizing connected component labeling algorithms",
  Paper LBNL-56864, 2005,
  Lawrence Berkeley National Laboratory
  (University of California),
  http://repositories.cdlib.org/lbnl/LBNL-56864

"""

# Tree operations implemented by an array as described in Wu et al.
# The term "forest" is used to indicate an array that stores one or more trees

DTYPE = np.intp


cdef DTYPE_t find_root(DTYPE_t *forest, DTYPE_t n):
    """Find the root of node n.

    """
    cdef DTYPE_t root = n
    while (forest[root] < root):
        root = forest[root]
    return root


cdef inline void set_root(DTYPE_t *forest, DTYPE_t n, DTYPE_t root):
    """
    Set all nodes on a path to point to new_root.

    """
    cdef DTYPE_t j
    while (forest[n] < n):
        j = forest[n]
        forest[n] = root
        n = j

    forest[n] = root


cdef inline void join_trees(DTYPE_t *forest, DTYPE_t n, DTYPE_t m):
    """Join two trees containing nodes n and m.

    """
    cdef DTYPE_t root = find_root(forest, n)
    cdef DTYPE_t root_m

    if (n != m):
        root_m = find_root(forest, m)

        if (root > root_m):
            root = root_m

        set_root(forest, n, root)
        set_root(forest, m, root)


cdef inline void link_bg(DTYPE_t *forest, DTYPE_t n, DTYPE_t *background_node):
    """
    Link a node to the background node.

    """
    if background_node[0] == -999:
        background_node[0] = n

    join_trees(forest, n, background_node[0])


# Connected components search as described in Fiorio et al.
def label(input, DTYPE_t neighbors=8, background=None, return_num=False):
    """Label connected regions of an integer array.

    Two pixels are connected when they are neighbors and have the same value.
    They can be neighbors either in a 4- or 8-connected sense::

      4-connectivity      8-connectivity

           [ ]           [ ]  [ ]  [ ]
            |               \  |  /
      [ ]--[ ]--[ ]      [ ]--[ ]--[ ]
            |               /  |  \\
           [ ]           [ ]  [ ]  [ ]

    Parameters
    ----------
    input : ndarray of dtype int
        Image to label.
    neighbors : {4, 8}, int, optional
        Whether to use 4- or 8-connectivity.
    background : int, optional
        Consider all pixels with this value as background pixels, and label
        them as -1. (Note: background pixels will be labeled as 0 starting with
        version 0.12).
    return_num : bool, optional
        Whether to return the number of assigned labels.

    Returns
    -------
    labels : ndarray of dtype int
        Labeled array, where all connected regions are assigned the
        same integer value.
    num : int, optional
        Number of labels, which equals the maximum label index and is only
        returned if return_num is `True`.

    Examples
    --------
    >>> x = np.eye(3).astype(int)
    >>> print(x)
    [[1 0 0]
     [0 1 0]
     [0 0 1]]

    >>> print(m.label(x, neighbors=4))
    [[0 1 1]
     [2 3 1]
     [2 2 4]]

    >>> print(m.label(x, neighbors=8))
    [[0 1 1]
     [1 0 1]
     [1 1 0]]

    >>> x = np.array([[1, 0, 0],
    ...               [1, 1, 5],
    ...               [0, 0, 0]])

    >>> print(m.label(x, background=0))
    [[ 0 -1 -1]
     [ 0  0  1]
     [-1 -1 -1]]

    """
    cdef DTYPE_t rows = input.shape[0]
    cdef DTYPE_t cols = input.shape[1]

    cdef cnp.ndarray[DTYPE_t, ndim=2] data = np.array(input, copy=True,
                                                      dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=2] forest

    forest = np.arange(data.size, dtype=DTYPE).reshape((rows, cols))

    cdef DTYPE_t *forest_p = <DTYPE_t*>forest.data
    cdef DTYPE_t *data_p = <DTYPE_t*>data.data

    cdef DTYPE_t i, j

    cdef DTYPE_t background_val

    if background is None:
        background_val = -1
        warnings.warn(DeprecationWarning(
                'The default value for `background` will change to 0 in v0.12'
            ))
    else:
        background_val = background

    cdef DTYPE_t background_node = -999

    if neighbors != 4 and neighbors != 8:
        raise ValueError('Neighbors must be either 4 or 8.')

    # Initialize the first row
    if data[0, 0] == background_val:
        link_bg(forest_p, 0, &background_node)

    for j in range(1, cols):
        if data[0, j] == background_val:
            link_bg(forest_p, j, &background_node)

        if data[0, j] == data[0, j-1]:
            join_trees(forest_p, j, j-1)

    for i in range(1, rows):
        # Handle the first column
        if data[i, 0] == background_val:
            link_bg(forest_p, i * cols, &background_node)

        if data[i, 0] == data[i-1, 0]:
            join_trees(forest_p, i*cols, (i-1)*cols)

        if neighbors == 8:
            if data[i, 0] == data[i-1, 1]:
                join_trees(forest_p, i*cols, (i-1)*cols + 1)

        for j in range(1, cols):
            if data[i, j] == background_val:
                link_bg(forest_p, i * cols + j, &background_node)

            if neighbors == 8:
                if data[i, j] == data[i-1, j-1]:
                    join_trees(forest_p, i*cols + j, (i-1)*cols + j - 1)

            if data[i, j] == data[i-1, j]:
                join_trees(forest_p, i*cols + j, (i-1)*cols + j)

            if neighbors == 8:
                if j < cols - 1:
                    if data[i, j] == data[i - 1, j + 1]:
                        join_trees(forest_p, i*cols + j, (i-1)*cols + j + 1)

            if data[i, j] == data[i, j-1]:
                join_trees(forest_p, i*cols + j, i*cols + j - 1)

    # Label output
    cdef DTYPE_t ctr = 0
    for i in range(rows):
        for j in range(cols):
            if (i*cols + j) == background_node:
                data[i, j] = -1
            elif (i*cols + j) == forest[i, j]:
                data[i, j] = ctr
                ctr = ctr + 1
            else:
                data[i, j] = data_p[forest[i, j]]

    # Work around a bug in ndimage's type checking on 32-bit platforms
    if data.dtype == np.int32:
        data = data.view(np.int32)

    if return_num:
        return data, ctr
    else:
        return data
