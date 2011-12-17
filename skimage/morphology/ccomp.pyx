# -*- python -*-
#cython: cdivision=True

import numpy as np
cimport numpy as np

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
  http://repositories.cdlib.org/lbnl/LBNL-56864.

"""

# Tree operations implemented by an array as described in Wu et al.

DTYPE = np.int
ctypedef np.int_t DTYPE_t

cdef DTYPE_t find_root(np.int_t *work, np.int_t n):
    """Find the root of node n.

    """
    cdef np.int_t root = n
    while (work[root] < root):
        root = work[root]
    return root

cdef set_root(np.int_t *work, np.int_t n, np.int_t root):
    """
    Set all nodes on a path to point to new_root.

    """
    cdef np.int_t j
    while (work[n] < n):
        j = work[n]
        work[n] = root
        n = j

    work[n] = root


cdef join_trees(np.int_t *work, np.int_t n, np.int_t m):
    """Join two trees containing nodes n and m.

    """
    cdef np.int_t root = find_root(work, n)
    cdef np.int_t root_m

    if (n != m):
        root_m = find_root(work, m)

        if (root > root_m):
            root = root_m

        set_root(work, n, root)
        set_root(work, m, root)

# Connected components search as described in Fiorio et al.

def label(np.ndarray[DTYPE_t, ndim=2] input,
          int neighbors=8):
    """Label connected regions of an integer array.

    Two pixels are connected when they are neighbors and have the same value.
    They can be neighbors either in a 4- or 8-connected sense::

      4-connectivity      8-connectivity

           [ ]           [ ]  [ ]  [ ]
            |               \  |  /
      [ ]--[ ]--[ ]      [ ]--[ ]--[ ]
            |               /  |  \ 
           [ ]           [ ]  [ ]  [ ]

    Parameters
    ----------
    input : ndarray of dtype int
        Image to label.
    neighbors : {4, 8}, int
        Whether to use 4- or 8-connectivity.

    Returns
    -------
    labels : ndarray of dtype int
        Labeled array, where all connected regions are assigned the
        same integer value.

    Examples
    --------
    >>> x = np.eye(3).astype(int)
    >>> print x
    [[1 0 0]
     [0 1 0]
     [0 0 1]]

    >>> print m.label(x, neighbors=4)
    [[0 1 1]
     [2 3 1]
     [2 2 4]]

    >>> print m.label(x, neighbors=8)
    [[0 1 1]
     [1 0 1]
     [1 1 0]]

    """
    cdef np.int_t rows = input.shape[0]
    cdef np.int_t cols = input.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=2] data = input.copy()
    cdef np.ndarray[DTYPE_t, ndim=2] work

    work = np.arange(data.size, dtype=DTYPE).reshape((rows, cols))

    cdef np.int_t *work_p = <np.int_t*>work.data
    cdef np.int_t *data_p = <np.int_t*>data.data

    cdef np.int_t i, j

    if neighbors != 4 and neighbors != 8:
        raise ValueError('Neighbors must be either 4 or 8.')

    # Initialize the first row
    for j in range(1, cols):
        if data[0, j] == data[0, j-1]:
            join_trees(work_p, j, j-1)

    for i in range(1, rows):
        # Handle the first column
        if data[i, 0] == data[i-1, 0]:
            join_trees(work_p, i*cols, (i-1)*cols)

        if neighbors == 8:
            if data[i, 0] == data[i-1, 1]:
                join_trees(work_p, i*cols, (i-1)*cols + 1)

        for j in range(1, cols):
            if neighbors == 8:
                if data[i, j] == data[i-1, j-1]:
                    join_trees(work_p, i*cols + j, (i-1)*cols + j - 1)

            if data[i, j] == data[i-1, j]:
                join_trees(work_p, i*cols + j, (i-1)*cols + j)

            if neighbors == 8:
                if j < cols - 1:
                    if data[i, j] == data[i - 1, j + 1]:
                        join_trees(work_p, i*cols + j, (i-1)*cols + j + 1)

            if data[i, j] == data[i, j-1]:
                join_trees(work_p, i*cols + j, i*cols + j - 1)

    # Label output

    cdef np.int_t ctr = 0
    for i in range(rows):
        for j in range(cols):
            if (i*cols + j) == work[i, j]:
                data[i, j] = ctr
                ctr = ctr + 1
            else:
                data[i, j] = data_p[work[i, j]]

    return data
