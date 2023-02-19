#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False


"""Cython code used in `remove_near_objects` function."""


import numpy as np
cimport numpy as cnp

from .._shared.fused_numerics cimport np_anyint, np_numeric


def _remove_near_objects(
    np_anyint[::1] labels not None,
    Py_ssize_t[::1] indices not None,
    kdtree,
    cnp.float64_t p_norm,
    cnp.float64_t minimal_distance,
    tuple shape,
):
    """Remove objects until a minimal distance is ensured.

    Iterates over all objects (connected pixels that are True) inside an image
    and removes neighboring objects until all remaining ones are at least a
    minimal distance from each other.

    Parameters
    ----------
    labels :
        An array with labels for each object in `image` matching it in shape.
    indices :
        Indices into `labels` that determine the iteration order and
        thus which objects take precedence.
    kdtree : scipy.spatial.cKDTree
        A KDTree containing the coordinates of all objects in `image`.
    minimal_distance :
        The minimal allowed distance between objects.
    p_norm :
        Which Minkowski p-norm to use to calculate the distance between
        objects. Defaults to 2 which corresponds to the Euclidean distance
        while 1 corresponds to the Manatten distance.
    shape :
        The shape of the unraveled `image`.

    Notes
    -----
    This function and its partner function :func:`~._remove_object` can deal
    with objects where `labels` is 0 inside objects as long as its enclosing
    surface points (in the sense of the neighborhood) are labeled.
    This significantly improves the performance by reducing number of queries
    to the KDTree and its size.
    This effect grows with the size to surface ratio of all evaluated objects.
    """
    cdef:
        Py_ssize_t i_indices, i_labels, j_indices, object_id, other_id
        list in_range

    for i_indices in range(indices.shape[0]):
        i_labels = indices[i_indices]
        object_id = labels[i_labels]

        # Skip if point is part of a removed object
        if object_id == 0:
            continue

        in_range = kdtree.query_ball_point(
            np.unravel_index(i_labels, shape),
            r=minimal_distance,
            p=p_norm,
        )

        # Remove objects in `in_range` that don't share the same label ID
        for j_indices in in_range:
            other_id = labels[indices[j_indices]]
            if other_id != 0 and other_id != object_id:
                _remove_object(labels, indices, j_indices)


cdef inline _remove_object(
    np_anyint[::1] labels,
    Py_ssize_t[::1] indices,
    Py_ssize_t starting_j_indices
):
    cdef:
        Py_ssize_t k_indices, k_labels, remove_id

    remove_id = labels[indices[starting_j_indices]]

    for k_indices in range(starting_j_indices, -1, -1):
        k_labels = indices[k_indices]
        if remove_id == labels[k_labels]:
            labels[k_labels] = 0
        else:
            break
    for k_indices in range(starting_j_indices + 1, indices.size):
        k_labels = indices[k_indices]
        if remove_id == labels[k_labels]:
            labels[k_labels] = 0
        else:
            break
