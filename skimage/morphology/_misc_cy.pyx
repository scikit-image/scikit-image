#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False


"""Cython code used in `remove_near_objects` function."""


import numpy as np
cimport numpy as cnp
from libcpp.set cimport set

from .._shared.fused_numerics cimport np_anyint


def _remove_near_objects(
    np_anyint[::1] out not None,
    Py_ssize_t[::1] boundary_indices not None,
    Py_ssize_t[::1] inner_indices not None,
    kdtree,
    cnp.float64_t p_norm,
    cnp.float64_t minimal_distance,
    tuple shape,
):
    """Remove objects until a minimal distance is ensured.

    Iterates over all objects (pixels that aren't zero) inside an image and
    removes "nearby" objects until all remaining ones are spaced more than a
    given minimal distance from each other.

    Parameters
    ----------
    out :
        An array with labels for each object in `image` matching it in shape.
    boundary_indices, inner_indices :
        Indices into `out` for the boundary of objects (`boundary_indices`) and
        the inner part of objects (`inner_indices`). `boundary_indices`
        determines the iteration order; objects that are indexed first are
        preserved.
    kdtree : scipy.spatial.cKDTree
        A KDTree containing the coordinates of all objects in `image`.
    minimal_distance :
        The minimal allowed distance between objects.
    p_norm :
        The Minkowski p-norm used to calculate the distance between objects.
        Defaults to 2 which corresponds to the Euclidean distance.
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
        Py_ssize_t i_indices, i_out, j_indices, object_id, other_id
        list neighborhood
        set[np_anyint] remove_inner

    for i_indices in range(boundary_indices.shape[0]):
        i_out = boundary_indices[i_indices]
        object_id = out[i_out]
        # Skip if point is part of a removed object
        if object_id == 0:
            continue

        neighborhood = kdtree.query_ball_point(
            np.unravel_index(i_out, shape),
            r=minimal_distance,
            p=p_norm,
        )
        for j_indices in neighborhood:
            # Check object IDs in neighborhood
            other_id = out[boundary_indices[j_indices]]
            if other_id != 0 and other_id != object_id:
                with nogil:
                    # If neighbor ID wasn't already removed or is the current one
                    # remove the boundary and remember the ID
                    _remove_object(out, boundary_indices, j_indices)
                    remove_inner.insert(other_id)

    with nogil:
        # Delete inner parts of remembered objects
        for j_indices in range(inner_indices.shape[0]):
            other_id = out[inner_indices[j_indices]]
            if other_id != 0 and remove_inner.find(other_id) != remove_inner.end():
                _remove_object(out, inner_indices, j_indices)


cdef inline void _remove_object(
    np_anyint[::1] out,
    Py_ssize_t[::1] indices,
    Py_ssize_t starting_j_indices
) noexcept nogil:
    cdef:
        Py_ssize_t k_indices, k_labels
        np_anyint remove_id

    remove_id = out[indices[starting_j_indices]]

    for k_indices in range(starting_j_indices, -1, -1):
        k_labels = indices[k_indices]
        if remove_id == out[k_labels]:
            out[k_labels] = 0
        else:
            break
    for k_indices in range(starting_j_indices + 1, indices.shape[0]):
        k_labels = indices[k_indices]
        if remove_id == out[k_labels]:
            out[k_labels] = 0
        else:
            break
