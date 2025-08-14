#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

"""Cython code used in `remove_objects_by_distance` function."""

cimport numpy as cnp

from .._shared.fused_numerics cimport np_anyint


def _remove_objects_by_distance(
    np_anyint[::1] out not None,
    Py_ssize_t[::1] border_indices not None,
    Py_ssize_t[::1] inner_indices not None,
    kdtree,
    cnp.float64_t p_norm,
    cnp.float64_t min_distance,
    tuple shape,
):
    """Remove objects, in specified order, until remaining are a minimum distance apart.

    Remove labeled objects from an image until the remaining ones are spaced
    more than a given distance from one another. By default, smaller objects
    are removed first.

    Parameters
    ----------
    out :
        An array with labels for each object in `image` matching it in shape.
    border_indices, inner_indices :
        Indices into `out` for the border of objects (`border_indices`) and
        the inner part of objects (`inner_indices`). `border_indices`
        determines the iteration order; objects that are indexed first are
        preserved. Indices must be sorted such, that indices pointing to the
        same object are next to each other.
    kdtree : scipy.spatial.cKDTree
        A KDTree containing the coordinates of all objects in `image`.
    min_distance :
        The minimal allowed distance between objects.
    p_norm :
        The Minkowski p-norm used to calculate the distance between objects.
        Defaults to 2 which corresponds to the Euclidean distance.
    shape :
        The shape of the unraveled `image`.
    """
    cdef:
        Py_ssize_t i_indices, j_indices  # Loop variables to index `indices`
        Py_ssize_t i_out  # Loop variable to index `out`
        np_anyint object_id, other_id
        list neighborhood
        set remembered_ids

    remembered_ids = set()
    for i_indices in range(border_indices.shape[0]):
        i_out = border_indices[i_indices]
        object_id = out[i_out]
        # Skip if sample is part of a removed object
        if object_id == 0:
            continue

        neighborhood = kdtree.query_ball_point(
            kdtree.data[i_indices, ...],
            r=min_distance,
            p=p_norm,
        )
        for j_indices in neighborhood:
            # Check object IDs in neighborhood
            other_id = out[border_indices[j_indices]]
            if other_id != 0 and other_id != object_id:
                # If neighbor ID wasn't already removed or is the current one
                # remove the boundary and remember the ID
                _remove_object(out, border_indices, j_indices)
                remembered_ids.add(other_id)

    # Delete inner parts of remembered objects
    for j_indices in range(inner_indices.shape[0]):
        object_id = out[inner_indices[j_indices]]
        if object_id != 0 and object_id in remembered_ids:
            _remove_object(out, inner_indices, j_indices)


cdef inline _remove_object(
    np_anyint[::1] out,
    Py_ssize_t[::1] indices,
    Py_ssize_t start
):
    """Delete an object.

    Starting from `start`, iterate the `indices` in both directions and assign
    0 until the object ID changes

    Parameters
    ----------
    out :
        An array with labels for each object in `image` matching it in shape.
    indices :
        Indices into `out` for objects. Indices must be sorted such, that
        indices pointing to the same object are next to each other.
    start :
        Index into `indices` for the object.
    """
    cdef:
        Py_ssize_t k_indices, k_labels
        np_anyint remove_id

    with nogil:
        remove_id = out[indices[start]]

        for k_indices in range(start, -1, -1):
            k_labels = indices[k_indices]
            if remove_id == out[k_labels]:
                out[k_labels] = 0
            else:
                break
        for k_indices in range(start + 1, indices.shape[0]):
            k_labels = indices[k_indices]
            if remove_id == out[k_labels]:
                out[k_labels] = 0
            else:
                break
