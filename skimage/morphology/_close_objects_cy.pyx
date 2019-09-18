#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False


"""Cython code used in `remove_close_objects` function."""


import numpy as np
cimport numpy as cnp


# Must be defined to use QueueWithHistory
ctypedef Py_ssize_t QueueItem


include "_queue_with_history.pxi"


def _remove_close_objects(
    cnp.uint8_t[::1] image not None,
    cnp.uint32_t[::1] labels not None,
    Py_ssize_t[::1] indices not None,
    Py_ssize_t[::1] neighbor_offsets not None,
    kdtree,
    cnp.float64_t minimal_distance,
    tuple shape,
):
    """Remove objects until a minimal distance is ensured.

    Iterates over all objects (connected pixels that are True) inside an image
    and removes neighboring objects until all remaining ones are at least a
    minimal euclidean distance from each other.

    Parameters
    ----------
    image :
        The raveled view of a n-dimensional array. which is modified inplace.
    labels :
        An array with labels for each object in `image` matching it in shape.
    indices :
        Indices into `image` that determines the priority.
    neighbor_offsets :
        A one-dimensional array that contains the offsets to find the
        connected neighbors for any index in `image`.
    kdtree : scipy.spatial.cKDTree
        A KDTree containing the coordinates of all objects in `image`.
    minimal_distance :
        The minimal allowed euclidean distance between objects.
    shape :
        The shape of the unraveled `image`.
    """
    cdef:
        Py_ssize_t i, j, index_i, index_j
        list in_range
        QueueWithHistory queue

    queue_init(&queue, 64)
    try:
        for i in range(indices.shape[0]):
            index_i = indices[i]
            if image[index_i] == 0:
                continue

            in_range = kdtree.query_ball_point(
                np.unravel_index(index_i, shape), minimal_distance
            )
            for j in in_range:
                index_j = indices[j]
                if (
                    image[index_j] > 0
                    and labels[index_i] != labels[index_j]
                ):
                    _remove_object(
                        image=image,
                        labels=labels,
                        start_index=index_j,
                        neighbor_offsets=neighbor_offsets,
                        queue_ptr=&queue,
                    )
    finally:
        queue_exit(&queue)


cdef inline void _remove_object(
    cnp.uint8_t[::1] image,
    cnp.uint32_t[::1] labels,
    Py_ssize_t start_index,
    Py_ssize_t[::1] neighbor_offsets,
    QueueWithHistory* queue_ptr,
):
    """Remove single connected object.
    
    Performs a flood-fill on the object with the value 0.

    Parameters
    ----------
    image :
        The raveled view of a n-dimensional array. which is modified inplace.
    start_index :
        Start position for the flood-fill.
    neighbor_offsets :
        A one-dimensional array that contains the offsets to find the
        connected neighbors for any index in `image`.
    queue_ptr :
        Pointer to initialized queue.
    """
    cdef Py_ssize_t i, point, neighbor, max_index
    cdef cnp.uint32_t label

    max_index = image.shape[0]
    queue_clear(queue_ptr)
    queue_push(queue_ptr, &start_index)
    image[start_index] = 0
    label = labels[start_index]
    while queue_pop(queue_ptr, &point):
        for i in range(neighbor_offsets.shape[0]):
            neighbor = point + neighbor_offsets[i]
            if not 0 <= neighbor < max_index:
                continue
            # The algorithm might cross the image edge when two objects are
            # neighbors in the raveled view -> check label to avoid that
            if image[neighbor] > 0 and labels[neighbor] == label:
                queue_push(queue_ptr, &neighbor)
                image[neighbor] = 0
