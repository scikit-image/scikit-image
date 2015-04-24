from multiprocessing import cpu_count

import dask.array as da

__all__ = ['process_chunks']


def _get_chunks(shape, ncpu):
    """
    Split the array into equal sized chunks based on the number of
    available processors. The last chunk in each dimension absorbs the
    remainder array elements if the number of cpus does not divide evenly into
    the number of array elements.
    """
    chunks = []

    for i in shape:
        regular_chunk = i // ncpu
        remainder_chunk = regular_chunk + (i % ncpu)

        if regular_chunk == 0:
            chunk_lens = (remainder_chunk,)
        else:
            chunk_lens = (regular_chunk,) * (ncpu - 1) + (remainder_chunk,)

        chunks.append(chunk_lens)
    return tuple(chunks)


def process_chunks(function, array, args=(), kwargs={} chunks=None, depth=0, mode=None):
    """Map a function in parallel across an array.

    Split an array into possibly overlapping chunks of a given depth and
    boundary type, call the given function in parallel on the chunks, combine
    the chunks and return the resulting array.

    Parameters
    ----------
    function : function
        Function to be mapped which takes an array as an argument.
    array : numpy array
        array which the function will be applied to.
    chunks : int, tuple, or tuple of tuples
        One tuple of length array.ndim or a list of tuples of length ndim.
        Where each subtuple adds to the size of the array in the corresponding
        dimension. If None, the array is broken up into chunks based on the
        number of available cpus.
    depth : int
        integer equal to the depth of the internal external padding
    mode : 'reflect', 'periodic', 'wrap', 'nearest'
        type of external boundary padding

    Notes
    -----
    Be careful choosing the depth so that it is never larger than the length of
    a chunk.

    """
    if chunks == None:
        shape = array.shape
        ncpu = cpu_count()
        chunks = _get_chunks(shape, ncpu)

    if mode == 'wrap':
        mode = 'periodic'

    def wrapped_func(arr):
        return function(arr, *args, **kwargs)

    darr = da.from_array(array, chunks=chunks)
    return darr.map_overlap(wrapped_func, depth, boundary=mode).compute()
