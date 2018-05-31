from math import ceil
from multiprocessing import cpu_count
from time import time
import numpy as np

__all__ = ['apply_parallel']


try:
    import dask.array as da
    dask_available = True
except ImportError:
    dask_available = False


def _get_chunks(shape, ncpu):
    """Split the array into equal sized chunks based on the number of
    available processors. The last chunk in each dimension absorbs the
    remainder array elements if the number of CPUs does not divide evenly into
    the number of array elements.

    Examples
    --------
    >>> _get_chunks((4, 4), 4)
    ((2, 2), (2, 2))
    >>> _get_chunks((4, 4), 2)
    ((2, 2), (4,))
    >>> _get_chunks((5, 5), 2)
    ((2, 3), (5,))
    >>> _get_chunks((2, 4), 2)
    ((1, 1), (4,))
    """
    chunks = []
    nchunks_per_dim = int(ceil(ncpu ** (1./len(shape))))

    used_chunks = 1
    for i in shape:
        if used_chunks < ncpu:
            regular_chunk = i // nchunks_per_dim
            remainder_chunk = regular_chunk + (i % nchunks_per_dim)

            if regular_chunk == 0:
                chunk_lens = (remainder_chunk,)
            else:
                chunk_lens = ((regular_chunk,) * (nchunks_per_dim - 1) +
                              (remainder_chunk,))
        else:
            chunk_lens = (i,)

        chunks.append(chunk_lens)
        used_chunks *= nchunks_per_dim
    return tuple(chunks)


def _ensure_dask_array(array, chunks=None):
    if isinstance(array, da.Array):
        return array

    return da.from_array(array, chunks=chunks)


def apply_parallel(function, array, chunks=None, depth=0, mode=None,
                   extra_arguments=(), extra_keywords={}, *, compute=True):
    """Map a function in parallel across an array.

    Split an array into possibly overlapping chunks of a given depth and
    boundary type, call the given function in parallel on the chunks, combine
    the chunks and return the resulting array.

    Parameters
    ----------
    function : function
        Function to be mapped which takes an array as an argument.
    array : numpy array or dask array
        Array which the function will be applied to.
    chunks : int, tuple, or tuple of tuples, optional
        A single integer is interpreted as the length of one side of a square
        chunk that should be tiled across the array.  One tuple of length
        ``array.ndim`` represents the shape of a chunk, and it is tiled across
        the array.  A list of tuples of length ``ndim``, where each sub-tuple
        is a sequence of chunk sizes along the corresponding dimension. If
        None, the array is broken up into chunks based on the number of
        available cpus. More information about chunks is in the documentation
        `here <https://dask.pydata.org/en/latest/array-design.html>`_.
    depth : int, optional
        Integer equal to the depth of the added boundary cells. Defaults to
        zero.
    mode : {'reflect', 'symmetric', 'periodic', 'wrap', 'nearest', 'edge'}, optional
        type of external boundary padding.
    extra_arguments : tuple, optional
        Tuple of arguments to be passed to the function.
    extra_keywords : dictionary, optional
        Dictionary of keyword arguments to be passed to the function.
    compute : bool, optional
        Whether to compute right away (default) or
        skip computing and return a dask Array.

    Returns
    -------
    out : ndarray or dask Array
        Returns the result of the applying the operation.
        Type is dependent on the ``compute`` argument.

    Notes
    -----
    Numpy edge modes 'symmetric', 'wrap', and 'edge' are converted to the
    equivalent ``dask`` boundary modes 'reflect', 'periodic' and 'nearest',
    respectively.
    Setting ``compute=False`` can be useful for chaining later operations.
    For example region selection to preview a result or storing large data
    to disk instead of loading in memory.

    """
    if not dask_available:
        raise RuntimeError("Could not import 'dask'.  Please install "
                           "using 'pip install dask'")

    if chunks is None:
        shape = array.shape
        try:
            ncpu = cpu_count()
        except NotImplementedError:
            ncpu = 4
        chunks = _get_chunks(shape, ncpu)

    if mode == 'wrap':
        mode = 'periodic'
    elif mode == 'symmetric':
        mode = 'reflect'
    elif mode == 'edge':
        mode = 'nearest'

    def wrapped_func(arr):
        return function(arr, *extra_arguments, **extra_keywords)

    darr = _ensure_dask_array(array, chunks=chunks)

    res = darr.map_overlap(wrapped_func, depth, boundary=mode)
    if compute:
        res = res.compute()

    return res


def check_parallel(function, im=None, shape=(1000, 1000),
                        dtype=np.float, depth_max=10, full_output=False,
                        verbose = True,
                        extra_arguments=(), extra_keywords={}):
    """
    Run a function on an image with and without chunking with
    ``apply_parallel``, and check whether results are consistent.
    Returns the smallest overlap size giving the same result as the original
    function.


    Parameters
    ----------
    function : function
        Function to be mapped which takes an array as an argument.
    im : numpy array
        Image array which the function will be applied to.
    shape: tuple
        Shape of the image. If ``im`` is defined, ``im.shape`` will be used.
    dtype: dtype
        dtype of the image. If ``im`` is defined, ``im.dtype`` is given.
    depth_max: int
        Maximum overlap size tested.
    full_output: bool
        If True, the output of apply_parallel is returned.
    verbose: bool
        Additonal comments.
    extra_arguments : tuple, optional
        Tuple of arguments to be passed to the function.
    extra_keywords : dictionary, optional
        Dictionary of keyword arguments to be passed to the function.
    """

    # Build an image with chosen dtype and shape
    if im is not None:
        shape = im.shape
        dtype = im.dtype
    if im is None:
        if dtype is np.float:
            im = np.random.random(shape)
        elif dtype is np.bool: # binary objects from data module
            from ..data import binary_blobs
            im = binary_blobs(shape[0])
        else:
            im = np.random.randint(0, 256, size=shape) 

    # Execute function without appyly_parallel and time it
    t_init = time()
    out = function(im, *extra_arguments, **extra_keywords)
    t_end = time()
    t_not_parallel = t_end - t_init
    res = [out]

    # Execute function with apply_parallel for different overlaps and time it
    l = im.shape[0] // 2
    is_equal = False
    depth = -1
    while not is_equal and depth < depth_max: # stop when same value as out
        depth += 1
        t_init = time()
        out_parallel = apply_parallel(function, im, chunks=l, depth=depth,
                                        mode='none',
                                        extra_arguments=extra_arguments,
                                        extra_keywords=extra_keywords)
        t_end = time()
        t_parallel = t_end - t_init
        is_equal = np.allclose(out, out_parallel)
        if full_output:
            res.append(out_parallel)


    if depth == depth_max:
        raise ValueError(
"""An overlap value resulting in the same output as the original (non-chunked)
function could not be found.

You can increase the max_depth parameter for a larger overlap, but it is
possible that this function is not suited for chunking.""")

    if verbose:
        print("""An overlap value of %d is safe to use with the input image.
Note that the overlap value can depend on the image, and not only on the
function.

Execution time of %s without chunking: %f
Execution time of %s with apply_parallel; %f"""
            %(depth, function.__name__, t_not_parallel,
                                function.__name__, t_parallel)
         )
    if full_output:
        return depth, np.array(res)
    else:
        return depth
