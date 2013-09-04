__all__ = ['process_windows']

import numpy as np
from skimage.util import pad, view_as_windows, view_as_blocks
from itertools import product
from multiprocessing import Pool
from functools import partial


def process_windows(im, window_shape, fn, fn_kwargs={}, step_size=1, n_jobs=1):
    """ Apply function to each distinct window in the image.

    Uses view_as_windows to break up the image and apply a function.

    Parameters
    ----------
    im: ndarray
        input image
    
    window_shape: tuple or list
        size of each piece of the image
    
    fn: function 
        function to be applied to each piece
    
    fn_kwargs: dict
        dictionary of input args for 'fn'
    
    step_size: int
        skip X pixels between blocks
    
    n_jobs: int
        how many cores to use for multiprocessing    

    Returns
    -------
    arr_out: ndarray
        Output of the applied function

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.data import lena
    >>> im = lena()
    >>> output = process_windows(im[:,:,0], (8,8), np.sum)
    >>> output2 = process_windows(im, (8,8,3), np.sum, {'axis':-1})
    """    


    # -- Check arguments
    if not (isinstance(window_shape, tuple) or isinstance(window_shape, list)):
        raise ValueError("block_shape must be a tuple or a list")

    if np.any(window_shape <= 0):
        raise ValueError("block_shape must be strictly positive")        

    if len(window_shape) != im.ndim:
        raise ValueError("im.ndim and len(block_shape) must be the same")

    # -- Break the image up into windows
    image_views = view_as_windows(im, tuple(window_shape), step_size)

    # -- Get output dimensionality
    n_dims = len(window_shape)
    dims = image_views.shape[:n_dims]
    dims_out = hstack([dims, -1])

    # -- Apply the function to the image blocks
    indicies = product(*[range(x) for x in dims])
    if n_jobs > 1:
        # - First apply arguments to the function
        fn_map = partial(fn, **fn_kwargs)
        # - Then apply the function to each window in parallel
        pool = Pool(processes=n_jobs)        
        output = array(pool.map(fn_map, [image_views[idx] for idx in indicies]))
    else:
        output = array([fn(image_views[idx], **fn_kwargs) for idx in indicies])
    # -- Reshape the image
    output = output.reshape(dims_out)

    return np.squeeze(output)

