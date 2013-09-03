
from numpy import array, any, all, mod, hstack, floor, ceil
from skimage.util import pad, view_as_windows, view_as_blocks
from itertools import product
from multiprocessing import Pool
from functools import partial


def process_blocks(im, block_shape, fn, fn_kwargs={}, pad_mode='reflect', n_jobs=1):
    ''' Split image into blocks and apply function 'fn' to each.

    im : input image
    block_shape : size of each piece of the image
    fn : function to be applied to each piece
    fn_kwargs : dictionary of input args for 'fn'
    pad_mode : if step_size=0 then choose how to pad the image
    n_jobs : how many cores to use for multiprocessing    

    example:
    >>> from skimage.data import lena
    >>> im = lena()
    >>> output = process_blocks(im[:,:,0], (8,8), np.sum)
    >>> output2 = process_blocks(im, (8,8,3), np.sum, [-1],
                            output_shape=[64,64,8,8])
    '''

    assert len(block_shape)==im.ndim, \
        ValueError("im.ndim and len(window_shape) should be the same")

    # If necessary, pad the image so it can be broken up evenly
    pad_size = block_shape - mod(im.shape, block_shape)
    pad_size[pad_size==block_shape] = 0
    im = pad(im, list((floor(x/2.), ceil(x/2.)) for x in pad_size), pad_mode)
    
    # Break the image up into blocks    
    image_views = view_as_blocks(im, tuple(block_shape))

    # Get output dimensionality
    n_dims = len(block_shape)
    dims = image_views.shape[:n_dims]
    dims_out = hstack([dims, -1])

    # Apply the function to the image blocks
    indicies = product(*[range(x) for x in dims])

    if n_jobs > 1:
        pool = Pool(processes=n_jobs)
        # First apply arguments to the function
        fn_map = partial(fn, **fn_kwargs)
        # Then apply the function to each window in parallel
        output = array(pool.map(fn_map, [image_views[idx] for idx in indicies]))
    else:
        output = array([fn(image_views[idx], **fn_kwargs) for idx in indicies])
    
    output = output.reshape(dims_out)


    return output

def process_windows(im, window_shape, fn, fn_kwargs={}, step_size=1, n_jobs=1):
    ''' Split image into windows and apply function 'fn' to each.

    im : input image
    window_shape : size of each piece of the image
    fn : function to be applied to each piece
    fn_kwargs : list of input args for 'fn'. Order dependent.
    step_size : if 0 use unique windows. Otherwise skip X pixels between blocks.
    n_jobs : how many cores to use for multiprocessing

    example:
    >>> from skimage.data import lena
    >>> im = lena()
    >>> fn = np.sum
    >>> output = process_windows(im[:,:,0], (8,8), fn, n_jobs=8)
    >>> output2 = process_windows(im, (8,8,3), fn, [-1],
                            output_shape=[64,64,8,8])
    '''

    assert len(window_shape)==im.ndim, \
        ValueError("im.ndim and len(window_shape) should be the same")

    # Break the image up into windows
    image_views = view_as_windows(im, tuple(window_shape), step_size)

    # Get output dimensionality
    n_dims = len(window_shape)
    dims = image_views.shape[:n_dims]
    dims_out = hstack([dims, -1])

    # Apply the function to the image blocks
    # indicies = product(*[range(x) for x in dims])
    indicies = product(*[range(x) for x in dims])
    if n_jobs > 1:
        pool = Pool(processes=n_jobs)
        # First apply arguments to the function
        fn_map = partial(fn, **fn_kwargs)
        # Then apply the function to each window in parallel
        output = array(pool.map(fn_map, [image_views[idx] for idx in indicies]))
    else:
        output = array([fn(image_views[idx], **fn_kwargs) for idx in indicies])
    output = output.reshape(dims_out)

    return output

