
import numpy as np
import skimage.util
import itertools as it


def blockproc(im, shape, fun, fun_params=[], overlap=True,
              n_jobs=1, output_shape=None):
    '''
    This function breaks an image into pieces and applies a function to each
    section. The pieces can be be overlapping or unique.

    Note: Only use n_jobs > 1 if you have a computationaly expensive function.
    Otherwise it takes longer to copy the image to seperate cores than to
    simply perform on one.

    im : input image [n x m ]
    shape : size of each piece of the image
    fun : function to be applied to each piece
    fun_params : input args for 'fun'. This is a list of inputs and not a
    dictionary. Thus they must be in the correct order as used in 'fun'.
    n_jobs : the number of cores that should process the image.
     (n_jobs=-1 automatically uses all cores)
    output_size : if the output of the function is not the same as the input

    Parallelism is done using joblib. Install this for better performance.
    pip install joblib

    example:
    >>> from skimage.data import lena
    >>> im = lena()
    >>> output = blockproc(im[:,:,0], (8,8), np.sum, overlap=False, n_jobs=1)
    >>> output2 = blockproc(im, (8,8,3), np.sum, [-1], overlap=False,
                            output_shape=[64,64,8,8])
    '''

    # Try to parallelize using joblib if it's installed
    try:
        from joblib import Parallel, delayed
        joblib_installed = True
    except:
        joblib_installed = False

    # Break the image up into blocks that either overlap or are unique
    if overlap:
        image_views = skimage.util.view_as_windows(im, tuple(shape))
    else:
        # Check that the image is divisible by shape
        if not np.all(np.mod(im.shape, shape) == 0):
            msg = "Error: block size should be a multiple of the image shape"
            raise ValueError(msg)

        image_views = skimage.util.view_as_blocks(im, tuple(shape))

    # Get output dimensionality
    n_dims = len(shape)
    dims = image_views.shape[:n_dims]
    if output_shape is None:
        output_shape = dims

    # Apply the function to the image blocks
    if joblib_installed:
        indicies = it.product(*[xrange(x) for x in dims])
        output = Parallel(n_jobs)(delayed(fun)(image_views[idx], *fun_params)
                                  for idx in indicies)
    else:
        output = np.array([[fun(x, *fun_params) for x in y]
                           for y in image_views])

    output = np.array(output).reshape(output_shape)

    return output
