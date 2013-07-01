import numpy as np
from .dtype import img_as_float


__all__ = ['random_noise']


def random_noise(image, mode='gaussian', seed=None, **kwargs):
    """
    Function to add random noise of various types to a floating-point image.

    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gaussian'  Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        'salt'      Replaces random pixels with 1.
        'pepper'    Replaces random pixels with 0.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image, where
                    n is uniform noise with specified mean & variance.
    seed : int
        If provided, this will set the random seed before generating noise.
    m : float
        Mean of random distribution. Used in 'gaussian' and 'speckle'.
    v : float
        Variance of random distribution. Used in 'gaussian' and 'speckle'.
        Note: variance = (standard deviation) ** 2
    d : float
        Proportion of image pixels to replace with noise on range [0, 1].
        Used in 'salt', 'pepper', and 'salt & pepper'.
    p : float
        Proportion of salt vs. pepper noise for 's&p' on range [0, 1].
        Higher values represent more salt.

    Returns
    -------
    out : ndarray
        Output floating-point image data on range [0, 1].

    """
    mode = mode.lower()
    image = img_as_float(image)
    if seed is not None:
        np.random.seed(seed=seed)

    allowedtypes = {
        'gaussian': 'gaussian_values',
        'poisson': '',
        'salt': 'sp_values',
        'pepper': 'sp_values',
        's&p': 's&p_values',
        'speckle': 'gaussian_values'}

    kwdefaults = {
        'm': 0.,
        'v': 0.01,
        'd': 0.05,
        'p': 0.5}

    allowedkwargs = {
        'gaussian_values': ['m', 'v'],
        'sp_values': ['d'],
        's&p_values': ['d', 'p']}

    for key in kwargs:
        if key not in allowedkwargs[allowedtypes[mode]]:
            raise ValueError('%s keyword not in allowed keywords %s' %
                             (key, allowedkwargs[allowedtypes[mode]]))

    # Set kwarg defaults
    for kw in allowedkwargs[allowedtypes[mode]]:
        kwargs.setdefault(kw, kwdefaults[kw])

    if mode == 'gaussian':
        noise = np.random.normal(kwargs['m'], kwargs['v'] ** 0.5, image.shape)
        out = np.clip(image + noise, 0., 1.)

    elif mode == 'poisson':
        # Generating noise for each unique value in image.
        out = np.zeros_like(image)
        for val in np.unique(image):
            # Generate mask for a unique value, replace w/values drawn from
            # Poisson distribution about the unique value
            mask = image == val
            out[mask] = np.poisson(val, mask.sum())

    elif mode == 'salt':
        # Re-call function with mode='s&p' and p=1 (all salt noise)
        out = random_noise(image, mode='s&p', seed=seed, d=kwargs['d'], p=1)

    elif mode == 'pepper':
        # Re-call function with mode='s&p' and p=1 (all pepper noise)
        out = random_noise(image, mode='s&p', seed=seed, d=kwargs['d'], p=0)

    elif mode == 's&p':
        out = image.copy()

        # Salt mode
        num_salt = np.ceil(kwargs['d'] * image.size * kwargs['p'])
        coords = [np.random.randint(0, i - 1, num_salt)
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(kwargs['d'] * image.size * (1. - kwargs['p']))
        coords = [np.random.randint(0, i - 1, num_pepper)
                  for i in image.shape]
        out[coords] = 0

    elif mode == 'speckle':
        noise = np.random.normal(kwargs['m'], kwargs['v'] ** 0.5, image.shape)
        out = np.clip(image + image * noise, 0., 1.)

    return out
