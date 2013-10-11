import numpy as np
from .dtype import img_as_float


__all__ = ['random_noise']


def _next_pow2(n):
    """
    Returns next integer power of two.
    """

    next_pow = 0
    if n == np.inf:
        return np.inf

    if n < 1:
        raise ValueError("Unable to determine next power of two for %i" % (n))

    while True:
        if 2 ** next_pow >= n:
            return next_pow
        else:
            next_pow += 1


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
        If provided, this will set the random seed before generating noise,
        for valid pseudo-random comparisons.
    mean : float
        Mean of random distribution. Used in 'gaussian' and 'speckle'.
        Default : 0.
    var : float
        Variance of random distribution. Used in 'gaussian' and 'speckle'.
        Note: variance = (standard deviation) ** 2. Default : 0.01
    amount : float
        Proportion of image pixels to replace with noise on range [0, 1].
        Used in 'salt', 'pepper', and 'salt & pepper'. Default : 0.05
    salt_vs_pepper : float
        Proportion of salt vs. pepper noise for 's&p' on range [0, 1].
        Higher values represent more salt. Default : 0.5 (equal amounts)

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
        'poisson': 'poisson_values',
        'salt': 'sp_values',
        'pepper': 'sp_values',
        's&p': 's&p_values',
        'speckle': 'gaussian_values'}

    kwdefaults = {
        'mean': 0.,
        'var': 0.01,
        'amount': 0.05,
        'salt_vs_pepper': 0.5}

    allowedkwargs = {
        'gaussian_values': ['mean', 'var'],
        'sp_values': ['amount'],
        's&p_values': ['amount', 'salt_vs_pepper'],
        'poisson_values': []}

    for key in kwargs:
        if key not in allowedkwargs[allowedtypes[mode]]:
            raise ValueError('%s keyword not in allowed keywords %s' %
                             (key, allowedkwargs[allowedtypes[mode]]))

    # Set kwarg defaults
    for kw in allowedkwargs[allowedtypes[mode]]:
        kwargs.setdefault(kw, kwdefaults[kw])

    if mode == 'gaussian':
        noise = np.random.normal(kwargs['mean'], kwargs['var'] ** 0.5,
                                 image.shape)
        out = np.clip(image + noise, 0., 1.)

    elif mode == 'poisson':
        # Determine unique values present in image
        vals = len(np.unique(image))

        # Calculate the next lowest power of two
        vals = 2 ** _next_pow2(vals)

        # Generating noise for each unique value in image.
        out = np.random.poisson(image * vals) / float(vals)

    elif mode == 'salt':
        # Re-call function with mode='s&p' and p=1 (all salt noise)
        out = random_noise(image, mode='s&p', seed=seed,
                           amount=kwargs['amount'], salt_vs_pepper=1.)

    elif mode == 'pepper':
        # Re-call function with mode='s&p' and p=1 (all pepper noise)
        out = random_noise(image, mode='s&p', seed=seed,
                           amount=kwargs['amount'], salt_vs_pepper=0.)

    elif mode == 's&p':
        # This mode makes no effort to avoid repeat sampling. Thus, the
        # exact number of replaced pixels is only approximate.
        out = image.copy()

        # Salt mode
        num_salt = np.ceil(
            kwargs['amount'] * image.size * kwargs['salt_vs_pepper'])
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(
            kwargs['amount'] * image.size * (1. - kwargs['salt_vs_pepper']))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0

    elif mode == 'speckle':
        noise = np.random.normal(kwargs['mean'], kwargs['var'] ** 0.5,
                                 image.shape)
        out = np.clip(image + image * noise, 0., 1.)

    return out
