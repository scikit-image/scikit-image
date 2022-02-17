import numpy as np

from skimage import data, filters
from skimage._shared.utils import _supported_float_type


def homomorphic(filter_func, image, func_kwargs={}, eps=0):
    """ Apply homomorphic filtering using a user-provided filter.

    Apply a specified filtering function, `filter_func` to the logarithm of an
    image. Specifically::

    ``out = np.exp(filter_func(np.log(image)))``

    Parameters
    ----------
    filter_func : callable
        A function that takes an image and returns a filtered version.
    image : ndarray
        The image to filter. Image values should be non-negative.
    func_kwargs : dict
        Additional keyword arguments to pass to `filter_func`.
    eps : float or None
        A small value to add to `image` to avoid ``log(0)`` in places where the
        image intensity is zero.

    Returns
    -------
    out : ndarray
        The homorphic-filtered image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Homomorphic_filtering
    .. [2] Gonzalez, R. and Woods, R. Digital image processing.
           Upper Saddle River, N.J: Prentice Hall, 2nd. ed., 2002.

    Examples
    --------
    >>> from functools import partial
    >>> from skimage import data, filters
    >>> eagle = data.eagle()
    >>>
    >>> # Define a high-pass filter with amplitude 0.3 in the stop-band.
    >>> filter_func = partial(filters.butterworth, cutoff_frequency_ratio=0.02,
    ...                       npad=32, amplitude_range=(0.3, 1))
    >>> # Apply homomorphic filtering with the filter created above.
    >>> # This will reduce illumination differences in the image.
    >>> eagle_filtered = filters.homomorphic(filter_func, eagle)
    """

    float_dtype = _supported_float_type(image.dtype)
    output = image.astype(float_dtype, copy=True)
    if eps:
        # add small offset to help avoid log(0)
        output += eps
    # apply homomorphic filtering
    np.log(output, out=output)
    output = filter_func(output, **func_kwargs)
    np.exp(output, out=output)
    return output
