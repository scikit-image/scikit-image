import numpy as np

import skimage as ski
from skimage._shared.utils import _supported_float_type
from skimage._shared._warnings import warn_external


def rescale_minmax(image):
    """Rescale `image` to the value range [0, 1].

    Rescaling values between [0, 1], or *min-max normalization* [1]_,
    is a simple method to ensure that data is inside a range.

    Parameters
    ----------
    image : ndarray
        Input image.
    out : ndarray, optional
        If given, the rescaled image will be stored in this array.

    Returns
    -------
    rescaled_image : ndarray
        Rescaled image, of same shape as input `image` but with a
        floating dtype (according to :func:`_supported_float_type`).

    Raises
    ------
    ValueError
        NaN and infinity values are not supported.
        Replace such values before rescaling.

    See Also
    --------
    rescale_legacy
        Rescale value range based on dtype (legacy `skimage` behavior).

    References
    ----------
    .. [1]: https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)

    Examples
    --------
    >>> import numpy as np
    >>> image = np.array([-10, 45, 100], dtype=np.int8)
    >>> rescale_minmax(image)
    array([0. , 0.5, 1. ])
    """
    # Prepare `out` array, `lower` and `higher` with exact dtype to avoid
    # unexpected promotion and / or precision problems during normalization
    dtype = _supported_float_type(image.dtype, allow_complex=False)
    out = image.astype(dtype)

    lower = out.min()
    higher = out.max()

    # Deal with unexpected or invalid `lower` and `higher` early
    if np.isnan(lower) or np.isnan(higher):
        msg = (
            "`image` contains NaN. "
            "Min-max normalization with NaN is not supported. "
            "Replace NaNs manually before rescaling."
        )
        raise ValueError(msg)

    if np.isinf(lower) or np.isinf(higher):
        msg = (
            "`image` contains inf. "
            "Min-max normalization with inf is not supported. "
            "Replace inf manually before rescaling."
        )
        raise ValueError(msg)

    if lower == higher:
        msg = "`image` is uniform, returning uniform array of 0's"
        warn_external(msg, category=RuntimeWarning)
        out = np.zeros_like(out)
        return out
    assert lower < higher

    # Actual normalization
    with np.errstate(all="raise"):
        try:
            peak_to_peak = higher - lower
            out -= lower
        except FloatingPointError as e:
            if "overflow" in e.args[0]:
                warn_external(
                    "Overflow while attempting to rescale. This could be due to "
                    "`image` containing unexpectedly large values. Dividing all "
                    "values by 2 before rescaling to avoid overflow.",
                    category=RuntimeWarning,
                )
                out /= 2
                lower /= 2
                higher /= 2
                peak_to_peak = higher - lower
                out -= lower
            else:
                raise

        out /= peak_to_peak

    return out


def rescale_legacy(image):
    """Rescale value range based on dtype (legacy `skimage` behavior).

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
        Otherwise, data is copied as necessary.

    Returns
    -------
    rescaled_image : ndarray
        Rescaled image, of same shape as input `image` but with a
        floating dtype (according to :func:`_supported_float_type`).

    See Also
    --------
    rescale_minmax
        Rescale `image` to the value range [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> image = np.array([0, 127, 255], dtype=np.uint8)
    >>> rescale_legacy(image)
    array([0.        , 0.49803922, 1.        ])
    """
    out = ski.util.img_as_float(image)
    return out


def _prescale_value_range(image, *, mode):
    """Rescale the value range of `image` according to the selected `mode`.

    For now, this private function handles *prescaling* (`prescale` parameter)
    for public API that needs a value range to be known and well-defined.

    Parameters
    ----------
    image : ndarray
        Image to rescale.
    mode : {'minmax', 'none', 'legacy'}, optional
        Controls the rescaling behavior for `image`.

        ``'minmax'``
            Normalize `image` between 0 and 1 regardless of dtype. After
            normalization, `rescaled_image` will have a floating dtype
            (according to :func:`_supported_float_type`).

        ``'none'``
            Don't rescale the value range of `image` at all and return a
            copy of `image`. Useful when `image` has already been rescaled.

        ``'legacy'``
            Normalize only if `image` has an integer dtype. If `image` is of
            floating dtype, it is left alone. See :func:`.img_as_float` for
            more details.

    Returns
    -------
    rescaled_image : ndarray
        The rescaled `image` of the same shape but possibly with a different
        dtype.

    Raises
    ------
    ValueError
        Rescaling an `image` with `mode='minmax'` that contains NaN or
        infinity is not supported for now. In those cases, consider replacing
        the unsupported values manually.

    Examples
    --------
    >>> import numpy as np
    >>> image = np.array([-10, 45, 100], dtype=np.int8)

    >>> _prescale_value_range(image, mode="minmax")
    array([0. , 0.5, 1. ])

    >>> _prescale_value_range(image, mode="legacy")
    array([-0.07874016,  0.35433071,  0.78740157])

    >>> _prescale_value_range(image, mode="none")
    array([-10, 45, 100], dtype=int8)
    """
    if mode == "none":
        # Exit early
        return image.copy()
    if mode == "legacy":
        return rescale_legacy(image)
    if mode == "minmax":
        return rescale_minmax(image)
    else:
        raise ValueError("unsupported mode")
