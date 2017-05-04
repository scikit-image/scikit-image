from concurrent import futures
from multiprocessing import cpu_count
from itertools import product

import numpy as np


def _generate_shifts(ndim, max_shifts=None, shift_steps=1):
    """Returns all combinations of shifts in n dimensions over the specified
    max_shifts and step sizes.

    Examples
    --------
    print(list(_generate_shifts(2, max_shifts=(1, 2), shift_steps=1)))
    >>> [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    """
    if max_shifts is None:
        max_shifts = (0, )*ndim
    elif np.isscalar(max_shifts):
        max_shifts = (max_shifts, )*ndim
    elif len(max_shifts) != ndim:
        raise ValueError("max_shifts should have length ndim")

    if shift_steps is None:
        shift_steps = (1, )*ndim
    elif np.isscalar(shift_steps):
        shift_steps = (shift_steps, )*ndim
    elif len(shift_steps) != ndim:
        raise ValueError("max_shifts should have length ndim")

    if np.any(np.asarray(shift_steps) < 1):
        raise ValueError("shift_steps must all be >= 1")
    return product(*([range(0, s+1, t) for
                      s, t in zip(max_shifts, shift_steps)]))


def _roll_axes(x, rolls, axes=None):
    """Apply np.roll along a set of axes.

    Parameters
    ----------
    x : array-like
        array to roll
    rolls : int or sequence
        The amount to roll along each axis in ``axes``
    axes : int or sequence
        The axes to roll.  Default is the first len(roll) axes of x.

    Returns
    -------
    x : array
        data with axes rolled
    """
    if np.isscalar(rolls):
        rolls = (rolls, )
    if len(rolls) > x.ndim:
        raise ValueError("length of rolls should not exceed x.ndim")
    if axes is None:
        axes = np.arange(len(rolls))
    if np.isscalar(axes):
        axes = (axes, )
    if len(rolls) != len(axes):
        raise ValueError("size mismatch between rolls and axes")
    try:
        # numpy>=1.12 supports tuple for rolls, axes
        return np.roll(x, rolls, axes)
    except TypeError:
        for r, a in zip(rolls, axes):
            x = np.roll(x, r, a)
    return x


def cycle_spin(x, func, max_shifts, shift_steps=1, num_workers=1,
               func_kw={}):
    """Cycle spinning (repeatedly apply func to shifted versions of x).

    Parameters
    ----------
    x : array-like
        data
    func : function
        A function to apply to circularly shifted versions of x.  Should take
        x as its first argument.  Any additional arguments can be supplied via
        ``func_kw``.
    max_shifts : int or tuple, optional
        If an integer, shifts in ``range(0, max_shifts+1)`` will be used along
        each axis of x. If a tuple, ``range(0, max_shifts[i]+1)`` will be used
        along axis i.
    shift_steps : int or tuple, optional
        The step size for the shifts applied along axis, i, are::
        ``range((0, max_shifts[i]+1, shift_steps[i]))``.  If an integer is
        provided, the same step size is used for all axes.
    func_kw : dict, optional
        Additional keyword arguments to supply to ``func``.

    Returns
    -------
    avg_y : np.ndarray
        The output of ``func(x, **func_kw)`` averaged over all
        combinations of the specified axis shifts.

    Notes
    -----
    Cycle spinning was proposed as a way to approach shift-invariance via
    performing several circular shifts of a shift-variant transform [1]_.

    For a n-level discrete wavelet transforms, one may wish to perform all
    shifts up to ``max_shifts = 2**n - 1``. In practice, much of the benefit
    can often be realized with only a small number of shifts per axis.

    For transforms such as the blockwise discrete cosine transform, one may
    wish to evaluate shifts up to the block size used by the transform.

    References
    ----------
    ..[1] R.R. Coifman and D.L. Donoho.  "Translation-Invariant De-Noising".
          Wavelets and Statistics, Lecture Notes in Statistics, vol.103.
          Springer, New York, 1995, pp.125-150.
          http://dx.doi.org/10.1007/978-1-4612-2544-7_9

    Examples
    --------
    >>> import skimage.data
    >>> from skimage import img_as_float
    >>> from skimage.restoration import denoise_wavelet, cycle_spin
    >>> img = img_as_float(skimage.data.camera())
    >>> sigma = 0.1
    >>> img = img + sigma * np.random.standard_normal(img.shape)
    >>> denoised = cycle_spin(img, func=denoise_wavelet, max_shifts=3)

    """
    x = np.asanyarray(x)
    all_shifts = _generate_shifts(x.ndim, max_shifts, shift_steps)
    all_shifts = list(all_shifts)
    nshifts = len(all_shifts)

    def _run_one_shift(shift):
        # shift, apply function, inverse shift
        xs = _roll_axes(x, shift)
        tmp = func(xs, **func_kw)
        return _roll_axes(tmp, -np.asarray(shift))

    avg_y = 0
    if num_workers == 1:
        # serial processing
        for shift in all_shifts:
            avg_y += _run_one_shift(shift)
    else:
        # multithread via concurrent.futures ThreadPoolExecutor
        if num_workers is None:
            num_workers = cpu_count()
        num_workers = np.clip(num_workers, 1, nshifts)
        with futures.ThreadPoolExecutor(max_workers=num_workers) as execute:
            for y in execute.map(_run_one_shift, all_shifts):
                avg_y += y
    avg_y /= nshifts
    return avg_y
