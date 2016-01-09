import numpy as np


def _astype_nocopy(a, dtype):
    """
    Workaround to enable "a.astype(dtype, copy=False) under Python 2.6
    """
    if a.dtype != dtype:
        return a.astype(dtype)
    else:
        return a


def line_integral_convolution(input, velocity, kernel, origin=0, order=3,
                              weighted='average', step_size='unit_length',
                              maximum_velocity=None):
    """
    Line integral convolution of an input image with an arbitrary kernel.
    Lines are defined by a velocity field.

    Parameters
    ----------
    input : array_like
        Random image to be convolved (can be created for instance with
        ``1. * (np.random.random(shape) > 0.5)``).
    velocity: array_like
        One velocity vector for each pixel. Must have shape
        ``input.shape + (input.ndim,)``.
        First dimensions are identical to the shape of the random input
        image, the last dimension defines the coordinate of the velocity.
    kernel: array_like
        1-D array with at least on element. Defines the convolution kernel
        (e.g. a Gaussian kernel constructed with
        scipy.stats.norm.norm.pdf(np.linspace(-3,3,50))).
        The flow direction can be visualized by using an asymmetric kernel
        in combination with an ``origin`` parameter equal to ``None`` or
        equivalently ``-(len(kernel) // 2)``.
    origin: int, optional
        Placement of the filter, by default 0 (which is correct for
        symmetric kernels). The flow direction can be visualized by using an
        asymmetric kernel in combination with an ``origin`` parameter equal to
        ``None`` or equivalently ``-(len(kernel) // 2)``.
    order: int, optional
        The order of the spline interpolation used for interpolating the input
        image and the velocity field, by default 3.
        See the documentation of
        ``scipy.ndimage.interpolation.map_coordinates`` for more details.
    weighted: string, optional
        Can be either ``'average'`` or ``'integral'``, by default
        ``'average'``. If set to ``'average'``, the weighted average is
        computed. If set to ``'integral'``, the weighted integral is computed.
        See the examples to see which parameter is appropriate each use case.
    step_size: str, optional
        Can be either ``'unit_length'`` or ``'unit_time'``, by default
        ``'unit_length'``. If set to ``'unit_length'``, the integration step is
        the velocity scaled to unit length. If set to ``'unit_time'``, the step
        equals the velocity.
    maximum_velocity: float, optional
        Is ``None`` by default. If it is not ``None``, the velocity field is
        mutiplied with a scalar variable s.t. the maximum velocity after
        multiplication equals ``maximum_velocity``.

    Returns
    -------
    line_integral_convolution : ndarray
        Returned array of same shape as `input`.

    See Also
    --------
    scipy.ndimage.filters.convolve1d : Calculate a one-dimensional convolution
                                       with a kernel.
    scipy.integrate.ode : Integrate an ordinary differential equation.

    References
    ----------
    .. [1] http://dl.acm.org/citation.cfm?id=166151

    Examples
    --------
    .. plot:: lic.py

    """
    from scipy.ndimage.interpolation import map_coordinates
    input = np.asarray(input)
    kernel = np.asarray(kernel)
    velocity = np.asarray(velocity)
    if (kernel.ndim != 1) or (len(kernel) < 1):
        raise ValueError('Kernel must be 1-D array of length at least 1')
    if velocity.shape != (input.shape + (input.ndim,)):
        raise ValueError('Shape of velocity not compatible with shape of '
                         'input image')
    float_type = np.result_type(velocity.dtype, kernel.dtype, input.dtype,
                                np.float32)
    if origin is None:
        center = 0
    else:
        center = (len(kernel) // 2) + origin
    if center < 0:
        raise ValueError('Origin parameter is too low for the chosen kernel '
                         'length')
    if center >= len(kernel):
        raise ValueError('Origin paremeter is too large for the kernel length')
    if maximum_velocity is not None:
        velocity = maximum_velocity * velocity / np.sqrt(np.max(
            np.sum(np.square(_astype_nocopy(velocity, float_type)), axis=-1)))
    if weighted not in ('integral', 'average'):
        raise ValueError("Weighted must be either 'intergral' or 'average'")
    if step_size not in ('unit_length', 'unit_time'):
        raise ValueError("Step_size must be either 'unit_length' "
                         "or 'unit_time'")
    if weighted == 'average':
        weight = np.zeros(input.shape, float_type)
    result = np.zeros(input.shape, float_type)
    # the kernel is divided into a part left of "center" and a part right of
    # "center".
    # The part on the "right" lies in positive flow direction (sign=1),
    # the part on the "left" lies in negative flow direction (sign=-1)
    for sign, ids in [(+1, range(center, len(kernel))),
                      (-1, reversed(range(0, center + 1)))]:
        # workaround to prevent pyflakes from complaining about undefined
        # variables
        v2, m1, v2l2 = None, None, None
        for i in ids:
            if i == center:
                # initialize position and mask
                pos = np.mgrid[[slice(None, s)
                                for s in input.shape]].astype(float_type)
                m = np.ones(input.shape, dtype=bool)
            else:
                # advance position using velocity
                if step_size == 'unit_length':
                    denominator = v2l2[m1][None, :]
                elif step_size == 'unit_time':
                    denominator = 1
                pos[:, m] += (sign * v2[:, m1] / denominator)
            # velocity at current position
            v2 = np.array([map_coordinates(velocity[..., axis], pos[:, m],
                                           order=order,
                                           output=float_type)
                           for axis in range(input.ndim)])
            # l2-norm of velocity
            v2l2 = np.sqrt(np.sum(np.square(v2), axis=0))
            # update mask
            m1 = (v2l2 > 0)
            m[m] = m1
            # update result
            if (sign == 1) or (i != center):
                if step_size == 'unit_length':
                    ki = kernel[i]
                elif step_size == 'unit_time':
                    ki = kernel[i] * v2l2[m1]
                if i == center:
                    result[m] += ki * input[m]
                else:
                    result[m] += ki * map_coordinates(input, pos[:, m],
                                                      order=order,
                                                      output=float_type)
                if weighted == 'average':
                    weight[m] += ki
    if weighted == 'average':
        return result / weight
    else:
        return result
