"""
Template matching using normalized cross-correlation.

We use fast normalized cross-correlation algorithm (see [1]_ and [2]_) to
compute match probability. This algorithm calculates the normalized
cross-correlation of an image, `I`, with a template `T` according to the
following equation::

                sum{ I(x, y) [T(x, y) - <T>] }
    -------------------------------------------------------
    sqrt(sum{ [I(x, y) - <I>]^2 } sum{ [T(x, y) - <T>]^2 })

where `<T>` is the average of the template, and `<I>` is the average of the
image *coincident with the template*, and sums are over the template and the
image window coincident with the template. Note that the numerator is simply
the cross-correlation of the image and the zero-mean template.

To speed up calculations, we use summed-area tables (a.k.a. integral images) to
quickly calculate sums of image windows inside the loop. This step relies on
the following relation (see Eq. 10 of [1])::

    sum{ [I(x, y) - <I>]^2 } =
        sum{ I^2(x, y) } - [sum{ I(x, y) }]^2 / N_x N_y

(Without this relation, you would need to subtract each image-window mean from
the image window *before* squaring.)

.. [1] Briechle and Hanebeck, "Template Matching using Fast Normalized
       Cross Correlation", Proceedings of the SPIE (2001).
.. [2] J. P. Lewis, "Fast Normalized Cross-Correlation", Industrial Light and
       Magic.
"""
import cython
cimport numpy as np
import numpy as np
from scipy.signal import fftconvolve
from skimage.transform import integral


cdef extern from "math.h":
    float sqrt(float x)
    float fabs(float x)


@cython.boundscheck(False)
cdef float sum_integral(np.ndarray[float, ndim=2,  mode="c"] sat,
        int r0, int c0, int r1, int c1):
    """
    Using a summed area table / integral image, calculate the sum
    over a given window.

    This function is the same as the `integrate` function in
    `skimage.transform.integrate`, but this Cython version significantly
    speeds up the code.

    Parameters
    ----------
    sat : ndarray of float
        Summed area table / integral image.
    r0, c0 : int
        Top-left corner of block to be summed.
    r1, c1 : int
        Bottom-right corner of block to be summed.

    Returns
    -------
    S : int
        Sum over the given window.
    """
    cdef float S = 0

    S += sat[r1, c1]

    if (r0 - 1 >= 0) and (c0 - 1 >= 0):
        S += sat[r0 - 1, c0 - 1]

    if (r0 - 1 >= 0):
        S -= sat[r0 - 1, c1]

    if (c0 - 1 >= 0):
        S -= sat[r1, c0 - 1]
    return S


@cython.boundscheck(False)
def match_template(np.ndarray[float, ndim=2, mode="c"] image,
                   np.ndarray[float, ndim=2, mode="c"] template):
    cdef np.ndarray[float, ndim=2, mode="c"] result
    cdef np.ndarray[float, ndim=2, mode="c"] integral_sum
    cdef np.ndarray[float, ndim=2, mode="c"] integral_sqr
    cdef float template_mean = np.mean(template)
    cdef float template_ssd
    cdef float inv_area

    integral_sum = integral.integral_image(image)
    integral_sqr = integral.integral_image(image**2)

    template -= template_mean
    template_ssd = np.sum(template**2)
    # use inversed area for accuracy
    inv_area = 1.0 / (template.shape[0] * template.shape[1])

    # when `dtype=float` is used, ascontiguousarray returns ``double``.
    result = np.ascontiguousarray(fftconvolve(image, np.fliplr(template),
                                              mode="valid"), dtype=np.float32)

    cdef int i, j
    cdef float num, den, window_sqr_sum, window_mean_sqr, window_sum,
    # move window through convolution results, normalizing in the process
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            num = result[i, j]
            # subtract 1 because `i_end` and `j_end` are used for indexing into
            # summed-area table, instead of slicing windows of the image.
            i_end = i + template.shape[0] - 1
            j_end = j + template.shape[1] - 1

            window_sum = sum_integral(integral_sum, i, j, i_end, j_end)
            window_mean_sqr = window_sum * window_sum * inv_area
            window_sqr_sum = sum_integral(integral_sqr, i, j, i_end, j_end)
            den = sqrt((window_sqr_sum - window_mean_sqr) * template_ssd)

            # enforce some limits
            if fabs(num) < den:
                num /= den
            elif fabs(num) < den * 1.125:
                if num > 0:
                    num = 1
                else:
                    num = -1
            else:
                num = 0
            result[i, j] = num
    return result

