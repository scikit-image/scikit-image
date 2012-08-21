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
from libc.math cimport sqrt, fabs
from skimage._shared.transform cimport integrate


@cython.boundscheck(False)
def match_template(np.ndarray[float, ndim=2, mode="c"] image,
                   np.ndarray[float, ndim=2, mode="c"] template):
    cdef np.ndarray[float, ndim=2, mode="c"] corr
    cdef np.ndarray[float, ndim=2, mode="c"] image_sat
    cdef np.ndarray[float, ndim=2, mode="c"] image_sqr_sat
    cdef float template_mean = np.mean(template)
    cdef float template_ssd
    cdef float inv_area

    image_sat = integral.integral_image(image)
    image_sqr_sat = integral.integral_image(image**2)

    template -= template_mean
    template_ssd = np.sum(template**2)
    # use inversed area for accuracy
    inv_area = 1.0 / (template.shape[0] * template.shape[1])

    # when `dtype=float` is used, ascontiguousarray returns ``double``.
    corr = np.ascontiguousarray(fftconvolve(image,
                                            template[::-1, ::-1],
                                            mode="valid"),
                                dtype=np.float32)

    cdef int i, j
    cdef float den, window_sqr_sum, window_mean_sqr, window_sum,
    # move window through convolution results, normalizing in the process
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            # subtract 1 because `i_end` and `j_end` are used for indexing into
            # summed-area table, instead of slicing windows of the image.
            i_end = i + template.shape[0] - 1
            j_end = j + template.shape[1] - 1

            window_sum = integrate(image_sat, i, j, i_end, j_end)
            window_mean_sqr = window_sum * window_sum * inv_area
            window_sqr_sum = integrate(image_sqr_sat, i, j, i_end, j_end)
            if window_sqr_sum <= window_mean_sqr:
                corr[i, j] = 0
                continue

            den = sqrt((window_sqr_sum - window_mean_sqr) * template_ssd)
            corr[i, j] /= den
    return corr

