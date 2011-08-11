"""template.py - Template matching
"""
import cython
cimport numpy as np
import numpy as np
import cv
from scipy.signal import fftconvolve

cdef extern from "math.h":
    double sqrt(double x)
    double fabs(double x)
    
cdef extern from "template.h":
    void iimg2(float *image, double *ii2, int imH, int imW)

cdef void integral_image2(float *image, double *ii2, int imH, int imW):
	double *prev_line = ii2;
	double s
	int x, y
	*ii2 = (*image)*(*image); 
	ii2 += 1
	image += 1;
	for (x = 1; x < imH; x++):
		*ii2 = (*image)*(*image) + *(ii2-1);
		ii2+=1; image+=1;
	for (y = 1; y < imW; y++):
		s = 0;
		for (x = 0; x < imH; x++):
			s += *image * *image;
			*ii2 = s + *prev_line;
			ii2+=1
			image+=1
			prev_line+=1

@cython.boundscheck(False)
cpdef np.ndarray integral_image(np.ndarray X, with_squared=False, squared=False):
    """Summed area table / integral image.

    The integral image contains the sum of all elements above and to the
    left of it, i.e.:

    .. math::

       S[m, n] = \sum_{i \leq m} \sum_{j \leq n} X[i, j]

    Parameters
    ----------
    X : ndarray of uint8
        Input image.

    Returns
    -------
    S : ndarray
        Summed area table.

    References
    ----------
    .. [1] F.C. Crow, "Summed-area tables for texture mapping,"
           ACM SIGGRAPH Computer Graphics, vol. 18, 1984, pp. 207-212.

    """
    cdef int s_rows, s_cols
    s_rows = X.shape[0]
    s_cols = X.shape[1]

    cdef np.ndarray[np.uint64_t, ndim=2] S
    cdef np.ndarray[np.uint64_t, ndim=2] S2
    
    if squared:
        S2 = np.zeros((s_rows, s_cols), dtype=np.uint64)  
        # First column
        S2[0, 0] = X[0, 0] * X[0, 0]
        for m in range(1, s_rows):
            S2[m, 0] = X[m, 0] * X[m, 0] + S2[m - 1, 0]
        # First row
        for n in range(1, s_cols):
            S2[0, n] = X[0, n] * X[0, n] + S2[0, n - 1]
        for m in range(1, s_rows):
            for n in range(1, s_cols):
                    S2[m, n] = S2[m - 1, n] + S2[m, n - 1] \
                                  - S2[m - 1, n - 1] + X[m, n] * X[m, n]
        return S2
    elif not with_squared:
        S = np.zeros((s_rows, s_cols), dtype=np.uint64)
        # First column
        S[0, 0] = X[0, 0]
        for m in range(1, s_rows):
            S[m, 0] = X[m, 0] + S[m - 1, 0]
        # First row
        for n in range(1, s_cols):
            S[0, n] = X[0, n] + S[0, n - 1]
        for m in range(1, s_rows):
            for n in range(1, s_cols):
                    S[m, n] = S[m - 1, n] + S[m, n - 1] \
                                  - S[m - 1, n - 1] + X[m, n]
        return S
    else:
        S = np.zeros((s_rows, s_cols), dtype=np.uint64)
        S2 = np.zeros((s_rows, s_cols), dtype=np.uint64)  
        # First column
        S[0, 0] = X[0, 0]
        S2[0, 0] = X[0, 0] ** 2
        for m in range(1, s_rows):
            S[m, 0] = X[m, 0] + S[m - 1, 0]
            S2[m, 0] = X[m, 0] ** 2 + S2[m - 1, 0]
        # First row
        for n in range(1, s_cols):
            S[0, n] = X[0, n] + S[0, n - 1]
            S2[0, n] = X[0, n] ** 2 + S2[0, n - 1]
        for m in range(1, s_rows):
            for n in range(1, s_cols):
                    S[m, n] = S[m - 1, n] + S[m, n - 1] \
                                  - S[m - 1, n - 1] + X[m, n] ** 2
                    S2[m, n] = S2[m - 1, n] + S2[m, n - 1] \
                                  - S2[m - 1, n - 1] + X[m, n] ** 2                                  
        return (S, S2)


@cython.boundscheck(False)
def match_template(np.ndarray [float, ndim=2, mode="c"] image, np.ndarray[float, ndim=2, mode="c"] template):
    #out = np.empty((image.shape[0] - template.shape[0] + 1,image.shape[1] - template.shape[0] + 1), dtype=np.float32)
    cdef np.ndarray result = np.ascontiguousarray(fftconvolve(image, template, mode="valid"))
    cdef np.ndarray integral = np.empty((image.shape[0]+1, image.shape[1]+1))
    cdef np.ndarray integral_sqr = np.empty((image.shape[0], image.shape[1]))
    size = image.shape
    template_size = template.shape
    #cv.Integral(image, integral, integral_sqr)
    #integral_sqr = integral_image(image, squared=True)
    
    cdef np.ndarray integral_sqr2 = np.empty((image.shape[0], image.shape[1]))
    iimg2(<float*>image.data, <double*>integral_sqr.data, image.shape[1], image.shape[0])
    
    cdef double inv_area = 1.0 / (template.shape[0] * template.shape[1])
    cdef double template_norm = sqrt((np.std(template) ** 2 + np.mean(template) ** 2)) / sqrt(inv_area)
    cdef int sqstep = integral_sqr.strides[0] / sizeof(double)
    cdef double* q0 = <double*> integral_sqr.data
    cdef double* q1 = q0 + template.shape[0] 
    cdef double* q2 = <double*>integral_sqr.data + template.shape[1] * sqstep
    cdef double* q3 = q2 + template.shape[0]
    cdef int i, j, k, index
    cdef double num, window_sum2, normed
    for i in range(result.shape[0]):
        index = i * sqstep;
        for j in range(result.shape[1]):
            window_sum2 = q0[index] - q1[index] - q2[index] + q3[index]
            normed = sqrt(window_sum2) * template_norm
            num = result[i, j]
            if fabs(num) < normed:
                num /= normed
            elif fabs(num) < normed*1.125:
                if num > 0:
                    num = 1
                else:
                    num = -1                    
            else:
                num = 0
            result[i, j] = num
            index += 1
    return result
    

