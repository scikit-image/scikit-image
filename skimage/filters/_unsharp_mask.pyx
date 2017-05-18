import sys
import numpy as np
from scipy import signal
from scipy import ndimage
from scipy import LowLevelCallable
from skimage import img_as_float

cimport numpy as np

from libc.stdint cimport intptr_t

DTYPE = np.int
ctypedef np.int_t DTYPE_t

def _precision_image(double[:,:,:,:] auto_corr, double[:,:,:] input_arr,
                     double[:,:,:] output_arr):
    """
    Along the third axis, calculate the dot product of the inverse of
    `auto_corr[i,j]`, which is a 2 by 2 symmetric matrix, and the vector
    `input_arr[i,j]`.
    """
    cdef:
        double a, b, d, x, y
        Py_ssize_t rowNum, colNum
        Py_ssize_t i, j
        double[:,:,:] Precision

    rowNum, colNum = auto_corr.shape[0], auto_corr.shape[1]
    for i in range(rowNum):
        for j in range(colNum):

            a, b = auto_corr[i,j,0,0], auto_corr[i,j,0,1]
            d = auto_corr[i,j,1,1]
            
            x, y = input_arr[i,j,0], input_arr[i,j,1]
            
            output_arr[i,j,0] = 1 / (a * d - b**2) * (d * x - b * y)
            output_arr[i,j,1] = 1 / (a * d - b**2) * (-b * x + a * y)

cdef api int local_var(double* input_arr_1d, intptr_t filter_size,
                          double* return_value, void* user_data):
    """
    Calculate the local variance of a pixel.
    """
    cdef:
        double sum1,sum2, mean, variance
        double[:, :] input_arr
        Py_ssize_t n, rowNum, colNum
        Py_ssize_t i, j

    n = 0
    input_arr = <double[:3, :3]> input_arr_1d
    rowNum, colNum = input_arr.shape[0], input_arr.shape[1]
    sum1 = 0
    sum2 = 0

    for i in range(rowNum):
        for j in range(colNum):
            n += 1
            sum1 += input_arr[i, j]

    mean = sum1 / n

    for i in range(rowNum):
        for j in range(colNum):
            sum2 += (input_arr[i, j] - mean)*(input_arr[i, j] - mean)

    variance = sum2 / n
    return_value[0] = variance

    return 1


def _gain(np.ndarray[double, ndim=2] var_img, double tau1, double tau2=200,
          double dh=4, double dl=3):
    """
    Depending on the local variance and threshold parameters, classify each
    pixel as belonging to smooth region if the local variance < `tau1`, a
    medium-contrast area if `tau1` <= local variance < `tau2`, and a high-
    contrast area otherwise. Then assign the gain of each pixel according to
    the classification.
    """
    cdef Py_ssize_t i, j
    cdef double val, out

    for i in range(var_img.shape[0]):
        for j in range(var_img.shape[1]):
            val = var_img[i, j]
        
            if val < tau1:
                out = 1
            if val >= tau2:
                out = dl
            else:
                out = dh

            var_img[i, j] = out
    

def _auto_correlation(double beta, np.ndarray[double, ndim=3] G):
    """ 
    Calculate the estimate of the autocorrelation matrix of G
    """
    cdef:
        Py_ssize_t m, n, j
        np.ndarray[double, ndim=4] ret_mtx, Gcorr

    m, n = G.shape[0], G.shape[1]
            
    ret_mtx = np.zeros((m,n, 2, 2))
    
    ret_mtx[:,0] = np.identity(2)

    Gcorr = np.multiply(G[..., np.newaxis], G[..., np.newaxis, :])

    for j in range(1, n):
        ret_mtx[:, j] = np.multiply((1 - beta), ret_mtx[:, j - 1]) + \
            np.multiply(beta, Gcorr[:, j])
        
    return ret_mtx      


def unsharp_mask(img, double tau1=0.005, double tau2=0.01, 
                   double dh=4, double dl=3, double mu=0.1, double beta=0.5,
                   Py_ssize_t iternum=1000):
    """Apply adaptive unsharp masking for contrast enhancement of input image.

    Parameters
    ----------
    image : ndarray
        Image to process.
    tau1, tau2 : double, optional
        `tau1` and `tau2` are two positive threshold values such that
        `tau1` < `tau2`. The input signal is classified as belonging to a
        smooth region if the local variance < `tau1`, a medium-contrast area
        if `tau1` <= local variance < `tau2`, and a high-contrast area
        otherwise.
    dh : double, optional
        Gain of an input signal belonging to a medium-contrast area.
        Must satify: `dh` > 1.
    dl : double, optional
        Gain of an input signal belonging to a high-contrast area
        Must satify: 1 < `dl` < `dh`.
    mu: double, optional
        A small, positive step size that controls the speed of convergence of
        the adaptive filter.
    beta: double, optional
        a positive convergence parameter that is less than 1
    iternum : int, optional
        Maximum number of iterations to run.
        
    Returns
    -------
    output : ndarray
        The contrast enchanced image.

    References:
    -------
    ..[1]Andrea Polesel, Giovanni Ramponi, and V. John Mathews (2000) "Image
         Enhancement via Adaptive Unsharp Masking" IEEE Trans. on Image
         Processing, 9(3): 505 - 510. DOI:10.1109/83.826787
    """

    cdef:
        Py_ssize_t m, n, k, i, j
        np.ndarray[DTYPE_t, ndim=2] g_filter, h_filter, v_filter
        np.ndarray[double, ndim=2] input_dynamics, correction_x, correction_y
        np.ndarray[double, ndim=2] variance_gain, desired_dynamics, g_zx, g_zy
        np.ndarray[double, ndim=2] desired_error, error, output_img
        np.ndarray[double, ndim=3] G, Alpha, Precision
        np.ndarray[double, ndim=4] R
        double error_norm_prev, error_norm
    
    img = img_as_float(img)
    m, n = img.shape[0], img.shape[1]

    g_filter = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])

    h_filter = np.array([
        [0, 0, 0],
        [-1, 2, -1],
        [0, 0, 0]
    ])

    v_filter = np.array([
        [0, -1, 0],
        [0, 2, 0],
        [0, -1, 0]
    ])
    
    # apply linear highpass filter on image to obtain local dynamics
    input_dynamics = signal.convolve(img, g_filter, mode='same')
    
    # apply horizontal and vertical Laplacian operator on image
    # to obtain correction signal
    correction_x = signal.convolve(img, h_filter, mode='same')
    correction_y = signal.convolve(img, v_filter, mode='same')
        
    variance_gain = ndimage.generic_filter(img,
        LowLevelCallable.from_cython(sys.modules[__name__], name='local_var'), size = 3)
    _gain(variance_gain, tau1, tau2, dh, dl)

    # desired activity level in the output image
    desired_dynamics = np.multiply(variance_gain, input_dynamics)

    # apply linear highpass filter
    g_zx = signal.convolve(correction_x, g_filter, mode='same')
    g_zy = signal.convolve(correction_y, g_filter, mode='same')
    
    G = np.dstack((g_zx, g_zy))
            
    # estimate of the autocorrelation matrix of G
    R = _auto_correlation(0.5, G)
    
    # initialize scaling vector for correction signal
    Alpha = np.zeros((m,n,2));
    
    Precision = np.empty_like(Alpha)
    _precision_image(R, G, Precision)

    error_norm_prev = np.infty
    desired_error = np.subtract(desired_dynamics, input_dynamics)

    # update scaling vector Alpha using Gauss–Newton algorithm 
    for k in range(iternum):
        error = desired_error.copy()
        
        # error -= np.sum(Alpha*G, axis=2)
        # leave this in as a sane explanation of the next line
        error -= np.einsum('ijk,ijk->ij', Alpha, G)
        error_norm = np.linalg.norm(error)

        if np.abs(np.linalg.norm(error) - error_norm_prev) < 1e-6:
            break
        error_norm_prev = error_norm

        Alpha[:, 1:] = (Alpha + np.multiply(2*mu, 
            np.multiply(error[...,np.newaxis], Precision)))[:, :-1]

    output_img = img + np.multiply(Alpha[..., 0], correction_x) + \
        np.multiply(Alpha[..., 1], correction_y)
    output_img = np.clip(output_img, 0, 1)
    
    return output_img
