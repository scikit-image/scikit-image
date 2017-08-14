# -*- coding: utf-8 -*-

from itertools import cycle

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, \
                        gaussian_filter, gaussian_gradient_magnitude

__all__ = ['morph_acwe', 'morph_gac', 'SIoIS', 'ISoSI', 'gborders']

class fcycle(object):
    
    def __init__(self, iterable):
        """Call functions from the iterable each time it is called."""
        self.funcs = cycle(iterable)
    
    def __call__(self, *args, **kwargs):
        f = next(self.funcs)
        return f(*args, **kwargs)
    

# SI and IS operators for 2D and 3D.
_P2 = [np.eye(3), np.array([[0,1,0]]*3), np.flipud(np.eye(3)), np.rot90([[0,1,0]]*3)]
_P3 = [np.zeros((3,3,3)) for i in range(9)]

_P3[0][:,:,1] = 1
_P3[1][:,1,:] = 1
_P3[2][1,:,:] = 1
_P3[3][:,[0,1,2],[0,1,2]] = 1
_P3[4][:,[0,1,2],[2,1,0]] = 1
_P3[5][[0,1,2],:,[0,1,2]] = 1
_P3[6][[0,1,2],:,[2,1,0]] = 1
_P3[7][[0,1,2],[0,1,2],:] = 1
_P3[8][[0,1,2],[2,1,0],:] = 1

_opbuffer = np.zeros((0))
def SI(u):
    """SI operator."""
    global _opbuffer
    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError("u has an invalid number of dimensions (should be 2 or 3)")
    
    if u.shape != _opbuffer.shape[1:]:
        _opbuffer = np.zeros((len(P),) + u.shape)
    
    for _opbuffer_i, P_i in zip(_opbuffer, P):
        _opbuffer_i[:] = binary_erosion(u, P_i)
    
    return _opbuffer.max(0)

def IS(u):
    """IS operator."""
    global _opbuffer
    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError("u has an invalid number of dimensions (should be 2 or 3)")
    
    if u.shape != _opbuffer.shape[1:]:
        _opbuffer = np.zeros((len(P),) + u.shape)
    
    for _opbuffer_i, P_i in zip(_opbuffer, P):
        _opbuffer_i[:] = binary_dilation(u, P_i)
    
    return _opbuffer.min(0)

# SIoIS operator.
SIoIS = lambda u: SI(IS(u))
ISoSI = lambda u: IS(SI(u))
curvop = fcycle([SIoIS, ISoSI])

def _check_input(image, init_level_set):
    
    if len(image.shape) not in [2, 3]:
        raise ValueError("Input image is not a 2D or 3D array.")
    
    if len(image.shape) != len(init_level_set.shape):
        raise ValueError("The dimensions of the initial level set do not "
                         "match the dimensions of the image.")

def _find_threshold(image, percentile=10):
    
    return np.percentile(image, percentile)

def gborders(img, alpha=1.0, sigma=1.0):
    """Stopping criterion for image borders."""
    # The norm of the gradient.
    gradnorm = gaussian_gradient_magnitude(img, sigma, mode='constant')
    return 1.0/np.sqrt(1.0 + alpha*gradnorm)

def morph_acwe(image, init_level_set, iterations,
              smoothing=1, lambda1=1, lambda2=1,
              iter_callback=lambda x: None):
    """Morphological active contours without edges (aka Morphological Chan-Vese).
    
    Active contours without edges implemented with morphological operators. It
    can be used to segment objects in images without well defined borders. It is
    required that the inside of the object looks different on average than the
    outside (i.e., the inner area of the object should be darker or lighter than
    the outer area on average).
    
    Parameters
    ----------
    image : (M, N) or (L, M, N) array
        Grayscale image to be segmented.
    init_level_set : (M, N) or (L, M, N) array
        Initial level set.
    iterations : uint
        Number of iterations to run
    smoothing : uint, optional
        Number of times the smoothing operator is applied per iteration.
        Reasonable values are around 1-4. Larger values lead to smoother
        segmentations.
    lambda1 : float, optional
        Weight parameter for the outer region. If `lambda1` is larger than
        `lambda2`, the outer region will contain a larger range of values than
        the inner region.
    lambda2 : float, optional
        Weight parameter for the inner region. If `lambda2` is larger than
        `lambda1`, the inner region will contain a larger range of values than
        the outer region.
    iter_callback : function, optional
        If given, this function is called once per iteration with the current
        level set as the only argument. This is useful for debugging or for
        plotting intermediate results during the evolution.
    
    Returns
    -------
    segmentation : (M, N) or (L, M, N) array
        Final segmentation (i.e., the final level set)
    
    Notes
    -----
    
    This is a version of the Chan-Vese algorithm that uses morphological
    operators instead of solving a partial differential equation (PDE) for the
    evolution of the contour. The set of morphological operators used in this
    algorithm are proved to be infinitesimally equivalent to the Chan-Vese PDE
    (see [1]_). However, morphological operators are do not suffer from the
    numerical stability issues typically found in PDEs (it is not necessary to
    find the right time step for the evolution), and are computationally faster.
    
    The algorithm and its theoretical derivation are described in [1]_.
    
    References
    ----------
    .. [1] A Morphological Approach to Curvature-based Evolution of Curves and
    Surfaces, Pablo Márquez-Neila, Luis Baumela, Luis Álvarez. In IEEE
    Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2014,
    DOI 10.1109/TPAMI.2013.106
    """
    
    # TODO: Make it work with "color" (or multi-channel) images.
    
    _check_input(image, init_level_set)
    
    u = np.int8(init_level_set > 0)
    
    iter_callback(u)
    
    for _ in range(iterations):
        
        # TODO: test speed
        # inside = u > 0
        # outside = u <= 0
        c0 = (image * (1 - u)).sum() / float((1 - u).sum())
        c1 = (image * u).sum() / float(u.sum())
        
        # Image attachment
        du = np.gradient(u)
        abs_du = np.abs(du).sum(0)
        aux = abs_du * (lambda1*(image - c1)**2 - lambda2*(image - c0)**2)
        
        u[aux < 0] = 1
        u[aux > 0] = 0
        
        # Smoothing
        for _ in range(smoothing):
            u = curvop(u)
        
        iter_callback(u)
    
    return u

def morph_gac(image, init_level_set, iterations,
             smoothing=1, threshold='auto', balloon=0,
             iter_callback=lambda x: None):
    """Morphological geodesic active contours.
    
    Geodesic active contours implemented with morphological operators. It can be
    used to segment objects with visible but noisy, cluttered, broken borders.
    
    Parameters
    ----------
    image : (M, N) or (L, M, N) array
        Grayscale image to be segmented. This is rarely the original image.
        Instead, this is usually a preprocessed version of the original image
        with the borders of the object to segment having low values. See
        `morphsnakes.gborders` as an example function to perform this task.
    init_level_set : (M, N) or (L, M, N) array
        Initial level set.
    iterations : uint
        Number of iterations to run.
    smoothing : uint, optional
        Number of times the smoothing operator is applied per iteration.
        Reasonable values are around 1-4. Larger values lead to smoother
        segmentations.
    threshold : float, optional
        Areas of the image with a value smaller than this threshold will be
        considered borders. The evolution of the contour will stop in this
        areas.
    balloon : float, optional
        Balloon force to guide the contour in non-informative areas of the
        image, i.e., areas where the gradient of the image is too small to push
        the contour towards a border. A negative value will shrink the contour,
        while a positive value will expand the contour in these areas. Setting
        this to zero will disable the balloon force.
    iter_callback : function, optional
        If given, this function is called once per iteration with the current
        level set as the only argument. This is useful for debugging or for
        plotting intermediate results during the evolution.
    
    Returns
    -------
    segmentation : (M, N) or (L, M, N) array
        Final segmentation (i.e., the final level set)
    
    References
    ----------
    .. [1] A Morphological Approach to Curvature-based Evolution of Curves and
    Surfaces, Pablo Márquez-Neila, Luis Baumela, Luis Álvarez. In IEEE
    Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2014,
    DOI 10.1109/TPAMI.2013.106
    
    """
    
    _check_input(image, init_level_set)
    
    if threshold == 'auto':
        threshold = _find_threshold(image)
    
    structure = np.ones((3,) * len(image.shape), dtype=np.int8)
    dimage = np.gradient(image)
    threshold_mask = image > threshold
    threshold_mask_balloon = image > threshold / np.abs(balloon)
    
    u = np.int8(init_level_set > 0)
    
    iter_callback(u)
    
    for _ in range(iterations):
        
        # Balloon
        if balloon > 0:
            aux = binary_dilation(u, structure)
        elif balloon < 0:
            aux = binary_erosion(u, structure)
        if balloon != 0:
            u[threshold_mask_balloon] = aux[threshold_mask_balloon]
        
        # Image attachment
        aux = np.zeros_like(image)
        du = np.gradient(u)
        for el1, el2 in zip(dimage, du):
            aux += el1 * el2
        u[aux > 0] = 1
        u[aux < 0] = 0
        
        # Smoothing
        for _ in range(smoothing):
            u = curvop(u)
        
        iter_callback(u)
    
    return u
