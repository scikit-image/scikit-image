import numpy as np
from libc.math cimport sqrt, sin, cos, floor, ceil, M_PI
from skimage._shared.interpolation cimport round


def cart2pol(img, center=None, full_output=False):
    """
    Polar transform of an image.

    Parameters
    ----------
    img : (M, N, D) ndarray
        Input image.
    center : (row, column) tuple or (2,) ndarray, optional
        Center of the polar transform.
        If None, the center is the center of the image.
    full_output : boolean, optional (default False)
        Extend the output to fully enclose the image.

    Returns
    -------
    polar : ndarray
        Polar transform.
    """
    cdef:
        int deltax, delta_y, radius
        int center0, center1, corner0, corner1
        Py_ssize_t imgshape0 = img.shape[0]
        Py_ssize_t imgshape1 = img.shape[1]
        double sina, cosa, a, r
        int angle = 360
        int i, j
        double pi_over_180 = M_PI / 180.

    # Default center.
    if not center:
        center0 = <int> round(imgshape0/2.)
        center1 = <int> round(imgshape1/2.)
    else:
        center0 = <int> center[0]
        center1 = <int> center[1]

    # Calculate the radius
    if full_output:
        # Pythagoras to find the largest radius
        deltax = max([center0, imgshape0 - center0])
        deltay = max([center1, imgshape1 - center1])
        radius = <int> ceil(sqrt(deltax**2 + deltay**2))

        corner1 = radius - center1
        corner0 = radius - center0

        # Prepare a larger enclosing image
        if img.ndim == 2:
            extended_img = np.zeros((2*radius, 2*radius), dtype=img.dtype)
        elif img.ndim == 3:
            extended_img = np.zeros((2*radius, 2*radius, img.shape[2]),
                                    dtype=img.dtype)

        extended_img[corner1:corner1+imgshape1, corner0:corner0+imgshape0] = img
        img = extended_img
        del extended_img
        center0 = radius
        center1 = radius

    else:
        radius = min((center0,
                      center1,
                      imgshape0 - center0,
                      imgshape1 - center1))

    # Allocate an image
    if img.ndim == 2:
        img2 = np.zeros((radius, angle), dtype=img.dtype)
    elif img.ndim == 3:
        img2 = np.zeros((radius, angle, img.shape[2]), dtype=img.dtype)

    # convert to polar coordinates
    i = 0
    for a in range(0,  angle, 1):
        a = a * pi_over_180
        j = 0
        sina = sin(a)
        cosa = cos(a)
        for r in range(0, radius, 1):
            img2[j, i] = img[center1 + round(r*sina), center0 + round(r*cosa)]
            j += 1
        i += 1

    return img2
