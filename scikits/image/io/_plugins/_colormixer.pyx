# -*- python -*-

"""Colour Mixer

NumPy does not do overflow checking when adding or multiplying
integers, so currently the only way to clip results efficiently
(without making copies of the data) is with an extension such as this
one.

"""

import numpy as np
cimport numpy as np

import cython

@cython.boundscheck(False)
def add(np.ndarray[np.uint8_t, ndim=3] img,
        np.ndarray[np.uint8_t, ndim=3] stateimg,
        int channel, int amount):
    """Add a given amount to a colour channel of `stateimg`, and
    store the result in `img`.  Overflow is clipped.

    Parameters
    ----------
    img : (M, N, 3) ndarray of uint8
        Output image.
    stateimg : (M, N, 3) ndarray of uint8
        Input image.
    channel : int
        Channel (0 for "red", 1 for "green", 2 for "blue").
    amount : int
        Value to add.

    """
    cdef int height = img.shape[0]
    cdef int width = img.shape[1]
    cdef int k = channel
    cdef int n = amount

    cdef np.int16_t op_result

    cdef int i, j
    for i in range(height):
        for j in range(width):
            op_result = <np.int16_t>(stateimg[i,j,k] + n)
            if op_result > 255:
                img[i, j, k] = 255
            elif op_result < 0:
                img[i, j, k] = 0
            else:
                img[i, j, k] = <np.uint8_t>op_result

@cython.boundscheck(False)
def multiply(np.ndarray[np.uint8_t, ndim=3] img,
             np.ndarray[np.uint8_t, ndim=3] stateimg,
             int channel, float amount):
    """Multiply a colour channel of `stateimg` by a certain amount, and
    store the result in `img`.  Overflow is clipped.

    Parameters
    ----------
    img : (M, N, 3) ndarray of uint8
        Output image.
    stateimg : (M, N, 3) ndarray of uint8
        Input image.
    channel : int
        Channel (0 for "red", 1 for "green", 2 for "blue").
    amount : float
        Multiplication factor.

    """
    cdef int height = img.shape[0]
    cdef int width = img.shape[1]
    cdef int k = channel
    cdef float n = amount

    cdef float op_result

    cdef int i, j
    for i in range(height):
        for j in range(width):
            op_result = <float>(stateimg[i,j,k] * n)
            if op_result > 255:
                img[i, j, k] = 255
            elif op_result < 0:
                img[i, j, k] = 0
            else:
                img[i, j, k] = <np.uint8_t>op_result

@cython.boundscheck(False)
def brightness(np.ndarray[np.uint8_t, ndim=3] img,
             np.ndarray[np.uint8_t, ndim=3] stateimg,
             int offset, float factor):
    """Modify the brightness of an image.
    'offset' is added to all channels, which are
    then multiplied by 'factor'. Overflow is clipped.

    Parameters
    ----------
    img : (M, N, 3) ndarray of uint8
        Output image.
    stateimg : (M, N, 3) ndarray of uint8
        Input image.
    offset : int
        Ammount to add to each channel.
    factor : float
        Multiplication factor.

    """

    cdef int height = img.shape[0]
    cdef int width = img.shape[1]

    cdef float op_result

    cdef int i, j, k
    for i in range(height):
        for j in range(width):
            for k in range(3):
                op_result = <float>((stateimg[i,j,k] + offset)*factor)
                if op_result > 255:
                    img[i, j, k] = 255
                elif op_result < 0:
                    img[i, j, k] = 0
                else:
                    img[i, j, k] = <np.uint8_t>op_result

cdef void rgb_2_hsv(float* RGB, float* HSV):
    '''Convert an HSV value to RGB.

    Automatic clipping.

    Parameters
    ----------
    R : float
        From 0. - 255.
    G : float
        From 0. - 255.
    B : float
        From 0. - 255.

    Returns
    -------
    out : (H, S, V) Floats
        Ranges (0...360), (0...1), (0...1)

    conversion convention from here:
    http://en.wikipedia.org/wiki/HSL_and_HSV

    '''

    cdef float R, G, B, H, S, V, MAX, MIN
    R = RGB[0]
    G = RGB[1]
    B = RGB[2]

    if R > 255:
        R = 255
    elif R < 0:
        R = 0
    else:
        pass

    if G > 255:
        G = 255
    elif G < 0:
        G = 0
    else:
        pass

    if B > 255:
        B = 255
    elif B < 0:
        B = 0
    else:
        pass

    if R < G:
        MIN = R
        MAX = G
    else:
        MIN = G
        MAX = R

    if B < MIN:
        MIN = B
    elif B > MAX:
        MAX = B
    else:
        pass

    V = MAX / 255.

    if MAX == MIN:
        H = 0.
    elif MAX == R:
        H = (60 * (G - B) / (MAX - MIN)) % 360
    elif MAX == G:
        H = 60 * (B - R) / (MAX - MIN) + 120
    else:
        H = 60 * (R - G) / (MAX - MIN) + 240

    if MAX == 0:
        S = 0
    else:
        S = 1 - MIN / MAX

    HSV[0] = H
    HSV[1] = S
    HSV[2] = V

cdef void hsv_2_rgb(float* HSV, float* RGB):
    '''Convert an HSV value to RGB.

    Automatic clipping.

    Parameters
    ----------
    H : float
        From 0. - 360.
    S : float
        From 0. - 1.
    V : float
        From 0. - 1.

    Returns
    -------
    out : (R, G, B) Floats
        Each from 0. - 1.

    conversion convention from here:
    http://en.wikipedia.org/wiki/HSL_and_HSV

    '''

    cdef float H, S, V
    cdef float f, p, q, t, r, g, b
    cdef int hi

    H = HSV[0]
    S = HSV[1]
    V = HSV[2]

    if H > 360:
        H = 360
    elif H < 0:
        H = 0
    else:
        pass

    if S > 1:
        S = 1
    elif S < 0:
        S = 0
    else:
        pass

    if V > 1:
        V = 1
    elif V < 0:
        V = 0
    else:
        pass

    hi = (<int>(H / 60.)) % 6
    f = (H / 60.) - (<int>(H / 60.))
    p = V * (1 - S)
    q = V * (1 - f * S)
    t = V * (1 - (1 -f) * S)

    if hi == 0:
        r = V
        g = t
        b = p
    elif hi == 1:
        r = q
        g = V
        b = p
    elif hi == 2:
        r = p
        g = V
        b = t
    elif hi == 3:
        r = p
        g = q
        b = V
    elif hi == 4:
        r = t
        g = p
        b = V
    else:
        r = V
        g = p
        b = q

    RGB[0] = r
    RGB[1] = g
    RGB[2] = b

@cython.boundscheck(False)
def hsv_add(np.ndarray[np.uint8_t, ndim=3] img,
            np.ndarray[np.uint8_t, ndim=3] stateimg,
            float h_amt, float s_amt, float v_amt):
    """Modify the image color by specifying additive HSV Values.

    Since the underlying images are RGB, all three values HSV
    must be specified at the same time.

    The RGB triplet in the image is converted to HSV, the operation
    is applied, and then the HSV triplet is converted back to RGB

    HSV values are scaled to H(0. - 360.), S(0. - 1.), V(0. - 1.)
    then the operation is performed and any overflow is clipped, then the
    reverse transform is performed. Those are the ranges to keep in mind,
    when passing in values.

    Parameters
    ----------
    img : (M, N, 3) ndarray of uint8
        Output image.
    stateimg : (M, N, 3) ndarray of uint8
        Input image.
    h_amt : float
        Ammount to add to H channel.
    s_amt : float
        Ammount to add to S channel.
    v_amt : float
        Ammount to add to V channel.


    """

    cdef int height = img.shape[0]
    cdef int width = img.shape[1]

    cdef float HSV[3]
    cdef float RGB[3]

    cdef int i, j

    for i in range(height):
        for j in range(width):
            RGB[0] = stateimg[i, j, 0]
            RGB[1] = stateimg[i, j, 1]
            RGB[2] = stateimg[i, j, 2]

            rgb_2_hsv(RGB, HSV)

            # Add operation
            HSV[0] += h_amt
            HSV[1] += s_amt
            HSV[2] += v_amt

            hsv_2_rgb(HSV, RGB)

            RGB[0] *= 255
            RGB[1] *= 255
            RGB[2] *= 255

            img[i, j, 0] = <np.uint8_t>RGB[0]
            img[i, j, 1] = <np.uint8_t>RGB[1]
            img[i, j, 2] = <np.uint8_t>RGB[2]

@cython.boundscheck(False)
def hsv_multiply(np.ndarray[np.uint8_t, ndim=3] img,
                 np.ndarray[np.uint8_t, ndim=3] stateimg,
                 float h_amt, float s_amt, float v_amt):
    """Modify the image color by specifying multiplicative HSV Values.

    Since the underlying images are RGB, all three values HSV
    must be specified at the same time.

    The RGB triplet in the image is converted to HSV, the operation
    is applied, and then the HSV triplet is converted back to RGB

    HSV values are scaled to H(0. - 360.), S(0. - 1.), V(0. - 1.)
    then the operation is performed and any overflow is clipped, then the
    reverse transform is performed. Those are the ranges to keep in mind,
    when passing in values.

    Parameters
    ----------
    img : (M, N, 3) ndarray of uint8
        Output image.
    stateimg : (M, N, 3) ndarray of uint8
        Input image.
    h_amt : float
        Ammount to add to H channel.
    s_amt : float
        Ammount to add to S channel.
    v_amt : float
        Ammount to add to V channel.


    """

    cdef int height = img.shape[0]
    cdef int width = img.shape[1]

    cdef float HSV[3]
    cdef float RGB[3]

    cdef int i, j

    for i in range(height):
        for j in range(width):
            RGB[0] = stateimg[i, j, 0]
            RGB[1] = stateimg[i, j, 1]
            RGB[2] = stateimg[i, j, 2]

            rgb_2_hsv(RGB, HSV)

            # Multiply operation
            HSV[0] *= h_amt
            HSV[1] *= s_amt
            HSV[2] *= v_amt

            hsv_2_rgb(HSV, RGB)

            RGB[0] *= 255
            RGB[1] *= 255
            RGB[2] *= 255

            img[i, j, 0] = <np.uint8_t>RGB[0]
            img[i, j, 1] = <np.uint8_t>RGB[1]
            img[i, j, 2] = <np.uint8_t>RGB[2]





