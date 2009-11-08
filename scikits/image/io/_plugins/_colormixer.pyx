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

cdef extern from "math.h":
    float exp(float) nogil
    float pow(float, float) nogil


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
    with nogil:
        for i from 0 <= i < height:
            for j from 0 <= j < width:
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
    with nogil:
        for i from 0 <= i < height:
            for j from 0 <= j < width:
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
             float factor, int offset):
    """Modify the brightness of an image.
    'factor' is multiplied to all channels, which are
    then added by 'amount'. Overflow is clipped.

    Parameters
    ----------
    img : (M, N, 3) ndarray of uint8
        Output image.
    stateimg : (M, N, 3) ndarray of uint8
        Input image.
    factor : float
        Multiplication factor.
    offset : int
        Ammount to add to each channel.

    """

    cdef int height = img.shape[0]
    cdef int width = img.shape[1]

    cdef float op_result

    cdef int i, j, k
    with nogil:
        for i from 0 <= i < height:
            for j from 0 <= j < width:
                for k from 0 <= k < 3:
                    op_result = <float>((stateimg[i,j,k] * factor + offset))

                    if op_result > 255:
                        img[i, j, k] = 255
                    elif op_result < 0:
                        img[i, j, k] = 0
                    else:
                        img[i, j, k] = <np.uint8_t>op_result


@cython.boundscheck(False)
def sigmoid_gamma(np.ndarray[np.uint8_t, ndim=3] img,
                  np.ndarray[np.uint8_t, ndim=3] stateimg,
                  float alpha, float beta):

    cdef int height = img.shape[0]
    cdef int width = img.shape[1]

    cdef float c1, c2, r, g, b
    cdef int i, j, k

    with nogil:
        for i from 0 <= i < height:
            for j from 0 <= j < width:
                r = <float>stateimg[i,j,0] / 255.
                g = <float>stateimg[i,j,1] / 255.
                b = <float>stateimg[i,j,2] / 255.

                c1 = 1 / (1 + exp(beta))
                c2 = 1 / (1 + exp(beta - alpha)) - c1

                r = 1 / (1 + exp(beta - r * alpha))
                r = (r - c1) / c2

                g = 1 / (1 + exp(beta - g * alpha))
                g = (g - c1) / c2

                b = 1 / (1 + exp(beta - b * alpha))
                b = (b - c1) / c2

                img[i,j,0] = <np.uint8_t>(r * 255)
                img[i,j,1] = <np.uint8_t>(g * 255)
                img[i,j,2] = <np.uint8_t>(b * 255)


@cython.boundscheck(False)
def gamma(np.ndarray[np.uint8_t, ndim=3] img,
          np.ndarray[np.uint8_t, ndim=3] stateimg,
          float gamma):

    cdef int height = img.shape[0]
    cdef int width = img.shape[1]

    cdef float r, g, b

    cdef int i, j

    gamma = 1./gamma

    with nogil:
        for i from 0 <= i < height:
            for j from 0 <= j < width:
                r = <float>stateimg[i,j,0] / 255.
                g = <float>stateimg[i,j,1] / 255.
                b = <float>stateimg[i,j,2] / 255.

                img[i,j,0] = <np.uint8_t>(pow(r, gamma) * 255)
                img[i,j,1] = <np.uint8_t>(pow(g, gamma) * 255)
                img[i,j,2] = <np.uint8_t>(pow(b, gamma) * 255)



cdef void rgb_2_hsv(float* RGB, float* HSV) nogil:
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
        H = (60 * (G - B) / (MAX - MIN) + 360) % 360
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


cdef void hsv_2_rgb(float* HSV, float* RGB) nogil:
    cdef float H, S, V
    cdef float f, p, q, t, r, g, b
    cdef int hi

    H = HSV[0]
    S = HSV[1]
    V = HSV[2]

    if H > 360:
        H = H % 360
    elif H < 0:
        H = 360 - ((-1 * H) % 360)
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


def py_hsv_2_rgb(H, S, V):
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
    out : (R, G, B) ints
        Each from 0 - 255

    conversion convention from here:
    http://en.wikipedia.org/wiki/HSL_and_HSV

    '''
    cdef float HSV[3]
    cdef float RGB[3]

    HSV[0] = H
    HSV[1] = S
    HSV[2] = V

    hsv_2_rgb(HSV, RGB)

    R = int(RGB[0] * 255)
    G = int(RGB[1] * 255)
    B = int(RGB[2] * 255)

    return (R, G, B)

def py_rgb_2_hsv(R, G, B):
    '''Convert an HSV value to RGB.

    Automatic clipping.

    Parameters
    ----------
    R : int
        From 0. - 255.
    G : int
        From 0. - 255.
    B : int
        From 0. - 255.

    Returns
    -------
    out : (H, S, V) floats
        Ranges (0...360), (0...1), (0...1)

    conversion convention from here:
    http://en.wikipedia.org/wiki/HSL_and_HSV

    '''
    cdef float HSV[3]
    cdef float RGB[3]

    RGB[0] = R
    RGB[1] = G
    RGB[2] = B

    rgb_2_hsv(RGB, HSV)

    H = HSV[0]
    S = HSV[1]
    V = HSV[2]

    return (H, S, V)


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

    with nogil:
        for i from 0 <= i < height:
            for j from 0 <= j < width:
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

    Note that since hue is in degrees, it makes no sense to multiply
    that channel, thus an add operation is performed on the hue. And the
    values given for h_amt, should be the same as for hsv_add

    Parameters
    ----------
    img : (M, N, 3) ndarray of uint8
        Output image.
    stateimg : (M, N, 3) ndarray of uint8
        Input image.
    h_amt : float
        Ammount to add to H channel.
    s_amt : float
        Ammount by which to multiply S channel.
    v_amt : float
        Ammount by which to multiply V channel.


    """

    cdef int height = img.shape[0]
    cdef int width = img.shape[1]

    cdef float HSV[3]
    cdef float RGB[3]

    cdef int i, j

    with nogil:
        for i from 0 <= i < height:
            for j from 0 <= j < width:
                RGB[0] = stateimg[i, j, 0]
                RGB[1] = stateimg[i, j, 1]
                RGB[2] = stateimg[i, j, 2]

                rgb_2_hsv(RGB, HSV)

                # Multiply operation
                HSV[0] += h_amt
                HSV[1] *= s_amt
                HSV[2] *= v_amt

                hsv_2_rgb(HSV, RGB)

                RGB[0] *= 255
                RGB[1] *= 255
                RGB[2] *= 255

                img[i, j, 0] = <np.uint8_t>RGB[0]
                img[i, j, 1] = <np.uint8_t>RGB[1]
                img[i, j, 2] = <np.uint8_t>RGB[2]






