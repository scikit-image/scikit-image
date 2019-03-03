#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

"""Color Mixer

NumPy does not do overflow checking when adding or multiplying
integers, so currently the only way to clip results efficiently
(without making copies of the data) is with an extension such as this
one.

"""

cimport numpy as cnp
from libc.math cimport exp, pow


def add(cnp.ndarray[cnp.uint8_t, ndim=3] img,
        cnp.ndarray[cnp.uint8_t, ndim=3] stateimg,
        Py_ssize_t channel, Py_ssize_t amount):
    """Add a given amount to a color channel of `stateimg`, and
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
    cdef Py_ssize_t height = img.shape[0]
    cdef Py_ssize_t width = img.shape[1]
    cdef Py_ssize_t k = channel
    cdef Py_ssize_t n = amount

    cdef cnp.int16_t op_result

    cdef cnp.uint8_t lut[256]

    cdef Py_ssize_t i, j, l

    with nogil:

        for l from 0 <= l < 256:
            op_result = <cnp.int16_t>(l + n)
            if op_result > 255:
                op_result = 255
            elif op_result < 0:
                op_result = 0
            else:
                pass
            lut[l] = <cnp.uint8_t>op_result

        for i from 0 <= i < height:
            for j from 0 <= j < width:
                img[i, j, k] = lut[stateimg[i,j,k]]


def multiply(cnp.ndarray[cnp.uint8_t, ndim=3] img,
             cnp.ndarray[cnp.uint8_t, ndim=3] stateimg,
             Py_ssize_t channel, float amount):
    """Multiply a color channel of `stateimg` by a certain amount, and
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
    cdef Py_ssize_t height = img.shape[0]
    cdef Py_ssize_t width = img.shape[1]
    cdef Py_ssize_t k = channel
    cdef float n = amount

    cdef float op_result

    cdef cnp.uint8_t lut[256]

    cdef Py_ssize_t i, j, l

    with nogil:

        for l from 0 <= l < 256:
            op_result = l * n
            if op_result > 255:
                op_result = 255
            elif op_result < 0:
                op_result = 0
            else:
                pass
            lut[l] = <cnp.uint8_t>op_result

        for i from 0 <= i < height:
            for j from 0 <= j < width:
                img[i,j,k] = lut[stateimg[i,j,k]]


def brightness(cnp.ndarray[cnp.uint8_t, ndim=3] img,
             cnp.ndarray[cnp.uint8_t, ndim=3] stateimg,
             float factor, Py_ssize_t offset):
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

    cdef Py_ssize_t height = img.shape[0]
    cdef Py_ssize_t width = img.shape[1]

    cdef float op_result
    cdef cnp.uint8_t lut[256]

    cdef Py_ssize_t i, j, k
    with nogil:

        for k from 0 <= k < 256:
            op_result = k * factor + offset
            if op_result > 255:
                op_result = 255
            elif op_result < 0:
                op_result = 0
            else:
                pass
            lut[k] = <cnp.uint8_t>op_result

        for i from 0 <= i < height:
            for j from 0 <= j < width:
                img[i,j,0] = lut[stateimg[i,j,0]]
                img[i,j,1] = lut[stateimg[i,j,1]]
                img[i,j,2] = lut[stateimg[i,j,2]]


def sigmoid_gamma(cnp.ndarray[cnp.uint8_t, ndim=3] img,
                  cnp.ndarray[cnp.uint8_t, ndim=3] stateimg,
                  float alpha, float beta):

    cdef Py_ssize_t height = img.shape[0]
    cdef Py_ssize_t width = img.shape[1]

    cdef Py_ssize_t i, j, k

    cdef float c1 = 1 / (1 + exp(beta))
    cdef float c2 = 1 / (1 + exp(beta - alpha)) - c1

    cdef cnp.uint8_t lut[256]

    with nogil:

        # compute the lut
        for k from 0 <= k < 256:
            lut[k] = <cnp.uint8_t>(((1 / (1 + exp(beta - (k / 255.) * alpha)))
                                    - c1) * 255 / c2)
        for i from 0 <= i < height:
            for j from 0 <= j < width:
                img[i,j,0] = lut[stateimg[i,j,0]]
                img[i,j,1] = lut[stateimg[i,j,1]]
                img[i,j,2] = lut[stateimg[i,j,2]]


def gamma(cnp.ndarray[cnp.uint8_t, ndim=3] img,
          cnp.ndarray[cnp.uint8_t, ndim=3] stateimg,
          float gamma):

    cdef Py_ssize_t height = img.shape[0]
    cdef Py_ssize_t width = img.shape[1]

    cdef cnp.uint8_t lut[256]

    cdef Py_ssize_t i, j, k

    if gamma == 0:
        gamma = 0.00000000000000000001
    gamma = 1./gamma

    with nogil:

        # compute the lut
        for k from 0 <= k < 256:
            lut[k] = <cnp.uint8_t>((pow((k / 255.), gamma) * 255))

        for i from 0 <= i < height:
            for j from 0 <= j < width:
                img[i,j,0] = lut[stateimg[i,j,0]]
                img[i,j,1] = lut[stateimg[i,j,1]]
                img[i,j,2] = lut[stateimg[i,j,2]]


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
    cdef Py_ssize_t hi

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
    https://en.wikipedia.org/wiki/HSL_and_HSV

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
    https://en.wikipedia.org/wiki/HSL_and_HSV

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


def hsv_add(cnp.ndarray[cnp.uint8_t, ndim=3] img,
            cnp.ndarray[cnp.uint8_t, ndim=3] stateimg,
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

    cdef Py_ssize_t height = img.shape[0]
    cdef Py_ssize_t width = img.shape[1]

    cdef float HSV[3]
    cdef float RGB[3]

    cdef Py_ssize_t i, j

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

                img[i, j, 0] = <cnp.uint8_t>RGB[0]
                img[i, j, 1] = <cnp.uint8_t>RGB[1]
                img[i, j, 2] = <cnp.uint8_t>RGB[2]


def hsv_multiply(cnp.ndarray[cnp.uint8_t, ndim=3] img,
                 cnp.ndarray[cnp.uint8_t, ndim=3] stateimg,
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

    cdef Py_ssize_t height = img.shape[0]
    cdef Py_ssize_t width = img.shape[1]

    cdef float HSV[3]
    cdef float RGB[3]

    cdef Py_ssize_t i, j

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

                img[i, j, 0] = <cnp.uint8_t>RGB[0]
                img[i, j, 1] = <cnp.uint8_t>RGB[1]
                img[i, j, 2] = <cnp.uint8_t>RGB[2]
