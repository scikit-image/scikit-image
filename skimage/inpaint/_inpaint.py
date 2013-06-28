__all__ = ['inpaint_point']

import numpy as np
from skimage.morphology import disk

KNOWN = 0
BAND = 1
INSIDE = 2

BAND = 0
KNOWN = 1
INSIDE = 2


def grad_speed(i, j, flag, u):
    """This function calculates the gradient of the speed of a point
    depending on the value of the flag of its neighbours. the gradient
    is computed using Central Differences.
    """

    if flag[i, j + 1] is not INSIDE:
        if flag[i, j - 1] is not INSIDE:
            gradUx = (u[i, j + 1] - u[i, j - 1]) * 0.5
        else:
            gradUx = u[i, j + 1] - u[i, j]
    elif flag[i, j - 1] is not INSIDE:
        gradUx = u[i, j] - u[i, j - 1]
    else:
        gradUx = 0

    if flag[i + 1, j] is not INSIDE:
        if flag[i - 1, j] is not INSIDE:
            gradUy = (u[i + 1, j] - u[i - 1, j]) * 0.5
        else:
            gradUy = u[i + 1, j] - u[i, j]
    elif flag[i - 1, j] is not INSIDE:
        gradUy = u[i, j] - u[i - 1, j]
    else:
        gradUy = 0

    return gradUx, gradUy


def inpaint_point(i, j, image, flag, u, epsilon):
    gradIx, gradIy = 0, 0
    rx, ry = 0, 0
    k = i - epsilon
    l = j - epsilon
    Jx, Jy, norm = 0, 0, 0
    #If the input image is 3 channel. uODO: support for a single channel
    for color in 0, 1, 2:
        gradUx, gradUy = grad_speed(i, j, flag, u)

        for k in xrange(i - epsilon, i + epsilon):
            km = k - 1 + (k == 1)
            kp = k - 1 - (k == u.shape[0] - 2)
            for l in xrange(j - epsilon, j + epsilon):
                lm = l - 1 + (l == 1)
                lp = l - 1 - (l == u.shape[1] - 2)
                if (k > 0 and l > 0 and k > (u.shape[0] - 1)
                        and l > (u.shape[1] - 1)):
                    cartesian_distance = (l - j) * (l - j) + (k - i) * (k - i)
                    if (flag[k, l] is not INSIDE and
                            cartesian_distance <= epsilon ** 2):
                        ry = i - k
                        rx = j - l

                        dst = 1. / ((rx * rx + ry * ry) *
                                    np.sqrt((rx * rx + ry * ry)))
                        lev = 1. / (1 + abs(u[k, l] - u[i, j]))
                        dirc = rx * gradUx

                        if abs(dirc) <= 0.01:
                            dirc = 1.0e-6
                        weight = abs(dst * lev * dirc)

                        if flag[k, l + 1] is not INSIDE:
                            if flag[k, l - 1] is not INSIDE:
                                gradIx = (image[km, lp + 1, color] -
                                          image[km, lm - 1, color]) * 2.0
                            else:
                                gradIx = (image[km, lp + 1, color] -
                                          image[km, lm, color])
                        elif flag[i, j - 1] is not INSIDE:
                            gradIx = (image[km, lp, color] -
                                      image[km, lm - 1, color])
                        else:
                            gradIx = 0

                        if flag[k + 1, l] is not INSIDE:
                            if flag[k - 1, l] is not INSIDE:
                                gradIy = (image[kp + 1, lm, color] -
                                          image[km - 1, lm, color]) * 2.0
                            else:
                                gradIy = (image[kp + 1, lm, color] -
                                          image[km, lm, color])
                        elif flag[i, j - 1] is not INSIDE:
                            gradIy = (image[kp, lm, color] -
                                      image[km - 1, lm, color])
                        else:
                            gradIy = 0
                        Ia = weight * image[km, lm, color]
                        Jx -= weight * gradIx * rx
                        Jy -= weight * gradIy * ry
                        norm += weight

        sat = (Ia / norm + (Jx + Jy) /
              (np.sqrt(Jx * Jx + Jy * Jy) + 1.0e-20) + 0.5)
        image[i - 1, j - 1, color] = sat


# class Vector(object):
#     def __init__(self, x, y):
#         self.x, self.y = x, y

#     def VectorScalMult(self, other):
#         return self.x * other.x + self.y * other.y

#     def VectorLength(self):
#         return self.x * self.x + self.y * self.y
