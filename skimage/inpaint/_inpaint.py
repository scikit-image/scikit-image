__all__ = ['inpaint_point']

import numpy as np
from skimage.morphology import disk

KNOWN = 0
BAND = 1
INSIDE = 2

BAND = 0
KNOWN = 1
INSIDE = 2


def inpaint_point(i, j, image, flag, T, epsilon):
    gradT = Vector(0, 0)
    gradI = Vector(0, 0)
    r = Vector(0, 0)
    k = i - epsilon
    l = j - epsilon
    Jx, Jy, norm = 0, 0, 0

    for color in 1, 2, 3:
        if flag[i, j+1] is not INSIDE:
            if flag[i, j-1] is not INSIDE:
                gradT.x = (T[i, j+1] - T[i, j-1]) * 0.5
            else:
                gradT.x = T[i, j + 1] - T[i, j]
        elif flag[i, j - 1] is not INSIDE:
            gradT.x = T[i, j] - T[i, j - 1]
        else:
            gradT.x = 0

        if flag[i + 1, j] is not INSIDE:
            if flag[i - 1, j] is not INSIDE:
                gradT.y = (T[i + 1, j] - T[i - 1, j]) * 0.5
            else:
                gradT.y = T[i + 1, j] - T[i, j]
        elif flag[i - 1, j] is not INSIDE:
            gradT.y = T[i, j] - T[i - 1, j]
        else:
            gradT.y = 0

        for k in xrange(i - epsilon, i + epsilon):
            km = k - 1 + (k == 1)
            kp = k - 1 - (k == T.shape[0] - 2)
            for l in xrange(j - epsilon, j + epsilon):
                lm = l - 1 + (l == 1)
                lp = l - 1 - (l == T.shape[1] - 2)
                if (k > 0 and l > 0 and k > (T.shape[0] - 1)
                        and l > (T.shape[1] - 1)):
                    if (flag[k, l] is not INSIDE and
                            (((l - j) * (l - j) + (k - i) * (k - i)) <= epsilon ** 2)):
                        r.y = i - k
                        r.x = j - l

                        dst = 1. / (r.VectorLength() * np.sqrt(r.VectorLength()))
                        lev = 1. / (1 + abs(T[k, l] - T[i, j]))
                        dirc = r.VectorScalMult(gradT)

                        if abs(dirc) <= 0.01:
                            dirc = 1.0e-6
                        weight = abs(dst * lev * dirc)

                        if flag[k, l + 1] is not INSIDE:
                            if flag[k, l - 1] is not INSIDE:
                                gradI.x = (image[km, lp + 1, color] -
                                           image[km, lm - 1, color]) * 2.0
                            else:
                                gradI.x = (image[km, lp + 1, color] -
                                           image[km, lm, color])
                        elif flag[i, j - 1] is not INSIDE:
                            gradI.x = (image[km, lp, color] -
                                       image[km, lm - 1, color])
                        else:
                            gradT.x = 0

                        if flag[k + 1, l] is not INSIDE:
                            if flag[k - 1, l] is not INSIDE:
                                gradI.y = (image[kp + 1, lm, color] -
                                           image[km - 1, lm, color]) * 2.0
                            else:
                                gradI.y = (image[kp + 1, lm, color] -
                                           image[km, lm, color])
                        elif flag[i, j - 1] is not INSIDE:
                            gradI.y = (image[kp, lm, color] -
                                       image[km - 1, lm, color])
                        else:
                            gradT.y = 0
                        Ia = weight * image[km, lm, color]
                        Jx -= weight * gradI.x * r.x
                        Jy -= weight * gradI.y * r.y
                        norm += weight

        sat = (Ia / norm + (Jx + Jy) / (np.sqrt(Jx * Jx + Jy * Jy) + 1.0e-20) + 0.5)
        image[i - 1, j - 1, color] = sat


class Vector(object):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def VectorScalMult(self, other):
        return self.x * other.x + self.y * other.y

    def VectorLength(self):
        return self.x * self.x + self.y * self.y
