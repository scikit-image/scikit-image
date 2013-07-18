import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, abs, round
from heapq import heappush, heappop
from skimage.morphology import dilation, disk

cdef:
    cnp.uint8_t KNOWN = 0
    cnp.uint8_t BAND = 1
    cnp.uint8_t INSIDE = 2


def initialise(_mask):

    mask = _mask.astype(np.uint8, order='C')
    outside = dilation(mask, disk(1))
    band = np.logical_xor(mask, outside).astype(np.uint8, order='C')

    flag = (2 * outside) - band
    u = np.where(flag == INSIDE, 1.0e6, 0)

    heap = []
    cdef cnp.uint8_t i_in, j_in
    indices = np.transpose(np.where(flag == BAND)).astype(dtype=np.uint8)

    for i_in, j_in in indices:
        heappush(heap, (u[i_in, j_in], (i_in, j_in)))

    return flag, u, heap


cdef cnp.float_t[:] grad_func(Py_ssize_t i, Py_ssize_t j, cnp.uint8_t[:, ::1] flag, cnp.float_t[:, ::1] array, Py_ssize_t channel=1):
    cdef:
        cnp.float_t factor,
        cnp.float_t[:] gradU = np.zeros(2, dtype=np.float)

    if channel == 0:
        factor = 2.0
    elif channel == 1:
        factor = 0.5

    if flag[i, j + 1] != INSIDE and flag[i, j - 1] != INSIDE:
        gradU[0] = (array[i, j + 1] - array[i, j - 1]) * factor
    elif flag[i, j + 1] != INSIDE and flag[i, j - 1] == INSIDE:
        gradU[0] = (array[i, j + 1] - array[i, j])
    elif flag[i, j + 1] == INSIDE and flag[i, j - 1] != INSIDE:
        gradU[0] = (array[i, j] - array[i, j - 1])
    elif flag[i, j + 1] == INSIDE and flag[i, j - 1] == INSIDE:
        gradU[0] = 0

    if flag[i + 1, j] != INSIDE and flag[i - 1, j] != INSIDE:
        gradU[1] = (array[i + 1, j] - array[i - 1, j]) * factor
    elif flag[i + 1, j] != INSIDE and flag[i - 1, j] == INSIDE:
        gradU[1] = (array[i + 1, j] - array[i, j])
    elif flag[i + 1, j] == INSIDE and flag[i - 1, j] != INSIDE:
        gradU[1] = (array[i, j] - array[i - 1, j])
    elif flag[i + 1, j] == INSIDE and flag[i - 1, j] == INSIDE:
        gradU[1] = 0

    return gradU


def ep_neighbor(Py_ssize_t i, Py_ssize_t j, Py_ssize_t h, Py_ssize_t w, Py_ssize_t epsilon):
    cdef:
        cnp.int8_t[:, ::1] center_ind
        Py_ssize_t i_ep, j_ep
        list nb = list()

    indices = np.transpose(np.where(disk(epsilon)))
    center_ind = np.ascontiguousarray(indices - [epsilon, epsilon] + [i, j], dtype=np.int8)

    for i_ep, j_ep in center_ind:
        if i_ep > 0 and j_ep > 0:
            if i_ep < h - 1 and j_ep < w - 1:
                nb.append([i_ep, j_ep])

    return nb


cdef inp_point(Py_ssize_t i, Py_ssize_t j, cnp.uint8_t[:, ::1] image, cnp.uint8_t[:, ::1] flag, cnp.float_t[:, ::1] u, Py_ssize_t epsilon):

    cdef:
        Py_ssize_t i_nb, j_nb
        cnp.int8_t rx, ry
        cnp.float_t dst, lev, dirc
        cnp.float_t Ia, weight, Jx, Jy, norm, sat
        cnp.float_t[:] gradU = np.zeros(2, dtype=np.float)
        cnp.float_t[:] gradI = np.zeros(2, dtype=np.float)

    Ia, Jx, Jy, norm = 0, 0, 0, 0
    gradU = grad_func(i, j, flag, u, channel=1)
    nb = ep_neighbor(i, j, image.shape[0], image.shape[1], epsilon)

    for [i_nb, j_nb] in nb:
        if flag[i_nb, j_nb] == KNOWN:
            rx = i - i_nb
            ry = j - j_nb

            dst = 1. / ((rx * rx + ry * ry) *
                        sqrt((rx * rx + ry * ry)))
            lev = 1. / (1 + abs(u[i_nb, j_nb] - u[i, j]))
            dirc = rx * gradU[0] + ry * gradU[1]

            if abs(dirc) <= 0.01:
                dirc = 1.0e-6
            weight = abs(dst * lev * dirc)

            gradI = grad_func(i_nb, j_nb, flag, np.ascontiguousarray(image, np.float), channel=0)

            Ia += weight * image[i_nb, j_nb]
            Jx -= weight * gradI[0] * rx
            Jy -= weight * gradI[1] * ry
            norm += weight

    sat = (Ia / norm + (Jx + Jy) / (sqrt(Jx * Jx + Jy * Jy) + 1.0e-20)
           + 0.5)
    image[i, j] = <cnp.uint8_t> round(sat)


cdef cnp.float_t eikonal(Py_ssize_t i1, Py_ssize_t j1, Py_ssize_t i2, Py_ssize_t j2, cnp.uint8_t[:, ::1] flag, cnp.float_t[:, ::1] u):
    cdef cnp.float_t u_out, u1, u2, r, s
    u_out = 1.0e6
    u1 = u[i1, j1]
    u2 = u[i2, j2]

    if flag[i1, j1] == KNOWN:
        if flag[i2, j2] == KNOWN:
            r = sqrt(2 - (u1 - u2) ** 2)
            s = (u1 + u2 - r) * 0.5
            if s >= u1 and s >= u2:
                u_out = s
            else:
                s += r
                if s >= u1 and s >= u2:
                    u_out = s
        else:
            u_out = 1 + u1
    elif flag[i2, j2] == KNOWN:
        u_out = 1 + u2

    return u_out


def fast_marching_method(cnp.uint8_t[:, ::1] image, cnp.uint8_t[:, ::1] flag, cnp.float_t[:, ::1] u, heap, _run_inpaint=True, Py_ssize_t epsilon=5):
    cdef Py_ssize_t i, j, i_nb, j_nb

    while len(heap):
        i, j = heappop(heap)[1]
        flag[i, j] = KNOWN

        if ((i <= 1) or (j <= 1) or (i >= image.shape[0] - 2)
                or (j >= image.shape[1] - 2)):
            continue

        for (i_nb, j_nb) in (i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1):

            if not flag[i_nb, j_nb] == KNOWN:
                u[i_nb, j_nb] = min(eikonal(i_nb - 1, j_nb,
                                            i_nb, j_nb - 1, flag, u),
                                    eikonal(i_nb + 1, j_nb,
                                            i_nb, j_nb - 1, flag, u),
                                    eikonal(i_nb - 1, j_nb,
                                            i_nb, j_nb + 1, flag, u),
                                    eikonal(i_nb + 1, j_nb,
                                            i_nb, j_nb + 1, flag, u))

                if flag[i_nb, j_nb] == INSIDE:
                    flag[i_nb, j_nb] = BAND
                    heappush(heap, (u[i_nb, j_nb], (i_nb, j_nb)))

                    if _run_inpaint:
                        inp_point(i_nb, j_nb, image, flag, u, epsilon)

    if not _run_inpaint:
        return u
    else:
        return image


cpdef inpaint(input_image,  inpaint_mask, epsilon=5):
    h, w = input_image.shape
    image = np.zeros((h + 2, w + 2), np.uint8)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    image[1: -1, 1: -1] = input_image
    mask[1: -1, 1: -1] = inpaint_mask

    flag, u, heap = initialise(mask)

    painted = fast_marching_method(image, flag, u, heap, epsilon=epsilon)

    return painted
