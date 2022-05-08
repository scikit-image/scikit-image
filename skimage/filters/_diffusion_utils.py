import numpy as np
# from numba import jit


# pythran export nonlinear_aniso_step(float64[:,:], float64[:,:], float64[:,:], float64[:,:],float64[:,:], float or int, uint8)
def nonlinear_aniso_step(src, dest, a, b, c, tau, border):
    for i in range(border, src.shape[0] - border - 1):
        for j in range(border, src.shape[1] - border - 1):
            dest[i, j] = src[i, j] + tau * (src[i - 1, j + 1] *
                ((abs(b[i - 1, j + 1]) - b[i - 1, j + 1]) / 4.0 + (abs(b[i, j]) - b[i, j]) / 4.0) + src[i, j + 1] *
                ((c[i, j + 1] + c[i, j]) / 2.0 - (abs(b[i, j + 1]) + abs(b[i, j])) / 2.0) + src[i + 1, j + 1] *
                ((abs(b[i + 1, j + 1]) + b[i + 1, j + 1]) / 4.0 + (abs(b[i, j]) + b[i, j]) / 4.0) + src[i - 1, j] *
                ((a[i - 1, j] + a[i, j]) / 2.0 - (abs(b[i - 1, j]) + abs(b[i, j])) / 2.0) + src[i, j] *
                ((-(a[i - 1, j] + 2 * a[i, j] + a[i + 1, j]) / 2.0 -
                    (abs(b[i - 1, j + 1]) - b[i - 1, j + 1] + abs(b[i + 1, j + 1]) + b[i + 1, j + 1]) / 4.0 -
                    (abs(b[i - 1, j - 1]) + b[i - 1, j - 1] + abs(b[i + 1, j - 1]) - b[i + 1, j - 1]) / 4.0 +
                    (abs(b[i - 1, j]) + abs(b[i + 1, j]) + abs(b[i, j - 1]) + abs(b[i, j + 1]) + 2.0 * abs(b[i, j])) / 2.0 -
                    (c[i, j - 1] + 2.0 * c[i, j] + c[i, j + 1]) / 2.0)) + src[i + 1, j] *
                ((a[i + 1, j] + a[i, j]) / 2.0 - (abs(b[i + 1, j]) + abs(b[i, j])) / 2.0) + src[i - 1, j - 1] *
                ((abs(b[i - 1, j - 1]) + b[i - 1, j - 1]) / 4.0 + (abs(b[i, j]) + b[i, j]) / 4.0) + src[i, j - 1] *
                ((c[i, j - 1] + c[i, j]) / 2.0 - (abs(b[i, j - 1]) + abs(b[i, j])) / 2.0) + src[i + 1, j - 1] *
                ((abs(b[i + 1, j - 1]) - b[i + 1, j - 1]) / 4.0 + (abs(b[i, j]) - b[i, j]) / 4.0))


# pythran export nonlinear_iso_step(float64[:,:], float64[:,:], float or int, float64[:,:], float64[:,:], float, str)
def nonlinear_iso_step(image, out_image, tau, gradX, gradY, alpha, type):
    h1 = h2 = 1
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            diff_ij = get_diffusivity(gradX[i, j], gradY[i, j], alpha, type)
            out_image[i, j] = (tau * (diff_ij + get_diffusivity(gradX[i + 1, j], gradY[i + 1, j], alpha, type)) /
                    2 * image[i + 1, j] / np.power(h1, 2)) + \
                        (tau * (diff_ij + get_diffusivity(gradX[i - 1, j], gradY[i - 1, j], alpha, type)) /
                        2 * image[i - 1, j] / np.power(h1, 2)) + (tau * (diff_ij + get_diffusivity(gradX[i, j + 1], gradY[i, j + 1], alpha, type)) /
                        2 * image[i, j + 1] / np.power(h2, 2)) + (tau * (diff_ij + get_diffusivity(gradX[i, j - 1], gradY[i, j - 1], alpha, type)) /
                        2 * image[i, j - 1] / np.power(h2, 2)) + (1 - (tau * (diff_ij + get_diffusivity(gradX[i + 1, j], gradY[i + 1, j], alpha, type)) /
                        2 / np.power(h1, 2)) - (tau * (diff_ij + get_diffusivity(gradX[i - 1, j], gradY[i - 1, j], alpha, type)) /
                        2 / np.power(h1, 2)) - (tau * (diff_ij + get_diffusivity(gradX[i, j + 1], gradY[i, j + 1], alpha, type)) /
                        2 / np.power(h2, 2)) - (tau * (diff_ij + get_diffusivity(gradX[i, j - 1], gradY[i, j - 1], alpha, type)) / 2 / np.power(h2, 2))) * image[i, j]


# pythran export linear_step(float64[:,:], float64[:,:], float or int)
def linear_step(image, out_image, tau):
    h1 = h2 = 1
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            out_image[i, j] = image[i, j] + tau * ((image[i + 1, j] - 2
                    * image[i, j] + image[i - 1, j]) / np.power(h1, 2)
                    + (image[i, j + 1] - 2 * image[i, j] + image[i, j
                       - 1]) / np.power(h2, 2))


# @ jit(nopython=True)
def get_diffusivity(gradX, gradY, lmbd, type):
    gradMag = np.sqrt(
        np.power(gradX, 2) + np.power(gradY, 2))
    if type == 'perona-malik':
        return 1 / (1 + (np.power(gradMag, 2) / np.power(lmbd, 2)))
    elif type == 'charbonnier':
        return 1 / np.square(1 + (np.power(gradMag, 2) / np.power(lmbd, 2)))
    elif type == 'exponential':
        return np.exp(-np.power(gradMag, 2) / 2 * np.power(lmbd, 2))
    else:
        raise ValueError(
            'invalid diffusivity type')


def slice_border(img, slice_val):
    return img[slice_val: - slice_val, slice_val: - slice_val]


# @jit(nopython=True)
def prepare_diagonals(Diag, UDiag, LDiag, data, data_i, n, tau, shift, sx):
    UDiag[0] = -tau * (data[get_coord(data_i, sx)] +
                       data[get_coord(data_i + shift, sx)])
    LDiag[0] = UDiag[0]
    Diag[0] = 1 - \
        UDiag[0]
    data_i += shift

    for i in range(1, n - 1):
        UDiag[i] = -tau * (data[get_coord(data_i, sx)] +
                           data[get_coord(data_i + shift, sx)])
        LDiag[i] = UDiag[i]
        Diag[i] = 1 - \
            UDiag[i] - \
            UDiag[i - 1]
        data_i += shift
    Diag[n - 1] = 1 - \
        UDiag[n - 2]


# @jit(nopython=True)
def add_and_avg(src, dst, dst_i, n, step, coef, sx):
    """
    Add values from one buffer to the other one and multiply
    them by a given coefficient.
    """
    for i in range(n):
        dst[get_coord(
            dst_i, sx)] += src[i]
        dst[get_coord(
            dst_i, sx)] *= coef
        dst_i += step


# @jit(nopython=True)
def aniso_diff_step_AOS(img, Da, Db, Dc, out, tau):
    # number of cols
    sx = img.shape[1]
    # number of rows
    sy = img.shape[0]
    n = max(sx, sy)
    r = np.zeros(n)
    Diag = np.zeros(
        n)
    UDiag = np.zeros(
        n - 1)
    LDiag = np.zeros(
        n - 1)
    solution = np.zeros(
        n)

    u = 0
    b = sx

    # create the image containing the right-hand side for the AOS scheme
    tmp_img = img.copy()
    tmp_i = 0

    # process first row
    tmp_img[0, :] = img[u,
                        :].copy()
    u += sx
    tmp_i += sx

    # process middle rows
    for y in range(1, sy - 1):
        # set left border same as input image
        tmp_img[get_coord(
            tmp_i, sx)] = img[get_coord(u, sx)]
        u += 1
        tmp_i += 1
        b += 1

        for x in range(1, sx - 1):
            tmp_img[get_coord(tmp_i, sx)] = img[get_coord(u, sx)] + \
                0.25 * tau * (
                img[get_coord(u - sx - 1, sx)] *
                (Db[get_coord(b - 1, sx)] + Db[get_coord(b - sx, sx)]) -
                img[get_coord(u - sx + 1, sx)] *
                (Db[get_coord(b + 1, sx)] + Db[get_coord(b - sx, sx)]) -
                img[get_coord(u + sx - 1, sx)] *
                (Db[get_coord(b - 1, sx)] + Db[get_coord(b + sx, sx)]) +
                img[get_coord(u + sx + 1, sx)] *
                (Db[get_coord(b + 1, sx)] + Db[get_coord(b + sx, sx)]))

            u += 1
            tmp_i += 1
            b += 1

        tmp_img[get_coord(
            tmp_i, sx)] = img[get_coord(u, sx)]

        u += 1
        tmp_i += 1
        b += 1

    # process last row
    tmp_img[sy - 1,
            :] = img[sy - 1, :].copy()

    # scan in the x axis
    a = 0
    tmp_i = 0
    out_i = 0

    for y in range(sy):
        # copy ith row
        r = tmp_img[tmp_i //
                    sx, :].copy()
        prepare_diagonals(
            Diag, UDiag, LDiag, Da, a, sx, tau, 1, sx)
        tridiagonal_matrix_solver(
            Diag, UDiag, LDiag, r, out[out_i // sx, :], sx)
        a += sx
        tmp_i += sx
        out_i += sx

    # scan in the y axis
    c = 0

    for x in range(sx):
        c = x
        tmp_i = x
        out_i = x

        # copy ith column
        r = tmp_img[:, tmp_i].copy(
        )
        prepare_diagonals(
            Diag, UDiag, LDiag, Dc, c, sy, tau, sx, sx)
        tridiagonal_matrix_solver(
            Diag, UDiag, LDiag, r, solution, sy)
        add_and_avg(solution, out,
                    out_i, sy, sx, 0.5, sx)


# @ jit(nopython=True)
def get_coord(n, sizeX):
    return (n // sizeX, n % sizeX)


# @ jit(nopython=True)
# pythran export tridiagonal_matrix_solver(float64[], float64[], float64[], float64[],float64[], int)
def tridiagonal_matrix_solver(diag_a, diag_b, diag_c, d, x, n):
    """
    Solve a tridiagonal system of linear equations using the Thomas algorithm.
    """
    for i in range(n - 1):
        diag_c[i] /= diag_a[i]
        diag_a[i + 1] -= diag_c[i] * \
            diag_b[i]

    for i in range(1, n):
        d[i] -= diag_c[i -
                       1] * d[i - 1]

    x[n - 1] = d[n -
                 1] / diag_a[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - diag_b[i] *
                x[i + 1]) / diag_a[i]
