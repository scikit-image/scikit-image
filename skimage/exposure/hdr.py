import numpy as np


def get_radiance(images, exposure, radiance_map):
    """
    Return the radiance for a series of images based upon a camera response
    function.

    Parameters
    ----------
    images: list
        List of images in the for of numpy arrays. Either mono or color
        (RGB, MxNx3).
    exposure : numpy 1D array
        Array of exposure times in seconds.
    radiacenMap : numpy array
        Array mapping the counts to radiance

    Returns
    -------
    hdr : numpy array
        Resulting image with radiance values.
    """

    den = np.ones(images[0].shape)
    num = np.zeros(images[0].shape)
    wf = np.vectorized(_weight_func)
    for idx, im in enumerate(images):
        gij = im.copy()
        # For colour
        if im.ndim == 3:
            for ii in range(im.shape[2]):
                gij[:, :, ii] = radiance_map[im[:, :, ii] + 1, ii]
        else:
            gij[:, :] = radiance_map[im[:, :, ii] + 1]
        gij = gij - np.log(exposure[idx])

        wij = wf(gij)
        num = num + wij * gij
        den = den + wij

    return num / den


def make_hdr(images, exposure, radiance_map, depth=16):
    """
    Compute the HDR image from a series of images and a racianceMap.

    Parameters
    ----------
    images: list
        List of images in the for of numpy arrays. Either grayscale or color
        (RGB).
    exposure : numpy 1D array
        Array of exposure times in seconds.
    radiance_map : numpy array
        Array (idx) mapping counts to radiance value, if input is RGB this must
        be Nx3.
    depth : int, optional
        Pixel depth.

    Returns
    -------
    hdr : numpy array
        The HDR image either grayscale or RGB depending on input in ln(E).

    References
    ----------
    .. [1]  Debevec, P. E., & Malik, J. (1997). SIGGRAPH 97 Conf. Proc.,
            August, 3-8. DOI:10.1145/258734.258884
    """
    B = np.log(np.array(exposure))

    if images[0].ndim == 3:
        sx, sy, sc = np.shape(images[0])
        hdr = np.zeros([sx, sy, sc], dtype=np.float)
        gray = False
    else:
        sx, sy = np.shape(images[0])
        hdr = np.zeros([sx, sy], dtype=np.float)
        gray = True

    images = np.asarray(images, dtype=np.uint64)

    sx, sy, sz = np.shape(images[0])
    w = _weight_func_arr(images, depth)

    if gray:
        num = np.zeros_like(hdr)
        den = np.zeros_like(hdr)

        for kk in range(images.shape[0]):
            g = np.reshape(
                radiance_map[images[kk, :, :].flatten()], [sx, sy])
            num[:, :] += w[kk, :, :] * (g - B[kk])
            den[:, :] += w[kk, :, :]
        hdr = num / den
    else:
        num = np.zeros_like(hdr)
        den = np.zeros_like(hdr)
        for cc in range(sc):
            for kk in range(images.shape[0]):
                g = np.reshape(
                    radiance_map[images[kk, :, :, cc].flatten(), cc], [sx, sy])
                num[:, :,
                    cc] += w[kk, :, :, cc] * (g - B[kk])
                den[:, :, cc] += w[kk, :, :, cc]
        hdr = num / den

    return np.exp(hdr)


def get_crf(images, exposure, depth=16, l=200, depth_max=10):
    """
    Compute the camera response function from a set of images and exposures.

    Parameters
    ----------
    images: list
        List of images in the for of numpy arrays. Either grayscale or color
        (RGB).
    exposure: numpy 1D array
        Array of exposure times in seconds.
    depth : int, optional
        Pixel depth.
    l : int, optional
        Smoothness parameter, default 200, increase for noisy images.
        Can help to increase this for better smoothness in large bit depths
        (depth_max > 10).
    depth_max : int, optional
        Depth used for the SVC is reduced to this if depth is larger.
        Used to reduce the size of the matrix solved by the SVC for
        images with more than 8 bits per colour channel.
        Note that the scaling of memory requirements and computational
        time with this parameter is highly non-linear.
        The resulting radiance is interpolated up to depth before being
        returned.

    Returns
    -------
    radiance_map : numpy array
        Array (idx) mapping counts to radiance value, if input is RGB
        this is 2**depth x 3.
        Order of colours is same as input.

    References
    ----------
    .. [1]  Debevec, P. E., & Malik, J. (1997). SIGGRAPH 97 Conf. Proc.,
            August, 3-8. DOI:10.1145/258734.258884
    """

    # Calculate number of samples from image necessary for an overdetermined
    # system (assuming Z_min = 0) using the four times the minimum requirement
    # in the article

    if depth > depth_max:
        div = depth - depth_max
    else:
        div = 0

    samples = int(4 * (2**(depth - div)) / (len(images) - 1))

    # Find if it is grayscale or colour
    colour = (images[0].ndim == 3)

    # Compute the camera response function
    rand_idx = np.floor(np.random.randn(samples) * 2**depth).astype(np.int)
    B = np.log(np.array(exposure))

    if colour:
        radiance_map = np.zeros([2**depth, 3])

        Z = np.zeros([rand_idx.size, len(images)])

        for ii in range(3):

            for jj in range(len(images)):
                Z[:, jj] = images[jj][:, :, ii].flatten()[rand_idx]
            radiance_map[:, ii], LE = _gsolve(Z, B, l, depth, depth_max)

    else:
        for jj in range(len(images)):
            Z[:, jj] = images[jj][:, :, ii].flatten()[rand_idx]
        radiance_map, LE = _gsolve(Z, B, l, depth, depth_max)

    return radiance_map


def _gsolve(Z, B, l, depth=16, depth_max=12):
    """
    Solves for the camera response function.

    Parameters
    ----------
    Z : numpy array
        2D array (i,j) with pixel i in image j.
    B : numpy array
        The ln of the shutter speed for image j.
    l : int
        l determines the amount of smoothness.
    depth : int, optional
        Pixel depth, default=16
    depth_max : int, optional
        Depth used for the SVC is reduced to this if depth is larger than
        this value.
        g is interpolated to depth if this is smaller than depth
        Used to reduce the size of the matrix solved by the SVC.

    Returns
    -------
    g : numpy array
        ln exposure corresponding to pixel value z.
    LE : numpy array
        ln film irradiance at pixel location i.

    References
    ----------
    .. [1]  Debevec, P. E., & Malik, J. (1997). SIGGRAPH 97 Conf. Proc.,
            August, 3-8. DOI:10.1145/258734.258884
    """
    # Reduce the bit depth to preserve memory and computational time
    if depth > depth_max:
        div = depth - depth_max
    else:
        div = 0
    n = 2**(depth - div)
    Z = np.array(Z / (2**div), dtype=np.int64)  # Make sure it stays int

    A = np.zeros([Z.size + n + 1, n + Z.shape[0]])
    b = np.zeros(A.shape[0])
    k = 0
    for ii in range(Z.shape[0]):
        for jj in range(Z.shape[1]):
            wij = _weight_func(Z[ii, jj] + 1, depth - div)
            A[k, Z[ii, jj] + 1] = wij
            A[k, n + ii] = -wij
            b[k] = wij * B[jj]
            k += 1

    #  Fix the curve by setting its middle value to 0 = ln(1)
    A[k, int(n / 2)] = 1
    k += 1

    for ii in range(n - 2):
        A[k, ii] = l * _weight_func(ii + 1, depth - div)
        A[k, ii + 1] = -2 * l * _weight_func(ii + 1, depth - div)
        A[k, ii + 2] = l * _weight_func(ii + 1, depth - div)
        k += 1

    # Solve the equations with SVD
    x, residuals, rank, s = np.linalg.lstsq(A, b)
    g = x[:n]
    LE = x[n::]

    # Interpolate the result if depth is larger than depth_max
    if div != 0:
        g = np.interp(
            np.arange(0, 2**depth), np.arange(0, 2**depth, 2**(div)),
            g)

    return g, LE


def _weight_func(intensity, depth=16):
    """
    Weight function.

    Parameters
    ----------
    intensity : int
        Intensity.
    depth : int, optional
        Pixel depth.

    Returns
    -------
    w : int
        Weight for given intensity.
    """

    # This assumes Z_min = 0
    if intensity <= (2**depth / 2):
        return intensity
    else:
        return (2**depth - 1) - intensity


def _weight_func_arr(intensity, depth=16):
    """
    Weight function for arrays.

    Parameters
    ----------
    intensity : array
        Intensities
    depth : int, optional
        Pixel depth, default=16

    Returns
    -------
    w : int
        Weight for given intensity.
    """

    # This assumes Z_min = 0
    Iout = intensity.copy()
    Iout[intensity > (2**depth / 2)] = (2**depth - 1) - intensity[intensity > (2**depth / 2)]
    return Iout
