import numpy as np
import warnings


def make_hdr(images, exposure, radiance_map, depth=16, channel_axis=None):
    """
    Compute the HDR image from a series of images with a given radiance
    mapping.

    Parameters
    ----------
    images: list of numpy arrays, or numpy ndarray or ImageCollection
        List of images in the form of numpy arrays, or a numpy array with the
        first dimension being the different images.
        Either greyscale or colour (RGB). Can't be float images
        List of images as numpy arrays. Either greyscale or colour
        (RGB).
        Can also be an ImageCollection
    exposure : numpy 1D array
        Array of exposure times in seconds.
        Images some times have these in the exif information.
    radiance_map : numpy array
        Array (idx) mapping counts to radiance value, if input is RGB this must
        be Nx3. See `get_crf`.
    depth : int, optional
        Pixel depth.
    channel_axis : int or None, optional
        If None, the images are assumed to be greyscale images.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels. This includes the list index, meaning the images
        with "normal" axis (X, Y, C) should set `channel_axis=3`

    Returns
    -------
    hdr : numpy array
        The HDR image, either greyscale or RGB depending on input, in ln(E).

    References
    ----------
    .. [1] Debevec and Malik, J. "Recovering high dynamic range radiance maps
       from photographs" (1997). DOI:10.1145/258734.258884

    .. [2] https://en.wikipedia.org/wiki/Radiance
    """
    # Check that we are not dealing with float images
    if np.issubdtype(images[0].dtype, float):
        raise ValueError("""Images can't be float images, they need to be
converted to int before using this method.""")

    # Calculating the logarithm of the exposure
    B = np.log(exposure)

    # Making sure the images are uint64
    images = np.asarray(images, dtype=np.uint64)

    # Initialization for RBG or greyscale images
    if channel_axis is None:
        # We have greyscale images
        srow, scol = np.shape(images[0])
        hdr = np.zeros([srow, scol], dtype=np.float64)
        grey = True
    elif images.ndim == 4:
        # We have a colour image
        # No check that the last index is colour:
        if channel_axis != 3:
            # It is not, we need to create one that has it in the correct place
            images = np.moveaxis(images, channel_axis, 3)

        srow, scol, sch = np.shape(images[0])
        if sch > 3:
            warnings.warn("The specified colour channel has a length of "
                          + str(sch) + """ which is greater than the expected 3
(RGB).""")
        hdr = np.zeros([srow, scol, sch], dtype=np.float64)
        grey = False
    elif images.ndim > 4:
        raise ValueError("""Individual images have more than 3 dimensions,
which is not supported""")

    # Calculating the weight
    w = _weight_func_arr(images, depth)

    # Initializing variables for the numerator and denominator
    num = np.zeros_like(hdr)
    den = np.zeros_like(hdr)

    if grey:
        # Looping over the images and computing the camera response
        # function for each of them.
        for kk in range(images.shape[0]):
            g = np.reshape(
                radiance_map[images[kk, :, :].flatten()], [srow, scol])
            num[:, :] += w[kk, :, :] * (g - B[kk])
            den[:, :] += w[kk, :, :]
        print(np.min(den))
        hdr = num / den
    else:
        # Looping over the colours
        for cc in range(sch):
            # Looping over the images and computing the camera response
            # function for each of them.
            for kk in range(images.shape[0]):
                g = np.reshape(
                    radiance_map[images[kk, :, :, cc].flatten(), cc],
                    [srow, scol])
                num[:, :, cc] += w[kk, :, :, cc] * (g - B[kk])
                den[:, :, cc] += w[kk, :, :, cc]
        # Calculating the HDR image
        print(np.min(den))
        hdr = num / den

    return np.exp(hdr)


def get_crf(images, exposure, depth=16, lambd=200, depth_max=10,
            channel_axis=None):
    """
    Compute the camera response function from a set of images and exposures.

    Parameters
    ----------
    images: list of numpy arrays or numpy ndarray or ImageCollection
        List of images in the form of numpy arrays, or a numpy array with the
        first dimension being the different images.
        Either greyscale or colour (RGB). Can't be float images
    exposure: numpy 1D array
        Array of exposure times in seconds.
    depth : int, optional
        Pixel depth.
    lambd : int, optional
        Smoothness parameter, default 200, increase for noiscol images.
        Can help to increase this for better smoothness in large bit depths
        (depth_max > 10).
    depth_max : int, optional
        Depth used for the SVC, depth will be reduced to depth_max if larger.
        Used to reduce the size of the matrix solved by the SVC for
        images with more than 8 bits per colour channel.
        Note that the scaling of memory requirements and computational
        time with this parameter is highly non-linear.
        The resulting radiance is interpolated up to depth before being
        returned.
    channel_axis : int or None, optional
        If None, the images are assumed to be greyscale images.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels. This includes the list index, meaning the images
        with "normal" axis (X, Y, C) should set `channel_axis=3`

    Returns
    -------
    radiance_map : numpy array
        Array (idx) mapping counts to radiance value, if input is RGB
        this is 2**depth x 3.
        Order of colours is same as input.

    References
    ----------
    .. [1] Debevec and Malik, J. "Recovering high dynamic range radiance maps
       from photographs" (1997). DOI:10.1145/258734.258884
    """

    # Check that we are not dealing with float images
    if np.issubdtype(images[0].dtype, float):
        raise ValueError("""Images can't be float images, they need to be
converted to int before using this method.""")

    # Making sure the images are uint64 and an array
    images = np.asanyarray(images, dtype=np.uint64)
    print(images.shape)

    # Determine which type of images we are working with
    if channel_axis is None:
        # We have greyscale images
        si, srow, scol = images.shape
        grey = True
    elif images.ndim == 4:
        # We have a colour image
        # No check that the last index is colour:
        if channel_axis != 3:
            # It is not, we need to create one that has it in the correct place
            images = np.moveaxis(images, channel_axis, 3)

        si, srow, scol, sch = images.shape
        if sch > 3:
            warnings.warn("The specified colour channel has a length of "
                          + str(sch) + """ which is greater than the expected 3
(RGB).""")
        grey = False
    elif images.ndim > 4:
        raise ValueError("""Individual images have more than 3 dimensions,
which is not supported""")

    # Calculate number of samples from image necessary for an overdetermined
    # scolstem (assuming Z_min = 0). We are using four times the minimum
    # requirement in the article

    if depth > depth_max:
        div = depth - depth_max
    else:
        div = 0

    samples = np.int32(4 * (2**(depth - div)) / (si - 1))
    # Compute the camera response function
    rng = np.random.default_rng()

    rand_idx = rng.choice(srow*scol, samples)
    B = np.log(np.array(exposure))
    print(B)

    Z = np.zeros([len(rand_idx), si])

    if grey:
        # Working with a greyscale image
        for jj in range(si):
            Z[:, jj] = images[jj, :, :].flatten()[rand_idx]
        radiance_map, LE = _gsolve(Z, B, lambd, depth, depth_max)
    else:
        # Working with a colour image
        radiance_map = np.zeros([2**depth, sch])
        # Looping over the colours
        for ii in range(sch):
            # Looping over the images
            for jj in range(si):
                Z[:, jj] = images[jj, :, :, ii].flatten()[rand_idx]
            print(Z.shape)
            radiance_map[:, ii], LE = _gsolve(Z, B, lambd, depth, depth_max)

    return radiance_map


def _gsolve(Z, B, lambd, depth=16, depth_max=12):
    """
    Solves for the camera response function.

    Parameters
    ----------
    Z : numpy array
        2D array (i,j) with pixel i in image j.
    B : numpy array
        The ln of the shutter speed for image j.
    lambd : int
        lambd determines the amount of smoothness.
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
    .. [1] Debevec and Malik, J. "Recovering high dynamic range radiance maps
       from photographs" (1997). DOI:10.1145/258734.258884
    """
    # Reduce the bit depth to preserve memory and computational time
    if depth > depth_max:
        div = depth - depth_max
    else:
        div = 0
    n = 2**(depth - div)
    Z = np.array(Z / (2**div), dtype=np.int64)  # Make sure it stays int

    print(B)
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
    A[k, np.int32(n / 2)] = 1
    k += 1

    for ii in range(n - 2):
        A[k, ii] = lambd * _weight_func(ii + 1, depth - div)
        A[k, ii + 1] = -2 * lambd * _weight_func(ii + 1, depth - div)
        A[k, ii + 2] = lambd * _weight_func(ii + 1, depth - div)
        k += 1

    # Solve the equations with SVD
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
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
    Iout[intensity > (2**depth / 2)] = (2**depth - 1) - \
        intensity[intensity > (2**depth / 2)]
    return Iout
