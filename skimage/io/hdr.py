import numpy as np


def getRadiance(ims, exp, radianceMap):
    """
    Return the radiance for a series of images based upon a camera response 
    function

    Parameters
    ----------
    ims : list
          list of images in the for of numpy arrays. Either mono or colour
          (RGB, MxNx3)

    exp : numpy 1D array
          array of exposure times in seconds

    radiacenMap : numpy array
                  array mapping the counts to radiance

    Returns
    ----------
    hdr : numpy array
          resulting image with radiance values
    """

    den = np.ones(ims[0].shape)
    num = np.zeros(ims[0].shape)
    wf = np.vectorized(weightFunc)
    for idx, im in enumerate(ims):
        gij = im.copy()
        # For colour
        if im.ndim == 3:
            for ii in range(im.shape[2]):
                gij[:, :, ii] = radianceMap[im[:, :, ii] + 1, ii]
        else:
            gij[:, :] = radianceMap[im[:, :, ii] + 1]
        gij = gij - np.log(exp[idx])

        wij = wf(zij)
        num = num + wij * gij
        den = den + wij

    return num / den


def makeHdr(ims, exp, radianceMap, depth=16):
    """
    Compute the HDR image from a series of images and a racianceMap
    Based on:
    Debevec, P. E., & Malik, J. (1997). SIGGRAPH 97 Conf. Proc., August, 3–8.
    doi:10.1145/258734.258884

    Parameters
    ----------
    ims : list
          list of images in the for of numpy arrays. Either grayscale or colour
          (RGB)

    exp : numpy 1D array
          array of exposure times in seconds

    radianceMap : numpy array
               array (idx) mapping counts to radiance value, if input is RGB 
               this must be Nx3 

    depth : int, optional
            pixel depth, default=16

    Returns
    ----------
    hdr : numpy array
          The HDR image either grayscale or RGB depending on input in ln(E)
    """
    B = np.log(np.array(exp))

    if ims[0].ndim == 3:
        sx, sy, sc = np.shape(ims[0])
        hdr = np.zeros([sx, sy, sc], dtype=np.float)
        gray = False
    else:
        sx, sy = np.shape(ims[0])
        hdr = np.zeros([sx, sy], dtype=np.float)
        gray = True

    ims = np.asarray(ims, dtype=np.int)

    sx, sy, sz = np.shape(ims[0])

    w = np.vectorize(weightFunc)

    for ii in range(sx):
        for jj in range(sy):
            if gray:
                zij = ims[:, ii, jj]
                g = radianceMap[zij]
                W = w(zij, depth)
                hdr[ii, jj] = np.sum(W * (g - B)) / np.sum(W)
            else:
                for cc in range(sc):
                    zij = ims[:, ii, jj, cc]
                    g = radianceMap[zij, cc]
                    W = w(zij, depth)
                    hdr[ii, jj, cc] = np.sum(W * (g - B)) / np.sum(W)

    return hdr


def getCRF(ims, exp, depth=16, l=200, depth_max=10):
    """
    Compute the camera response function from a set of images and exposures.
    Based on:
    Debevec, P. E., & Malik, J. (1997). SIGGRAPH 97 Conf. Proc., August, 3–8.
    doi:10.1145/258734.258884


    Parameters
    ----------
    ims : list
          list of images in the for of numpy arrays. Either grayscale or colour
          (RGB)

    exp : numpy 1D array
          array of exposure times in seconds

    depth : int, optional
            pixel depth, default=16

    l : int, optional
        Smoothness parameter, default 200, increase for noisy images
        Can help to increase this for better smoothness in large bit depths 
        (depth_max > 10) 

    depth_max : int, optional
              Depth used for the SVC is reduced to this if depth is larger.
              Used to reduce the size of the matrix solved by the SVC for 
              images with more than 8 bits per colour channel.
              Note that the scaling of memory requirements and computational 
              time with this parameter is highly non-linear.
              The resulting radiance is interpolated up to depth before being 
              returned
              default = 10

    Returns
    ----------  
    radianceMap : numpy array
               array (idx) mapping counts to radiance value, if input is RGB 
               this is 2**depth x 3
               Order of colours is same as input
    """

    # Calculate number of samples from image neccesarry for an overdetermined
    # system (assuming Z_min = 0) using the two times the minimum requirement
    # in the article

    if depth > depth_max:
        div = depth - depth_max
    else:
        div = 0

    samples = 2 * (2**(depth - div)) // (len(ims) - 1)

    # Find if it is grayscale or colour
    colour = (ims[0].ndim == 3)

    # Compute the camera response function
    randIdx = np.floor(np.random.randn(samples) * 2**depth).astype(np.int)
    B = np.log(np.array(exp))

    if colour:
        radianceMap = np.zeros([2**depth, 3])

        Z = np.zeros([randIdx.size, len(ims)])

        for ii in range(3):

            for jj in range(len(ims)):
                Z[:, jj] = ims[jj][:, :, ii].flatten()[randIdx]
            radianceMap[:, ii], LE = gsolve(Z, B, l, depth, depth_max)

    else:
        for jj in range(len(ims)):
            Z[:, jj] = ims[jj][:, :, ii].flatten()[randIdx]
        radianceMap, LE = gsolve(Z, B, l, depth, depth_max)

    return radianceMap


def gsolve(Z, B, l, depth=16, depth_max=12):
    """
    Solves for the camera response function, based upon the code in:
    Debevec, P. E., & Malik, J. (1997). SIGGRAPH 97 Conf. Proc., August, 3–8.
    doi:10.1145/258734.258884


    Parameters
    ----------
    Z : numpy array
        2D array (i,j) with pixel i in image j

    B : numpy array
        the ln of the shutter speed for image j

    l : int
        lambda determines the amount of smoothness

    depth : int, optional
            pixel depth, default=16

    depth_max : int, optional
              Depth used for the SVC is reduced to this if depth is larger than
              this value.
              g is interpolated to depth if this is smaller than depth
              Used to reduce the size of the matrix solved by the SVC.
              default = 12

    Returns
    ----------
    g : numpy array 
        ln exposure corresponding to pixel value z 

    LE : numpy array 
         ln film irradiance at pixel location i
    """
    # Reduce the bit depth to preserve memory and computational time
    if depth > depth_max:
        div = depth - depth_max
    else:
        div = 0
    n = 2**(depth - div)
    Z = Z / (2**div)

    A = np.zeros([Z.size + n + 1, n + Z.shape[0]])
    b = np.zeros(A.shape[0])
    k = 0
    for ii in range(Z.shape[0]):
        for jj in range(Z.shape[1]):
            wij = weightFunc(Z[ii, jj] + 1, depth - div)
            A[k, Z[ii, jj] + 1] = wij
            A[k, n + ii] = -wij
            b[k] = wij * B[jj]
            k += 1

    #  Fix the curve by setting its middle value to 0 = ln(1)
    A[k, n / 2] = 1
    k += 1

    for ii in range(n - 2):
        A[k, ii] = l * weightFunc(ii + 1, depth - div)
        A[k, ii + 1] = -2 * l * weightFunc(ii + 1, depth - div)
        A[k, ii + 2] = l * weightFunc(ii + 1, depth - div)
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


def weightFunc(I, depth=16):
    """
    Weight function for the gsolve, based on:
    Debevec, P. E., & Malik, J. (1997). SIGGRAPH 97 Conf. Proc., August, 3–8.
    doi:10.1145/258734.258884

    Parameters
    ----------
    I : int
        intensity

    depth : int, optional
            pixel depth, default=16

    Returns
    ----------
    w : int
        weight for given intensity
    """

    if I <= (2**depth / 2 + 1):
        return I
    else:
        return (2**depth - 1) - I
