import numpy as np
from scipy import sqrt, pi, arctan2, cos, sin, exp
from scipy.ndimage import gaussian_filter
from scipy.special import iv


def daisy(img, step=4, radius=15, rings=3, histograms=8, orientations=8,
          normalization='l1', sigmas=None, ring_radii=None):
    '''Extract DAISY feature descriptors densely for the given image.

    DAISY is a feature descriptor similar to SIFT formulated in a way that
    allows for fast dense extraction. Typically, this is practical for
    bag-of-features image representations.

    The implementation follows Tola et al. [1] but deviate on the following
    points:
        * Histogram bin contribution are smoothed with a circular Gaussian
          window over the tonal range (the angular range).
        * The sigma values of the spatial Gaussian smoothing in this code do
          not match the sigma values in the original code by Tola et al. [2].
          In their code, spatial smoothing is applied to both the input image
          and the center histogram. However, this smoothing is not documented
          in [1] and, therefore, it is omitted.

    Parameters
    ----------
    img : (M, N) array
        Input image (greyscale).
    step : int, optional
        Distance between descriptor sampling points.
    radius : int, optional
        Radius (in pixels) of the outermost ring.
    rings : int, optional
        Number of rings.
    histograms  : int, optional
        Number of histograms sampled per ring.
    orientations : int, optional
        Number of orientations (bins) per histogram.
    normalization : {'l1', 'l2', 'daisy', 'off'}, optional
        How to normalize the descriptors:
            * 'l1': L1-normalization of each descriptor.
            * 'l2': L2-normalization of each descriptor.
            * 'daisy': L2-normalization of individual histograms.
            * 'off': Disable normalization.
    sigmas : 1D array of float, optional
        Standard deviation of spatial Gaussian smoothing for the center
        histogram and for each ring of histograms. The array of sigmas should
        be sorted from the center and out. I.e. the first sigma value defines
        the spatial smoothing of the center histogram and the last sigma value
        defines the spatial smoothing of the outermost ring. Specifying sigmas
        overrides the following parameter:
            rings = len(sigmas)-1
    ring_radii : 1D array of int, optional
        Radius (in pixels) for each ring. Specifying ring_radii overrides the
        following two parameters:
            rings = len(ring_radii)
            radius = ring_radii[-1]
        If both sigmas and ring_radii are given, they must satisfy
            len(ring_radii) == len(sigmas)+1
        since no radius is needed for the center histogram.

    Returns
    -------
    descs : array
        Grid of DAISY descriptors for the given image as an array
        dimensionality  (P, Q, R) where
            P = ceil((M-radius*2)/step)
            Q = ceil((N-radius*2)/step)
            R = (rings*histograms + 1)*orientations

    References
    ----------
    [1]: Tola et al. "Daisy: An efficient dense descriptor applied to
         wide-baseline stereo." Pattern Analysis and Machine Intelligence,
         IEEE Transactions on 32.5 (2010): 815-830.
    [2]: http://cvlab.epfl.ch/alumni/tola/daisy.html
    '''

    # Validate image format.
    if img.ndim > 2:
        raise ValueError('Only grey-level images are supported.')
    if img.dtype.kind == 'u':
        img = img.astype(float)
        img = img / 255.

    # Validate parameters.
    if sigmas is not None and ring_radii is not None \
            and len(sigmas) - 1 != len(ring_radii):
        raise ValueError('len(sigmas)-1 != len(ring_radii)')
    if ring_radii is not None:
        rings = len(ring_radii)
        radius = ring_radii[-1]
    if sigmas is not None:
        rings = len(sigmas) - 1
    if sigmas is None:
        sigmas = [radius * (i + 1) / float(2 * rings) for i in range(rings)]
    if ring_radii is None:
        ring_radii = [radius * (i + 1) / float(rings) for i in range(rings)]
    if normalization not in ['l1', 'l2', 'daisy', 'off']:
        raise ValueError('Invalid normalization method.')

    # Compute image derivatives.
    dx = np.zeros(img.shape)
    dy = np.zeros(img.shape)
    dx[:, :-1] = np.diff(img, n=1, axis=1)
    dy[:-1, :] = np.diff(img, n=1, axis=0)

    # Compute gradient orientation and magnitude and their contribution
    # to the histograms.
    grad_mag = sqrt(dx ** 2 + dy ** 2)
    grad_ori = arctan2(dy, dx)
    hist_sigma = pi / orientations
    kappa = 1. / hist_sigma
    bessel = iv(0, kappa)
    hist = np.empty((orientations,) + img.shape, dtype=float)
    for i in range(orientations):
        mu = 2 * i * pi / orientations - pi
        # Weigh bin contribution by the circular normal distribution
        hist[i, :, :] = exp(kappa * cos(grad_ori - mu)) / (2 * pi * bessel)
        # Weigh bin contribution by the gradient magnitude
        hist[i, :, :] = np.multiply(hist[i, :, :], grad_mag)

    # Smooth orientation histograms for the center and all rings.
    sigmas = [sigmas[0]] + sigmas
    hist_smooth = np.empty((rings + 1,) + hist.shape, dtype=float)
    for i in range(rings + 1):
        for j in range(orientations):
            hist_smooth[i, j, :, :] = gaussian_filter(hist[j, :, :],
                                                      sigma=sigmas[i])

    # Assemble descriptor grid.
    theta = [2 * pi * j / histograms for j in range(histograms)]
    desc_dims = (rings * histograms + 1) * orientations
    descs = np.empty((desc_dims, img.shape[0] - 2 * radius,
                      img.shape[1] - 2 * radius))
    descs[:orientations, :, :] = hist_smooth[0, :, radius:-radius,
                                             radius:-radius]
    idx = orientations
    for i in range(rings):
        for j in range(histograms):
            y_min = radius + int(round(ring_radii[i] * sin(theta[j])))
            y_max = descs.shape[1] + y_min
            x_min = radius + int(round(ring_radii[i] * cos(theta[j])))
            x_max = descs.shape[2] + x_min
            descs[idx:idx + orientations, :, :] = hist_smooth[i + 1, :,
                                                              y_min:y_max,
                                                              x_min:x_max]
            idx += orientations
    descs = descs[:, ::step, ::step]
    descs = descs.swapaxes(0, 1).swapaxes(1, 2)

    # Normalize descriptors.
    if normalization != 'off':
        descs += 1e-10
        if normalization == 'l1':
            descs /= np.sum(descs, axis=2)[:, :, np.newaxis]
        elif normalization == 'l2':
            descs /= sqrt(np.sum(descs ** 2, axis=2))[:, :, np.newaxis]
        elif normalization == 'daisy':
            for i in range(0, desc_dims, orientations):
                norms = sqrt(np.sum(descs[:, :, i:i + orientations] ** 2,
                                    axis=2))
                descs[:, :, i:i + orientations] /= norms[:, :, np.newaxis]

    return descs
