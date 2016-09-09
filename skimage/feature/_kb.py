import numpy as np
from scipy import ndimage as ndi
from ..util import img_as_ubyte
from .._shared.utils import assert_nD


def kb(image, min_scale=10, max_scale=25, saliency_threshold=0.6, clustering_threshold=7):
    """Finds salient regions in the given grayscale image.For each point x
    the method picks a scale s and calculates salient score Y(x,s).
    By comparing Y(x,s) of different points x the detector can rank
    the saliency of points and pick the most representative ones.

    Parameters
    ----------
    image : ndarray
        Input grayscale image
    min_scale : int, optional
        Minimum scale for the keypoints detected.
    max_scale : int, optional
        Maximum scale for the keypoints detected.
    saliency_threshold : float, optional.
        Features with saliency score satisfying threshold will be considered.
    clustering_threshold : int, optional
        Variance among the regions should be smaller than this threshold
        for sufficient clustering.

    Returns
    -------
    A : (n, 3) ndarray
        A 2d array with each row representing 3 values, '(y, x, scale)'
        where '(y,x)' are coordinates of the region and 'scale' is the
        size of corresponding salient region.

    References
    ----------
    https://en.wikipedia.org/wiki/Kadir–Brady_saliency_detector

    Examples
    --------
    >>> from skimage.feature import kb
    >>> from skimage.color import rgb2gray
    >>> from skimage.data import astronaut
    >>> image = astronaut()[100:300, 100:300]
    >>> kb(rgb2gray(image))

    Notes
    -----
    The radius of each region is 'scale/2'.
    """

    assert_nD(image, 2)

    # scales for keypoints
    scales = np.array([_ for _ in range(min_scale, max_scale, 2)])
    # detect keypoints in scale-space
    base_regions = detect(image, scales)
    # pruning based on thresholds
    regions = prune(base_regions, saliency_threshold, clustering_threshold, K=7)

    return regions.T


def detect(image, scales):
    """ Calculates the Shannon entropy of local image attributes for each region
    over a range of scales, and weight them based on the difference between
    the local descriptors around scale-space.

    Parameters
    ----------
    image : ndarray
        Input grayscale image
    scales : ndarray

    Returns
    -------
    A : (4, n) ndarray
        A 2d array with each column representing 4 values, '(gamma, scale, y, x)'
        where '(y, x)' are coordinates of the region, 'scale' is the
        size of salient region, and 'gamma' is the saliency score.
    """
    image = img_as_ubyte(image)
    nr, nc = image.shape
    # find pixels that we are going to examine
    mask = np.ones((nr, nc))
    r, c = np.nonzero(mask)
    nPix = len(r)

    nScales = len(scales)
    intensity_edges = [_ for _ in range(0, 257, 16)]

    previous_h = np.zeros((len(intensity_edges)-1, nPix))
    weights = np.zeros((nScales, nPix))
    entropy = np.zeros((nScales, nPix))

    # iterate through every possible region
    # iterate through scales
    for s_count in range(nScales):
        scale_size = scales[s_count]
        radius = int((scale_size)/2)

        # iterate through pixels
        for i in range(nPix):
            min_r, max_r = r[i]-radius, r[i]+radius
            min_c, max_c = c[i]-radius, c[i]+radius

            if min_r < 0:
                min_r = 0
            if max_r > nr:
                max_r = nr
            if min_c < 0:
                min_c = 0
            if max_c > nc:
                max_c = nc

            # compute the histogram of intensity values in this region
            patch = image[min_r:max_r, min_c:max_c]
            h = np.histogram(patch, bins=intensity_edges)[0]
            h = np.array([_/sum(h) for _ in h])

            # index of histogram values greater than zero
            idx = np.nonzero(h > 0)
            entropy[s_count, i] = -sum(h[idx]*np.log(h[idx]))

            if s_count >= 1:
                # first derivative in entropy space to calculate weights
                dif = abs(h - previous_h[:, i])
                factor = scales[s_count]**2/(2*scales[s_count]-1)
                weights[s_count, i] = factor * sum(dif)
                if s_count == 1:
                    weights[s_count-1, i] = weights[s_count, i]
            previous_h[:, i] = h

    # find local maxima in scale space by selecting second derivatives that are less than zero
    # second derivative kernel
    fxx = np.array([1, -2, 1])
    # evaluate second derivative using the kernel
    d_entropy = np.transpose(ndi.correlate1d(entropy.T, fxx, mode='nearest'))
    d_entropy[0, :] = 0
    int_pts = np.nonzero(d_entropy < 0)

    # weighting interest points by scale size times the first derivative of the scale-space function
    gamma = entropy[int_pts] * weights[int_pts]
    scale = scales[int_pts[0]]
    row = r[int_pts[1]]
    column = c[int_pts[1]]

    return np.array([gamma, scale, row, column])


def prune(candidate_regions, saliency_threshold, v_th, K=7):
    """ It selects highly salient points that have local support
    i.e. nearby points with similar saliency and scale. Each region is
    sufficiently distant from all others (in Scale-space regions) to
    qualify as a separate entity.

    Parameters
    ----------
    candidate_regions : (4, n) ndarray
        A 2d array with each column representing gamma,scale,y,x.
    saliency_threshold : float, optional.
        Features with saliency score more than this will be considered.
    v_th : int, optional
        Variance among the regions should be smaller than this threshold
        for sufficient clustering.

    Returns
    -------
<<<<<<< HEAD
    A : (3, n) ndarray
        A 2d array with each column representing 3 values, '(y, x, scale)'
        where '(y,x)' are coordinates of the region and 'scale' is the
        size of corresponding salient region.
=======
    A : (4, n) ndarray
        A 2d array with each column representing 4 values, '(gamma, scale, y, x)'
        where '(y, x)' are coordinates of the region, 'scale' is the
        size of salient region, and 'gamma' is the saliency score.
>>>>>>> 365d9ad9d47663ef095acc601cb7ec14577f972d
    """

    gamma, scale, row, column = (np.array([]) for i in range(4))
    cgamma, cscale, cr, cc = candidate_regions

    # apply a global threshold to the gamma values
    threshold = saliency_threshold * max(cgamma)
    t = np.nonzero(cgamma > threshold)
    t_gamma, t_scale, t_r, t_c = cgamma[t], cscale[t], cr[t], cc[t]

    # sort the gamma values and order the rest by that
    s_i = np.argsort(t_gamma)[::-1]
    t_gamma, t_scale, t_r, t_c = t_gamma[s_i], t_scale[s_i], t_r[s_i], t_c[s_i]

    # create a Distance matrix
    n = max(t_gamma.shape)
    D = np.zeros((n, n))
    pts = np.array(list(zip(t_c, t_r, t_scale)))

    # fill it with distances
    for i in range(n):
        pt = pts[i][:]
        # calculate the distances b/w regions
        dists = np.sqrt(((pts-np.tile(pt, (pts.shape[0], 1)))**2).sum(axis=1))
        D[i, :], D[:, i] = dists.T, dists

    nReg = 0
    # clusters matrix
    cluster = np.zeros((3, K+1))

    # pruning process
    for index in range(n):
        cluster[0, 0] = t_c[index]
        cluster[1, 0] = t_r[index]
        cluster[2, 0] = t_scale[index]
        s_i = np.argsort(D[index, :])
        # fill in the neighbouring regions
        for j in range(K):
            cluster[0, j+1] = t_c[s_i[j+1]]
            cluster[1, j+1] = t_r[s_i[j+1]]
            cluster[2, j+1] = t_scale[s_i[j+1]]

        # clusters center point
        center = np.array([np.mean(cluster, axis=1)])

        # check if the regions are "suffiently clustered", if variance is less than threshold
        v = np.var(np.sqrt(((cluster - np.tile(center.T, (1, K+1)))**2).sum(axis=0)))
        if v > v_th:
            continue

        center = np.mean(cluster, axis=1)
        if nReg > 0:
            # make sure the region is "far enough" from already clustered regions
            d = np.sqrt(((np.array(list(zip(column, row, scale)))
                            - np.tile(center.T, (nReg, 1)))**2).sum(axis=1))
            if (center[2] >= d).sum() == 0:
                nReg = nReg+1
                column = np.append(column, center[0])
                row = np.append(row, center[1])
                scale = np.append(scale, center[2])
                gamma = np.append(gamma, t_gamma[index])
        else:
            nReg = nReg+1
            column = np.append(column, center[0])
            row = np.append(row, center[1])
            scale = np.append(scale, center[2])
            gamma = np.append(gamma, t_gamma[index])

    return np.array([row, column, scale])