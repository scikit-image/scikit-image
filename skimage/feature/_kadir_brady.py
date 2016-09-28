from __future__ import print_function, division
import numpy as np
from scipy import ndimage as ndi
from ..util import img_as_ubyte, view_as_windows
from .._shared.utils import assert_nD

def saliency_kadir_brady(image, min_scale=5, max_scale=13, saliency_threshold=0.6, clustering_threshold=2):
    """Find salient regions in the given grayscale image.

    For each point x, the method picks a scale s and calculates salient score Y(x,s).
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
    A : (N, 3) ndarray of float
        A 2d array with each row representing 3 values, '(y, x, scale)'
        where '(y,x)' are coordinates of the region and 'scale' is the
        size of corresponding salient region.

    References
    ----------
    https://en.wikipedia.org/wiki/Kadir-Brady_saliency_detector

    Examples
    --------
    >>> from skimage.feature import saliency_kadir_brady
    >>> from skimage.color import rgb2gray
    >>> from skimage.data import astronaut
    >>> image = astronaut()[100:300, 100:300]
    >>> saliency_kadir_brady(rgb2gray(image))
    array([[ 129.8125   70.125     7.5   ]
           [  85.25     22.8125    7.375 ]
           [ 110.125    13.        7.5   ]
           [ 155.1875  106.4375   11.    ]
           [  44.875   142.1875    9.    ]
           [ 137.1875  193.25      8.625 ]
           [ 100.375    45.3125    9.25  ]
           [ 158.5625  128.4375   10.625 ]
           [  51.75     69.125     7.5   ]
           [  76.625    60.1875    8.5   ]
           [ 106.1875   28.6875    8.25  ]
           [  73.8125   88.        7.375 ]
           [ 149.9375   93.625     8.375 ]
           [  16.0625   76.5      10.25  ]
           [ 118.875    23.5       8.125 ]
           [ 164.875   150.75     10.875 ]
           [  90.1875   54.        7.625 ]
           [ 152.125   184.75      9.75  ]
           [   9.0625  192.5625   10.5   ]
           [  85.4375  121.25      8.    ]
           [ 109.0625   61.0625   10.25  ]
           [ 133.3125   34.1875    8.25  ]])

    Notes
    -----
    The radius of each region is 'scale/2'.
    """

    assert_nD(image, 2)

    # scales for keypoints
    scales = np.arange(min_scale, max_scale, 2)
    # detect keypoints in scale-space
    base_regions = detect(image, scales)
    # pruning based on thresholds
    regions = prune(base_regions, saliency_threshold, clustering_threshold, k=15)

    return regions.T


def _detect(image, scales):
    """Find keypoints in the given grayscale image.

    Calculates the Shannon entropy of local image attributes for each region
    over a range of scales, and weight them based on the difference between
    the local descriptors around scale-space.

    Parameters
    ----------
    image : ndarray
        Input grayscale image
    scales : ndarray
        Scales for key regions

    Returns
    -------
    A : (4, N) ndarray of float
        A 2d array with each column representing 4 values, '(gamma, scale, y, x)'
        where '(y, x)' are coordinates of the region, 'scale' is the
        size of salient region, and 'gamma' is the saliency score.
    """

    entropy, weights, r, c = _saliency_param(image, scales)

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


def _saliency_param(image, scales):
    """Calculate parameters for Saliency metric.

    Parameters
    ----------
    image : ndarray
        Input grayscale image
    scales : ndarray
        Scales for key regions

    Returns
    -------
    entropy : (M, N) ndarray of float
        Matrix containing entropy value of region over scales.
    weights : (M, N) ndarray of float
        Matrix containing abs difference of PDF over scales.
    r : (N,) ndarray of float
        Row index of regions.
    c : (N,) ndarray of float
        Column index of regions.
    """
    
    image = img_as_ubyte(image)
    nr, nc = image.shape
    # find pixels that we are going to examine
    mask = np.ones((nr - max(scales), nc - max(scales)))
    r, c = np.nonzero(mask) + max(scales) / 2 + 1

    n_pix = len(r)

    n_scales = len(scales)
    intensity_edges = np.arange(0, 257, 16)

    previous_h = np.zeros((len(intensity_edges) - 1, n_pix))
    weights = np.zeros((n_scales, n_pix))
    entropy = np.zeros((n_scales, n_pix))

    # iterate through every possible region
    # iterate through scales
    for s_idx, scale in enumerate(scales):
        # shape for windows according to current scale
        radius = scale + 1
        window_shape = (radius, radius)

        init = int(0 + max(scales) / 2 - scale / 2)
        end = int(nr - max(scales) / 2 + scale / 2)
        patches = view_as_windows(image[init:end,init:end], window_shape)

        # iterate through window patches
        i=0
        for row in patches:
            for patch in row:
                h = np.histogram(patch, bins=intensity_edges)[0]
                h = h / np.sum(h)

                # index of histogram values greater than zero
                idx = np.nonzero(h > 0)
                entropy[s_idx, i] = -sum(h[idx] * np.log(h[idx]))

                if s_idx >= 1:
                    # first derivative in entropy space to calculate weights
                    dif = abs(h - previous_h[:, i])
                    factor = scales[s_idx] ** 2 / (2 * scales[s_idx] - 1)
                    weights[s_idx, i] = factor * sum(dif)
                    if s_idx == 1:
                        weights[s_idx - 1, i] = weights[s_idx, i]
                previous_h[:, i] = h
                i+=1

    return entropy, weights, r, c


def _salient_regions(candidate_regions, saliency_threshold):
    """Selects regions above saliency threshold.

    Parameters
    ----------
    candidate_regions : (4, N) ndarray of float
        A 2d array with each column representing gamma,scale,y,x.
    saliency_threshold : float, optional.
        Features with saliency score more than this will be considered.

    Returns
    -------
    t_gamma : (N,) ndarray of float
        Gamma values above threshold.
    t_scale : (N,) ndarray of float
        Scales of selected regions.
    t_r : (N,) ndarray of float
        Row index of selected regions.
    t_c : (N,) ndarray of float
        Cloumn index of selected regions.
    D : (N, N) ndarray of float
        Matrix with distances between regions 
    """

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

    # fill it with distance
    for i in range(n):
        pt = pts[i][:]
        # calculate the distance b/w regions
        dists = np.sqrt(((pts - np.tile(pt, (pts.shape[0], 1))) ** 2).sum(axis=1))
        D[i, :], D[:, i] = dists.T, dists

    return t_gamma, t_scale, t_r, t_c, D


def _prune(candidate_regions, saliency_threshold, v_th, k):
    """Clusters the detected keypoints.

    It selects highly salient points that have local support
    i.e. nearby points with similar saliency and scale. Each region is
    sufficiently distant from all others (in Scale-space regions) to
    qualify as a separate entity.

    Parameters
    ----------
    candidate_regions : (4, n) ndarray of float
        A 2d array with each column representing gamma,scale,y,x.
    saliency_threshold : float, optional.
        Features with saliency score more than this will be considered.
    v_th : int, optional
        Variance among the regions should be smaller than this threshold
        for sufficient clustering.

    Returns
    -------
    A : (3, N) ndarray of float
        A 2d array with each column representing 3 values, '(y, x, scale)'
        where '(y,x)' are coordinates of the region and 'scale' is the
        size of corresponding salient region.
    """

    t_gamma, t_scale, t_r, t_c, D = _salient_regions(candidate_regions, saliency_threshold)
    gamma, scale, row, column = (np.array([]) for i in range(4))

    n_reg = 0
    # clusters matrix
    cluster = np.zeros((3, k + 1))

    # pruning process
    for index in range(n):
        cluster[0, 0] = t_c[index]
        cluster[1, 0] = t_r[index]
        cluster[2, 0] = t_scale[index]
        s_i = np.argsort(D[index, :])
        # fill in the neighbouring regions
        for j in range(k):
            cluster[0, j+1] = t_c[s_i[j+1]]
            cluster[1, j+1] = t_r[s_i[j+1]]
            cluster[2, j+1] = t_scale[s_i[j+1]]

        # clusters center point
        center = np.array([np.mean(cluster, axis=1)])

        # check if the regions are "suffiently clustered", if variance is less than threshold
        v = np.var(np.sqrt(((cluster - np.tile(center.T, (1, k + 1))) ** 2).sum(axis=0)))
        if v > v_th:
            continue

        center = np.mean(cluster, axis=1)
        if n_reg > 0:
            # make sure the region is "far enough" from already clustered regions
            d = np.sqrt(((np.array(list(zip(column, row, scale)))
                            - np.tile(center.T, (n_reg, 1))) ** 2).sum(axis=1))
            if (center[2] >= 0.6*d).sum() == 0:
                n_reg = n_reg + 1
                column = np.append(column, center[0])
                row = np.append(row, center[1])
                scale = np.append(scale, center[2])
                gamma = np.append(gamma, t_gamma[index])
        else:
            n_reg = n_reg + 1
            column = np.append(column, center[0])
            row = np.append(row, center[1])
            scale = np.append(scale, center[2])
            gamma = np.append(gamma, t_gamma[index])

    return np.array([row, column, scale])
