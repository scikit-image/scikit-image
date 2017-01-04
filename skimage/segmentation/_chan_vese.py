import numpy as np
from scipy.ndimage import distance_transform_edt as distance


def _distance_from_zero_level(phi):
    H = _cv_heavyside(phi)
    return distance(H) - distance(H-1)


def _cv_curvature1(phi):
    """
    Returns the 'curvature' of a level set 'phi'.
    """
    P = np.pad(phi, 1, mode='mean')
    P = _distance_from_zero_level(P)
    fy = (P[2:, 1:-1] - P[0:-2, 1:-1]) / 2.0
    fx = (P[1:-1, 2:] - P[1:-1, 0:-2]) / 2.0
    K = np.sqrt(fx*fx+fy*fy)
    return _cv_delta(phi)*K


def _cv_curvature(phi):
    """
    Returns the 'curvature' of a level set 'phi'.
    """
    P = np.pad(phi, 1, mode='edge')
    P = _distance_from_zero_level(P)
    fy = (P[2:, 1:-1] - P[0:-2, 1:-1]) / 2.0
    fx = (P[1:-1, 2:] - P[1:-1, 0:-2]) / 2.0
    fyy = P[2:, 1:-1] + P[0:-2, 1:-1] - 2*phi
    fxx = P[1:-1, 2:] + P[1:-1, 0:-2] - 2*phi
    fxy = .25 * (P[2:, 2:]+P[:-2, :-2]-P[:-2, 2:]-P[2:, :-2])
    grad2 = fx**2 + fy**2
    K = ((fxx*fy**2 - 2*fxy*fx*fy + fyy*fx**2) /
         (grad2*np.sqrt(grad2) + 1e-10))
    return _cv_delta(phi)*K


def _cv_calculate_variation(img, phi, mu, lambda1, lambda2, dt):
    """
    Returns the variation of level set 'phi' based on algorithm
    parameters.
    """
    eta = 1.0
    P = np.pad(phi, 1, mode='mean')

    phixp = P[1:-1, 2:] - P[1:-1, 1:-1]
    phixn = P[1:-1, 1:-1] - P[1:-1, :-2]
    phix0 = (P[1:-1, 2:] - P[1:-1, :-2]) / 2.0

    phiyp = P[2:, 1:-1] - P[1:-1, 1:-1]
    phiyn = P[1:-1, 1:-1] - P[:-2, 1:-1]
    phiy0 = (P[2:, 1:-1] - P[:-2, 1:-1]) / 2.0

    C1 = 1. / np.sqrt(eta + phixp**2 + phiy0**2)
    C2 = 1. / np.sqrt(eta + phixn**2 + phiy0**2)
    C3 = 1. / np.sqrt(eta + phix0**2 + phiyp**2)
    C4 = 1. / np.sqrt(eta + phix0**2 + phiyn**2)

    K = (P[1:-1, 2:]*C1 + P[1:-1, :-2]*C2 +
         P[2:, 1:-1]*C3 + P[:-2, 1:-1]*C4)

    Hphi = _cv_heavyside(phi)
    (c1, c2) = _cv_calculate_averages(img, Hphi)

    difference_from_average_term = (- lambda1*(img-c1)**2 +
                                    lambda2*(img-c2)**2)
    new_phi = (phi + (dt*_cv_delta(phi)) *
               (mu*K + difference_from_average_term))
    return new_phi / (1 + mu*dt*_cv_delta(phi)*(C1+C2+C3+C4))


def _cv_heavyside(x, eps=1.):
    """
    Returns the result of a regularised heavyside function of the
    input value(s).
    """
    return 0.5 * (1. + 2./np.pi * np.arctan(x/eps))


def _cv_delta(x, eps=1.):
    """
    Returns the result of a regularised dirac function of the
    input value(s).
    """
    return eps / (eps**2 + x**2)


def _cv_calculate_averages(img, Hphi):
    """
    Returns the average values 'inside' and 'outside'
    """
    H = Hphi
    Hinv = 1. - H
    Hsum = np.sum(H)
    Hinvsum = np.sum(Hinv)
    avg_inside = np.sum(img * H)
    avg_oustide = np.sum(img * Hinv)
    if Hsum != 0:
        avg_inside = avg_inside / Hsum
    if Hinvsum != 0:
        avg_oustide = avg_oustide / Hinvsum
    return (avg_inside, avg_oustide)


def _cv_difference_from_average_term(img, Hphi, lambda_pos, lambda_neg):
    """
    Returns the 'energy' contribution due to the difference from
    the average value within a region at each point.
    """
    (c1, c2) = _cv_calculate_averages(img, Hphi)
    Hinv = 1.-Hphi
    return (lambda_pos * (img-c1)**2 * Hphi +
            lambda_neg * (img-c2)**2 * Hinv)


def _cv_edge_length_term(phi, mu):
    """
    Returns the 'energy' contribution due to the length of the
    edge between regions at each point, multiplied by a factor 'mu'.
    """
    toret = _cv_curvature(phi)
    return mu * toret


def _cv_energy(img, phi, mu, lambda1, lambda2, heavyside=_cv_heavyside):
    """
    Returns the total 'energy' of the current level set function
    """
    H = heavyside(phi)
    avgenergy = _cv_difference_from_average_term(img, H, lambda1, lambda2)
    lenenergy = _cv_edge_length_term(phi, mu)
    return np.sum(avgenergy) + np.abs(np.sum(lenenergy))


def _cv_reset_level_set(phi):
    """
    This is a placeholder function as resetting the level set is not
    strictly necessary, and has not been done for this implementation.
    """
    return phi


def _cv_checkerboard(image_size, square_size):
    """
    Generates a checkerboard level set function. According to
    Pascal Getreuer, such a level set function has fast convergence.
    """
    yv = np.arange(image_size[0]).reshape(image_size[0], 1)
    xv = np.arange(image_size[1])
    return (np.sin(np.pi/square_size * yv) *
            np.sin(np.pi/square_size * xv))


def _cv_large_disk(image_size):
    """
    Generates a disk level set function. The disk covers the whole
    image along its smallest dimension.
    """
    res = np.ones(image_size)
    centerY = int((image_size[0]-1) / 2)
    centerX = int((image_size[1]-1) / 2)
    res[centerY, centerX] = 0.
    radius = float(min(centerX, centerY))
    return (radius-distance(res)) / radius


def _cv_small_disk(image_size):
    """
    Generates a disk level set function. The disk covers half of
    the image along its smallest dimension.
    """
    res = np.ones(image_size)
    centerY = int((image_size[0]-1) / 2)
    centerX = int((image_size[1]-1) / 2)
    res[centerY, centerX] = 0.
    radius = float(min(centerX, centerY))
    return (radius/2 - distance(res)) / (radius*1.5)


def _cv_initial_shape(starting_level_set, img):
    """
    Generates a checkerboard level set function. According to
    Pascal Getreuer, such a level set function has fast convergence.
    """
    res = []
    if type(starting_level_set) == str:
        if starting_level_set == 'checkerboard':
            res = _cv_checkerboard(img.shape, 5)
        elif starting_level_set == 'disk':
            res = _cv_large_disk(img.shape)
        elif starting_level_set == 'small disk':
            res = _cv_small_disk(img.shape)
        else:
            raise ValueError("Incorrect name for starting level set preset.")
    else:
        res = starting_level_set
    return res


def chan_vese(img, mu=0.2, lambda1=1.0, lambda2=1.0, tol=1e-3, maxiter=100,
              dt=1.0, starting_level_set='checkerboard',
              extended_output=False):
    """Chan-vese algorithm.

    Active contour model by evolving a level set. Supports 2D
    grayscale images only, and does not implement the area term
    described in the original article.

    Parameters
    ----------
    img : (M, N) 2D array
        Grayscale image to be segmented.
    mu : float
        'edge length' weight parameter. Higher mu values will produce
        a 'round' edge, while values closer to zero will detect
        smaller objects.
    lambda1 : float, optional
        'difference from average' weight parameter for the output
        region with value 'True'. If it is lower than lambda2, this
        region will have a larger range of values than the other.
    lambda2 : float, optional
        'difference from average' weight parameter for the output
        region with value 'False'. If it is lower than lambda1, this
        region will have a larger range of values than the other.
    tol : float, positive, optional
        'energy' variation tolerance between iterations. If the
        difference of 'energy' between two iterations is below this
        value, the algorithm will assume that the solution was
        reached.
    maxiter : uint, optional
        Maximum number of iterations allowed before the algorithm
        interrupts itself.
    dt : float, optional
        A multiplication factor applied at calculations for each step,
        serves to accelerate the algorithm. While higher values may
        speed up the algorithm, they may also lead to convergence
        problems.
    starting_level_set : str or (M, N) 2D array, optional
        Defines the starting level set used by the algorithm.
        If a string is inputted, a level set that matches the image
        size will automatically be generated. Alternatively, it is
        possible to define a custom level set, which should be an
        array of float values, with the same size as 'img'.
        Accepted String values are as follows.

        'checkerboard'
            the starting level set is defined as
            sin(x/5*pi)*sin(y/5*pi), where x and y are pixel
            coordinates. This level set has fast convergence, but may
            fail to detect implicit edges.
        'disk'
            the starting level set is defined as the opposite
            of the distance from the center of the image minus half of
            the minimum value between image width and image height.
            This is somewhat slower, but is more likely to properly
            detect implicit edges.
        'small disk'
            the starting level set is defined as the
            opposite of the distance from the center of the image
            minus a quarter of the minimum value between image width
            and image height.
    extended_output : bool, optional
        If set to True, the return value will be a tuple containing
        the three return values (see below). If set to False which
        is the default value, only the 'segmentation' array will be
        returned.

    Returns
    -------
    segmentation : (M, N) 2D array, bool
        Segmentation produced by the algorithm.
    phi : (M, N) 2D array, float
        Final level set computed by the algorithm.
    energies : list of floats
        Shows the evolution of the 'energy' for each step of the
        algorithm. This should allow to check whether the algorithm
        converged.

    Note
    ----
    The 'energy' which this algorithm tries to minimize is defined
    as the sum of the differences from the average within the region
    weighed by the 'lambda' factors to which is added the length of
    the contour multiplied by the 'mu' factor.

    References
    ----------
    .. [1] An Active Contour Model without Edges, Tony Chan and
           Luminita Vese, Scale-Space Theories in Computer Vision,
           1999
    .. [2] Chan-Vese Segmentation, Pascal Getreuer Image Processing On
           Line, 2 (2012), pp. 214-224.
    .. [3] The Chan-Vese Algorithm - Project Report, Rami Cohen
           http://arxiv.org/abs/1107.2782 , 2011
    """
    if len(img.shape) != 2:
        raise ValueError("Input image should be a 2D array.")
    phi = _cv_initial_shape(starting_level_set, img)
    if type(phi) != np.ndarray or phi.shape != img.shape:
        raise ValueError("Dimensions of initial level set does not "
                         "match dimensions of image.")

    img = img - np.min(img)
    if np.max(img) != 0:
        img = img / np.max(img)

    i = 0
    delta = np.finfo(np.float).max
    old_energy = _cv_energy(img, phi, mu, lambda1, lambda2)
    energies = []
    phivar = tol + 1
    segmentation = phi > 0
    segchange = True
    area = img.shape[0] * img.shape[1]
    
    while((segchange or phivar > tol) and i < maxiter):
        print i
        # Save old values
        oldphi = phi
        oldseg = segmentation
        # Calculate new level set
        phi = _cv_calculate_variation(img, phi, mu, lambda1, lambda2, dt)
        phi = _cv_reset_level_set(phi)
        phivar = np.sum((phi-oldphi)**2) / area
        # Extract energy and compare to previous level set and
        # segmentation to see if continuin is necessary
        segmentation = phi > 0
        segchange = not np.array_equal(oldseg, segmentation)
        new_energy = _cv_energy(img, phi, mu, lambda1, lambda2)
        # Save old values
        energies.append(old_energy)
        delta = np.abs(new_energy - old_energy)
        old_energy = new_energy
        i += 1

    if extended_output:
        return (segmentation, phi, energies)
    else:
        return segmentation
