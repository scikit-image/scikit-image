import numpy as np
from numpy.linalg import norm

from ..metrics._structural_similarity import structural_similarity as ssim


def E(X, Y):
    return (Y - X) / norm(Y - X)


def optimize_func(minimize, img, _lambda, iters, sigma):
    if len(img.shape) != 3:
        img = img[0]

    """
    Compute the image with the highest or lowest SSIM to a reference image
    on the L-2 ball of the reference image. Follows a two step iterative
    gradient descent procedure.

    Parameters
    ----------
    minimize: bool
        If True, image with lowest ssim is found. Else, highest ssim is found
    img: ndarray
        The reference image
    _lambda: float
        The L-2 distance from the reference image
    iters: int
        The number of iterations allowed to find image with highest/lowest ssim
    sigma: int
        A step size scalar

    Returns
    -------
    adv_x: ndarray
        Optimized image

    Notes
    ------
    First step of algorithm is modified to increase memory efficiency. Behavior
    of the algorithm is unchanged.

    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.
       https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
       :DOI:`10.1109/TIP.2003.819861`

    """

    if len(img.shape) != 3:
        raise ValueError(
            "Array should have dimensions (width, height, channel)\n")

    if np.max(img) > 1.0:
        raise ValueError(
            "Array values should me in range [0, 1]")

    noise = np.random.randint(0, 255, size = img.shape)/255.
    noise = noise * _lambda/norm(noise)
    adv_x = np.clip(img + noise, 0, 1)

    prev_ssim = ssim(img, adv_x, multichannel=True)
    prev_img = np.copy(adv_x)

    for i in range(iters):
        # ----------------- step 1 -----------------
        grad = ssim(img, adv_x, multichannel=True, gradient=True)[1].flatten()
        grad = np.expand_dims(grad, axis=1)

        diff = E(img, adv_x).flatten()
        diff = np.expand_dims(diff, axis=1)

        diff_t = diff.T

        update = sigma * (grad - diff.dot((diff_t.dot(grad))))
        update = update.reshape(adv_x.shape)

        if minimize:
            adv_x = adv_x - update
        else:
            adv_x = adv_x + update

        # ----------------- step 2 -----------------
        adv_x = np.clip(img + _lambda * E(img, adv_x), 0, 1)

        # ----------- Conditional update -----------
        ssim_score = ssim(img, adv_x, multichannel=True)

        if minimize:
            if ssim_score > prev_ssim:
                adv_x = np.copy(prev_img)
                sigma *= 0.9
            else:
                prev_img = np.copy(adv_x)
                prev_ssim = ssim_score
        else:
            if ssim_score < prev_ssim:
                adv_x = np.copy(prev_img)
                sigma *= 0.9
            else:
                prev_img = np.copy(adv_x)
                prev_ssim = ssim_score

    return adv_x


def minimize_ssim(img, _lambda, iters=50, sigma=30):
    adv_x = optimize_func(True, img, _lambda, iters, sigma)
    return adv_x


def maximize_ssim(img, _lambda, iters=50, sigma=30):
    adv_x = optimize_func(False, img, _lambda, iters, sigma)
    return adv_x
