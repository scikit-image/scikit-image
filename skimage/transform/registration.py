import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import basinhopping, minimize

from .pyramids import pyramid_reduce
from ._warps import warp, SimilarityTransform

__all__ = ['register', 'p_to_matrix', 'matrix_to_p']


def _draw_setup(img):
    fig, ax = plt.subplots()
    axim = ax.imshow(img)
    plt.show(block=False)
    return fig, ax, axim


def _action(package, img, p):
    fig, ax, axim = package
    matrix = p_to_matrix(p)
    img = warp(img, matrix)

    plt.imshow(img)
    fig.canvas.draw()
    return fig, ax, axim


def _gaussian_pyramid(image, levels=6):
    pyramid = levels*[None]
    pyramid[-1] = image

    for level in range(levels-2, -1, -1):
        image = pyramid_reduce(image, sigma=2/3)
        pyramid[level] = (image)

    return pyramid


def _mse(img1, img2):
    return ((img1-img2)**2).sum()


def _cost_mse(param, reference_image, target_image):
    transformation = p_to_matrix(param)
    transformed = warp(target_image, transformation, order=3)
    return _mse(reference_image, transformed)


def p_to_matrix(param):
    r, tc, tr = param
    return SimilarityTransform(rotation=r, translation=(tc, tr))


def matrix_to_p(matrix):
    m = matrix.params
    return (np.arccos(m[0][0])*180/np.pi, m[0][2], m[1][2])


def register(reference, target, *, cost=_cost_mse, nlevels=7, method='Powell', draw=False):
    assert method in ['Powell', 'BH']
    pyramid_ref = _gaussian_pyramid(reference, levels=nlevels)
    pyramid_tgt = _gaussian_pyramid(target, levels=nlevels)
    levels = range(nlevels, 0, -1)
    image_pairs = zip(pyramid_ref, pyramid_tgt)
    p = np.zeros(3)

    if draw:
        drawing_package = _draw_setup(reference)

    for n, (ref, tgt) in zip(levels, image_pairs):
        p[1] *= 2
        p[2] *= 2
        if method.upper() == 'BH':
            res = basinhopping(cost, p,
                               minimizer_kwargs={'args': (ref, tgt)})
            if n <= 4:  # avoid basin-hopping in lower levels
                method = 'Powell'
        else:
            res = minimize(cost, p, args=(ref, tgt), method='Powell')
        p = res.x
        if draw:
            _action(drawing_package, tgt, p)

    matrix = p_to_matrix(p)

    return matrix
