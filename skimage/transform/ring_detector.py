from ._ring_detector import RidgeHoughTransform

import matplotlib.pyplot as plt


__all__ = ['ring_detector']


def ring_detector(image, sigma, r_min, r_max, curv_thresh, circle_thresh,
                  vote_thresh, dr):
    """

    References
    ----------
    [1] Afik, E. Robust and highly performant ring detection algorithm for 3d particle tracking using 2d microscope imaging. Sci. Rep. 5,
        13584; doi: 10.1038/srep13584 (2015).
    """
    ridge_hough = RidgeHoughTransform(image)
    ridge_hough.params['sigma'] = sigma
    ridge_hough.params['Rmin'] = r_min
    ridge_hough.params['Rmax'] = r_max
    ridge_hough.params['curv_thresh'] = curv_thresh
    ridge_hough.params['circle_thresh'] = circle_thresh
    ridge_hough.params['vote_thresh'] = vote_thresh
    ridge_hough.params['dr'] = dr
    ridge_hough.img_preprocess()
    ridge_hough.rings_detection()
    detected_rings = ridge_hough.output['rings']

    return detected_rings