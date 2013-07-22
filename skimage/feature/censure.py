import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter
from skimage.transform import integral_image
import time

def _get_filtered_image(image, n, mode='DoB'):
    if mode == 'DoB':
        inner_wt = (1.0 / (2*n + 1)**2)
        outer_wt = (1.0 / (12*n**2 + 4*n))
        integral_img = integral_image(image)
        filtered_image = np.zeros(image.shape)
        # TODO : Outsource to Cython
        start = time.time()
        for i in range(2 * n, image.shape[0] - 2 * n):
            for j in range(2 * n, image.shape[1] - 2 * n):
                inner = integral_img[i + n, j + n] + integral_img[i - n, j - n] - integral_img[i + n, j - n] - integral_img[i - n, j + n]
                outer = integral_img[i + 2 * n, j + 2 * n] + integral_img[i - 2 * n, j - 2 * n] - integral_img[i + 2 * n, j - 2 * n] - integral_img[i - 2 * n, j + 2 * n]
                filtered_image[i, j] = outer_wt * outer - (inner_wt + outer_wt) * inner
        print time.time() - start
        return filtered_image


def censure_keypoints(image, mode='DoB', threshold=0.1):
    # TODO : Decide mode for convolve function
    # TODO : Decide number of scales. Image-size dependent?
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError("Only 2-D gray-scale images supported.")
    # Generating all the scales
    scale1 = _get_filtered_image(image, 1, mode)
    scale2 = _get_filtered_image(image, 2, mode)
    scale3 = _get_filtered_image(image, 3, mode)
    scale4 = _get_filtered_image(image, 4, mode)
    scale5 = _get_filtered_image(image, 5, mode)
    scale6 = _get_filtered_image(image, 6, mode)
    scale7 = _get_filtered_image(image, 7, mode)
    # Stacking all the scales in the 3rd dimension
    scales = np.dstack((scale1, scale2, scale3, scale4, scale5, scale6, scale7))
    # Suppressing points that are neither minima or maxima in their 3 x 3 x 3
    # neighbourhood to zero
    minimas = (minimum_filter(scales, (3, 3, 3)) == scales).astype(int) * scales
    maximas = (maximum_filter(scales, (3, 3, 3)) == scales).astype(int) * scales
    # Suppressing minimas and maximas weaker than threshold
    minimas[np.abs(minimas) < threshold] = 0
    maximas[np.abs(maximas) < threshold] = 0
    response = maximas + np.abs(minimas)
    return response
