import numpy as np
from ..util import img_as_float


def corner_fast(image, n=9, threshold=0.15):

    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError("Only 2-D gray-scale images supported.")

    image = img_as_float(image)
    corner_mask = np.zeros(image.shape, dtype=bool)

    test_pixels = np.asarray([[-3, 0], [-3, 1], [-2, 2], [-1, 3], [0, 3],
                              [1, 3], [2, 2], [3, 1], [3, 0], [3, -1],
                              [2, -2], [1, -3], [0, -3], [-1, -3],
                              [-2, -2], [-1, -3]])

    # TODO : Outsource to Cython
    for i in range(3, image.shape[0] - 3):
        for j in range(3, image.shape[1] - 3):
            test_x = i + test_pixels[:, 0]
            test_y = j + test_pixels[:, 1]
            intensities = image[test_x, test_y]
            low = intensities < image[i, j] - threshold
            high = intensities > image[i, j] + threshold
            low = np.concatenate(low, low)
            high = np.concatenate(high, high)
            # How to check if a sequence n * [True] exists in low/high ?
            # if n * [True] in low or n * True in high:
            #     corner_mask[i, j] = True
    
    corner_x, corner_y = np.where(corner_mask == True)
    corners = np.dstack(corner_x, corner_y)
    return corners
