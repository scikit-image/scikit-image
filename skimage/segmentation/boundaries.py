import numpy as np
from ..morphology import dilation, square
from ..util import img_as_float


def find_boundaries(label_img):
    boundaries = np.zeros(label_img.shape, dtype=np.bool)
    boundaries[1:, :] += label_img[1:, :] != label_img[:-1, :]
    boundaries[:, 1:] += label_img[:, 1:] != label_img[:, :-1]
    return boundaries


def visualize_boundaries(img, label_img):
    img = img_as_float(img, force_copy=True)
    boundaries = find_boundaries(label_img)
    outer_boundaries = dilation(boundaries.astype(np.uint8), square(2))
    img[outer_boundaries != 0, :] = np.array([0, 0, 0])  # black
    img[boundaries, :] = np.array([1, 1, 0])  # yellow
    return img
