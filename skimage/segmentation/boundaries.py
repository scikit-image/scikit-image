import numpy as np
import matplotlib.pyplot as plt

from ..util import img_as_float
from .._shared.utils import deprecated
from .._shared.mpl import ClearColormap
from ..morphology import dilation, square


__all__ = ['find_boundaries', 'visualize_boundaries', 'overlay_boundaries']


def find_boundaries(label_img):
    """Return bool array where boundaries between labeled regions are True."""
    boundaries = np.zeros(label_img.shape, dtype=np.bool)
    boundaries[1:, :] += label_img[1:, :] != label_img[:-1, :]
    boundaries[:, 1:] += label_img[:, 1:] != label_img[:, :-1]
    return boundaries


@deprecated('skimage.segmentation.boundaries.overlay_boundaries')
def visualize_boundaries(img, label_img):
    img = img_as_float(img, force_copy=True)
    boundaries = find_boundaries(label_img)
    outer_boundaries = dilation(boundaries.astype(np.uint8), square(2))
    img[outer_boundaries != 0, :] = np.array([0, 0, 0])  # black
    img[boundaries, :] = np.array([1, 1, 0])  # yellow
    return img


def overlay_boundaries(label_img, color=None, ax=None):
    """Plot boundaries over current image plot.

    Parameters
    ----------
    label_img : (M, N) array
        Labeled array, where all connected regions are assigned the same
        integer value.
    color : RGB or RGBA tuple
        Color of overlay.
    ax : matplotlib.axes.Axes
        Axes where boundaries are plotted.
    """
    if color is None:
        color = (1, 1, 0)
    cmap = ClearColormap(color)

    ax = ax if ax is not None else plt.gca()

    boundaries = find_boundaries(label_img)
    ax.imshow(boundaries, cmap=cmap)
