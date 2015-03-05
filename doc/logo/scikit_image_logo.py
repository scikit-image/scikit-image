"""
Script to draw skimage logo using Scipy logo as stencil. The easiest
starting point is the `plot_colorized_logo`.

Original snake image from pixabay [1]_

.. [1] http://pixabay.com/en/snake-green-toxic-close-yellow-3237/
"""
import sys
if len(sys.argv) != 2 or sys.argv[1] != '--no-plot':
    print("Run with '--no-plot' flag to generate logo silently.")
else:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import skimage.io as sio
from skimage import img_as_float
from skimage.color import gray2rgb, rgb2gray
from skimage.exposure import rescale_intensity
from skimage.filters import sobel

import scipy_logo

# Utility functions
# =================

def colorize(image, color, whiten=False):
    """Return colorized image from gray scale image.

    The colorized image has values from ranging between black at the lowest
    intensity to `color` at the highest. If `whiten=True`, then the color
    ranges from `color` to white.
    """
    color = np.asarray(color)[np.newaxis, np.newaxis, :]
    image = image[:, :, np.newaxis]
    if whiten:
        # truncate and stretch intensity range to enhance contrast
        image = rescale_intensity(image, in_range=(0.3, 1))
        return color * (1 - image) + image
    else:
        return image * color


def prepare_axes(ax):
    plt.sca(ax)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


# Logo generating classes
# =======================

class LogoBase(object):

    def __init__(self):
        self.logo = scipy_logo.ScipyLogo(radius=self.radius)
        self.mask_1 = self.logo.get_mask(self.image.shape, 'upper left')
        self.mask_2 = self.logo.get_mask(self.image.shape, 'lower right')

        edges = np.array([sobel(img) for img in self.image.T]).T
        # truncate and stretch intensity range to enhance contrast
        self.edges = rescale_intensity(edges, in_range=(0, 0.4))

    def _crop_image(self, image):
        w = 2 * self.radius
        x, y = self.origin
        return image[y:y + w, x:x + w]

    def plot_curve(self, **kwargs):
        self.logo.plot_snake_curve(**kwargs)


class SnakeLogo(LogoBase):

    radius = 250
    origin = (420, 0)

    def __init__(self):
        image = sio.imread('data/snake_pixabay.jpg')
        image = self._crop_image(image)
        self.image = img_as_float(image)

        LogoBase.__init__(self)


snake_color = SnakeLogo()
snake = SnakeLogo()
# turn RGB image into gray image
snake.image = rgb2gray(snake.image)
snake.edges = rgb2gray(snake.edges)


# Demo plotting functions
# =======================

def plot_colorized_logo(logo, color, edges='light', whiten=False):
    """Convenience function to plot artificially-colored logo.

    The upper-left half of the logo is an edge filtered image, while the
    lower-right half is unfiltered.

    Parameters
    ----------
    logo : LogoBase instance
    color : length-3 sequence of floats or 2 length-3 sequences
        RGB color spec. Float values should be between 0 and 1.
    edges : {'light'|'dark'}
        Specifies whether Sobel edges are drawn light or dark
    whiten : bool or 2 bools
        If True, a color value less than 1 increases the image intensity.
    """
    if not hasattr(color[0], '__iter__'):
        color = [color] * 2  # use same color for upper-left & lower-right
    if not hasattr(whiten, '__iter__'):
        whiten = [whiten] * 2  # use same setting for upper-left & lower-right

    image = gray2rgb(np.ones_like(logo.image))
    mask_img = gray2rgb(logo.mask_2)
    mask_edge = gray2rgb(logo.mask_1)

    # Compose image with colorized image and edge-image.
    if edges == 'dark':
        logo_edge = colorize(1 - logo.edges, color[0], whiten=whiten[0])
    else:
        logo_edge = colorize(logo.edges, color[0], whiten=whiten[0])
    logo_img = colorize(logo.image, color[1], whiten=whiten[1])
    image[mask_img] = logo_img[mask_img]
    image[mask_edge] = logo_edge[mask_edge]

    logo.plot_curve(lw=5, color='w')  # plot snake curve on current axes
    plt.imshow(image)


if __name__ == '__main__':
    # Colors to use for the logo:
    red = (1, 0, 0)
    blue = (0.35, 0.55, 0.85)
    green_orange = ((0.6, 0.8, 0.3), (1, 0.5, 0.1))

    def plot_all():
        color_list = [red, blue, green_orange]
        edge_list = ['light', 'dark']
        f, axes = plt.subplots(nrows=len(edge_list), ncols=len(color_list))
        for axes_row, edges in zip(axes, edge_list):
            for ax, color in zip(axes_row, color_list):
                prepare_axes(ax)
                plot_colorized_logo(snake, color, edges=edges)
        plt.tight_layout()

    def plot_official_logo():
        f, ax = plt.subplots()
        prepare_axes(ax)
        plot_colorized_logo(snake, green_orange, edges='dark',
                            whiten=(False, True))
        plt.savefig('green_orange_snake.png', bbox_inches='tight')

    plot_all()
    plot_official_logo()

    plt.show()
