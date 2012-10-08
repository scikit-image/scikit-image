"""
Script to draw skimage logo using Scipy logo as stencil. The easiest
starting point is the `plot_colorized_logo`; the "if-main" demonstrates its use.

Original snake image from pixabay [1]_

.. [1] http://pixabay.com/en/snake-green-toxic-close-yellow-3237/
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

import skimage.io as sio
import skimage.filter as imfilt

import scipy_logo


# Utility functions
# =================

def get_edges(img):
    edge = np.empty(img.shape)
    if len(img.shape) == 3:
        for i in range(3):
            edge[:, :, i] = imfilt.sobel(img[:, :, i])
    else:
        edge = imfilt.sobel(img)
    edge = rescale_intensity(edge)
    return edge

def rescale_intensity(img):
    i_range = float(img.max() - img.min())
    img = (img - img.min()) / i_range * 255
    return np.uint8(img)

def colorize(img, color, whiten=False):
    """Return colorized image from gray scale image

    Parameters
    ----------
    img : N x M array
        grayscale image
    color : length-3 sequence of floats
        RGB color spec. Float values should be between 0 and 1.
    whiten : bool
        If True, a color value less than 1 increases the image intensity.
    """
    color = np.asarray(color)[np.newaxis, np.newaxis, :]
    img = img[:, :, np.newaxis]
    if whiten:
        # truncate and stretch intensity range to enhance contrast
        img = np.clip(img, 80, 255)
        img = rescale_intensity(img)
        return np.uint8(color * (255 - img) + img)
    else:
        return np.uint8(img * color)


def prepare_axes(ax):
    plt.sca(ax)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.itervalues():
        spine.set_visible(False)


_rgb_stack = np.ones((1, 1, 3), dtype=bool)
def gray2rgb(arr):
    """Return RGB image from a grayscale image.

    Expand h x w image to h x w x 3 image where color channels are simply copies
    of the grayscale image.
    """
    return arr[:, :, np.newaxis] * _rgb_stack


# Logo generating classes
# =======================

class LogoBase(object):

    def __init__(self):
        self.logo = scipy_logo.ScipyLogo(radius=self.radius)
        self.mask_1 = self.logo.get_mask(self.img.shape, 'upper left')
        self.mask_2 = self.logo.get_mask(self.img.shape, 'lower right')
        self.edges = get_edges(self.img)
        # truncate and stretch intensity range to enhance contrast
        self.edges = np.clip(self.edges, 0, 100)
        self.edges = rescale_intensity(self.edges)


    def _crop_image(self, img):
        w = 2 * self.radius
        x, y = self.origin
        return img[y:y+w, x:x+w]

    def get_canvas(self):
        return 255 * np.ones(self.img.shape, dtype=np.uint8)

    def plot_curve(self, **kwargs):
        self.logo.plot_snake_curve(**kwargs)


class SnakeLogo(LogoBase):

    def __init__(self):
        self.radius = 250
        self.origin = (420, 0)
        img = sio.imread('data/snake_pixabay.jpg')        
        img = self._crop_image(img)

        img = img.astype(float) * 1.1
        img[img > 255] = 255
        self.img = img.astype(np.uint8)

        LogoBase.__init__(self)


snake_color = SnakeLogo()
snake = SnakeLogo()
# turn RGB image into gray image
snake.img = np.mean(snake.img, axis=2)
snake.edges = np.mean(snake.edges, axis=2)


# Demo plotting functions
# =======================

def plot_colorized_logo(logo, color, edges='light', switch=False, whiten=False):
    """Convenience function to plot artificially colored logo.

    Parameters
    ----------
    logo : subclass of LogoBase
    color : length-3 sequence of floats
        RGB color spec. Float values should be between 0 and 1.
    edges : {'light'|'dark'}
        Specifies whether Sobel edges are drawn light or dark
    switch : bool
        If False, the image is drawn on the southeast half of the Scipy curve
        and the edge image is drawn on northwest half.
    whiten : bool
        If True, a color value less than 1 increases the image intensity.
    """
    if not hasattr(color[0], '__iter__'):
        color = [color] * 2
    if not hasattr(whiten, '__iter__'):
        whiten = [whiten] * 2
    img = gray2rgb(logo.get_canvas())
    mask_img = gray2rgb(logo.mask_2)
    mask_edge = gray2rgb(logo.mask_1)
    if switch:
        mask_img, mask_edge = mask_edge, mask_img
    if edges == 'dark':
        lg_edge = colorize(255 - logo.edges, color[0], whiten=whiten[0])
    else:
        lg_edge = colorize(logo.edges, color[0], whiten=whiten[0])
    lg_img = colorize(logo.img, color[1], whiten=whiten[1])
    img[mask_img] = lg_img[mask_img]
    img[mask_edge] = lg_edge[mask_edge]
    logo.plot_curve(lw=5, color='w')
    plt.imshow(img)


def red_light_edges(logo, **kwargs):
    plot_colorized_logo(logo, (1, 0, 0), edges='light', **kwargs)


def red_dark_edges(logo, **kwargs):
    plot_colorized_logo(logo, (1, 0, 0), edges='dark', **kwargs)

def blue_light_edges(logo, **kwargs):
    plot_colorized_logo(logo, (0.35, 0.55, 0.85), edges='light', **kwargs)


def blue_dark_edges(logo, **kwargs):
    plot_colorized_logo(logo, (0.35, 0.55, 0.85), edges='dark', **kwargs)


def green_orange_light_edges(logo, **kwargs):
    colors = ((0.6, 0.8, 0.3), (1, 0.5, 0.1))
    plot_colorized_logo(logo, colors, edges='light', **kwargs)

def green_orange_dark_edges(logo, **kwargs):
    colors = ((0.6, 0.8, 0.3), (1, 0.5, 0.1))
    plot_colorized_logo(logo, colors, edges='dark', **kwargs)


if __name__ == '__main__':

    import sys
    plot = False
    if len(sys.argv) < 2 or sys.argv[1] != '--no-plot':
        plot = True

        print "Run with '--no-plot' flag to generate logo silently."

    def plot_all():
        plotters = (red_light_edges, red_dark_edges,
                    blue_light_edges, blue_dark_edges,
                    green_orange_light_edges, green_orange_dark_edges)

        f, axes_array = plt.subplots(nrows=2, ncols=len(plotters))
        for plot, ax_col in zip(plotters, axes_array.T):
                prepare_axes(ax_col[0])
                plot(snake)
                prepare_axes(ax_col[1])
                plot(snake, whiten=True)
        plt.tight_layout()

    def plot_snake():

        f, ax = plt.subplots()
        prepare_axes(ax)
        green_orange_dark_edges(snake, whiten=(False, True))
        plt.savefig('green_orange_snake.png', bbox_inches='tight')

    if plot:
        plot_all()

    plot_snake()

    if plot:
        plt.show()

