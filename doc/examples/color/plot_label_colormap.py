from skimage.color.colorlabel import label_colormap

import matplotlib.pyplot as plt
import numpy


def draw_colormap(colormap):
    """Draw colormap as an image.

    Parameters
    ----------
    colormap: (N, 3)
        Colormap for N colors.
    """
    n_colors = len(colormap)
    ret = numpy.zeros((n_colors, 10 * 10, 3))
    for i in xrange(n_colors):
        ret[i, ...] = colormap[i]
    return ret.reshape((n_colors * 10, 10, 3))


def main():
    # Get colormap for number of labels (ex. 8)
    colormap = label_colormap(n_labels=8)

    # Draw colormap
    img_viz = draw_colormap(colormap)

    plt.imshow(img_viz)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
