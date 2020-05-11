"""
==================================
3D adaptive histogram equalization
==================================

Adaptive histogram equalization (AHE) can be used to improve the local
contrast of an image [1]_. Specifically, AHE can be useful for normalizing
intensities across images. This example compares the results of applying AHE
to a 3D image and a degraded version of it.

.. [1] https://en.wikipedia.org/wiki/Histogram_equalization
"""

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from skimage import exposure
from scipy import ndimage


#############
# Prepare data and apply AHE
#############

# create random 3d data by zooming into random array
a = 5
np.random.seed(100)
im_orig = np.random.randint(0, 1000,
                            (a, a, a)).astype(np.float64) / 1000.
im_orig = ndimage.zoom(im_orig, 50, order=1)


# degrade image by applying uneven intensity gradient along x
sigmoid = (1 / (1 + np.exp(5 * (np.linspace(0, 1, im_orig.shape[0]) - 0.5))))
im_degraded = (im_orig.T * sigmoid).T

# recover local contrast with AHE
kernel_size, clip_limit = 55, 0.5
im_orig_ahe = exposure.equalize_adapthist(im_orig,
                                          kernel_size=kernel_size,
                                          clip_limit=clip_limit)
im_degraded_ahe = exposure.equalize_adapthist(im_degraded,
                                              kernel_size=kernel_size,
                                              clip_limit=clip_limit)


#############
# define functions to help plot the data
#############

def get_rgba(scalars, cmap, vmin, vmax, alpha=0.2):
    """
    Convert array of scalars into array of corresponding RGBA values.
    """
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgbas = scalar_map.to_rgba(scalars)
    rgbas[:, 3] = alpha
    return rgbas


def plt_render_volume(vol, fig_ax, ref_vol=None,
                      vmin=0, vmax=1, bin_width=10, n_levels=50):
    """
    Render a volume in a 3D matplotlib scatter plot.
    Better would be to use napari.
    """

    xs, ys, zs = np.mgrid[0:vol.shape[0]:bin_width,
                          0:vol.shape[1]:bin_width,
                          0:vol.shape[2]:bin_width]
    vol_scaled = vol[::bin_width, ::bin_width, ::bin_width].flatten()
    ref_vol_scaled = ref_vol[::bin_width, ::bin_width, ::bin_width].flatten()

    # define alpha transfer function
    levels = np.linspace(vmin, vmax, n_levels)
    alphas = np.mean([levels[1:], levels[:-1]], 0)
    alphas = alphas ** 3
    alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min())
    alphas *= 0.8
    alphas = np.clip(alphas, 0.2, 1)

    # group pixels by intensity and plot separately, as 3D scatter
    # does not accept arrays of alpha values
    for il in range(1, len(levels)):
        sel = (ref_vol_scaled >= levels[il - 1])
        sel *= (ref_vol_scaled < levels[il])
        if not len(sel):
            continue
        c = get_rgba(vol_scaled[sel], 'viridis',
                     vmin=0, vmax=1, alpha=alphas[il - 1])
        fig_ax.scatter(xs.flatten()[sel],
                       ys.flatten()[sel],
                       zs.flatten()[sel],
                       marker='o', c=c, s=2 * bin_width, linewidth=0)


#############
# create nice plot
#############

fig = plt.figure(figsize=(8, 7))
axs = [fig.add_subplot(2, 2, i + 1, projection=Axes3D.name)
       for i in range(4)]

ims = [im_orig, im_degraded, im_orig_ahe, im_degraded_ahe]
titles = ['Original image\n(randomly generated)',
          'Degraded image\n(sigmoid in x)',
          '3D AHE\n(original image)',
          '3D AHE\n(degraded image)']

for iax, ax in enumerate(axs):
    plt_render_volume(ims[iax], ax, ref_vol=ims[0], bin_width=10)
    ax.grid(False)
    ax.set_title(titles[iax])

    lpad = -15
    ax.set_xlabel('x', labelpad=lpad)
    ax.set_ylabel('y', labelpad=lpad)
    ax.set_zlabel('z', labelpad=lpad)

    # Get rid of the panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

plt.tight_layout()
plt.show()
