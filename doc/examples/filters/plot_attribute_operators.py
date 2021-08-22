"""
===================
Attribute operators
===================

Attribute operators (or connected operators) [1]_ is a family of contour
preserving filtering operations in mathematical morphology. They can be
implemented by max-trees [2]_, a compact hierarchical representation of the
image.

Here, we show how to use diameter closing [3]_ [4]_, which is compared to
morphological closing. Comparing the two results, we observe that the
difference between image and morphological closing also extracts the long line.
A thin but long line cannot contain the structuring element. The diameter
closing stops the filling as soon as a maximal extension is reached. The line
is therefore not filled and therefore not extracted by the difference.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import diameter_closing
from skimage import data
from skimage.morphology import closing
from skimage.morphology import square

datasets = {
    'retina': {'image': data.microaneurysms(),
               'figsize': (15, 9),
               'diameter': 10,
               'vis_factor': 3,
               'title': 'Detection of microaneurysm'},
    'page': {'image': data.page(),
             'figsize': (15, 7),
             'diameter': 23,
             'vis_factor': 1,
             'title': 'Text detection'}
}

for dataset in datasets.values():
    # image with printed letters
    image = dataset['image']
    figsize = dataset['figsize']
    diameter = dataset['diameter']

    fig, ax = plt.subplots(2, 3, figsize=figsize)
    # Original image
    ax[0, 0].imshow(image, cmap='gray', aspect='equal',
                    vmin=0, vmax=255)
    ax[0, 0].set_title('Original', fontsize=16)
    ax[0, 0].axis('off')

    ax[1, 0].imshow(image, cmap='gray', aspect='equal',
                    vmin=0, vmax=255)
    ax[1, 0].set_title('Original', fontsize=16)
    ax[1, 0].axis('off')

    # Diameter closing : we remove all dark structures with a maximal
    # extension of less than <diameter> (12 or 23). I.e. in closed_attr, all
    # local minima have at least a maximal extension of <diameter>.
    closed_attr = diameter_closing(image, diameter, connectivity=2)

    # We then calculate the difference to the original image.
    tophat_attr = closed_attr - image

    ax[0, 1].imshow(closed_attr, cmap='gray', aspect='equal',
                    vmin=0, vmax=255)
    ax[0, 1].set_title('Diameter Closing', fontsize=16)
    ax[0, 1].axis('off')

    ax[0, 2].imshow(dataset['vis_factor'] * tophat_attr, cmap='gray',
                    aspect='equal', vmin=0, vmax=255)
    ax[0, 2].set_title('Tophat (Difference)', fontsize=16)
    ax[0, 2].axis('off')

    # A morphological closing removes all dark structures that cannot
    # contain a structuring element of a certain size.
    closed = closing(image, square(diameter))

    # Again we calculate the difference to the original image.
    tophat = closed - image

    ax[1, 1].imshow(closed, cmap='gray', aspect='equal',
                    vmin=0, vmax=255)
    ax[1, 1].set_title('Morphological Closing', fontsize=16)
    ax[1, 1].axis('off')

    ax[1, 2].imshow(dataset['vis_factor'] * tophat, cmap='gray',
                    aspect='equal', vmin=0, vmax=255)
    ax[1, 2].set_title('Tophat (Difference)', fontsize=16)
    ax[1, 2].axis('off')
    fig.suptitle(dataset['title'], fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

plt.show()


#####################################################################
# References
# ----------
# .. [1] Salembier, P., Oliveras, A., & Garrido, L. (1998). Antiextensive
#        Connected Operators for Image and Sequence Processing.
#        IEEE Transactions on Image Processing, 7(4), 555-570.
#        :DOI:`10.1109/83.663500`
# .. [2] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
#        Component Tree Computation Algorithms. IEEE Transactions on Image
#        Processing, 23(9), 3885-3895.
#        :DOI:`10.1109/TIP.2014.2336551`
# .. [3] Vincent L., Proc. "Grayscale area openings and closings,
#        their efficient implementation and applications",
#        EURASIP Workshop on Mathematical Morphology and its
#        Applications to Signal Processing, Barcelona, Spain, pp.22-27,
#        May 1993.
# .. [4] Walter, T., & Klein, J.-C. (2002). Automatic Detection of
#        Microaneurysms in Color Fundus Images of the Human Retina by Means
#        of the Bounding Box Closing. In A. Colosimo, P. Sirabella,
#        A. Giuliani (Eds.), Medical Data Analysis. Lecture Notes in Computer
#        Science, vol 2526, pp. 210-220. Springer Berlin Heidelberg.
#        :DOI:`10.1007/3-540-36104-9_23`
