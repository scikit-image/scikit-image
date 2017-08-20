"""
===============
Ridge operators
===============

Ridge filters can be used to detect continuous edges, such as vessels,
neurites, wrinkles, rivers, and other tube-like structures. The present
class of ridge filters relies on the eigenvalues of the Hessian matrix of
image intensities to detect ridge structures where the intensity changes
perpendicular but not along the structure.
"""

from skimage.data import page
from skimage.filters import identity, meijering, sato, frangi, hessian
import matplotlib.pyplot as plt

image = page()

cmap = plt.cm.gray

kwargs = {}
kwargs['scale_range'] = (1, 3)
kwargs['scale_step'] = 5

for i, black_ridges in enumerate([1, 0]):

    for j, func in enumerate([identity, meijering, sato, frangi, hessian]):

        kwargs['black_ridges'] = black_ridges

        plt.subplot(2, 5, 1 + 5 * i + j)

        plt.imshow(func(image, **kwargs), cmap=cmap, aspect='auto')

        if i == 0:
            plt.title(['Original image', 'Meijering neuriteness\n',
                       'Sato tubeness', 'Frangi vesselness\n',
                       'Hessian vesselness'][j])

        if j == 0:
            plt.ylabel('black_ridges = ' + str(bool(black_ridges)))

        plt.xticks([])
        plt.yticks([])
