"""
===============
Ridge operators
===============

Ridge filters can be used to detect continuous edges, such as neurites 
_[1], tubes _[2], vessels _[3], wrinkles _[4], rivers, and other rigdge-like 
structures. The present class of ridge filters relies on the eigenvalues of 
the Hessian matrix of image intensities to detect ridge structures where the 
intensity changes perpendicular but not along the structure.

.. [1] Meijering, E., Jacob, M., Sarria, J. C., Steiner, P., Hirling, H., 
Unser, M. (2004). Design and validation of a tool for neurite tracing and 
analysis in fluorescence microscopy images. Cytometry Part A, 58(2), 167-176.

.. [2] Sato, Y., Nakajima, S., Shiraga, N., Atsumi, H., Yoshida, S., Koller, 
T., ..., Kikinis, R. (1998). Three-dimensional multi-scale line filter for
segmentation and visualization of curvilinear structures in medical images.
Medical image analysis, 2(2), 143-168.

.. [3] Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A. (1998,
October). Multiscale vessel enhancement filtering. In International Conference
on Medical Image Computing and Computer-Assisted Intervention (pp. 130-137).
Springer Berlin Heidelberg.

.. [4] Ng, C. C., Yap, M. H., Costen, N., & Li, B. (2014, November). Automatic
wrinkle detection using hybrid Hessian filter. In Asian Conference on Computer
Vision (pp. 609-622). Springer International Publishing.
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
