"""
===============
Ridge operators
===============

Ridge filters can be used to detect ridge-like structures, such as neurites
[1]_, tubes [2]_, vessels [3]_, wrinkles [4]_ or rivers.

Different ridge filters may be suited for detecting different structures,
e.g., depending on contrast or noise level.

The present class of ridge filters relies on the eigenvalues of
the Hessian matrix of image intensities to detect ridge structures where the
intensity changes perpendicular but not along the structure.

Note that, due to edge effects, results for Meijering and Frangi filters
are cropped by 4 pixels on each edge to get a proper rendering.

References
----------

.. [1] Meijering, E., Jacob, M., Sarria, J. C., Steiner, P., Hirling, H.,
       Unser, M. (2004). Design and validation of a tool for neurite tracing
       and analysis in fluorescence microscopy images. Cytometry Part A, 58(2),
       167-176.
       :DOI:`10.1002/cyto.a.20022`

.. [2] Sato, Y., Nakajima, S., Shiraga, N., Atsumi, H., Yoshida, S.,
       Koller, T., ..., Kikinis, R. (1998). Three-dimensional multi-scale line
       filter for segmentation and visualization of curvilinear structures in
       medical images. Medical image analysis, 2(2), 143-168.
       :DOI:`10.1016/S1361-8415(98)80009-1`

.. [3] Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A. (1998,
       October). Multiscale vessel enhancement filtering. In International
       Conference on Medical Image Computing and Computer-Assisted Intervention
       (pp. 130-137). Springer Berlin Heidelberg.
       :DOI:`10.1007/BFb0056195`

.. [4] Ng, C. C., Yap, M. H., Costen, N., & Li, B. (2014, November). Automatic
       wrinkle detection using hybrid Hessian filter. In Asian Conference on
       Computer Vision (pp. 609-622). Springer International Publishing.
       :DOI:`10.1007/978-3-319-16811-1_40`
"""

from skimage import data
from skimage import color
from skimage.filters import meijering, sato, frangi, hessian
import matplotlib.pyplot as plt


def identity(image, **kwargs):
    """Return the original image, ignoring any kwargs."""
    return image


image = color.rgb2gray(data.retina())[300:700, 700:900]
cmap = plt.cm.gray

kwargs = {'sigmas': [1], 'mode': 'reflect'}

fig, axes = plt.subplots(2, 5)
for i, black_ridges in enumerate([1, 0]):
    for j, func in enumerate([identity, meijering, sato, frangi, hessian]):
        kwargs['black_ridges'] = black_ridges
        result = func(image, **kwargs)
        axes[i, j].imshow(result, cmap=cmap, aspect='auto')
        if i == 0:
            axes[i, j].set_title(['Original\nimage', 'Meijering\nneuriteness',
                                  'Sato\ntubeness', 'Frangi\nvesselness',
                                  'Hessian\nvesselness'][j])
        if j == 0:
            axes[i, j].set_ylabel('black_ridges = ' + str(bool(black_ridges)))
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

plt.tight_layout()
plt.show()
