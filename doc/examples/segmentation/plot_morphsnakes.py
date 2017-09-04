"""
====================
Morphological Snakes
====================

Morphological snakes [1]_ are a family of methods for image segmentation. Their
behavior is similar to that of active contours (for example, Geodesic Active
Contours [2]_ or Active Contours without Edges [3]_). However, Morphological
Snakes use morphological operators (such as dilation or erosion) over a binary
array instead of solving PDEs over a floating point array, which is the standard
approach for active contours. This makes Morphological Snakes faster and
numerically more stable than their traditional counterpart.

There are two Morphological Snakes methods available in this implementation:
Morphological Geodesic Active Contours (``morph_gac``) and Morphological Active
Contours without Edges (``morph_acwe``).

``morph_gac`` is suitable for images with visible contours, even when these
contours might be noisy, cluttered, or partially unclear. It requires, however,
that the image is preprocessed to highlight the contours. This can be done using
the function ``skimage.segmentation.morphsnakes.gborders``, although the user
might want to define its own version. Note that the quality of the segmentation
provided by ``morph_gac`` depends greatly on this preprocessing.

On the contrary, ``morph_acwe`` works well when the pixel values of the inside
and the outside regions of the object to segment have different averages. Unlike
``morph_gac``, ``morph_acwe`` does not require that the contours of the object
are well defined and it works over the original image without any preceding
processing. This makes ``morph_acwe`` easier to use and tune than ``morph_gac``.

References
----------

.. [1] A Morphological Approach to Curvature-based Evolution of Curves and
       Surfaces, Pablo Márquez-Neila, Luis Baumela and Luis Álvarez. In IEEE
       Transactions on Pattern Analysis and Machine Intelligence (PAMI),
       2014, DOI 10.1109/TPAMI.2013.106
.. [2] Geodesic Active Contours, Vicent Caselles, Ron Kimmel and Guillermo
       Sapiro. In International Journal of Computer Vision (IJCV), 1997,
       DOI:10.1023/A:1007979827043
.. [3] Active Contours without Edges, Tony Chan and Luminita Vese. In IEEE
       Transactions on Image Processing, 2001, DOI:10.1109/83.902291

"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import morph_acwe, checkerboard_level_set

image = img_as_float(data.coins())
init_ls = None

