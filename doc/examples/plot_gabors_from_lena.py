"""
=======================================================
Gabors / Primary Visual Cortex "Simple Cells" from Lena
=======================================================

(under construction)

How to build a (bio-plausible) "sparse" dictionary (or 'codebook', or
'filterbank') for e.g. image classification without any fancy math and
with just standard python scientific librairies?

Please find below a short answer ;-)

This simple example shows how to get Gabor-like filters [1]_ using just
the famous Lena image. Gabor filters are good approximations of the
"Simple Cells" [2]_ receptive fields [3]_ found in the mammalian primary
visual cortex (V1) (for details, see e.g. the Nobel-prize winning work of Hubel
& Wiesel done in the 60s).

Here we use McQueen's 'kmeans' algorithm [4]_, as a simple bio-plausible
hebbian-like learning rule and we apply it (a) to patches of the
original Lena image (retinal projection), and (b) to patches of an
LGN-like [5]_ Lena image using a simple difference of gaussians (DoG)
approximation.

Enjoy ;-) And keep in mind that getting Gabors on natural image patches
is not rocket science.

.. [1] http://en.wikipedia.org/wiki/Gabor_filter
.. [2] http://en.wikipedia.org/wiki/Simple_cell
.. [3] http://en.wikipedia.org/wiki/Receptive_field
.. [4] http://en.wikipedia.org/wiki/K-means_clustering
.. [5] http://en.wikipedia.org/wiki/Lateral_geniculate_nucleus

References
----------
D. H. Hubel and T. N. Wiesel Receptive Fields of Single Neurones in the
Cat's Striate Cortex J. Physiol. pp. 574-591 (148) 1959

D. H. Hubel and T. N. Wiesel Receptive Fields, Binocular Interaction and
Functional Architecture in the Cat's Visual Cortex J. Physiol. 160 pp.
106-154 1962
"""

import numpy as np
from scipy import misc
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt

from skimage.util.shape import view_as_windows
from skimage.util.montage import montage2d
from scipy import ndimage as ndi

np.random.seed(42)

patch_shape = 8, 8
n_filters = 49

lena = misc.lena() / 255.

# -- filterbank1 on original Lena
patches1 = view_as_windows(lena, patch_shape)
patches1 = patches1.reshape(-1, patch_shape[0] * patch_shape[1])[::8]
fb1, _ = kmeans2(patches1, n_filters, minit='points')
fb1 = fb1.reshape((-1,) + patch_shape)
fb1_montage = montage2d(fb1)

# -- filterbank2 LGN-like Lena
lena_dog = ndi.gaussian_filter(lena, .5) - ndi.gaussian_filter(lena, 1)
patches2 = view_as_windows(lena_dog, patch_shape)
patches2 = patches2.reshape(-1, patch_shape[0] * patch_shape[1])[::8]
fb2, _ = kmeans2(patches2, n_filters, minit='points')
fb2 = fb2.reshape((-1,) + patch_shape)
fb2_montage = montage2d(fb2)

# --
plt.figure(figsize=(9, 3))


plt.subplot(2, 2, 1)
plt.imshow(lena, cmap=plt.cm.gray)
plt.axis('off')
plt.title("Lena (original)")

plt.subplot(2, 2, 2)
plt.imshow(fb1_montage, cmap=plt.cm.gray)
plt.axis('off')
plt.title("K-means filterbank (codebook) on Lena (original)")

plt.subplot(2, 2, 3)
plt.imshow(lena_dog, cmap=plt.cm.gray)
plt.axis('off')
plt.title("Lena (LGN-like DoG)")

plt.subplot(2, 2, 4)
plt.imshow(fb2_montage, cmap=plt.cm.gray)
plt.axis('off')
plt.title("K-means filterbank (codebook) on Lena (LGN-like DoG)")

plt.show()
