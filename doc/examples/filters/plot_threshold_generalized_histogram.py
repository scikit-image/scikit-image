"""
================================================================
Generalized Histogram Thresholding (GHT)
================================================================

Thresholding is used to create a binary image from a grayscale image [1]_.

There are number of histograms based methods such as Otsu's method,
Minimum Error Thresholding and weighted percentile thresholding.
GHT provides an generalized framework to cover above 3 standard methods as special cases

GHT works by performing approximate maximum of a posteriori estimation
of a mixture of Gaussians with appropriate priors [2]_.

One drawback of this approach is that user has to control 4 parameters
and find the best combination suited for the problem at hand. However The algorithm provides
Thoritical gaurantees that the said solution with provide better or atleast same quality of results
as compare to other 3 Methods.



.. [1] https://en.wikipedia.org/wiki/Thresholding_%28image_processing%29

.. [2] A Generalization of Otsu's Method and Minimum Error Thresholding
         Jonathan T. Barron, ECCV, 2020

"""

import matplotlib.pyplot as plt

from skimage import data
from skimage.exposure import histogram
from skimage.filters import theshold_generalized_histogram, threshold_otsu



image = data.page()

titles = ["Original", "A Good Threshold", "Otsu's Method", "MET", "Percentile",
			"A random threshold", "Otsu explict implementation"
			]

# (nu, tau, kappa, omega,)
hyperparameters = [
			(1e-30, 12.589254, 1e+30, 0.11), # a good Threshold
			( 1e30, 1e-30, 1e-30, 0.5),  # otsu
			(1e-30, 1.0, 1e-30, 0.5),   # met
			(1e-30, 1.0, 1e30, 0.5), #percentile
			(73533, 0.28867513459481287, 73533.0, 0.5) # a random
			]
# create histogram
counts, bin_centers = histogram(image.ravel(), 256, source_range="image")

#creating images
imgs = [image]
for nu, tau, kappa, omega in hyperparameters:
	t, _ = theshold_generalized_histogram(counts, bin_centers,
			nu, tau, kappa, omega)
	imgs.append(image>=t)
imgs.append(image > threshold_otsu(image))

fig, axes = plt.subplots(4, 2, figsize=(8, 8))
ax = axes.ravel()

for i in range(0, len(imgs)):
	ax[i].imshow(imgs[i], cmap=plt.cm.gray)
	ax[i].set_title(titles[i])
	ax[i].axis('off')

plt.tight_layout()
plt.show()
