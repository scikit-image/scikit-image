Announcement: scikits-image 0.8.0
=================================

We're happy to announce the 8th version of scikit-image!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

    http://scikit-image.org


New Features
------------

- New rank filter package with many new functions and a very fast underlying
  local histogram algorithm, especially for large structuring elements
  `skimage.filter.rank.*`
- New function for small object removal
  `skimage.morphology.remove_small_objects`
- New circular hough transformation `skimage.transform.hough_circle`
- New function to draw circle perimeter `skimage.draw.circle_perimeter` and
  ellipse perimeter `skimage.draw.ellipse_perimeter`
- New dense DAISY feature descriptor `skimage.feature.daisy`
- New bilateral filter `skimage.filter.denoise_bilateral`
- New faster TV denoising filter based on split-Bregman algorithm
  `skimage.filter.denoise_tv_bregman`
- New linear hough peak detection `skimage.transform.hough_peaks`
- New Scharr edge detection `skimage.filter.scharr`
- New geometric image scaling as convenience function
  `skimage.transform.rescale`
- New theme for documentation and website
- Faster median filter through vectorization `skimage.filter.median_filter`
- Grayscale images supported for SLIC segmentation
- Unified peak detection with more options `skimage.feature.peak_local_max`
- `imread` can read images via URL and knows more formats `skimage.io.imread`

Additionally, this release adds lots of bug fixes, new examples, and
performance enhancements.


Contributors to this release
----------------------------

This release was only possible due to the efforts of many contributors, both
new and old.

- Adam Ginsburg
- Anders Boesen Lindbo Larsen
- Andreas Mueller
- Christoph Gohlke
- Christos Psaltis
- Colin Lea
- François Boulogne
- Jan Margeta
- Johannes Schönberger
- Josh Warner (Mac)
- Juan Nunez-Iglesias
- Luis Pedro Coelho
- Marianne Corvellec
- Matt McCormick
- Nicolas Pinto
- Olivier Debeir
- Paul Ivanov
- Sergey Karayev
- Stefan van der Walt
- Steven Silvester
- Thouis (Ray) Jones
- Tony S Yu
