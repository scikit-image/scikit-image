Announcement: scikit-image 0.X.0
================================

We're happy to announce the release of scikit-image v0.X.0!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

http://scikit-image.org


New Features
------------
- manual segmentation with matplotlib (#2584)
- hysteresis thresholding in filters (#2665)
- lookfor function (#2713)
- montage function (#2626)
- 2D and 3D segmentation with morphological snakes (#2791)
- Haar-like feature (#2848)


Improvements
------------
- VisuShrink method for wavelet denoising (#2470)
- ``skimage.transform.resize`` and ``skimage.transform.rescale`` have a new
  ``anti_aliasing`` option to avoid aliasing artifacts when down-sampling
  images (#2802)


API Changes
-----------
- ``skimage.util.montage.montage2d`` is now available as
  ``skimage.util.montage2d``.


Deprecations
------------
- ``skimage.util.montage2d`` is deprecated and will be removed in 0.15.
  Use ``skimage.util.montage`` instead.
- ``skimage.novice`` is deprecated and will be removed in 0.16.
- ``skimage.transform.resize`` and ``skimage.transform.rescale`` have a new
  ``anti_aliasing`` option that avoids aliasing artifacts when down-sampling
  images. This option will be enabled by default in 0.15.


Contributors to this release
----------------------------
