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
- Frangi (vesselness) filter for 3D data
- Meijering (neuriteness) filter for nD data
- Sato (tubeness) filter for 2D and 3D data


Improvements
------------
- VisuShrink method for wavelet denoising (#2470)
- Correct bright ridge detection for Frangi filter (#2700)


API Changes
-----------
- ``skimage.util.montage.montage2d`` is now available as ``skimage.util.montage2d``.


Deprecations
------------
- ``skimage.util.montage2d`` is deprecated and will be removed in 0.15.
  Use ``skimage.util.montage`` instead.


Contributors to this release
----------------------------
