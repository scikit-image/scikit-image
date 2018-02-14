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
- nD support for image moments (#2603)
- inertia tensor and its eigenvalues can now be computed outside of
  regionprops; available in ``skimage.measure.inertia_tensor`` (#2603)
- cycle-spinning function for approximating shift-invariance by averaging
  results from a series of spatial shifts (#2647)
- Haar-like feature (#2848)
- subset of LFW database (#2905)


Improvements
------------
- VisuShrink method for wavelet denoising (#2470)
- ``skimage.transform.resize`` and ``skimage.transform.rescale`` have a new
  ``anti_aliasing`` option to avoid aliasing artifacts when down-sampling
  images (#2802)
- Support for multichannel images for ``skimage.feature.hog`` (#2870)
- Non-local means denoising (``denoise_nl_means``) has a new optional
  parameter, `sigma`, that can be used to specify the noise standard deviation.
  This enables noise-robust patch distance estimation. (#2890)


API Changes
-----------
- ``skimage.util.montage.montage2d`` is now available as
  ``skimage.util.montage2d``.
- ``skimage.morphology.binary_erosion`` now uses ``True`` as border
  value, and is now consistent with ``skimage.morphology.erosion``.

Deprecations
------------
- ``skimage.util.montage2d`` is deprecated and will be removed in 0.15.
  Use ``skimage.util.montage`` instead.
- ``skimage.novice`` is deprecated and will be removed in 0.16.
- ``skimage.transform.resize`` and ``skimage.transform.rescale`` have a new
  ``anti_aliasing`` option that avoids aliasing artifacts when down-sampling
  images. This option will be enabled by default in 0.15.
- ``regionprops`` will use row-column coordinates in 0.16. You can start
  using them now with ``regionprops(..., coordinates='rc')``. You can silence
  warning messages, and retain the old behavior, with
  ``regionprops(..., coordinates='xy')``. However, that option will go away
  in 0.16 and result in an error. This change has a number of consequences.
  Specifically, the "orientation" region property will measure the
  anticlockwise angle from a *vertical* line, i.e. from the vector (1, 0) in
  row-column coordinates.
- ``skimage.morphology.remove_small_holes`` ``min_size`` argument is deprecated
  and will be removed in 0.16. Use ``area_threshold`` instead.


Contributors to this release
----------------------------
