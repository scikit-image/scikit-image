Announcement: scikit-image 0.X.0
================================

We're happy to announce the release of scikit-image v0.X.0!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

https://scikit-image.org

Starting from release 0.16, scikit-image follows the spirit of the recently
introduced NumPy deprecation policy -- NEP 29
(https://github.com/numpy/numpy/blob/master/doc/neps/nep-0029-deprecation_policy.rst). 
This release of scikit-image officially supports Python 3.6, 3.7, and
3.8.

New Features
------------


Improvements
------------


API Changes
-----------
- When used with floating point inputs, ``denoise_wavelet`` no longer rescales
  the range of the data or clips the output to the range [0, 1] or [-1, 1].
  For non-float inputs, rescaling and clipping still occurs as in prior
  releases (although with a bugfix related to the scaling of ``sigma``).
- For 2D input, edge filters (Sobel, Scharr, Prewitt, Roberts, and Farid)
  no longer set the boundary pixels to 0 when a mask is not supplied. This was
  changed because the boundary mode for `scipy.ndimage.convolve` is now
  ``'reflect'``, which allows meaningful values at the borders for these
  filters. To retain the old behavior, pass
  ``mask=np.ones(image.shape, dtype=bool)`` (#4347)


Bugfixes
--------
- ``denoise_wavelet``: For user-supplied `sigma`, if the input image gets
  rescaled via ``img_as_float``, the same scaling will be applied to `sigma` to
  preserve the relative scale of the noise estimate. To restore the old,
  behaviour, the user can manually specify ``rescale_sigma=False``.


Deprecations
------------
- Parameter ``inplace`` in skimage.morphology.flood_fill has been deprecated
  in favor of ``in_place`` and will be removed in version scikit-image 0.19.0.
- ``skimage.segmentation.circle_level_set`` has been deprecated and will be
  removed in 0.19. Use ``skimage.segmentation.disk_level_set`` instead.
- ``skimage.draw.circle`` has been deprecated and will be removed in 0.19.
  Use ``skimage.draw.disk`` instead.


Contributors to this release
----------------------------
