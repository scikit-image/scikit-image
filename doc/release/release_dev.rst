Announcement: scikit-image 0.X.0
================================

We're happy to announce the release of scikit-image v0.X.0!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

https://scikit-image.org

Starting from this release, scikit-image will follow the spirit of the recently
introduced numpy deprecation policy -- NEP 29
(https://github.com/numpy/numpy/blob/master/doc/neps/nep-0029-deprecation_policy.rst). 
Accordingly, scikit-image 0.16 drops the support for Python 3.5.
This release of scikit-image officially supports Python 3.6 and 3.7.

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


Contributors to this release
----------------------------
